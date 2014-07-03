// Copyright 2009 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// Garbage collector (GC).
//
// GC is:
// - mark&sweep
// - mostly precise (with the exception of some C-allocated objects, assembly frames/arguments, etc)
// - parallel (up to MaxGcproc threads)
// - partially concurrent (mark is stop-the-world, while sweep is concurrent)
// - non-moving/non-compacting
// - full (non-partial)
//
// GC rate.
// Next GC is after we've allocated an extra amount of memory proportional to
// the amount already in use. The proportion is controlled by GOGC environment variable
// (100 by default). If GOGC=100 and we're using 4M, we'll GC again when we get to 8M
// (this mark is tracked in next_gc variable). This keeps the GC cost in linear
// proportion to the allocation cost. Adjusting GOGC just changes the linear constant
// (and also the amount of extra memory used).
//
// Concurrent sweep.
// The sweep phase proceeds concurrently with normal program execution.
// The heap is swept span-by-span both lazily (when a goroutine needs another span)
// and concurrently in a background goroutine (this helps programs that are not CPU bound).
// However, at the end of the stop-the-world GC phase we don't know the size of the live heap,
// and so next_gc calculation is tricky and happens as follows.
// At the end of the stop-the-world phase next_gc is conservatively set based on total
// heap size; all spans are marked as "needs sweeping".
// Whenever a span is swept, next_gc is decremented by GOGC*newly_freed_memory.
// The background sweeper goroutine simply sweeps spans one-by-one bringing next_gc
// closer to the target value. However, this is not enough to avoid over-allocating memory.
// Consider that a goroutine wants to allocate a new span for a large object and
// there are no free swept spans, but there are small-object unswept spans.
// If the goroutine naively allocates a new span, it can surpass the yet-unknown
// target next_gc value. In order to prevent such cases (1) when a goroutine needs
// to allocate a new small-object span, it sweeps small-object spans for the same
// object size until it frees at least one object; (2) when a goroutine needs to
// allocate large-object span from heap, it sweeps spans until it frees at least
// that many pages into heap. Together these two measures ensure that we don't surpass
// target next_gc value by a large margin. There is an exception: if a goroutine sweeps
// and frees two nonadjacent one-page spans to the heap, it will allocate a new two-page span,
// but there can still be other one-page unswept spans which could be combined into a two-page span.
// It's critical to ensure that no operations proceed on unswept spans (that would corrupt
// mark bits in GC bitmap). During GC all mcaches are flushed into the central cache,
// so they are empty. When a goroutine grabs a new span into mcache, it sweeps it.
// When a goroutine explicitly frees an object or sets a finalizer, it ensures that
// the span is swept (either by sweeping it, or by waiting for the concurrent sweep to finish).
// The finalizer goroutine is kicked off only when all spans are swept.
// When the next GC starts, it sweeps all not-yet-swept spans (if any).

#include "runtime.h"
#include "arch_GOARCH.h"
#include "malloc.h"
#include "stack.h"
#include "mgc0.h"
#include "chan.h"
#include "race.h"
#include "type.h"
#include "typekind.h"
#include "funcdata.h"
#include "../../cmd/ld/textflag.h"

enum {
	Debug = 0,
	CollectStats = 0,
	ConcurrentSweep = 1,

	WorkbufSize	= 16*1024,
	FinBlockSize	= 4*1024,

	handoffThreshold = 4,
	IntermediateBufferCapacity = 64,

	// Bits in type information
	PRECISE = 1,
	LOOP = 2,
	PC_BITS = PRECISE | LOOP,

	RootData	= 0,
	RootBss		= 1,
	RootFinalizers	= 2,
	RootSpanTypes	= 3,
	RootFlushCaches = 4,
	RootCount	= 5,
};

#define GcpercentUnknown (-2)

// Initialized from $GOGC.  GOGC=off means no gc.
static int32 gcpercent = GcpercentUnknown;

static FuncVal* poolcleanup;

void
sync·runtime_registerPoolCleanup(FuncVal *f)
{
	poolcleanup = f;
}

static void
clearpools(void)
{
	P *p, **pp;
	MCache *c;
	int32 i;

	// clear sync.Pool's
	if(poolcleanup != nil)
		reflect·call(poolcleanup, nil, 0, 0);

	for(pp=runtime·allp; p=*pp; pp++) {
		// clear tinyalloc pool
		c = p->mcache;
		if(c != nil) {
			c->tiny = nil;
			c->tinysize = 0;
		}
		// clear defer pools
		for(i=0; i<nelem(p->deferpool); i++)
			p->deferpool[i] = nil;
	}
}

// Holding worldsema grants an M the right to try to stop the world.
// The procedure is:
//
//	runtime·semacquire(&runtime·worldsema);
//	m->gcing = 1;
//	runtime·stoptheworld();
//
//	... do stuff ...
//
//	m->gcing = 0;
//	runtime·semrelease(&runtime·worldsema);
//	runtime·starttheworld();
//
uint32 runtime·worldsema = 1;

typedef struct Obj Obj;
struct Obj
{
	byte	*p;	// data pointer
	uintptr	n;	// size of data in bytes
	uintptr	ti;	// type info
};

typedef struct Workbuf Workbuf;
struct Workbuf
{
#define SIZE (WorkbufSize-sizeof(LFNode)-sizeof(uintptr))
	LFNode  node; // must be first
	uintptr nobj;
	Obj     obj[SIZE/sizeof(Obj) - 1];
	uint8   _padding[SIZE%sizeof(Obj) + sizeof(Obj)];
#undef SIZE
};

typedef struct Finalizer Finalizer;
struct Finalizer
{
	FuncVal *fn;
	void *arg;
	uintptr nret;
	Type *fint;
	PtrType *ot;
};

typedef struct FinBlock FinBlock;
struct FinBlock
{
	FinBlock *alllink;
	FinBlock *next;
	int32 cnt;
	int32 cap;
	Finalizer fin[1];
};

extern byte data[];
extern byte edata[];
extern byte bss[];
extern byte ebss[];

extern byte gcdata[];
extern byte gcbss[];

static Lock	finlock;	// protects the following variables
static FinBlock	*finq;		// list of finalizers that are to be executed
static FinBlock	*finc;		// cache of free blocks
static FinBlock	*allfin;	// list of all blocks
bool	runtime·fingwait;
bool	runtime·fingwake;

static Lock	gclock;
static G*	fing;

static void	runfinq(void);
static void	bgsweep(void);
static Workbuf* getempty(Workbuf*);
static Workbuf* getfull(Workbuf*);
static void	putempty(Workbuf*);
static Workbuf* handoff(Workbuf*);
static void	gchelperstart(void);
static void	flushallmcaches(void);
static bool	scanframe(Stkframe *frame, void *wbufp);
static void	addstackroots(G *gp, Workbuf **wbufp);

static FuncVal runfinqv = {runfinq};
static FuncVal bgsweepv = {bgsweep};

static struct {
	uint64	full;  // lock-free list of full blocks
	uint64	empty; // lock-free list of empty blocks
	byte	pad0[CacheLineSize]; // prevents false-sharing between full/empty and nproc/nwait
	uint32	nproc;
	int64	tstart;
	volatile uint32	nwait;
	volatile uint32	ndone;
	Note	alldone;
	ParFor	*markfor;

	Lock;
	byte	*chunk;
	uintptr	nchunk;
} work;

enum {
	GC_DEFAULT_PTR = GC_NUM_INSTR,
	GC_CHAN,

	GC_NUM_INSTR2
};

static struct {
	struct {
		uint64 sum;
		uint64 cnt;
	} ptr;
	uint64 nbytes;
	struct {
		uint64 sum;
		uint64 cnt;
		uint64 notype;
		uint64 typelookup;
	} obj;
	uint64 rescan;
	uint64 rescanbytes;
	uint64 instr[GC_NUM_INSTR2];
	uint64 putempty;
	uint64 getfull;
	struct {
		uint64 foundbit;
		uint64 foundword;
		uint64 foundspan;
	} flushptrbuf;
	struct {
		uint64 foundbit;
		uint64 foundword;
		uint64 foundspan;
	} markonly;
	uint32 nbgsweep;
	uint32 npausesweep;
} gcstats;

// markonly marks an object. It returns true if the object
// has been marked by this function, false otherwise.
// This function doesn't append the object to any buffer.
static bool
markonly(void *obj)
{
	byte *p;
	uintptr *bitp, bits, shift, x, xbits, off;
	MSpan *s;
	PageID k;

	// Words outside the arena cannot be pointers.
	if(obj < runtime·mheap.arena_start || obj >= runtime·mheap.arena_used)
		return false;

	// obj may be a pointer to a live object.
	// Try to find the beginning of the object.

	// Round down to word boundary.
	obj = (void*)((uintptr)obj & ~((uintptr)PtrSize-1));

	// Find bits for this word.
	off = (uintptr*)obj - (uintptr*)runtime·mheap.arena_start;
	bitp = (uintptr*)runtime·mheap.arena_start - off/wordsPerBitmapWord - 1;
	shift = off % wordsPerBitmapWord;
	xbits = *bitp;
	bits = xbits >> shift;

	// Pointing at the beginning of a block?
	if((bits & (bitAllocated|bitBlockBoundary)) != 0) {
		if(CollectStats)
			runtime·xadd64(&gcstats.markonly.foundbit, 1);
		goto found;
	}

	// Otherwise consult span table to find beginning.
	// (Manually inlined copy of MHeap_LookupMaybe.)
	k = (uintptr)obj>>PageShift;
	x = k;
	x -= (uintptr)runtime·mheap.arena_start>>PageShift;
	s = runtime·mheap.spans[x];
	if(s == nil || k < s->start || obj >= s->limit || s->state != MSpanInUse)
		return false;
	p = (byte*)((uintptr)s->start<<PageShift);
	if(s->sizeclass == 0) {
		obj = p;
	} else {
		uintptr size = s->elemsize;
		int32 i = ((byte*)obj - p)/size;
		obj = p+i*size;
	}

	// Now that we know the object header, reload bits.
	off = (uintptr*)obj - (uintptr*)runtime·mheap.arena_start;
	bitp = (uintptr*)runtime·mheap.arena_start - off/wordsPerBitmapWord - 1;
	shift = off % wordsPerBitmapWord;
	xbits = *bitp;
	bits = xbits >> shift;
	if(CollectStats)
		runtime·xadd64(&gcstats.markonly.foundspan, 1);

found:
	// Now we have bits, bitp, and shift correct for
	// obj pointing at the base of the object.
	// Only care about allocated and not marked.
	if((bits & (bitAllocated|bitMarked)) != bitAllocated)
		return false;
	if(work.nproc == 1)
		*bitp |= bitMarked<<shift;
	else {
		for(;;) {
			x = *bitp;
			if(x & (bitMarked<<shift))
				return false;
			if(runtime·casp((void**)bitp, (void*)x, (void*)(x|(bitMarked<<shift))))
				break;
		}
	}

	// The object is now marked
	return true;
}

// PtrTarget is a structure used by intermediate buffers.
// The intermediate buffers hold GC data before it
// is moved/flushed to the work buffer (Workbuf).
// The size of an intermediate buffer is very small,
// such as 32 or 64 elements.
typedef struct PtrTarget PtrTarget;
struct PtrTarget
{
	void *p;
	uintptr ti;
};

typedef	struct Scanbuf Scanbuf;
struct	Scanbuf
{
	struct {
		PtrTarget *begin;
		PtrTarget *end;
		PtrTarget *pos;
	} ptr;
	struct {
		Obj *begin;
		Obj *end;
		Obj *pos;
	} obj;
	Workbuf *wbuf;
	Obj *wp;
	uintptr nobj;
};

typedef struct BufferList BufferList;
struct BufferList
{
	PtrTarget ptrtarget[IntermediateBufferCapacity];
	Obj obj[IntermediateBufferCapacity];
	uint32 busy;
	byte pad[CacheLineSize];
};
#pragma dataflag NOPTR
static BufferList bufferList[MaxGcproc];

static Type *itabtype;

static void enqueue(Obj obj, Workbuf **_wbuf, Obj **_wp, uintptr *_nobj);

// flushptrbuf moves data from the PtrTarget buffer to the work buffer.
// The PtrTarget buffer contains blocks irrespective of whether the blocks have been marked or scanned,
// while the work buffer contains blocks which have been marked
// and are prepared to be scanned by the garbage collector.
//
// _wp, _wbuf, _nobj are input/output parameters and are specifying the work buffer.
//
// A simplified drawing explaining how the todo-list moves from a structure to another:
//
//     scanblock
//  (find pointers)
//    Obj ------> PtrTarget (pointer targets)
//     ↑          |
//     |          |
//     `----------'
//     flushptrbuf
//  (find block start, mark and enqueue)
static void
flushptrbuf(Scanbuf *sbuf)
{
	byte *p, *arena_start, *obj;
	uintptr size, *bitp, bits, shift, x, xbits, off, nobj, ti, n;
	MSpan *s;
	PageID k;
	Obj *wp;
	Workbuf *wbuf;
	PtrTarget *ptrbuf;
	PtrTarget *ptrbuf_end;

	arena_start = runtime·mheap.arena_start;

	wp = sbuf->wp;
	wbuf = sbuf->wbuf;
	nobj = sbuf->nobj;

	ptrbuf = sbuf->ptr.begin;
	ptrbuf_end = sbuf->ptr.pos;
	n = ptrbuf_end - sbuf->ptr.begin;
	sbuf->ptr.pos = sbuf->ptr.begin;

	if(CollectStats) {
		runtime·xadd64(&gcstats.ptr.sum, n);
		runtime·xadd64(&gcstats.ptr.cnt, 1);
	}

	// If buffer is nearly full, get a new one.
	if(wbuf == nil || nobj+n >= nelem(wbuf->obj)) {
		if(wbuf != nil)
			wbuf->nobj = nobj;
		wbuf = getempty(wbuf);
		wp = wbuf->obj;
		nobj = 0;

		if(n >= nelem(wbuf->obj))
			runtime·throw("ptrbuf has to be smaller than WorkBuf");
	}

	while(ptrbuf < ptrbuf_end) {
		obj = ptrbuf->p;
		ti = ptrbuf->ti;
		ptrbuf++;

		// obj belongs to interval [mheap.arena_start, mheap.arena_used).
		if(Debug > 1) {
			if(obj < runtime·mheap.arena_start || obj >= runtime·mheap.arena_used)
				runtime·throw("object is outside of mheap");
		}

		// obj may be a pointer to a live object.
		// Try to find the beginning of the object.

		// Round down to word boundary.
		if(((uintptr)obj & ((uintptr)PtrSize-1)) != 0) {
			obj = (void*)((uintptr)obj & ~((uintptr)PtrSize-1));
			ti = 0;
		}

		// Find bits for this word.
		off = (uintptr*)obj - (uintptr*)arena_start;
		bitp = (uintptr*)arena_start - off/wordsPerBitmapWord - 1;
		shift = off % wordsPerBitmapWord;
		xbits = *bitp;
		bits = xbits >> shift;

		// Pointing at the beginning of a block?
		if((bits & (bitAllocated|bitBlockBoundary)) != 0) {
			if(CollectStats)
				runtime·xadd64(&gcstats.flushptrbuf.foundbit, 1);
			goto found;
		}

		ti = 0;

		// Otherwise consult span table to find beginning.
		// (Manually inlined copy of MHeap_LookupMaybe.)
		k = (uintptr)obj>>PageShift;
		x = k;
		x -= (uintptr)arena_start>>PageShift;
		s = runtime·mheap.spans[x];
		if(s == nil || k < s->start || obj >= s->limit || s->state != MSpanInUse)
			continue;
		p = (byte*)((uintptr)s->start<<PageShift);
		if(s->sizeclass == 0) {
			obj = p;
		} else {
			size = s->elemsize;
			int32 i = ((byte*)obj - p)/size;
			obj = p+i*size;
		}

		// Now that we know the object header, reload bits.
		off = (uintptr*)obj - (uintptr*)arena_start;
		bitp = (uintptr*)arena_start - off/wordsPerBitmapWord - 1;
		shift = off % wordsPerBitmapWord;
		xbits = *bitp;
		bits = xbits >> shift;
		if(CollectStats)
			runtime·xadd64(&gcstats.flushptrbuf.foundspan, 1);

	found:
		// Now we have bits, bitp, and shift correct for
		// obj pointing at the base of the object.
		// Only care about allocated and not marked.
		if((bits & (bitAllocated|bitMarked)) != bitAllocated)
			continue;
		if(work.nproc == 1)
			*bitp |= bitMarked<<shift;
		else {
			for(;;) {
				x = *bitp;
				if(x & (bitMarked<<shift))
					goto continue_obj;
				if(runtime·casp((void**)bitp, (void*)x, (void*)(x|(bitMarked<<shift))))
					break;
			}
		}

		// If object has no pointers, don't need to scan further.
		if((bits & bitScan) == 0)
			continue;

		// Ask span about size class.
		// (Manually inlined copy of MHeap_Lookup.)
		x = (uintptr)obj >> PageShift;
		x -= (uintptr)arena_start>>PageShift;
		s = runtime·mheap.spans[x];

		PREFETCH(obj);

		*wp = (Obj){obj, s->elemsize, ti};
		wp++;
		nobj++;
	continue_obj:;
	}

	// If another proc wants a pointer, give it some.
	if(work.nwait > 0 && nobj > handoffThreshold && work.full == 0) {
		wbuf->nobj = nobj;
		wbuf = handoff(wbuf);
		nobj = wbuf->nobj;
		wp = wbuf->obj + nobj;
	}

	sbuf->wp = wp;
	sbuf->wbuf = wbuf;
	sbuf->nobj = nobj;
}

static void
flushobjbuf(Scanbuf *sbuf)
{
	uintptr nobj, off;
	Obj *wp, obj;
	Workbuf *wbuf;
	Obj *objbuf;
	Obj *objbuf_end;

	wp = sbuf->wp;
	wbuf = sbuf->wbuf;
	nobj = sbuf->nobj;

	objbuf = sbuf->obj.begin;
	objbuf_end = sbuf->obj.pos;
	sbuf->obj.pos = sbuf->obj.begin;

	while(objbuf < objbuf_end) {
		obj = *objbuf++;

		// Align obj.b to a word boundary.
		off = (uintptr)obj.p & (PtrSize-1);
		if(off != 0) {
			obj.p += PtrSize - off;
			obj.n -= PtrSize - off;
			obj.ti = 0;
		}

		if(obj.p == nil || obj.n == 0)
			continue;

		// If buffer is full, get a new one.
		if(wbuf == nil || nobj >= nelem(wbuf->obj)) {
			if(wbuf != nil)
				wbuf->nobj = nobj;
			wbuf = getempty(wbuf);
			wp = wbuf->obj;
			nobj = 0;
		}

		*wp = obj;
		wp++;
		nobj++;
	}

	// If another proc wants a pointer, give it some.
	if(work.nwait > 0 && nobj > handoffThreshold && work.full == 0) {
		wbuf->nobj = nobj;
		wbuf = handoff(wbuf);
		nobj = wbuf->nobj;
		wp = wbuf->obj + nobj;
	}

	sbuf->wp = wp;
	sbuf->wbuf = wbuf;
	sbuf->nobj = nobj;
}

// Program that scans the whole block and treats every block element as a potential pointer
static uintptr defaultProg[2] = {PtrSize, GC_DEFAULT_PTR};

// Hchan program
static uintptr chanProg[2] = {0, GC_CHAN};

// Local variables of a program fragment or loop
typedef struct Frame Frame;
struct Frame {
	uintptr count, elemsize, b;
	uintptr *loop_or_ret;
};

// Sanity check for the derived type info objti.
static void
checkptr(void *obj, uintptr objti)
{
	uintptr *pc1, *pc2, type, tisize, i, j, x;
	byte *objstart;
	Type *t;
	MSpan *s;

	if(!Debug)
		runtime·throw("checkptr is debug only");

	if(obj < runtime·mheap.arena_start || obj >= runtime·mheap.arena_used)
		return;
	type = runtime·gettype(obj);
	t = (Type*)(type & ~(uintptr)(PtrSize-1));
	if(t == nil)
		return;
	x = (uintptr)obj >> PageShift;
	x -= (uintptr)(runtime·mheap.arena_start)>>PageShift;
	s = runtime·mheap.spans[x];
	objstart = (byte*)((uintptr)s->start<<PageShift);
	if(s->sizeclass != 0) {
		i = ((byte*)obj - objstart)/s->elemsize;
		objstart += i*s->elemsize;
	}
	tisize = *(uintptr*)objti;
	// Sanity check for object size: it should fit into the memory block.
	if((byte*)obj + tisize > objstart + s->elemsize) {
		runtime·printf("object of type '%S' at %p/%p does not fit in block %p/%p\n",
			       *t->string, obj, tisize, objstart, s->elemsize);
		runtime·throw("invalid gc type info");
	}
	if(obj != objstart)
		return;
	// If obj points to the beginning of the memory block,
	// check type info as well.
	if(t->string == nil ||
		// Gob allocates unsafe pointers for indirection.
		(runtime·strcmp(t->string->str, (byte*)"unsafe.Pointer") &&
		// Runtime and gc think differently about closures.
		runtime·strstr(t->string->str, (byte*)"struct { F uintptr") != t->string->str)) {
		pc1 = (uintptr*)objti;
		pc2 = (uintptr*)t->gc;
		// A simple best-effort check until first GC_END.
		for(j = 1; pc1[j] != GC_END && pc2[j] != GC_END; j++) {
			if(pc1[j] != pc2[j]) {
				runtime·printf("invalid gc type info for '%s', type info %p [%d]=%p, block info %p [%d]=%p\n",
					       t->string ? (int8*)t->string->str : (int8*)"?", pc1, (int32)j, pc1[j], pc2, (int32)j, pc2[j]);
				runtime·throw("invalid gc type info");
			}
		}
	}
}					

// scanblock scans a block of n bytes starting at pointer b for references
// to other objects, scanning any it finds recursively until there are no
// unscanned objects left.  Instead of using an explicit recursion, it keeps
// a work list in the Workbuf* structures and loops in the main function
// body.  Keeping an explicit work list is easier on the stack allocator and
// more efficient.
static void
scanblock(Workbuf *wbuf, bool keepworking)
{
	byte *b, *arena_start, *arena_used;
	uintptr n, i, end_b, elemsize, size, ti, objti, count, type, nobj;
	uintptr *pc, precise_type, nominal_size;
	uintptr *chan_ret, chancap;
	void *obj;
	Type *t, *et;
	Slice *sliceptr;
	String *stringptr;
	Frame *stack_ptr, stack_top, stack[GC_STACK_CAPACITY+4];
	BufferList *scanbuffers;
	Scanbuf sbuf;
	Eface *eface;
	Iface *iface;
	Hchan *chan;
	ChanType *chantype;
	Obj *wp;

	if(sizeof(Workbuf) % WorkbufSize != 0)
		runtime·throw("scanblock: size of Workbuf is suboptimal");

	// Memory arena parameters.
	arena_start = runtime·mheap.arena_start;
	arena_used = runtime·mheap.arena_used;

	stack_ptr = stack+nelem(stack)-1;

	precise_type = false;
	nominal_size = 0;

	if(wbuf) {
		nobj = wbuf->nobj;
		wp = &wbuf->obj[nobj];
	} else {
		nobj = 0;
		wp = nil;
	}

	// Initialize sbuf
	scanbuffers = &bufferList[g->m->helpgc];

	sbuf.ptr.begin = sbuf.ptr.pos = &scanbuffers->ptrtarget[0];
	sbuf.ptr.end = sbuf.ptr.begin + nelem(scanbuffers->ptrtarget);

	sbuf.obj.begin = sbuf.obj.pos = &scanbuffers->obj[0];
	sbuf.obj.end = sbuf.obj.begin + nelem(scanbuffers->obj);

	sbuf.wbuf = wbuf;
	sbuf.wp = wp;
	sbuf.nobj = nobj;

	// (Silence the compiler)
	chan = nil;
	chantype = nil;
	chan_ret = nil;

	goto next_block;

	for(;;) {
		// Each iteration scans the block b of length n, queueing pointers in
		// the work buffer.

		if(CollectStats) {
			runtime·xadd64(&gcstats.nbytes, n);
			runtime·xadd64(&gcstats.obj.sum, sbuf.nobj);
			runtime·xadd64(&gcstats.obj.cnt, 1);
		}

		if(ti != 0) {
			if(Debug > 1) {
				runtime·printf("scanblock %p %D ti %p\n", b, (int64)n, ti);
			}
			pc = (uintptr*)(ti & ~(uintptr)PC_BITS);
			precise_type = (ti & PRECISE);
			stack_top.elemsize = pc[0];
			if(!precise_type)
				nominal_size = pc[0];
			if(ti & LOOP) {
				stack_top.count = 0;	// 0 means an infinite number of iterations
				stack_top.loop_or_ret = pc+1;
			} else {
				stack_top.count = 1;
			}
			if(Debug) {
				// Simple sanity check for provided type info ti:
				// The declared size of the object must be not larger than the actual size
				// (it can be smaller due to inferior pointers).
				// It's difficult to make a comprehensive check due to inferior pointers,
				// reflection, gob, etc.
				if(pc[0] > n) {
					runtime·printf("invalid gc type info: type info size %p, block size %p\n", pc[0], n);
					runtime·throw("invalid gc type info");
				}
			}
		} else if(UseSpanType) {
			if(CollectStats)
				runtime·xadd64(&gcstats.obj.notype, 1);

			type = runtime·gettype(b);
			if(type != 0) {
				if(CollectStats)
					runtime·xadd64(&gcstats.obj.typelookup, 1);

				t = (Type*)(type & ~(uintptr)(PtrSize-1));
				switch(type & (PtrSize-1)) {
				case TypeInfo_SingleObject:
					pc = (uintptr*)t->gc;
					precise_type = true;  // type information about 'b' is precise
					stack_top.count = 1;
					stack_top.elemsize = pc[0];
					break;
				case TypeInfo_Array:
					pc = (uintptr*)t->gc;
					if(pc[0] == 0)
						goto next_block;
					precise_type = true;  // type information about 'b' is precise
					stack_top.count = 0;  // 0 means an infinite number of iterations
					stack_top.elemsize = pc[0];
					stack_top.loop_or_ret = pc+1;
					break;
				case TypeInfo_Chan:
					chan = (Hchan*)b;
					chantype = (ChanType*)t;
					chan_ret = nil;
					pc = chanProg;
					break;
				default:
					if(Debug > 1)
						runtime·printf("scanblock %p %D type %p %S\n", b, (int64)n, type, *t->string);
					runtime·throw("scanblock: invalid type");
					return;
				}
				if(Debug > 1)
					runtime·printf("scanblock %p %D type %p %S pc=%p\n", b, (int64)n, type, *t->string, pc);
			} else {
				pc = defaultProg;
				if(Debug > 1)
					runtime·printf("scanblock %p %D unknown type\n", b, (int64)n);
			}
		} else {
			pc = defaultProg;
			if(Debug > 1)
				runtime·printf("scanblock %p %D no span types\n", b, (int64)n);
		}

		if(IgnorePreciseGC)
			pc = defaultProg;

		pc++;
		stack_top.b = (uintptr)b;
		end_b = (uintptr)b + n - PtrSize;

	for(;;) {
		if(CollectStats)
			runtime·xadd64(&gcstats.instr[pc[0]], 1);

		obj = nil;
		objti = 0;
		switch(pc[0]) {
		case GC_PTR:
			obj = *(void**)(stack_top.b + pc[1]);
			objti = pc[2];
			if(Debug > 2)
				runtime·printf("gc_ptr @%p: %p ti=%p\n", stack_top.b+pc[1], obj, objti);
			pc += 3;
			if(Debug)
				checkptr(obj, objti);
			break;

		case GC_SLICE:
			sliceptr = (Slice*)(stack_top.b + pc[1]);
			if(Debug > 2)
				runtime·printf("gc_slice @%p: %p/%D/%D\n", sliceptr, sliceptr->array, (int64)sliceptr->len, (int64)sliceptr->cap);
			if(sliceptr->cap != 0) {
				obj = sliceptr->array;
				// Can't use slice element type for scanning,
				// because if it points to an array embedded
				// in the beginning of a struct,
				// we will scan the whole struct as the slice.
				// So just obtain type info from heap.
			}
			pc += 3;
			break;

		case GC_APTR:
			obj = *(void**)(stack_top.b + pc[1]);
			if(Debug > 2)
				runtime·printf("gc_aptr @%p: %p\n", stack_top.b+pc[1], obj);
			pc += 2;
			break;

		case GC_STRING:
			stringptr = (String*)(stack_top.b + pc[1]);
			if(Debug > 2)
				runtime·printf("gc_string @%p: %p/%D\n", stack_top.b+pc[1], stringptr->str, (int64)stringptr->len);
			if(stringptr->len != 0)
				markonly(stringptr->str);
			pc += 2;
			continue;

		case GC_EFACE:
			eface = (Eface*)(stack_top.b + pc[1]);
			pc += 2;
			if(Debug > 2)
				runtime·printf("gc_eface @%p: %p %p\n", stack_top.b+pc[1], eface->type, eface->data);
			if(eface->type == nil)
				continue;

			// eface->type
			t = eface->type;
			if((void*)t >= arena_start && (void*)t < arena_used) {
				*sbuf.ptr.pos++ = (PtrTarget){t, 0};
				if(sbuf.ptr.pos == sbuf.ptr.end)
					flushptrbuf(&sbuf);
			}

			// eface->data
			if(eface->data >= arena_start && eface->data < arena_used) {
				if(t->size <= sizeof(void*)) {
					if((t->kind & KindNoPointers))
						continue;

					obj = eface->data;
					if((t->kind & ~KindNoPointers) == KindPtr) {
						// Only use type information if it is a pointer-containing type.
						// This matches the GC programs written by cmd/gc/reflect.c's
						// dgcsym1 in case TPTR32/case TPTR64. See rationale there.
						et = ((PtrType*)t)->elem;
						if(!(et->kind & KindNoPointers))
							objti = (uintptr)((PtrType*)t)->elem->gc;
					}
				} else {
					obj = eface->data;
					objti = (uintptr)t->gc;
				}
			}
			break;

		case GC_IFACE:
			iface = (Iface*)(stack_top.b + pc[1]);
			pc += 2;
			if(Debug > 2)
				runtime·printf("gc_iface @%p: %p/%p %p\n", stack_top.b+pc[1], iface->tab, nil, iface->data);
			if(iface->tab == nil)
				continue;
			
			// iface->tab
			if((void*)iface->tab >= arena_start && (void*)iface->tab < arena_used) {
				*sbuf.ptr.pos++ = (PtrTarget){iface->tab, (uintptr)itabtype->gc};
				if(sbuf.ptr.pos == sbuf.ptr.end)
					flushptrbuf(&sbuf);
			}

			// iface->data
			if(iface->data >= arena_start && iface->data < arena_used) {
				t = iface->tab->type;
				if(t->size <= sizeof(void*)) {
					if((t->kind & KindNoPointers))
						continue;

					obj = iface->data;
					if((t->kind & ~KindNoPointers) == KindPtr) {
						// Only use type information if it is a pointer-containing type.
						// This matches the GC programs written by cmd/gc/reflect.c's
						// dgcsym1 in case TPTR32/case TPTR64. See rationale there.
						et = ((PtrType*)t)->elem;
						if(!(et->kind & KindNoPointers))
							objti = (uintptr)((PtrType*)t)->elem->gc;
					}
				} else {
					obj = iface->data;
					objti = (uintptr)t->gc;
				}
			}
			break;

		case GC_DEFAULT_PTR:
			while(stack_top.b <= end_b) {
				obj = *(byte**)stack_top.b;
				if(Debug > 2)
					runtime·printf("gc_default_ptr @%p: %p\n", stack_top.b, obj);
				stack_top.b += PtrSize;
				if(obj >= arena_start && obj < arena_used) {
					*sbuf.ptr.pos++ = (PtrTarget){obj, 0};
					if(sbuf.ptr.pos == sbuf.ptr.end)
						flushptrbuf(&sbuf);
				}
			}
			goto next_block;

		case GC_END:
			if(--stack_top.count != 0) {
				// Next iteration of a loop if possible.
				stack_top.b += stack_top.elemsize;
				if(stack_top.b + stack_top.elemsize <= end_b+PtrSize) {
					pc = stack_top.loop_or_ret;
					continue;
				}
				i = stack_top.b;
			} else {
				// Stack pop if possible.
				if(stack_ptr+1 < stack+nelem(stack)) {
					pc = stack_top.loop_or_ret;
					stack_top = *(++stack_ptr);
					continue;
				}
				i = (uintptr)b + nominal_size;
			}
			if(!precise_type) {
				// Quickly scan [b+i,b+n) for possible pointers.
				for(; i<=end_b; i+=PtrSize) {
					if(*(byte**)i != nil) {
						// Found a value that may be a pointer.
						// Do a rescan of the entire block.
						enqueue((Obj){b, n, 0}, &sbuf.wbuf, &sbuf.wp, &sbuf.nobj);
						if(CollectStats) {
							runtime·xadd64(&gcstats.rescan, 1);
							runtime·xadd64(&gcstats.rescanbytes, n);
						}
						break;
					}
				}
			}
			goto next_block;

		case GC_ARRAY_START:
			i = stack_top.b + pc[1];
			count = pc[2];
			elemsize = pc[3];
			pc += 4;

			// Stack push.
			*stack_ptr-- = stack_top;
			stack_top = (Frame){count, elemsize, i, pc};
			continue;

		case GC_ARRAY_NEXT:
			if(--stack_top.count != 0) {
				stack_top.b += stack_top.elemsize;
				pc = stack_top.loop_or_ret;
			} else {
				// Stack pop.
				stack_top = *(++stack_ptr);
				pc += 1;
			}
			continue;

		case GC_CALL:
			// Stack push.
			*stack_ptr-- = stack_top;
			stack_top = (Frame){1, 0, stack_top.b + pc[1], pc+3 /*return address*/};
			pc = (uintptr*)((byte*)pc + *(int32*)(pc+2));  // target of the CALL instruction
			continue;

		case GC_REGION:
			obj = (void*)(stack_top.b + pc[1]);
			size = pc[2];
			objti = pc[3];
			pc += 4;

			if(Debug > 2)
				runtime·printf("gc_region @%p: %D %p\n", stack_top.b+pc[1], (int64)size, objti);
			*sbuf.obj.pos++ = (Obj){obj, size, objti};
			if(sbuf.obj.pos == sbuf.obj.end)
				flushobjbuf(&sbuf);
			continue;

		case GC_CHAN_PTR:
			chan = *(Hchan**)(stack_top.b + pc[1]);
			if(Debug > 2 && chan != nil)
				runtime·printf("gc_chan_ptr @%p: %p/%D/%D %p\n", stack_top.b+pc[1], chan, (int64)chan->qcount, (int64)chan->dataqsiz, pc[2]);
			if(chan == nil) {
				pc += 3;
				continue;
			}
			if(markonly(chan)) {
				chantype = (ChanType*)pc[2];
				if(!(chantype->elem->kind & KindNoPointers)) {
					// Start chanProg.
					chan_ret = pc+3;
					pc = chanProg+1;
					continue;
				}
			}
			pc += 3;
			continue;

		case GC_CHAN:
			// There are no heap pointers in struct Hchan,
			// so we can ignore the leading sizeof(Hchan) bytes.
			if(!(chantype->elem->kind & KindNoPointers)) {
				// Channel's buffer follows Hchan immediately in memory.
				// Size of buffer (cap(c)) is second int in the chan struct.
				chancap = ((uintgo*)chan)[1];
				if(chancap > 0) {
					// TODO(atom): split into two chunks so that only the
					// in-use part of the circular buffer is scanned.
					// (Channel routines zero the unused part, so the current
					// code does not lead to leaks, it's just a little inefficient.)
					*sbuf.obj.pos++ = (Obj){(byte*)chan+runtime·Hchansize, chancap*chantype->elem->size,
						(uintptr)chantype->elem->gc | PRECISE | LOOP};
					if(sbuf.obj.pos == sbuf.obj.end)
						flushobjbuf(&sbuf);
				}
			}
			if(chan_ret == nil)
				goto next_block;
			pc = chan_ret;
			continue;

		default:
			runtime·printf("runtime: invalid GC instruction %p at %p\n", pc[0], pc);
			runtime·throw("scanblock: invalid GC instruction");
			return;
		}

		if(obj >= arena_start && obj < arena_used) {
			*sbuf.ptr.pos++ = (PtrTarget){obj, objti};
			if(sbuf.ptr.pos == sbuf.ptr.end)
				flushptrbuf(&sbuf);
		}
	}

	next_block:
		// Done scanning [b, b+n).  Prepare for the next iteration of
		// the loop by setting b, n, ti to the parameters for the next block.

		if(sbuf.nobj == 0) {
			flushptrbuf(&sbuf);
			flushobjbuf(&sbuf);

			if(sbuf.nobj == 0) {
				if(!keepworking) {
					if(sbuf.wbuf)
						putempty(sbuf.wbuf);
					return;
				}
				// Emptied our buffer: refill.
				sbuf.wbuf = getfull(sbuf.wbuf);
				if(sbuf.wbuf == nil)
					return;
				sbuf.nobj = sbuf.wbuf->nobj;
				sbuf.wp = sbuf.wbuf->obj + sbuf.wbuf->nobj;
			}
		}

		// Fetch b from the work buffer.
		--sbuf.wp;
		b = sbuf.wp->p;
		n = sbuf.wp->n;
		ti = sbuf.wp->ti;
		sbuf.nobj--;
	}
}

// Append obj to the work buffer.
// _wbuf, _wp, _nobj are input/output parameters and are specifying the work buffer.
static void
enqueue(Obj obj, Workbuf **_wbuf, Obj **_wp, uintptr *_nobj)
{
	uintptr nobj, off;
	Obj *wp;
	Workbuf *wbuf;

	if(Debug > 1)
		runtime·printf("append obj(%p %D %p)\n", obj.p, (int64)obj.n, obj.ti);

	// Align obj.b to a word boundary.
	off = (uintptr)obj.p & (PtrSize-1);
	if(off != 0) {
		obj.p += PtrSize - off;
		obj.n -= PtrSize - off;
		obj.ti = 0;
	}

	if(obj.p == nil || obj.n == 0)
		return;

	// Load work buffer state
	wp = *_wp;
	wbuf = *_wbuf;
	nobj = *_nobj;

	// If another proc wants a pointer, give it some.
	if(work.nwait > 0 && nobj > handoffThreshold && work.full == 0) {
		wbuf->nobj = nobj;
		wbuf = handoff(wbuf);
		nobj = wbuf->nobj;
		wp = wbuf->obj + nobj;
	}

	// If buffer is full, get a new one.
	if(wbuf == nil || nobj >= nelem(wbuf->obj)) {
		if(wbuf != nil)
			wbuf->nobj = nobj;
		wbuf = getempty(wbuf);
		wp = wbuf->obj;
		nobj = 0;
	}

	*wp = obj;
	wp++;
	nobj++;

	// Save work buffer state
	*_wp = wp;
	*_wbuf = wbuf;
	*_nobj = nobj;
}

static void
enqueue1(Workbuf **wbufp, Obj obj)
{
	Workbuf *wbuf;

	wbuf = *wbufp;
	if(wbuf->nobj >= nelem(wbuf->obj))
		*wbufp = wbuf = getempty(wbuf);
	wbuf->obj[wbuf->nobj++] = obj;
}

static void
markroot(ParFor *desc, uint32 i)
{
	Workbuf *wbuf;
	FinBlock *fb;
	MHeap *h;
	MSpan **allspans, *s;
	uint32 spanidx, sg;
	G *gp;
	void *p;

	USED(&desc);
	wbuf = getempty(nil);
	// Note: if you add a case here, please also update heapdump.c:dumproots.
	switch(i) {
	case RootData:
		enqueue1(&wbuf, (Obj){data, edata - data, (uintptr)gcdata});
		break;

	case RootBss:
		enqueue1(&wbuf, (Obj){bss, ebss - bss, (uintptr)gcbss});
		break;

	case RootFinalizers:
		for(fb=allfin; fb; fb=fb->alllink)
			enqueue1(&wbuf, (Obj){(byte*)fb->fin, fb->cnt*sizeof(fb->fin[0]), 0});
		break;

	case RootSpanTypes:
		// mark span types and MSpan.specials (to walk spans only once)
		h = &runtime·mheap;
		sg = h->sweepgen;
		allspans = h->allspans;
		for(spanidx=0; spanidx<runtime·mheap.nspan; spanidx++) {
			Special *sp;
			SpecialFinalizer *spf;

			s = allspans[spanidx];
			if(s->sweepgen != sg) {
				runtime·printf("sweep %d %d\n", s->sweepgen, sg);
				runtime·throw("gc: unswept span");
			}
			if(s->state != MSpanInUse)
				continue;
			// The garbage collector ignores type pointers stored in MSpan.types:
			//  - Compiler-generated types are stored outside of heap.
			//  - The reflect package has runtime-generated types cached in its data structures.
			//    The garbage collector relies on finding the references via that cache.
			if(s->types.compression == MTypes_Words || s->types.compression == MTypes_Bytes)
				markonly((byte*)s->types.data);
			for(sp = s->specials; sp != nil; sp = sp->next) {
				if(sp->kind != KindSpecialFinalizer)
					continue;
				// don't mark finalized object, but scan it so we
				// retain everything it points to.
				spf = (SpecialFinalizer*)sp;
				// A finalizer can be set for an inner byte of an object, find object beginning.
				p = (void*)((s->start << PageShift) + spf->offset/s->elemsize*s->elemsize);
				enqueue1(&wbuf, (Obj){p, s->elemsize, 0});
				enqueue1(&wbuf, (Obj){(void*)&spf->fn, PtrSize, 0});
				enqueue1(&wbuf, (Obj){(void*)&spf->fint, PtrSize, 0});
				enqueue1(&wbuf, (Obj){(void*)&spf->ot, PtrSize, 0});
			}
		}
		break;

	case RootFlushCaches:
		flushallmcaches();
		break;

	default:
		// the rest is scanning goroutine stacks
		if(i - RootCount >= runtime·allglen)
			runtime·throw("markroot: bad index");
		gp = runtime·allg[i - RootCount];
		// remember when we've first observed the G blocked
		// needed only to output in traceback
		if((gp->status == Gwaiting || gp->status == Gsyscall) && gp->waitsince == 0)
			gp->waitsince = work.tstart;
		addstackroots(gp, &wbuf);
		break;
		
	}

	if(wbuf)
		scanblock(wbuf, false);
}

// Get an empty work buffer off the work.empty list,
// allocating new buffers as needed.
static Workbuf*
getempty(Workbuf *b)
{
	if(b != nil)
		runtime·lfstackpush(&work.full, &b->node);
	b = (Workbuf*)runtime·lfstackpop(&work.empty);
	if(b == nil) {
		// Need to allocate.
		runtime·lock(&work);
		if(work.nchunk < sizeof *b) {
			work.nchunk = 1<<20;
			work.chunk = runtime·SysAlloc(work.nchunk, &mstats.gc_sys);
			if(work.chunk == nil)
				runtime·throw("runtime: cannot allocate memory");
		}
		b = (Workbuf*)work.chunk;
		work.chunk += sizeof *b;
		work.nchunk -= sizeof *b;
		runtime·unlock(&work);
	}
	b->nobj = 0;
	return b;
}

static void
putempty(Workbuf *b)
{
	if(CollectStats)
		runtime·xadd64(&gcstats.putempty, 1);

	runtime·lfstackpush(&work.empty, &b->node);
}

// Get a full work buffer off the work.full list, or return nil.
static Workbuf*
getfull(Workbuf *b)
{
	int32 i;

	if(CollectStats)
		runtime·xadd64(&gcstats.getfull, 1);

	if(b != nil)
		runtime·lfstackpush(&work.empty, &b->node);
	b = (Workbuf*)runtime·lfstackpop(&work.full);
	if(b != nil || work.nproc == 1)
		return b;

	runtime·xadd(&work.nwait, +1);
	for(i=0;; i++) {
		if(work.full != 0) {
			runtime·xadd(&work.nwait, -1);
			b = (Workbuf*)runtime·lfstackpop(&work.full);
			if(b != nil)
				return b;
			runtime·xadd(&work.nwait, +1);
		}
		if(work.nwait == work.nproc)
			return nil;
		if(i < 10) {
			g->m->gcstats.nprocyield++;
			runtime·procyield(20);
		} else if(i < 20) {
			g->m->gcstats.nosyield++;
			runtime·osyield();
		} else {
			g->m->gcstats.nsleep++;
			runtime·usleep(100);
		}
	}
}

static Workbuf*
handoff(Workbuf *b)
{
	int32 n;
	Workbuf *b1;

	// Make new buffer with half of b's pointers.
	b1 = getempty(nil);
	n = b->nobj/2;
	b->nobj -= n;
	b1->nobj = n;
	runtime·memmove(b1->obj, b->obj+b->nobj, n*sizeof b1->obj[0]);
	g->m->gcstats.nhandoff++;
	g->m->gcstats.nhandoffcnt += n;

	// Put b on full list - let first half of b get stolen.
	runtime·lfstackpush(&work.full, &b->node);
	return b1;
}

extern byte pclntab[]; // base for f->ptrsoff

BitVector
runtime·stackmapdata(StackMap *stackmap, int32 n)
{
	if(n < 0 || n >= stackmap->n)
		runtime·throw("stackmapdata: index out of range");
	return (BitVector){stackmap->nbit, stackmap->data + n*((stackmap->nbit+31)/32)};
}

// Scans an interface data value when the interface type indicates
// that it is a pointer.
static void
scaninterfacedata(uintptr bits, byte *scanp, void *wbufp)
{
	Itab *tab;
	Type *type;

	if(runtime·precisestack) {
		if(bits == BitsIface) {
			tab = *(Itab**)scanp;
			if(tab->type->size <= sizeof(void*) && (tab->type->kind & KindNoPointers))
				return;
		} else { // bits == BitsEface
			type = *(Type**)scanp;
			if(type->size <= sizeof(void*) && (type->kind & KindNoPointers))
				return;
		}
	}
	enqueue1(wbufp, (Obj){scanp+PtrSize, PtrSize, 0});
}

// Starting from scanp, scans words corresponding to set bits.
static void
scanbitvector(Func *f, bool precise, byte *scanp, BitVector *bv, void *wbufp)
{
	uintptr word, bits;
	uint32 *wordp;
	int32 i, remptrs;
	byte *p;

	wordp = bv->data;
	for(remptrs = bv->n; remptrs > 0; remptrs -= 32) {
		word = *wordp++;
		if(remptrs < 32)
			i = remptrs;
		else
			i = 32;
		i /= BitsPerPointer;
		for(; i > 0; i--) {
			bits = word & 3;
			switch(bits) {
			case BitsDead:
				if(runtime·debug.gcdead)
					*(uintptr*)scanp = PoisonGC;
				break;
			case BitsScalar:
				break;
			case BitsPointer:
				p = *(byte**)scanp;
				if(p != nil) {
					if(Debug > 2)
						runtime·printf("frame %s @%p: ptr %p\n", runtime·funcname(f), scanp, p);
					if(precise && (p < (byte*)PageSize || (uintptr)p == PoisonGC || (uintptr)p == PoisonStack)) {
						// Looks like a junk value in a pointer slot.
						// Liveness analysis wrong?
						g->m->traceback = 2;
						runtime·printf("bad pointer in frame %s at %p: %p\n", runtime·funcname(f), scanp, p);
						runtime·throw("bad pointer in scanbitvector");
					}
					enqueue1(wbufp, (Obj){scanp, PtrSize, 0});
				}
				break;
			case BitsMultiWord:
				p = scanp;
				word >>= BitsPerPointer;
				scanp += PtrSize;
				i--;
				if(i == 0) {
					// Get next chunk of bits
					remptrs -= 32;
					word = *wordp++;
					if(remptrs < 32)
						i = remptrs;
					else
						i = 32;
					i /= BitsPerPointer;
				}
				switch(word & 3) {
				case BitsString:
					if(Debug > 2)
						runtime·printf("frame %s @%p: string %p/%D\n", runtime·funcname(f), p, ((String*)p)->str, (int64)((String*)p)->len);
					if(((String*)p)->len != 0)
						markonly(((String*)p)->str);
					break;
				case BitsSlice:
					word >>= BitsPerPointer;
					scanp += PtrSize;
					i--;
					if(i == 0) {
						// Get next chunk of bits
						remptrs -= 32;
						word = *wordp++;
						if(remptrs < 32)
							i = remptrs;
						else
							i = 32;
						i /= BitsPerPointer;
					}
					if(Debug > 2)
						runtime·printf("frame %s @%p: slice %p/%D/%D\n", runtime·funcname(f), p, ((Slice*)p)->array, (int64)((Slice*)p)->len, (int64)((Slice*)p)->cap);
					if(((Slice*)p)->cap < ((Slice*)p)->len) {
						g->m->traceback = 2;
						runtime·printf("bad slice in frame %s at %p: %p/%p/%p\n", runtime·funcname(f), p, ((byte**)p)[0], ((byte**)p)[1], ((byte**)p)[2]);
						runtime·throw("slice capacity smaller than length");
					}
					if(((Slice*)p)->cap != 0)
						enqueue1(wbufp, (Obj){p, PtrSize, 0});
					break;
				case BitsIface:
				case BitsEface:
					if(*(byte**)p != nil) {
						if(Debug > 2) {
							if((word&3) == BitsEface)
								runtime·printf("frame %s @%p: eface %p %p\n", runtime·funcname(f), p, ((uintptr*)p)[0], ((uintptr*)p)[1]);
							else
								runtime·printf("frame %s @%p: iface %p %p\n", runtime·funcname(f), p, ((uintptr*)p)[0], ((uintptr*)p)[1]);
						}
						scaninterfacedata(word & 3, p, wbufp);
					}
					break;
				}
			}
			word >>= BitsPerPointer;
			scanp += PtrSize;
		}
	}
}

// Scan a stack frame: local variables and function arguments/results.
static bool
scanframe(Stkframe *frame, void *wbufp)
{
	Func *f;
	StackMap *stackmap;
	BitVector bv;
	uintptr size;
	uintptr targetpc;
	int32 pcdata;
	bool precise;

	f = frame->fn;
	targetpc = frame->continpc;
	if(targetpc == 0) {
		// Frame is dead.
		return true;
	}
	if(targetpc != f->entry)
		targetpc--;
	pcdata = runtime·pcdatavalue(f, PCDATA_StackMapIndex, targetpc);
	if(pcdata == -1) {
		// We do not have a valid pcdata value but there might be a
		// stackmap for this function.  It is likely that we are looking
		// at the function prologue, assume so and hope for the best.
		pcdata = 0;
	}

	// Scan local variables if stack frame has been allocated.
	// Use pointer information if known.
	precise = false;
	stackmap = runtime·funcdata(f, FUNCDATA_LocalsPointerMaps);
	if(stackmap == nil) {
		// No locals information, scan everything.
		size = frame->varp - (byte*)frame->sp;
		if(Debug > 2)
			runtime·printf("frame %s unsized locals %p+%p\n", runtime·funcname(f), frame->varp-size, size);
		enqueue1(wbufp, (Obj){frame->varp - size, size, 0});
	} else if(stackmap->n < 0) {
		// Locals size information, scan just the locals.
		size = -stackmap->n;
		if(Debug > 2)
			runtime·printf("frame %s conservative locals %p+%p\n", runtime·funcname(f), frame->varp-size, size);
		enqueue1(wbufp, (Obj){frame->varp - size, size, 0});
	} else if(stackmap->n > 0) {
		// Locals bitmap information, scan just the pointers in
		// locals.
		if(pcdata < 0 || pcdata >= stackmap->n) {
			// don't know where we are
			runtime·printf("pcdata is %d and %d stack map entries for %s (targetpc=%p)\n",
				pcdata, stackmap->n, runtime·funcname(f), targetpc);
			runtime·throw("scanframe: bad symbol table");
		}
		bv = runtime·stackmapdata(stackmap, pcdata);
		size = (bv.n * PtrSize) / BitsPerPointer;
		precise = true;
		scanbitvector(f, true, frame->varp - size, &bv, wbufp);
	}

	// Scan arguments.
	// Use pointer information if known.
	stackmap = runtime·funcdata(f, FUNCDATA_ArgsPointerMaps);
	if(stackmap != nil) {
		bv = runtime·stackmapdata(stackmap, pcdata);
		scanbitvector(f, precise, frame->argp, &bv, wbufp);
	} else {
		if(Debug > 2)
			runtime·printf("frame %s conservative args %p+%p\n", runtime·funcname(f), frame->argp, (uintptr)frame->arglen);
		enqueue1(wbufp, (Obj){frame->argp, frame->arglen, 0});
	}
	return true;
}

static void
addstackroots(G *gp, Workbuf **wbufp)
{
	M *mp;
	int32 n;
	Stktop *stk;
	uintptr sp, guard;

	switch(gp->status){
	default:
		runtime·printf("unexpected G.status %d (goroutine %p %D)\n", gp->status, gp, gp->goid);
		runtime·throw("mark - bad status");
	case Gdead:
		return;
	case Grunning:
		runtime·throw("mark - world not stopped");
	case Grunnable:
	case Gsyscall:
	case Gwaiting:
		break;
	}

	if(gp == g)
		runtime·throw("can't scan our own stack");
	if((mp = gp->m) != nil && mp->helpgc)
		runtime·throw("can't scan gchelper stack");

	if(gp->syscallstack != (uintptr)nil) {
		// Scanning another goroutine that is about to enter or might
		// have just exited a system call. It may be executing code such
		// as schedlock and may have needed to start a new stack segment.
		// Use the stack segment and stack pointer at the time of
		// the system call instead, since that won't change underfoot.
		sp = gp->syscallsp;
		stk = (Stktop*)gp->syscallstack;
		guard = gp->syscallguard;
	} else {
		// Scanning another goroutine's stack.
		// The goroutine is usually asleep (the world is stopped).
		sp = gp->sched.sp;
		stk = (Stktop*)gp->stackbase;
		guard = gp->stackguard;
	}
	if(ScanStackByFrames) {
		USED(sp);
		USED(stk);
		USED(guard);
		runtime·gentraceback(~(uintptr)0, ~(uintptr)0, 0, gp, 0, nil, 0x7fffffff, scanframe, wbufp, false);
	} else {
		n = 0;
		while(stk) {
			if(sp < guard-StackGuard || (uintptr)stk < sp) {
				runtime·printf("scanstack inconsistent: g%D#%d sp=%p not in [%p,%p]\n", gp->goid, n, sp, guard-StackGuard, stk);
				runtime·throw("scanstack");
			}
			if(Debug > 2)
				runtime·printf("conservative stack %p+%p\n", (byte*)sp, (uintptr)stk-sp);
			enqueue1(wbufp, (Obj){(byte*)sp, (uintptr)stk - sp, (uintptr)defaultProg | PRECISE | LOOP});
			sp = stk->gobuf.sp;
			guard = stk->stackguard;
			stk = (Stktop*)stk->stackbase;
			n++;
		}
	}
}

void
runtime·queuefinalizer(byte *p, FuncVal *fn, uintptr nret, Type *fint, PtrType *ot)
{
	FinBlock *block;
	Finalizer *f;

	runtime·lock(&finlock);
	if(finq == nil || finq->cnt == finq->cap) {
		if(finc == nil) {
			finc = runtime·persistentalloc(FinBlockSize, 0, &mstats.gc_sys);
			finc->cap = (FinBlockSize - sizeof(FinBlock)) / sizeof(Finalizer) + 1;
			finc->alllink = allfin;
			allfin = finc;
		}
		block = finc;
		finc = block->next;
		block->next = finq;
		finq = block;
	}
	f = &finq->fin[finq->cnt];
	finq->cnt++;
	f->fn = fn;
	f->nret = nret;
	f->fint = fint;
	f->ot = ot;
	f->arg = p;
	runtime·fingwake = true;
	runtime·unlock(&finlock);
}

void
runtime·iterate_finq(void (*callback)(FuncVal*, byte*, uintptr, Type*, PtrType*))
{
	FinBlock *fb;
	Finalizer *f;
	uintptr i;

	for(fb = allfin; fb; fb = fb->alllink) {
		for(i = 0; i < fb->cnt; i++) {
			f = &fb->fin[i];
			callback(f->fn, f->arg, f->nret, f->fint, f->ot);
		}
	}
}

void
runtime·MSpan_EnsureSwept(MSpan *s)
{
	uint32 sg;

	// Caller must disable preemption.
	// Otherwise when this function returns the span can become unswept again
	// (if GC is triggered on another goroutine).
	if(g->m->locks == 0 && g->m->mallocing == 0 && g != g->m->g0)
		runtime·throw("MSpan_EnsureSwept: m is not locked");

	sg = runtime·mheap.sweepgen;
	if(runtime·atomicload(&s->sweepgen) == sg)
		return;
	if(runtime·cas(&s->sweepgen, sg-2, sg-1)) {
		runtime·MSpan_Sweep(s);
		return;
	}
	// unfortunate condition, and we don't have efficient means to wait
	while(runtime·atomicload(&s->sweepgen) != sg)
		runtime·osyield();  
}

// Sweep frees or collects finalizers for blocks not marked in the mark phase.
// It clears the mark bits in preparation for the next GC round.
// Returns true if the span was returned to heap.
bool
runtime·MSpan_Sweep(MSpan *s)
{
	int32 cl, n, npages, nfree;
	uintptr size, off, *bitp, shift, bits;
	uint32 sweepgen;
	byte *p;
	MCache *c;
	byte *arena_start;
	MLink head, *end;
	byte *type_data;
	byte compression;
	uintptr type_data_inc;
	MLink *x;
	Special *special, **specialp, *y;
	bool res, sweepgenset;

	// It's critical that we enter this function with preemption disabled,
	// GC must not start while we are in the middle of this function.
	if(g->m->locks == 0 && g->m->mallocing == 0 && g != g->m->g0)
		runtime·throw("MSpan_Sweep: m is not locked");
	sweepgen = runtime·mheap.sweepgen;
	if(s->state != MSpanInUse || s->sweepgen != sweepgen-1) {
		runtime·printf("MSpan_Sweep: state=%d sweepgen=%d mheap.sweepgen=%d\n",
			s->state, s->sweepgen, sweepgen);
		runtime·throw("MSpan_Sweep: bad span state");
	}
	arena_start = runtime·mheap.arena_start;
	cl = s->sizeclass;
	size = s->elemsize;
	if(cl == 0) {
		n = 1;
	} else {
		// Chunk full of small blocks.
		npages = runtime·class_to_allocnpages[cl];
		n = (npages << PageShift) / size;
	}
	res = false;
	nfree = 0;
	end = &head;
	c = g->m->mcache;
	sweepgenset = false;

	// mark any free objects in this span so we don't collect them
	for(x = s->freelist; x != nil; x = x->next) {
		// This is markonly(x) but faster because we don't need
		// atomic access and we're guaranteed to be pointing at
		// the head of a valid object.
		off = (uintptr*)x - (uintptr*)runtime·mheap.arena_start;
		bitp = (uintptr*)runtime·mheap.arena_start - off/wordsPerBitmapWord - 1;
		shift = off % wordsPerBitmapWord;
		*bitp |= bitMarked<<shift;
	}

	// Unlink & free special records for any objects we're about to free.
	specialp = &s->specials;
	special = *specialp;
	while(special != nil) {
		// A finalizer can be set for an inner byte of an object, find object beginning.
		p = (byte*)(s->start << PageShift) + special->offset/size*size;
		off = (uintptr*)p - (uintptr*)arena_start;
		bitp = (uintptr*)arena_start - off/wordsPerBitmapWord - 1;
		shift = off % wordsPerBitmapWord;
		bits = *bitp>>shift;
		if((bits & (bitAllocated|bitMarked)) == bitAllocated) {
			// Find the exact byte for which the special was setup
			// (as opposed to object beginning).
			p = (byte*)(s->start << PageShift) + special->offset;
			// about to free object: splice out special record
			y = special;
			special = special->next;
			*specialp = special;
			if(!runtime·freespecial(y, p, size, false)) {
				// stop freeing of object if it has a finalizer
				*bitp |= bitMarked << shift;
			}
		} else {
			// object is still live: keep special record
			specialp = &special->next;
			special = *specialp;
		}
	}

	type_data = (byte*)s->types.data;
	type_data_inc = sizeof(uintptr);
	compression = s->types.compression;
	switch(compression) {
	case MTypes_Bytes:
		type_data += 8*sizeof(uintptr);
		type_data_inc = 1;
		break;
	}

	// Sweep through n objects of given size starting at p.
	// This thread owns the span now, so it can manipulate
	// the block bitmap without atomic operations.
	p = (byte*)(s->start << PageShift);
	for(; n > 0; n--, p += size, type_data+=type_data_inc) {
		off = (uintptr*)p - (uintptr*)arena_start;
		bitp = (uintptr*)arena_start - off/wordsPerBitmapWord - 1;
		shift = off % wordsPerBitmapWord;
		bits = *bitp>>shift;

		if((bits & bitAllocated) == 0)
			continue;

		if((bits & bitMarked) != 0) {
			*bitp &= ~(bitMarked<<shift);
			continue;
		}

		if(runtime·debug.allocfreetrace)
			runtime·tracefree(p, size);

		// Clear mark and scan bits.
		*bitp &= ~((bitScan|bitMarked)<<shift);

		if(cl == 0) {
			// Free large span.
			runtime·unmarkspan(p, 1<<PageShift);
			s->needzero = 1;
			// important to set sweepgen before returning it to heap
			runtime·atomicstore(&s->sweepgen, sweepgen);
			sweepgenset = true;
			// See note about SysFault vs SysFree in malloc.goc.
			if(runtime·debug.efence)
				runtime·SysFault(p, size);
			else
				runtime·MHeap_Free(&runtime·mheap, s, 1);
			c->local_nlargefree++;
			c->local_largefree += size;
			runtime·xadd64(&mstats.next_gc, -(uint64)(size * (gcpercent + 100)/100));
			res = true;
		} else {
			// Free small object.
			switch(compression) {
			case MTypes_Words:
				*(uintptr*)type_data = 0;
				break;
			case MTypes_Bytes:
				*(byte*)type_data = 0;
				break;
			}
			if(size > 2*sizeof(uintptr))
				((uintptr*)p)[1] = (uintptr)0xdeaddeaddeaddeadll;	// mark as "needs to be zeroed"
			else if(size > sizeof(uintptr))
				((uintptr*)p)[1] = 0;

			end->next = (MLink*)p;
			end = (MLink*)p;
			nfree++;
		}
	}

	// We need to set s->sweepgen = h->sweepgen only when all blocks are swept,
	// because of the potential for a concurrent free/SetFinalizer.
	// But we need to set it before we make the span available for allocation
	// (return it to heap or mcentral), because allocation code assumes that a
	// span is already swept if available for allocation.

	if(!sweepgenset && nfree == 0) {
		// The span must be in our exclusive ownership until we update sweepgen,
		// check for potential races.
		if(s->state != MSpanInUse || s->sweepgen != sweepgen-1) {
			runtime·printf("MSpan_Sweep: state=%d sweepgen=%d mheap.sweepgen=%d\n",
				s->state, s->sweepgen, sweepgen);
			runtime·throw("MSpan_Sweep: bad span state after sweep");
		}
		runtime·atomicstore(&s->sweepgen, sweepgen);
	}
	if(nfree > 0) {
		c->local_nsmallfree[cl] += nfree;
		c->local_cachealloc -= nfree * size;
		runtime·xadd64(&mstats.next_gc, -(uint64)(nfree * size * (gcpercent + 100)/100));
		res = runtime·MCentral_FreeSpan(&runtime·mheap.central[cl], s, nfree, head.next, end);
		//MCentral_FreeSpan updates sweepgen
	}
	return res;
}

// State of background sweep.
// Pretected by gclock.
static struct
{
	G*	g;
	bool	parked;

	MSpan**	spans;
	uint32	nspan;
	uint32	spanidx;
} sweep;

// background sweeping goroutine
static void
bgsweep(void)
{
	g->issystem = 1;
	for(;;) {
		while(runtime·sweepone() != -1) {
			gcstats.nbgsweep++;
			runtime·gosched();
		}
		runtime·lock(&gclock);
		if(!runtime·mheap.sweepdone) {
			// It's possible if GC has happened between sweepone has
			// returned -1 and gclock lock.
			runtime·unlock(&gclock);
			continue;
		}
		sweep.parked = true;
		g->isbackground = true;
		runtime·parkunlock(&gclock, "GC sweep wait");
		g->isbackground = false;
	}
}

// sweeps one span
// returns number of pages returned to heap, or -1 if there is nothing to sweep
uintptr
runtime·sweepone(void)
{
	MSpan *s;
	uint32 idx, sg;
	uintptr npages;

	// increment locks to ensure that the goroutine is not preempted
	// in the middle of sweep thus leaving the span in an inconsistent state for next GC
	g->m->locks++;
	sg = runtime·mheap.sweepgen;
	for(;;) {
		idx = runtime·xadd(&sweep.spanidx, 1) - 1;
		if(idx >= sweep.nspan) {
			runtime·mheap.sweepdone = true;
			g->m->locks--;
			return -1;
		}
		s = sweep.spans[idx];
		if(s->state != MSpanInUse) {
			s->sweepgen = sg;
			continue;
		}
		if(s->sweepgen != sg-2 || !runtime·cas(&s->sweepgen, sg-2, sg-1))
			continue;
		if(s->incache)
			runtime·throw("sweep of incache span");
		npages = s->npages;
		if(!runtime·MSpan_Sweep(s))
			npages = 0;
		g->m->locks--;
		return npages;
	}
}

static void
dumpspan(uint32 idx)
{
	int32 sizeclass, n, npages, i, column;
	uintptr size;
	byte *p;
	byte *arena_start;
	MSpan *s;
	bool allocated;

	s = runtime·mheap.allspans[idx];
	if(s->state != MSpanInUse)
		return;
	arena_start = runtime·mheap.arena_start;
	p = (byte*)(s->start << PageShift);
	sizeclass = s->sizeclass;
	size = s->elemsize;
	if(sizeclass == 0) {
		n = 1;
	} else {
		npages = runtime·class_to_allocnpages[sizeclass];
		n = (npages << PageShift) / size;
	}
	
	runtime·printf("%p .. %p:\n", p, p+n*size);
	column = 0;
	for(; n>0; n--, p+=size) {
		uintptr off, *bitp, shift, bits;

		off = (uintptr*)p - (uintptr*)arena_start;
		bitp = (uintptr*)arena_start - off/wordsPerBitmapWord - 1;
		shift = off % wordsPerBitmapWord;
		bits = *bitp>>shift;

		allocated = ((bits & bitAllocated) != 0);

		for(i=0; i<size; i+=sizeof(void*)) {
			if(column == 0) {
				runtime·printf("\t");
			}
			if(i == 0) {
				runtime·printf(allocated ? "(" : "[");
				runtime·printf("%p: ", p+i);
			} else {
				runtime·printf(" ");
			}

			runtime·printf("%p", *(void**)(p+i));

			if(i+sizeof(void*) >= size) {
				runtime·printf(allocated ? ") " : "] ");
			}

			column++;
			if(column == 8) {
				runtime·printf("\n");
				column = 0;
			}
		}
	}
	runtime·printf("\n");
}

// A debugging function to dump the contents of memory
void
runtime·memorydump(void)
{
	uint32 spanidx;

	for(spanidx=0; spanidx<runtime·mheap.nspan; spanidx++) {
		dumpspan(spanidx);
	}
}

void
runtime·gchelper(void)
{
	uint32 nproc;

	g->m->traceback = 2;
	gchelperstart();

	// parallel mark for over gc roots
	runtime·parfordo(work.markfor);

	// help other threads scan secondary blocks
	scanblock(nil, true);

	bufferList[g->m->helpgc].busy = 0;
	nproc = work.nproc;  // work.nproc can change right after we increment work.ndone
	if(runtime·xadd(&work.ndone, +1) == nproc-1)
		runtime·notewakeup(&work.alldone);
	g->m->traceback = 0;
}

static void
cachestats(void)
{
	MCache *c;
	P *p, **pp;

	for(pp=runtime·allp; p=*pp; pp++) {
		c = p->mcache;
		if(c==nil)
			continue;
		runtime·purgecachedstats(c);
	}
}

static void
flushallmcaches(void)
{
	P *p, **pp;
	MCache *c;

	// Flush MCache's to MCentral.
	for(pp=runtime·allp; p=*pp; pp++) {
		c = p->mcache;
		if(c==nil)
			continue;
		runtime·MCache_ReleaseAll(c);
	}
}

void
runtime·updatememstats(GCStats *stats)
{
	M *mp;
	MSpan *s;
	int32 i;
	uint64 stacks_inuse, smallfree;
	uint64 *src, *dst;

	if(stats)
		runtime·memclr((byte*)stats, sizeof(*stats));
	stacks_inuse = 0;
	for(mp=runtime·allm; mp; mp=mp->alllink) {
		stacks_inuse += mp->stackinuse*FixedStack;
		if(stats) {
			src = (uint64*)&mp->gcstats;
			dst = (uint64*)stats;
			for(i=0; i<sizeof(*stats)/sizeof(uint64); i++)
				dst[i] += src[i];
			runtime·memclr((byte*)&mp->gcstats, sizeof(mp->gcstats));
		}
	}
	mstats.stacks_inuse = stacks_inuse;
	mstats.mcache_inuse = runtime·mheap.cachealloc.inuse;
	mstats.mspan_inuse = runtime·mheap.spanalloc.inuse;
	mstats.sys = mstats.heap_sys + mstats.stacks_sys + mstats.mspan_sys +
		mstats.mcache_sys + mstats.buckhash_sys + mstats.gc_sys + mstats.other_sys;
	
	// Calculate memory allocator stats.
	// During program execution we only count number of frees and amount of freed memory.
	// Current number of alive object in the heap and amount of alive heap memory
	// are calculated by scanning all spans.
	// Total number of mallocs is calculated as number of frees plus number of alive objects.
	// Similarly, total amount of allocated memory is calculated as amount of freed memory
	// plus amount of alive heap memory.
	mstats.alloc = 0;
	mstats.total_alloc = 0;
	mstats.nmalloc = 0;
	mstats.nfree = 0;
	for(i = 0; i < nelem(mstats.by_size); i++) {
		mstats.by_size[i].nmalloc = 0;
		mstats.by_size[i].nfree = 0;
	}

	// Flush MCache's to MCentral.
	flushallmcaches();

	// Aggregate local stats.
	cachestats();

	// Scan all spans and count number of alive objects.
	for(i = 0; i < runtime·mheap.nspan; i++) {
		s = runtime·mheap.allspans[i];
		if(s->state != MSpanInUse)
			continue;
		if(s->sizeclass == 0) {
			mstats.nmalloc++;
			mstats.alloc += s->elemsize;
		} else {
			mstats.nmalloc += s->ref;
			mstats.by_size[s->sizeclass].nmalloc += s->ref;
			mstats.alloc += s->ref*s->elemsize;
		}
	}

	// Aggregate by size class.
	smallfree = 0;
	mstats.nfree = runtime·mheap.nlargefree;
	for(i = 0; i < nelem(mstats.by_size); i++) {
		mstats.nfree += runtime·mheap.nsmallfree[i];
		mstats.by_size[i].nfree = runtime·mheap.nsmallfree[i];
		mstats.by_size[i].nmalloc += runtime·mheap.nsmallfree[i];
		smallfree += runtime·mheap.nsmallfree[i] * runtime·class_to_size[i];
	}
	mstats.nmalloc += mstats.nfree;

	// Calculate derived stats.
	mstats.total_alloc = mstats.alloc + runtime·mheap.largefree + smallfree;
	mstats.heap_alloc = mstats.alloc;
	mstats.heap_objects = mstats.nmalloc - mstats.nfree;
}

// Structure of arguments passed to function gc().
// This allows the arguments to be passed via runtime·mcall.
struct gc_args
{
	int64 start_time; // start time of GC in ns (just before stoptheworld)
	bool  eagersweep;
};

static void gc(struct gc_args *args);
static void mgc(G *gp);

static int32
readgogc(void)
{
	byte *p;

	p = runtime·getenv("GOGC");
	if(p == nil || p[0] == '\0')
		return 100;
	if(runtime·strcmp(p, (byte*)"off") == 0)
		return -1;
	return runtime·atoi(p);
}

// force = 1 - do GC regardless of current heap usage
// force = 2 - go GC and eager sweep
void
runtime·gc(int32 force)
{
	struct gc_args a;
	int32 i;

	// The atomic operations are not atomic if the uint64s
	// are not aligned on uint64 boundaries. This has been
	// a problem in the past.
	if((((uintptr)&work.empty) & 7) != 0)
		runtime·throw("runtime: gc work buffer is misaligned");
	if((((uintptr)&work.full) & 7) != 0)
		runtime·throw("runtime: gc work buffer is misaligned");

	// The gc is turned off (via enablegc) until
	// the bootstrap has completed.
	// Also, malloc gets called in the guts
	// of a number of libraries that might be
	// holding locks.  To avoid priority inversion
	// problems, don't bother trying to run gc
	// while holding a lock.  The next mallocgc
	// without a lock will do the gc instead.
	if(!mstats.enablegc || g == g->m->g0 || g->m->locks > 0 || runtime·panicking)
		return;

	if(gcpercent == GcpercentUnknown) {	// first time through
		runtime·lock(&runtime·mheap);
		if(gcpercent == GcpercentUnknown)
			gcpercent = readgogc();
		runtime·unlock(&runtime·mheap);
	}
	if(gcpercent < 0)
		return;

	runtime·semacquire(&runtime·worldsema, false);
	if(force==0 && mstats.heap_alloc < mstats.next_gc) {
		// typically threads which lost the race to grab
		// worldsema exit here when gc is done.
		runtime·semrelease(&runtime·worldsema);
		return;
	}

	// Ok, we're doing it!  Stop everybody else
	a.start_time = runtime·nanotime();
	a.eagersweep = force >= 2;
	g->m->gcing = 1;
	runtime·stoptheworld();
	
	clearpools();

	// Run gc on the g0 stack.  We do this so that the g stack
	// we're currently running on will no longer change.  Cuts
	// the root set down a bit (g0 stacks are not scanned, and
	// we don't need to scan gc's internal state).  Also an
	// enabler for copyable stacks.
	for(i = 0; i < (runtime·debug.gctrace > 1 ? 2 : 1); i++) {
		if(i > 0)
			a.start_time = runtime·nanotime();
		// switch to g0, call gc(&a), then switch back
		g->param = &a;
		g->status = Gwaiting;
		g->waitreason = "garbage collection";
		runtime·mcall(mgc);
	}

	// all done
	g->m->gcing = 0;
	g->m->locks++;
	runtime·semrelease(&runtime·worldsema);
	runtime·starttheworld();
	g->m->locks--;

	// now that gc is done, kick off finalizer thread if needed
	if(!ConcurrentSweep) {
		// give the queued finalizers, if any, a chance to run
		runtime·gosched();
	}
}

static void
mgc(G *gp)
{
	gc(gp->param);
	gp->param = nil;
	gp->status = Grunning;
	runtime·gogo(&gp->sched);
}

static void
gc(struct gc_args *args)
{
	int64 t0, t1, t2, t3, t4;
	uint64 heap0, heap1, obj, ninstr;
	GCStats stats;
	uint32 i;
	Eface eface;

	if(runtime·debug.allocfreetrace)
		runtime·tracegc();

	g->m->traceback = 2;
	t0 = args->start_time;
	work.tstart = args->start_time; 

	if(CollectStats)
		runtime·memclr((byte*)&gcstats, sizeof(gcstats));

	g->m->locks++;	// disable gc during mallocs in parforalloc
	if(work.markfor == nil)
		work.markfor = runtime·parforalloc(MaxGcproc);
	g->m->locks--;

	if(itabtype == nil) {
		// get C pointer to the Go type "itab"
		runtime·gc_itab_ptr(&eface);
		itabtype = ((PtrType*)eface.type)->elem;
	}

	t1 = 0;
	if(runtime·debug.gctrace)
		t1 = runtime·nanotime();

	// Sweep what is not sweeped by bgsweep.
	while(runtime·sweepone() != -1)
		gcstats.npausesweep++;

	work.nwait = 0;
	work.ndone = 0;
	work.nproc = runtime·gcprocs();
	runtime·parforsetup(work.markfor, work.nproc, RootCount + runtime·allglen, nil, false, markroot);
	if(work.nproc > 1) {
		runtime·noteclear(&work.alldone);
		runtime·helpgc(work.nproc);
	}

	t2 = 0;
	if(runtime·debug.gctrace)
		t2 = runtime·nanotime();

	gchelperstart();
	runtime·parfordo(work.markfor);
	scanblock(nil, true);

	t3 = 0;
	if(runtime·debug.gctrace)
		t3 = runtime·nanotime();

	bufferList[g->m->helpgc].busy = 0;
	if(work.nproc > 1)
		runtime·notesleep(&work.alldone);

	cachestats();
	// next_gc calculation is tricky with concurrent sweep since we don't know size of live heap
	// estimate what was live heap size after previous GC (for tracing only)
	heap0 = mstats.next_gc*100/(gcpercent+100);
	// conservatively set next_gc to high value assuming that everything is live
	// concurrent/lazy sweep will reduce this number while discovering new garbage
	mstats.next_gc = mstats.heap_alloc+mstats.heap_alloc*gcpercent/100;

	t4 = runtime·nanotime();
	mstats.last_gc = runtime·unixnanotime();  // must be Unix time to make sense to user
	mstats.pause_ns[mstats.numgc%nelem(mstats.pause_ns)] = t4 - t0;
	mstats.pause_total_ns += t4 - t0;
	mstats.numgc++;
	if(mstats.debuggc)
		runtime·printf("pause %D\n", t4-t0);

	if(runtime·debug.gctrace) {
		heap1 = mstats.heap_alloc;
		runtime·updatememstats(&stats);
		if(heap1 != mstats.heap_alloc) {
			runtime·printf("runtime: mstats skew: heap=%D/%D\n", heap1, mstats.heap_alloc);
			runtime·throw("mstats skew");
		}
		obj = mstats.nmalloc - mstats.nfree;

		stats.nprocyield += work.markfor->nprocyield;
		stats.nosyield += work.markfor->nosyield;
		stats.nsleep += work.markfor->nsleep;

		runtime·printf("gc%d(%d): %D+%D+%D+%D us, %D -> %D MB, %D (%D-%D) objects,"
				" %d/%d/%d sweeps,"
				" %D(%D) handoff, %D(%D) steal, %D/%D/%D yields\n",
			mstats.numgc, work.nproc, (t1-t0)/1000, (t2-t1)/1000, (t3-t2)/1000, (t4-t3)/1000,
			heap0>>20, heap1>>20, obj,
			mstats.nmalloc, mstats.nfree,
			sweep.nspan, gcstats.nbgsweep, gcstats.npausesweep,
			stats.nhandoff, stats.nhandoffcnt,
			work.markfor->nsteal, work.markfor->nstealcnt,
			stats.nprocyield, stats.nosyield, stats.nsleep);
		gcstats.nbgsweep = gcstats.npausesweep = 0;
		if(CollectStats) {
			runtime·printf("scan: %D bytes, %D objects, %D untyped, %D types from MSpan\n",
				gcstats.nbytes, gcstats.obj.cnt, gcstats.obj.notype, gcstats.obj.typelookup);
			if(gcstats.ptr.cnt != 0)
				runtime·printf("avg ptrbufsize: %D (%D/%D)\n",
					gcstats.ptr.sum/gcstats.ptr.cnt, gcstats.ptr.sum, gcstats.ptr.cnt);
			if(gcstats.obj.cnt != 0)
				runtime·printf("avg nobj: %D (%D/%D)\n",
					gcstats.obj.sum/gcstats.obj.cnt, gcstats.obj.sum, gcstats.obj.cnt);
			runtime·printf("rescans: %D, %D bytes\n", gcstats.rescan, gcstats.rescanbytes);

			runtime·printf("instruction counts:\n");
			ninstr = 0;
			for(i=0; i<nelem(gcstats.instr); i++) {
				runtime·printf("\t%d:\t%D\n", i, gcstats.instr[i]);
				ninstr += gcstats.instr[i];
			}
			runtime·printf("\ttotal:\t%D\n", ninstr);

			runtime·printf("putempty: %D, getfull: %D\n", gcstats.putempty, gcstats.getfull);

			runtime·printf("markonly base lookup: bit %D word %D span %D\n", gcstats.markonly.foundbit, gcstats.markonly.foundword, gcstats.markonly.foundspan);
			runtime·printf("flushptrbuf base lookup: bit %D word %D span %D\n", gcstats.flushptrbuf.foundbit, gcstats.flushptrbuf.foundword, gcstats.flushptrbuf.foundspan);
		}
	}

	// We cache current runtime·mheap.allspans array in sweep.spans,
	// because the former can be resized and freed.
	// Otherwise we would need to take heap lock every time
	// we want to convert span index to span pointer.

	// Free the old cached array if necessary.
	if(sweep.spans && sweep.spans != runtime·mheap.allspans)
		runtime·SysFree(sweep.spans, sweep.nspan*sizeof(sweep.spans[0]), &mstats.other_sys);
	// Cache the current array.
	runtime·mheap.sweepspans = runtime·mheap.allspans;
	runtime·mheap.sweepgen += 2;
	runtime·mheap.sweepdone = false;
	sweep.spans = runtime·mheap.allspans;
	sweep.nspan = runtime·mheap.nspan;
	sweep.spanidx = 0;

	// Temporary disable concurrent sweep, because we see failures on builders.
	if(ConcurrentSweep && !args->eagersweep) {
		runtime·lock(&gclock);
		if(sweep.g == nil)
			sweep.g = runtime·newproc1(&bgsweepv, nil, 0, 0, runtime·gc);
		else if(sweep.parked) {
			sweep.parked = false;
			runtime·ready(sweep.g);
		}
		runtime·unlock(&gclock);
	} else {
		// Sweep all spans eagerly.
		while(runtime·sweepone() != -1)
			gcstats.npausesweep++;
	}

	// Shrink a stack if not much of it is being used.
	// TODO: do in a parfor
	for(i = 0; i < runtime·allglen; i++)
		runtime·shrinkstack(runtime·allg[i]);

	runtime·MProf_GC();
	g->m->traceback = 0;
}

extern uintptr runtime·sizeof_C_MStats;

void
runtime·ReadMemStats(MStats *stats)
{
	// Have to acquire worldsema to stop the world,
	// because stoptheworld can only be used by
	// one goroutine at a time, and there might be
	// a pending garbage collection already calling it.
	runtime·semacquire(&runtime·worldsema, false);
	g->m->gcing = 1;
	runtime·stoptheworld();
	runtime·updatememstats(nil);
	// Size of the trailing by_size array differs between Go and C,
	// NumSizeClasses was changed, but we can not change Go struct because of backward compatibility.
	runtime·memcopy(runtime·sizeof_C_MStats, stats, &mstats);
	g->m->gcing = 0;
	g->m->locks++;
	runtime·semrelease(&runtime·worldsema);
	runtime·starttheworld();
	g->m->locks--;
}

void
runtime∕debug·readGCStats(Slice *pauses)
{
	uint64 *p;
	uint32 i, n;

	// Calling code in runtime/debug should make the slice large enough.
	if(pauses->cap < nelem(mstats.pause_ns)+3)
		runtime·throw("runtime: short slice passed to readGCStats");

	// Pass back: pauses, last gc (absolute time), number of gc, total pause ns.
	p = (uint64*)pauses->array;
	runtime·lock(&runtime·mheap);
	n = mstats.numgc;
	if(n > nelem(mstats.pause_ns))
		n = nelem(mstats.pause_ns);
	
	// The pause buffer is circular. The most recent pause is at
	// pause_ns[(numgc-1)%nelem(pause_ns)], and then backward
	// from there to go back farther in time. We deliver the times
	// most recent first (in p[0]).
	for(i=0; i<n; i++)
		p[i] = mstats.pause_ns[(mstats.numgc-1-i)%nelem(mstats.pause_ns)];

	p[n] = mstats.last_gc;
	p[n+1] = mstats.numgc;
	p[n+2] = mstats.pause_total_ns;	
	runtime·unlock(&runtime·mheap);
	pauses->len = n+3;
}

int32
runtime·setgcpercent(int32 in) {
	int32 out;

	runtime·lock(&runtime·mheap);
	if(gcpercent == GcpercentUnknown)
		gcpercent = readgogc();
	out = gcpercent;
	if(in < 0)
		in = -1;
	gcpercent = in;
	runtime·unlock(&runtime·mheap);
	return out;
}

static void
gchelperstart(void)
{
	if(g->m->helpgc < 0 || g->m->helpgc >= MaxGcproc)
		runtime·throw("gchelperstart: bad m->helpgc");
	if(runtime·xchg(&bufferList[g->m->helpgc].busy, 1))
		runtime·throw("gchelperstart: already busy");
	if(g != g->m->g0)
		runtime·throw("gchelper not running on g0 stack");
}

static void
runfinq(void)
{
	Finalizer *f;
	FinBlock *fb, *next;
	byte *frame;
	uint32 framesz, framecap, i;
	Eface *ef, ef1;

	// This function blocks for long periods of time, and because it is written in C
	// we have no liveness information. Zero everything so that uninitialized pointers
	// do not cause memory leaks.
	f = nil;
	fb = nil;
	next = nil;
	frame = nil;
	framecap = 0;
	framesz = 0;
	i = 0;
	ef = nil;
	ef1.type = nil;
	ef1.data = nil;
	
	// force flush to memory
	USED(&f);
	USED(&fb);
	USED(&next);
	USED(&framesz);
	USED(&i);
	USED(&ef);
	USED(&ef1);

	for(;;) {
		runtime·lock(&finlock);
		fb = finq;
		finq = nil;
		if(fb == nil) {
			runtime·fingwait = true;
			g->isbackground = true;
			runtime·parkunlock(&finlock, "finalizer wait");
			g->isbackground = false;
			continue;
		}
		runtime·unlock(&finlock);
		if(raceenabled)
			runtime·racefingo();
		for(; fb; fb=next) {
			next = fb->next;
			for(i=0; i<fb->cnt; i++) {
				f = &fb->fin[i];
				framesz = sizeof(Eface) + f->nret;
				if(framecap < framesz) {
					runtime·free(frame);
					// The frame does not contain pointers interesting for GC,
					// all not yet finalized objects are stored in finq.
					// If we do not mark it as FlagNoScan,
					// the last finalized object is not collected.
					frame = runtime·mallocgc(framesz, 0, FlagNoScan|FlagNoInvokeGC);
					framecap = framesz;
				}
				if(f->fint == nil)
					runtime·throw("missing type in runfinq");
				if(f->fint->kind == KindPtr) {
					// direct use of pointer
					*(void**)frame = f->arg;
				} else if(((InterfaceType*)f->fint)->mhdr.len == 0) {
					// convert to empty interface
					ef = (Eface*)frame;
					ef->type = f->ot;
					ef->data = f->arg;
				} else {
					// convert to interface with methods, via empty interface.
					ef1.type = f->ot;
					ef1.data = f->arg;
					if(!runtime·ifaceE2I2((InterfaceType*)f->fint, ef1, (Iface*)frame))
						runtime·throw("invalid type conversion in runfinq");
				}
				reflect·call(f->fn, frame, framesz, framesz);
				f->fn = nil;
				f->arg = nil;
				f->ot = nil;
			}
			fb->cnt = 0;
			runtime·lock(&finlock);
			fb->next = finc;
			finc = fb;
			runtime·unlock(&finlock);
		}

		// Zero everything that's dead, to avoid memory leaks.
		// See comment at top of function.
		f = nil;
		fb = nil;
		next = nil;
		i = 0;
		ef = nil;
		ef1.type = nil;
		ef1.data = nil;
		runtime·gc(1);	// trigger another gc to clean up the finalized objects, if possible
	}
}

void
runtime·createfing(void)
{
	if(fing != nil)
		return;
	// Here we use gclock instead of finlock,
	// because newproc1 can allocate, which can cause on-demand span sweep,
	// which can queue finalizers, which would deadlock.
	runtime·lock(&gclock);
	if(fing == nil)
		fing = runtime·newproc1(&runfinqv, nil, 0, 0, runtime·gc);
	runtime·unlock(&gclock);
}

G*
runtime·wakefing(void)
{
	G *res;

	res = nil;
	runtime·lock(&finlock);
	if(runtime·fingwait && runtime·fingwake) {
		runtime·fingwait = false;
		runtime·fingwake = false;
		res = fing;
	}
	runtime·unlock(&finlock);
	return res;
}

void
runtime·marknogc(void *v)
{
	uintptr *b, off, shift;

	off = (uintptr*)v - (uintptr*)runtime·mheap.arena_start;  // word offset
	b = (uintptr*)runtime·mheap.arena_start - off/wordsPerBitmapWord - 1;
	shift = off % wordsPerBitmapWord;
	*b = (*b & ~(bitAllocated<<shift)) | bitBlockBoundary<<shift;
}

void
runtime·markscan(void *v)
{
	uintptr *b, off, shift;

	off = (uintptr*)v - (uintptr*)runtime·mheap.arena_start;  // word offset
	b = (uintptr*)runtime·mheap.arena_start - off/wordsPerBitmapWord - 1;
	shift = off % wordsPerBitmapWord;
	*b |= bitScan<<shift;
}

// mark the block at v as freed.
void
runtime·markfreed(void *v)
{
	uintptr *b, off, shift, xbits;

	if(0)
		runtime·printf("markfreed %p\n", v);

	if((byte*)v > (byte*)runtime·mheap.arena_used || (byte*)v < runtime·mheap.arena_start)
		runtime·throw("markfreed: bad pointer");

	off = (uintptr*)v - (uintptr*)runtime·mheap.arena_start;  // word offset
	b = (uintptr*)runtime·mheap.arena_start - off/wordsPerBitmapWord - 1;
	shift = off % wordsPerBitmapWord;
	if(!g->m->gcing || work.nproc == 1) {
		// During normal operation (not GC), the span bitmap is not updated concurrently,
		// because either the span is cached or accesses are protected with MCentral lock.
		*b = (*b & ~(bitMask<<shift)) | (bitAllocated<<shift);
	} else {
		// During GC other threads concurrently mark heap.
		for(;;) {
			xbits = *b;
			if(runtime·casp((void**)b, (void*)xbits, (void*)((xbits & ~(bitMask<<shift)) | (bitAllocated<<shift))))
				break;
		}
	}
}

// check that the block at v of size n is marked freed.
void
runtime·checkfreed(void *v, uintptr n)
{
	uintptr *b, bits, off, shift;

	if(!runtime·checking)
		return;

	if((byte*)v+n > (byte*)runtime·mheap.arena_used || (byte*)v < runtime·mheap.arena_start)
		return;	// not allocated, so okay

	off = (uintptr*)v - (uintptr*)runtime·mheap.arena_start;  // word offset
	b = (uintptr*)runtime·mheap.arena_start - off/wordsPerBitmapWord - 1;
	shift = off % wordsPerBitmapWord;

	bits = *b>>shift;
	if((bits & bitAllocated) != 0) {
		runtime·printf("checkfreed %p+%p: off=%p have=%p\n",
			v, n, off, bits & bitMask);
		runtime·throw("checkfreed: not freed");
	}
}

// mark the span of memory at v as having n blocks of the given size.
// if leftover is true, there is left over space at the end of the span.
void
runtime·markspan(void *v, uintptr size, uintptr n, bool leftover)
{
	uintptr *b, *b0, off, shift, i, x;
	byte *p;

	if((byte*)v+size*n > (byte*)runtime·mheap.arena_used || (byte*)v < runtime·mheap.arena_start)
		runtime·throw("markspan: bad pointer");

	if(runtime·checking) {
		// bits should be all zero at the start
		off = (byte*)v + size - runtime·mheap.arena_start;
		b = (uintptr*)(runtime·mheap.arena_start - off/wordsPerBitmapWord);
		for(i = 0; i < size/PtrSize/wordsPerBitmapWord; i++) {
			if(b[i] != 0)
				runtime·throw("markspan: span bits not zero");
		}
	}

	p = v;
	if(leftover)	// mark a boundary just past end of last block too
		n++;

	b0 = nil;
	x = 0;
	for(; n-- > 0; p += size) {
		// Okay to use non-atomic ops here, because we control
		// the entire span, and each bitmap word has bits for only
		// one span, so no other goroutines are changing these
		// bitmap words.
		off = (uintptr*)p - (uintptr*)runtime·mheap.arena_start;  // word offset
		b = (uintptr*)runtime·mheap.arena_start - off/wordsPerBitmapWord - 1;
		shift = off % wordsPerBitmapWord;
		if(b0 != b) {
			if(b0 != nil)
				*b0 = x;
			b0 = b;
			x = 0;
		}
		x |= bitAllocated<<shift;
	}
	*b0 = x;
}

// unmark the span of memory at v of length n bytes.
void
runtime·unmarkspan(void *v, uintptr n)
{
	uintptr *p, *b, off;

	if((byte*)v+n > (byte*)runtime·mheap.arena_used || (byte*)v < runtime·mheap.arena_start)
		runtime·throw("markspan: bad pointer");

	p = v;
	off = p - (uintptr*)runtime·mheap.arena_start;  // word offset
	if(off % wordsPerBitmapWord != 0)
		runtime·throw("markspan: unaligned pointer");
	b = (uintptr*)runtime·mheap.arena_start - off/wordsPerBitmapWord - 1;
	n /= PtrSize;
	if(n%wordsPerBitmapWord != 0)
		runtime·throw("unmarkspan: unaligned length");
	// Okay to use non-atomic ops here, because we control
	// the entire span, and each bitmap word has bits for only
	// one span, so no other goroutines are changing these
	// bitmap words.
	n /= wordsPerBitmapWord;
	while(n-- > 0)
		*b-- = 0;
}

void
runtime·MHeap_MapBits(MHeap *h)
{
	// Caller has added extra mappings to the arena.
	// Add extra mappings of bitmap words as needed.
	// We allocate extra bitmap pieces in chunks of bitmapChunk.
	enum {
		bitmapChunk = 8192
	};
	uintptr n;

	n = (h->arena_used - h->arena_start) / wordsPerBitmapWord;
	n = ROUND(n, bitmapChunk);
	n = ROUND(n, PhysPageSize);
	if(h->bitmap_mapped >= n)
		return;

	runtime·SysMap(h->arena_start - n, n - h->bitmap_mapped, h->arena_reserved, &mstats.gc_sys);
	h->bitmap_mapped = n;
}
