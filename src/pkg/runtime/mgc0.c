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
	Debug		= 0,
	ConcurrentSweep	= 1,
	PreciseScan	= 1,

	WorkbufSize	= 4*1024,
	FinBlockSize	= 4*1024,
	RootData	= 0,
	RootBss		= 1,
	RootFinalizers	= 2,
	RootSpans	= 3,
	RootFlushCaches = 4,
	RootCount	= 5,
};

#define ScanConservatively ((byte*)1)

// Initialized from $GOGC.  GOGC=off means no gc.
extern int32 runtime·gcpercent;

static FuncVal* poolcleanup;

void
sync·runtime_registerPoolCleanup(FuncVal *f)
{
	poolcleanup = f;
}

void
runtime·clearpools(void)
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
			c->sudogcache = nil;
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

typedef struct Workbuf Workbuf;
struct Workbuf
{
	LFNode	node; // must be first
	uintptr	nobj;
	byte*	obj[(WorkbufSize-sizeof(LFNode)-sizeof(uintptr))/PtrSize];
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
byte*	runtime·gcdatamask;
byte*	runtime·gcbssmask;

static Lock	gclock;

static void	runfinq(void);
static void	bgsweep(void);
static Workbuf* getempty(Workbuf*);
static Workbuf* getfull(Workbuf*);
static void	putempty(Workbuf*);
static Workbuf* handoff(Workbuf*);
static void	gchelperstart(void);
static void	flushallmcaches(void);
static bool	scanframe(Stkframe *frame, void *unused);
static void	scanstack(G *gp);
static byte*	unrollglobgcprog(byte *prog, uintptr size);

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
	ParFor*	markfor;

	// Copy of mheap.allspans for marker or sweeper.
	MSpan**	spans;
	uint32	nspan;
} work;

// scanblock scans a block of n bytes starting at pointer b for references
// to other objects, scanning any it finds recursively until there are no
// unscanned objects left.  Instead of using an explicit recursion, it keeps
// a work list in the Workbuf* structures and loops in the main function
// body.  Keeping an explicit work list is easier on the stack allocator and
// more efficient.
static void
scanblock(byte *b, uintptr n, byte *ptrmask)
{
	byte *obj, *p, *arena_start, *arena_used, **wp, *scanbuf[8], *ptrbitp, *bitp, bits, xbits, shift, cached;
	uintptr i, nobj, size, idx, x, off, scanbufpos;
	intptr ncached;
	Workbuf *wbuf;
	String *str;
	Slice *slice;
	Iface *iface;
	Eface *eface;
	Type *typ;
	MSpan *s;
	PageID k;
	bool keepworking;

	// Cache memory arena parameters in local vars.
	arena_start = runtime·mheap.arena_start;
	arena_used = runtime·mheap.arena_used;

	wbuf = getempty(nil);
	nobj = wbuf->nobj;
	wp = &wbuf->obj[nobj];
	keepworking = b == nil;
	scanbufpos = 0;
	for(i = 0; i < nelem(scanbuf); i++)
		scanbuf[i] = nil;

	ptrbitp = nil;
	cached = 0;
	ncached = 0;

	// ptrmask can have 3 possible values:
	// 1. nil - obtain pointer mask from GC bitmap.
	// 2. ScanConservatively - don't use any mask, scan conservatively.
	// 3. pointer to a compact mask (for stacks and data).
	if(b != nil)
		goto scanobj;
	for(;;) {
		if(nobj == 0) {
			// Out of work in workbuf.
			// First, see is there is any work in scanbuf.
			for(i = 0; i < nelem(scanbuf); i++) {
				b = scanbuf[scanbufpos];
				scanbuf[scanbufpos++] = nil;
				if(scanbufpos == nelem(scanbuf))
					scanbufpos = 0;
				if(b != nil) {
					n = arena_used - b; // scan until bitBoundary or BitsDead
					ptrmask = nil; // use GC bitmap for pointer info
					goto scanobj;
				}
			}
			if(!keepworking) {
				putempty(wbuf);
				return;
			}
			// Refill workbuf from global queue.
			wbuf = getfull(wbuf);
			if(wbuf == nil)
				return;
			nobj = wbuf->nobj;
			wp = &wbuf->obj[nobj];
		}

		// If another proc wants a pointer, give it some.
		if(work.nwait > 0 && nobj > 4 && work.full == 0) {
			wbuf->nobj = nobj;
			wbuf = handoff(wbuf);
			nobj = wbuf->nobj;
			wp = &wbuf->obj[nobj];
		}

		wp--;
		nobj--;
		b = *wp;
		n = arena_used - b; // scan until next bitBoundary or BitsDead
		ptrmask = nil; // use GC bitmap for pointer info

	scanobj:
		if(!PreciseScan) {
			if(ptrmask == nil) {
				// Heap obj, obtain real size.
				if(!runtime·mlookup(b, &p, &n, nil))
					continue; // not an allocated obj
				if(b != p)
					runtime·throw("bad heap object");
			}
			ptrmask = ScanConservatively;
		}
		// Find bits of the beginning of the object.
		if(ptrmask == nil) {
			off = (uintptr*)b - (uintptr*)arena_start;
			ptrbitp = arena_start - off/wordsPerBitmapByte - 1;
			shift = (off % wordsPerBitmapByte) * gcBits;
			cached = *ptrbitp >> shift;
			cached &= ~bitBoundary;
			ncached = (8 - shift)/gcBits;
		}
		for(i = 0; i < n; i += PtrSize) {
			obj = nil;
			// Find bits for this word.
			if(ptrmask == nil) {
				// Check is we have reached end of span.
				if((((uintptr)b+i)%PageSize) == 0 &&
					runtime·mheap.spans[(b-arena_start)>>PageShift] != runtime·mheap.spans[(b+i-arena_start)>>PageShift])
					break;
				// Consult GC bitmap.
				if(ncached <= 0) {
					// Refill cache.
					cached = *--ptrbitp;
					ncached = 2;
				}
				bits = cached;
				cached >>= gcBits;
				ncached--;
				if((bits&bitBoundary) != 0)
					break; // reached beginning of the next object
				bits = (bits>>2)&BitsMask;
				if(bits == BitsDead)
					break; // reached no-scan part of the object
			} else if(ptrmask != ScanConservatively) // dense mask (stack or data)
				bits = (ptrmask[(i/PtrSize)/4]>>(((i/PtrSize)%4)*BitsPerPointer))&BitsMask;
			else
				bits = BitsPointer;

			if(bits == BitsScalar || bits == BitsDead)
				continue;
			if(bits == BitsPointer) {
				obj = *(byte**)(b+i);
				goto markobj;
			}
			// Find the next pair of bits.
			if(ptrmask == nil) {
				if(ncached <= 0) {
					// Refill cache.
					cached = *--ptrbitp;
					ncached = 2;
				}
				bits = (cached>>2)&BitsMask;
			} else
				bits = (ptrmask[((i+PtrSize)/PtrSize)/4]>>((((i+PtrSize)/PtrSize)%4)*BitsPerPointer))&BitsMask;

			switch(bits) {
			case BitsString:
				str = (String*)(b+i);
				if(str->len > 0)
					obj = str->str;
				break;
			case BitsSlice:
				slice = (Slice*)(b+i);
				if(Debug && slice->cap < slice->len) {
					g->m->traceback = 2;
					runtime·printf("bad slice in object %p: %p/%p/%p\n",
						b, slice->array, slice->len, slice->cap);
					runtime·throw("bad slice in heap object");
				}
				if(slice->cap > 0)
					obj = slice->array;
				break;
			case BitsIface:
				iface = (Iface*)(b+i);
				if(iface->tab != nil) {
					typ = iface->tab->type;
					if(!(typ->kind&KindDirectIface) || !(typ->kind&KindNoPointers))
						obj = iface->data;
				}
				break;
			case BitsEface:
				eface = (Eface*)(b+i);
				typ = eface->type;
				if(typ != nil) {
					if(!(typ->kind&KindDirectIface) || !(typ->kind&KindNoPointers))
						obj = eface->data;
				}
				break;
			}

			if(bits == BitsSlice) {
				i += 2*PtrSize;
				if(ncached == 2)
					ncached = 0;
				else if(ptrmask == nil) {
					// Refill cache and consume one quadruple.
					cached = *--ptrbitp;
					cached >>= gcBits;
					ncached = 1;
				}
			} else {
				i += PtrSize;
				cached >>= gcBits;
				ncached--;
			}

		markobj:
			// At this point we have extracted the next potential pointer.
			// Check if it points into heap.
			if(obj == nil || obj < arena_start || obj >= arena_used)
				continue;
			// Mark the object.
			off = (uintptr*)obj - (uintptr*)arena_start;
			bitp = arena_start - off/wordsPerBitmapByte - 1;
			shift = (off % wordsPerBitmapByte) * gcBits;
			xbits = *bitp;
			bits = (xbits >> shift) & bitMask;
			if((bits&bitBoundary) == 0) {
				// Not a beginning of a block, consult span table to find the block beginning.
				k = (uintptr)obj>>PageShift;
				x = k;
				x -= (uintptr)arena_start>>PageShift;
				s = runtime·mheap.spans[x];
				if(s == nil || k < s->start || obj >= s->limit || s->state != MSpanInUse)
					continue;
				p = (byte*)((uintptr)s->start<<PageShift);
				if(s->sizeclass != 0) {
					size = s->elemsize;
					idx = ((byte*)obj - p)/size;
					p = p+idx*size;
				}
				if(p == obj) {
					runtime·printf("runtime: failed to find block beginning for %p s=%p s->limit=%p\n",
						p, s->start*PageSize, s->limit);
					runtime·throw("failed to find block beginning");
				}
				obj = p;
				goto markobj;
			}

			// Now we have bits, bitp, and shift correct for
			// obj pointing at the base of the object.
			// Only care about not marked objects.
			if((bits&bitMarked) != 0)
				continue;
			// If obj size is greater than 8, then each byte of GC bitmap
			// contains info for at most one object. In such case we use
			// non-atomic byte store to mark the object. This can lead
			// to double enqueue of the object for scanning, but scanning
			// is an idempotent operation, so it is OK. This cannot lead
			// to bitmap corruption because the single marked bit is the
			// only thing that can change in the byte.
			// For 8-byte objects we use non-atomic store, if the other
			// quadruple is already marked. Otherwise we resort to CAS
			// loop for marking.
			if((xbits&(bitMask|(bitMask<<gcBits))) != (bitBoundary|(bitBoundary<<gcBits)) ||
				work.nproc == 1)
				*bitp = xbits | (bitMarked<<shift);
			else
				runtime·atomicor8(bitp, bitMarked<<shift);

			if(((xbits>>(shift+2))&BitsMask) == BitsDead)
				continue;  // noscan object

			// Queue the obj for scanning.
			PREFETCH(obj);
			obj = (byte*)((uintptr)obj & ~(PtrSize-1));
			p = scanbuf[scanbufpos];
			scanbuf[scanbufpos++] = obj;
			if(scanbufpos == nelem(scanbuf))
				scanbufpos = 0;
			if(p == nil)
				continue;

			// If workbuf is full, obtain an empty one.
			if(nobj >= nelem(wbuf->obj)) {
				wbuf->nobj = nobj;
				wbuf = getempty(wbuf);
				nobj = wbuf->nobj;
				wp = &wbuf->obj[nobj];
			}
			*wp = p;
			wp++;
			nobj++;
		}

		if(Debug && ptrmask == nil) {
			// For heap objects ensure that we did not overscan.
			n = 0;
			p = nil;
			if(!runtime·mlookup(b, &p, &n, nil) || b != p || i > n) {
				runtime·printf("runtime: scanned (%p,%p), heap object (%p,%p)\n", b, i, p, n);
				runtime·throw("scanblock: scanned invalid object");
			}
		}
	}
}

static void
markroot(ParFor *desc, uint32 i)
{
	FinBlock *fb;
	MSpan *s;
	uint32 spanidx, sg;
	G *gp;
	void *p;

	USED(&desc);
	// Note: if you add a case here, please also update heapdump.c:dumproots.
	switch(i) {
	case RootData:
		scanblock(data, edata - data, runtime·gcdatamask);
		break;

	case RootBss:
		scanblock(bss, ebss - bss, runtime·gcbssmask);
		break;

	case RootFinalizers:
		for(fb=allfin; fb; fb=fb->alllink)
			scanblock((byte*)fb->fin, fb->cnt*sizeof(fb->fin[0]), ScanConservatively);
		break;

	case RootSpans:
		// mark MSpan.specials
		sg = runtime·mheap.sweepgen;
		for(spanidx=0; spanidx<work.nspan; spanidx++) {
			Special *sp;
			SpecialFinalizer *spf;

			s = work.spans[spanidx];
			if(s->state != MSpanInUse)
				continue;
			if(s->sweepgen != sg) {
				runtime·printf("sweep %d %d\n", s->sweepgen, sg);
				runtime·throw("gc: unswept span");
			}
			for(sp = s->specials; sp != nil; sp = sp->next) {
				if(sp->kind != KindSpecialFinalizer)
					continue;
				// don't mark finalized object, but scan it so we
				// retain everything it points to.
				spf = (SpecialFinalizer*)sp;
				// A finalizer can be set for an inner byte of an object, find object beginning.
				p = (void*)((s->start << PageShift) + spf->special.offset/s->elemsize*s->elemsize);
				scanblock(p, s->elemsize, nil);
				scanblock((void*)&spf->fn, PtrSize, ScanConservatively);
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
		// Shrink a stack if not much of it is being used.
		runtime·shrinkstack(gp);
		scanstack(gp);
		break;
		
	}
}

// Get an empty work buffer off the work.empty list,
// allocating new buffers as needed.
static Workbuf*
getempty(Workbuf *b)
{
	MCache *c;

	if(b != nil)
		runtime·lfstackpush(&work.full, &b->node);
	b = nil;
	c = g->m->mcache;
	if(c->gcworkbuf != nil) {
		b = c->gcworkbuf;
		c->gcworkbuf = nil;
	}
	if(b == nil)
		b = (Workbuf*)runtime·lfstackpop(&work.empty);
	if(b == nil)
		b = runtime·persistentalloc(sizeof(*b), CacheLineSize, &mstats.gc_sys);
	b->nobj = 0;
	return b;
}

static void
putempty(Workbuf *b)
{
	MCache *c;

	c = g->m->mcache;
	if(c->gcworkbuf == nil) {
		c->gcworkbuf = b;
		return;
	}
	runtime·lfstackpush(&work.empty, &b->node);
}

void
runtime·gcworkbuffree(void *b)
{
	if(b != nil)
		putempty(b);
}

// Get a full work buffer off the work.full list, or return nil.
static Workbuf*
getfull(Workbuf *b)
{
	int32 i;

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

BitVector
runtime·stackmapdata(StackMap *stackmap, int32 n)
{
	if(n < 0 || n >= stackmap->n)
		runtime·throw("stackmapdata: index out of range");
	return (BitVector){stackmap->nbit, stackmap->data + n*((stackmap->nbit+31)/32)};
}

// Scan a stack frame: local variables and function arguments/results.
static bool
scanframe(Stkframe *frame, void *unused)
{
	Func *f;
	StackMap *stackmap;
	BitVector bv;
	uintptr size;
	uintptr targetpc;
	int32 pcdata;

	USED(unused);
	f = frame->fn;
	targetpc = frame->continpc;
	if(targetpc == 0) {
		// Frame is dead.
		return true;
	}
	if(Debug > 1)
		runtime·printf("scanframe %s\n", runtime·funcname(f));
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
	stackmap = runtime·funcdata(f, FUNCDATA_LocalsPointerMaps);
	if(stackmap == nil) {
		// No locals information, scan everything.
		size = frame->varp - (byte*)frame->sp;
		if(Debug > 2)
			runtime·printf("frame %s unsized locals %p+%p\n", runtime·funcname(f), frame->varp-size, size);
		scanblock(frame->varp - size, size, ScanConservatively);
	} else if(stackmap->n < 0) {
		// Locals size information, scan just the locals.
		size = -stackmap->n;
		if(Debug > 2)
			runtime·printf("frame %s conservative locals %p+%p\n", runtime·funcname(f), frame->varp-size, size);
		scanblock(frame->varp - size, size, ScanConservatively);
	} else if(stackmap->n > 0) {
		// Locals bitmap information, scan just the pointers in locals.
		if(pcdata < 0 || pcdata >= stackmap->n) {
			// don't know where we are
			runtime·printf("pcdata is %d and %d stack map entries for %s (targetpc=%p)\n",
				pcdata, stackmap->n, runtime·funcname(f), targetpc);
			runtime·throw("scanframe: bad symbol table");
		}
		bv = runtime·stackmapdata(stackmap, pcdata);
		size = (bv.n * PtrSize) / BitsPerPointer;
		scanblock(frame->varp - size, bv.n/BitsPerPointer*PtrSize, (byte*)bv.data);
	}

	// Scan arguments.
	// Use pointer information if known.
	stackmap = runtime·funcdata(f, FUNCDATA_ArgsPointerMaps);
	if(stackmap != nil) {
		bv = runtime·stackmapdata(stackmap, pcdata);
		scanblock(frame->argp, bv.n/BitsPerPointer*PtrSize, (byte*)bv.data);
	} else {
		if(Debug > 2)
			runtime·printf("frame %s conservative args %p+%p\n", runtime·funcname(f), frame->argp, (uintptr)frame->arglen);
		scanblock(frame->argp, frame->arglen, ScanConservatively);
	}
	return true;
}

static void
scanstack(G *gp)
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
		runtime·gentraceback(~(uintptr)0, ~(uintptr)0, 0, gp, 0, nil, 0x7fffffff, scanframe, nil, false);
	} else {
		n = 0;
		while(stk) {
			if(sp < guard-StackGuard || (uintptr)stk < sp) {
				runtime·printf("scanstack inconsistent: g%D#%d sp=%p not in [%p,%p]\n", gp->goid, n, sp, guard-StackGuard, stk);
				runtime·throw("scanstack");
			}
			if(Debug > 2)
				runtime·printf("conservative stack %p+%p\n", (byte*)sp, (uintptr)stk-sp);
			scanblock((byte*)sp, (uintptr)stk - sp, ScanConservatively);
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
		runtime·MSpan_Sweep(s, false);
		return;
	}
	// unfortunate condition, and we don't have efficient means to wait
	while(runtime·atomicload(&s->sweepgen) != sg)
		runtime·osyield();  
}

// Sweep frees or collects finalizers for blocks not marked in the mark phase.
// It clears the mark bits in preparation for the next GC round.
// Returns true if the span was returned to heap.
// If preserve=true, don't return it to heap nor relink in MCentral lists;
// caller takes care of it.
bool
runtime·MSpan_Sweep(MSpan *s, bool preserve)
{
	int32 cl, n, npages, nfree;
	uintptr size, off, step;
	uint32 sweepgen;
	byte *p, *bitp, shift, xbits, bits;
	MCache *c;
	byte *arena_start;
	MLink head, *end, *link;
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

	// Mark any free objects in this span so we don't collect them.
	for(link = s->freelist; link != nil; link = link->next) {
		off = (uintptr*)link - (uintptr*)arena_start;
		bitp = arena_start - off/wordsPerBitmapByte - 1;
		shift = (off % wordsPerBitmapByte) * gcBits;
		*bitp |= bitMarked<<shift;
	}

	// Unlink & free special records for any objects we're about to free.
	specialp = &s->specials;
	special = *specialp;
	while(special != nil) {
		// A finalizer can be set for an inner byte of an object, find object beginning.
		p = (byte*)(s->start << PageShift) + special->offset/size*size;
		off = (uintptr*)p - (uintptr*)arena_start;
		bitp = arena_start - off/wordsPerBitmapByte - 1;
		shift = (off % wordsPerBitmapByte) * gcBits;
		bits = (*bitp>>shift) & bitMask;
		if((bits&bitMarked) == 0) {
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

	// Sweep through n objects of given size starting at p.
	// This thread owns the span now, so it can manipulate
	// the block bitmap without atomic operations.
	p = (byte*)(s->start << PageShift);
	// Find bits for the beginning of the span.
	off = (uintptr*)p - (uintptr*)arena_start;
	bitp = arena_start - off/wordsPerBitmapByte - 1;
	shift = 0;
	step = size/(PtrSize*wordsPerBitmapByte);
	// Rewind to the previous quadruple as we move to the next
	// in the beginning of the loop.
	bitp += step;
	if(step == 0) {
		// 8-byte objects.
		bitp++;
		shift = gcBits;
	}
	for(; n > 0; n--, p += size) {
		bitp -= step;
		if(step == 0) {
			if(shift != 0)
				bitp--;
			shift = gcBits - shift;
		}

		xbits = *bitp;
		bits = (xbits>>shift) & bitMask;

		// Allocated and marked object, reset bits to allocated.
		if((bits&bitMarked) != 0) {
			*bitp &= ~(bitMarked<<shift);
			continue;
		}
		// At this point we know that we are looking at garbage object
		// that needs to be collected.
		if(runtime·debug.allocfreetrace)
			runtime·tracefree(p, size);
		// Reset to allocated+noscan.
		*bitp = (xbits & ~((bitMarked|(BitsMask<<2))<<shift)) | ((uintptr)BitsDead<<(shift+2));
		if(cl == 0) {
			// Free large span.
			if(preserve)
				runtime·throw("can't preserve large span");
			runtime·unmarkspan(p, s->npages<<PageShift);
			s->needzero = 1;
			// important to set sweepgen before returning it to heap
			runtime·atomicstore(&s->sweepgen, sweepgen);
			sweepgenset = true;
			// NOTE(rsc,dvyukov): The original implementation of efence
			// in CL 22060046 used SysFree instead of SysFault, so that
			// the operating system would eventually give the memory
			// back to us again, so that an efence program could run
			// longer without running out of memory. Unfortunately,
			// calling SysFree here without any kind of adjustment of the
			// heap data structures means that when the memory does
			// come back to us, we have the wrong metadata for it, either in
			// the MSpan structures or in the garbage collection bitmap.
			// Using SysFault here means that the program will run out of
			// memory fairly quickly in efence mode, but at least it won't
			// have mysterious crashes due to confused memory reuse.
			// It should be possible to switch back to SysFree if we also
			// implement and then call some kind of MHeap_DeleteSpan.
			if(runtime·debug.efence) {
				s->limit = nil;	// prevent mlookup from finding this span
				runtime·SysFault(p, size);
			} else
				runtime·MHeap_Free(&runtime·mheap, s, 1);
			c->local_nlargefree++;
			c->local_largefree += size;
			runtime·xadd64(&mstats.next_gc, -(uint64)(size * (runtime·gcpercent + 100)/100));
			res = true;
		} else {
			// Free small object.
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
		runtime·xadd64(&mstats.next_gc, -(uint64)(nfree * size * (runtime·gcpercent + 100)/100));
		res = runtime·MCentral_FreeSpan(&runtime·mheap.central[cl].mcentral, s, nfree, head.next, end, preserve);
		// MCentral_FreeSpan updates sweepgen
	}
	return res;
}

// State of background sweep.
// Pretected by gclock.
static struct
{
	G*	g;
	bool	parked;

	uint32	spanidx;	// background sweeper position

	uint32	nbgsweep;
	uint32	npausesweep;
} sweep;

// background sweeping goroutine
static void
bgsweep(void)
{
	g->issystem = 1;
	for(;;) {
		while(runtime·sweepone() != -1) {
			sweep.nbgsweep++;
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
		runtime·parkunlock(&gclock, runtime·gostringnocopy((byte*)"GC sweep wait"));
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
		if(idx >= work.nspan) {
			runtime·mheap.sweepdone = true;
			g->m->locks--;
			return -1;
		}
		s = work.spans[idx];
		if(s->state != MSpanInUse) {
			s->sweepgen = sg;
			continue;
		}
		if(s->sweepgen != sg-2 || !runtime·cas(&s->sweepgen, sg-2, sg-1))
			continue;
		npages = s->npages;
		if(!runtime·MSpan_Sweep(s, false))
			npages = 0;
		g->m->locks--;
		return npages;
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
	scanblock(nil, 0, nil);

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
		runtime·stackcache_clear(c);
	}
}

static void
flushallmcaches_m(G *gp)
{
	flushallmcaches();
	runtime·gogo(&gp->sched);
}

void
runtime·updatememstats(GCStats *stats)
{
	M *mp;
	MSpan *s;
	int32 i;
	uint64 smallfree;
	uint64 *src, *dst;

	if(stats)
		runtime·memclr((byte*)stats, sizeof(*stats));
	for(mp=runtime·allm; mp; mp=mp->alllink) {
		if(stats) {
			src = (uint64*)&mp->gcstats;
			dst = (uint64*)stats;
			for(i=0; i<sizeof(*stats)/sizeof(uint64); i++)
				dst[i] += src[i];
			runtime·memclr((byte*)&mp->gcstats, sizeof(mp->gcstats));
		}
	}
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
	if(g == g->m->g0)
		flushallmcaches();
	else
		runtime·mcall(flushallmcaches_m);

	// Aggregate local stats.
	cachestats();

	// Scan all spans and count number of alive objects.
	runtime·lock(&runtime·mheap.lock);
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
	runtime·unlock(&runtime·mheap.lock);

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

int32
runtime·readgogc(void)
{
	byte *p;

	p = runtime·getenv("GOGC");
	if(p == nil || p[0] == '\0')
		return 100;
	if(runtime·strcmp(p, (byte*)"off") == 0)
		return -1;
	return runtime·atoi(p);
}

void
runtime·gcinit(void)
{
	if(sizeof(Workbuf) != WorkbufSize)
		runtime·throw("runtime: size of Workbuf is suboptimal");

	work.markfor = runtime·parforalloc(MaxGcproc);
	runtime·gcpercent = runtime·readgogc();
	runtime·gcdatamask = unrollglobgcprog(gcdata, edata - data);
	runtime·gcbssmask = unrollglobgcprog(gcbss, ebss - bss);
}

// force = 1 - do GC regardless of current heap usage
// force = 2 - go GC and eager sweep
void
runtime·gc(int32 force)
{
	struct gc_args a;
	int32 i;

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

	if(runtime·gcpercent < 0)
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
	
	runtime·clearpools();

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
		g->waitreason = runtime·gostringnocopy((byte*)"garbage collection");
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

void
runtime·gc_m(void)
{
	struct gc_args a;
	G *gp;

	gp = g->m->curg;
	gp->status = Gwaiting;
	gp->waitreason = runtime·gostringnocopy((byte*)"garbage collection");

	a.start_time = (uint64)(g->m->scalararg[0]) | ((uint64)(g->m->scalararg[1]) << 32);
	a.eagersweep = g->m->scalararg[2];
	gc(&a);

	gp->status = Grunning;
}

static void
gc(struct gc_args *args)
{
	int64 t0, t1, t2, t3, t4;
	uint64 heap0, heap1, obj;
	GCStats stats;

	if(runtime·debug.allocfreetrace)
		runtime·tracegc();

	g->m->traceback = 2;
	t0 = args->start_time;
	work.tstart = args->start_time; 

	t1 = 0;
	if(runtime·debug.gctrace)
		t1 = runtime·nanotime();

	// Sweep what is not sweeped by bgsweep.
	while(runtime·sweepone() != -1)
		sweep.npausesweep++;

	// Cache runtime.mheap.allspans in work.spans to avoid conflicts with
	// resizing/freeing allspans.
	// New spans can be created while GC progresses, but they are not garbage for
	// this round:
	//  - new stack spans can be created even while the world is stopped.
	//  - new malloc spans can be created during the concurrent sweep

	// Even if this is stop-the-world, a concurrent exitsyscall can allocate a stack from heap.
	runtime·lock(&runtime·mheap.lock);
	// Free the old cached sweep array if necessary.
	if(work.spans != nil && work.spans != runtime·mheap.allspans)
		runtime·SysFree(work.spans, work.nspan*sizeof(work.spans[0]), &mstats.other_sys);
	// Cache the current array for marking.
	runtime·mheap.gcspans = runtime·mheap.allspans;
	work.spans = runtime·mheap.allspans;
	work.nspan = runtime·mheap.nspan;
	runtime·unlock(&runtime·mheap.lock);

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
	scanblock(nil, 0, nil);

	t3 = 0;
	if(runtime·debug.gctrace)
		t3 = runtime·nanotime();

	if(work.nproc > 1)
		runtime·notesleep(&work.alldone);

	cachestats();
	// next_gc calculation is tricky with concurrent sweep since we don't know size of live heap
	// estimate what was live heap size after previous GC (for tracing only)
	heap0 = mstats.next_gc*100/(runtime·gcpercent+100);
	// conservatively set next_gc to high value assuming that everything is live
	// concurrent/lazy sweep will reduce this number while discovering new garbage
	mstats.next_gc = mstats.heap_alloc+mstats.heap_alloc*runtime·gcpercent/100;

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
			work.nspan, sweep.nbgsweep, sweep.npausesweep,
			stats.nhandoff, stats.nhandoffcnt,
			work.markfor->nsteal, work.markfor->nstealcnt,
			stats.nprocyield, stats.nosyield, stats.nsleep);
		sweep.nbgsweep = sweep.npausesweep = 0;
	}

	// See the comment in the beginning of this function as to why we need the following.
	// Even if this is still stop-the-world, a concurrent exitsyscall can allocate a stack from heap.
	runtime·lock(&runtime·mheap.lock);
	// Free the old cached mark array if necessary.
	if(work.spans != nil && work.spans != runtime·mheap.allspans)
		runtime·SysFree(work.spans, work.nspan*sizeof(work.spans[0]), &mstats.other_sys);
	// Cache the current array for sweeping.
	runtime·mheap.gcspans = runtime·mheap.allspans;
	runtime·mheap.sweepgen += 2;
	runtime·mheap.sweepdone = false;
	work.spans = runtime·mheap.allspans;
	work.nspan = runtime·mheap.nspan;
	sweep.spanidx = 0;
	runtime·unlock(&runtime·mheap.lock);

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
			sweep.npausesweep++;
	}

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
	runtime·memmove(stats, &mstats, runtime·sizeof_C_MStats);

	// Stack numbers are part of the heap numbers, separate those out for user consumption
	stats->stacks_sys = stats->stacks_inuse;
	stats->heap_inuse -= stats->stacks_inuse;
	stats->heap_sys -= stats->stacks_inuse;
	
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
	runtime·lock(&runtime·mheap.lock);
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
	runtime·unlock(&runtime·mheap.lock);
	pauses->len = n+3;
}

void
runtime·setgcpercent_m(void) {
	int32 in;
	int32 out;

	in = (int32)(intptr)g->m->scalararg[0];

	runtime·lock(&runtime·mheap.lock);
	out = runtime·gcpercent;
	if(in < 0)
		in = -1;
	runtime·gcpercent = in;
	runtime·unlock(&runtime·mheap.lock);

	g->m->scalararg[0] = (uintptr)(intptr)out;
}

static void
gchelperstart(void)
{
	if(g->m->helpgc < 0 || g->m->helpgc >= MaxGcproc)
		runtime·throw("gchelperstart: bad m->helpgc");
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
			runtime·parkunlock(&finlock, runtime·gostringnocopy((byte*)"finalizer wait"));
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
					// The frame does not contain pointers interesting for GC,
					// all not yet finalized objects are stored in finq.
					// If we do not mark it as FlagNoScan,
					// the last finalized object is not collected.
					frame = runtime·mallocgc(framesz, 0, FlagNoScan);
					framecap = framesz;
				}
				if(f->fint == nil)
					runtime·throw("missing type in runfinq");
				if((f->fint->kind&KindMask) == KindPtr) {
					// direct use of pointer
					*(void**)frame = f->arg;
				} else if(((InterfaceType*)f->fint)->mhdr.len == 0) {
					// convert to empty interface
					ef = (Eface*)frame;
					ef->type = &f->ot->typ;
					ef->data = f->arg;
				} else {
					// convert to interface with methods, via empty interface.
					ef1.type = &f->ot->typ;
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
	if(runtime·fing != nil)
		return;
	// Here we use gclock instead of finlock,
	// because newproc1 can allocate, which can cause on-demand span sweep,
	// which can queue finalizers, which would deadlock.
	runtime·lock(&gclock);
	if(runtime·fing == nil)
		runtime·fing = runtime·newproc1(&runfinqv, nil, 0, 0, runtime·gc);
	runtime·unlock(&gclock);
}

void
runtime·createfingM(G *gp)
{
	runtime·createfing();
	runtime·gogo(&gp->sched);
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
		res = runtime·fing;
	}
	runtime·unlock(&finlock);
	return res;
}

// Recursively unrolls GC program in prog.
// mask is where to store the result.
// ppos is a pointer to position in mask, in bits.
// sparse says to generate 4-bits per word mask for heap (2-bits for data/bss otherwise).
static byte*
unrollgcprog1(byte *mask, byte *prog, uintptr *ppos, bool inplace, bool sparse)
{
	uintptr pos, siz, i, off;
	byte *arena_start, *prog1, v, *bitp, shift;

	arena_start = runtime·mheap.arena_start;
	pos = *ppos;
	for(;;) {
		switch(prog[0]) {
		case insData:
			prog++;
			siz = prog[0];
			prog++;
			for(i = 0; i < siz; i++) {
				v = prog[i/PointersPerByte];
				v >>= (i%PointersPerByte)*BitsPerPointer;
				v &= BitsMask;
				if(inplace) {
					// Store directly into GC bitmap.
					off = (uintptr*)(mask+pos) - (uintptr*)arena_start;
					bitp = arena_start - off/wordsPerBitmapByte - 1;
					shift = (off % wordsPerBitmapByte) * gcBits;
					if(shift==0)
						*bitp = 0;
					*bitp |= v<<(shift+2);
					pos += PtrSize;
				} else if(sparse) {
					// 4-bits per word
					v <<= (pos%8)+2;
					mask[pos/8] |= v;
					pos += gcBits;
				} else {
					// 2-bits per word
					v <<= pos%8;
					mask[pos/8] |= v;
					pos += BitsPerPointer;
				}
			}
			prog += ROUND(siz*BitsPerPointer, 8)/8;
			break;
		case insArray:
			prog++;
			siz = 0;
			for(i = 0; i < PtrSize; i++)
				siz = (siz<<8) + prog[PtrSize-i-1];
			prog += PtrSize;
			prog1 = nil;
			for(i = 0; i < siz; i++)
				prog1 = unrollgcprog1(mask, prog, &pos, inplace, sparse);
			if(prog1[0] != insArrayEnd)
				runtime·throw("unrollgcprog: array does not end with insArrayEnd");
			prog = prog1+1;
			break;
		case insArrayEnd:
		case insEnd:
			*ppos = pos;
			return prog;
		default:
			runtime·throw("unrollgcprog: unknown instruction");
		}
	}
}

// Unrolls GC program prog for data/bss, returns dense GC mask.
static byte*
unrollglobgcprog(byte *prog, uintptr size)
{
	byte *mask;
	uintptr pos, masksize;

	masksize = ROUND(ROUND(size, PtrSize)/PtrSize*BitsPerPointer, 8)/8;
	mask = runtime·persistentalloc(masksize+1, 0, &mstats.gc_sys);
	mask[masksize] = 0xa1;
	pos = 0;
	prog = unrollgcprog1(mask, prog, &pos, false, false);
	if(pos != size/PtrSize*BitsPerPointer) {
		runtime·printf("unrollglobgcprog: bad program size, got %D, expect %D\n",
			(uint64)pos, (uint64)size/PtrSize*BitsPerPointer);
		runtime·throw("unrollglobgcprog: bad program size");
	}
	if(prog[0] != insEnd)
		runtime·throw("unrollglobgcprog: program does not end with insEnd");
	if(mask[masksize] != 0xa1)
		runtime·throw("unrollglobgcprog: overflow");
	return mask;
}

void
runtime·unrollgcproginplace_m(void)
{
	uintptr size, size0, pos, off;
	byte *arena_start, *prog, *bitp, shift;
	Type *typ;
	void *v;

	v = g->m->ptrarg[0];
	typ = g->m->ptrarg[1];
	size = g->m->scalararg[0];
	size0 = g->m->scalararg[1];
	g->m->ptrarg[0] = nil;
	g->m->ptrarg[1] = nil;

	pos = 0;
	prog = (byte*)typ->gc[1];
	while(pos != size0)
		unrollgcprog1(v, prog, &pos, true, true);
	// Mark first word as bitAllocated.
	arena_start = runtime·mheap.arena_start;
	off = (uintptr*)v - (uintptr*)arena_start;
	bitp = arena_start - off/wordsPerBitmapByte - 1;
	shift = (off % wordsPerBitmapByte) * gcBits;
	*bitp |= bitBoundary<<shift;
	// Mark word after last as BitsDead.
	if(size0 < size) {
		off = (uintptr*)((byte*)v + size0) - (uintptr*)arena_start;
		bitp = arena_start - off/wordsPerBitmapByte - 1;
		shift = (off % wordsPerBitmapByte) * gcBits;
		*bitp &= ~(bitPtrMask<<shift) | ((uintptr)BitsDead<<(shift+2));
	}
}

// Unrolls GC program in typ->gc[1] into typ->gc[0]
void
runtime·unrollgcprog_m(void)
{
	static Lock lock;
	Type *typ;
	byte *mask, *prog;
	uintptr pos;
	uint32 x;

	typ = g->m->ptrarg[0];
	g->m->ptrarg[0] = nil;

	runtime·lock(&lock);
	mask = (byte*)typ->gc[0];
	if(mask[0] == 0) {
		pos = 8;  // skip the unroll flag
		prog = (byte*)typ->gc[1];
		prog = unrollgcprog1(mask, prog, &pos, false, true);
		if(prog[0] != insEnd)
			runtime·throw("unrollgcprog: program does not end with insEnd");
		if(((typ->size/PtrSize)%2) != 0) {
			// repeat the program twice
			prog = (byte*)typ->gc[1];
			unrollgcprog1(mask, prog, &pos, false, true);
		}
		// atomic way to say mask[0] = 1
		x = ((uint32*)mask)[0];
		runtime·atomicstore((uint32*)mask, x|1);
	}
	runtime·unlock(&lock);
}

// mark the span of memory at v as having n blocks of the given size.
// if leftover is true, there is left over space at the end of the span.
void
runtime·markspan(void *v, uintptr size, uintptr n, bool leftover)
{
	uintptr i, off, step;
	byte *b;

	if((byte*)v+size*n > (byte*)runtime·mheap.arena_used || (byte*)v < runtime·mheap.arena_start)
		runtime·throw("markspan: bad pointer");

	// Find bits of the beginning of the span.
	off = (uintptr*)v - (uintptr*)runtime·mheap.arena_start;  // word offset
	b = runtime·mheap.arena_start - off/wordsPerBitmapByte - 1;
	if((off%wordsPerBitmapByte) != 0)
		runtime·throw("markspan: unaligned length");

	// Okay to use non-atomic ops here, because we control
	// the entire span, and each bitmap byte has bits for only
	// one span, so no other goroutines are changing these bitmap words.

	if(size == PtrSize) {
		// Possible only on 64-bits (minimal size class is 8 bytes).
		// Poor man's memset(0x11).
		if(0x11 != ((bitBoundary+BitsDead)<<gcBits) + (bitBoundary+BitsDead))
			runtime·throw("markspan: bad bits");
		if((n%(wordsPerBitmapByte*PtrSize)) != 0)
			runtime·throw("markspan: unaligned length");
		b = b - n/wordsPerBitmapByte + 1;	// find first byte
		if(((uintptr)b%PtrSize) != 0)
			runtime·throw("markspan: unaligned pointer");
		for(i = 0; i != n; i += wordsPerBitmapByte*PtrSize, b += PtrSize)
			*(uintptr*)b = (uintptr)0x1111111111111111ULL;  // bitBoundary+BitsDead
		return;
	}

	if(leftover)
		n++;	// mark a boundary just past end of last block too
	step = size/(PtrSize*wordsPerBitmapByte);
	for(i = 0; i != n; i++, b -= step)
		*b = bitBoundary|(BitsDead<<2);
}

// unmark the span of memory at v of length n bytes.
void
runtime·unmarkspan(void *v, uintptr n)
{
	uintptr off;
	byte *b;

	if((byte*)v+n > (byte*)runtime·mheap.arena_used || (byte*)v < runtime·mheap.arena_start)
		runtime·throw("markspan: bad pointer");

	off = (uintptr*)v - (uintptr*)runtime·mheap.arena_start;  // word offset
	if((off % (PtrSize*wordsPerBitmapByte)) != 0)
		runtime·throw("markspan: unaligned pointer");
	b = runtime·mheap.arena_start - off/wordsPerBitmapByte - 1;
	n /= PtrSize;
	if(n%(PtrSize*wordsPerBitmapByte) != 0)
		runtime·throw("unmarkspan: unaligned length");
	// Okay to use non-atomic ops here, because we control
	// the entire span, and each bitmap word has bits for only
	// one span, so no other goroutines are changing these
	// bitmap words.
	n /= wordsPerBitmapByte;
	runtime·memclr(b - n + 1, n);
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

	n = (h->arena_used - h->arena_start) / (PtrSize*wordsPerBitmapByte);
	n = ROUND(n, bitmapChunk);
	n = ROUND(n, PhysPageSize);
	if(h->bitmap_mapped >= n)
		return;

	runtime·SysMap(h->arena_start - n, n - h->bitmap_mapped, h->arena_reserved, &mstats.gc_sys);
	h->bitmap_mapped = n;
}

static bool
getgcmaskcb(Stkframe *frame, void *ctxt)
{
	Stkframe *frame0;

	frame0 = ctxt;
	if(frame0->sp >= (uintptr)frame->varp - frame->sp && frame0->sp < (uintptr)frame->varp) {
		*frame0 = *frame;
		return false;
	}
	return true;
}

// Returns GC type info for object p for testing.
void
runtime·getgcmask(byte *p, Type *t, byte **mask, uintptr *len)
{
	Stkframe frame;
	uintptr i, n, off;
	byte *base, bits, shift, *b;

	*mask = nil;
	*len = 0;

	// data
	if(p >= data && p < edata) {
		n = ((PtrType*)t)->elem->size;
		*len = n/PtrSize;
		*mask = runtime·mallocgc(*len, nil, 0);
		for(i = 0; i < n; i += PtrSize) {
			off = (p+i-data)/PtrSize;
			bits = (runtime·gcdatamask[off/PointersPerByte] >> ((off%PointersPerByte)*BitsPerPointer))&BitsMask;
			(*mask)[i/PtrSize] = bits;
		}
		return;
	}
	// bss
	if(p >= bss && p < ebss) {
		n = ((PtrType*)t)->elem->size;
		*len = n/PtrSize;
		*mask = runtime·mallocgc(*len, nil, 0);
		for(i = 0; i < n; i += PtrSize) {
			off = (p+i-bss)/PtrSize;
			bits = (runtime·gcbssmask[off/PointersPerByte] >> ((off%PointersPerByte)*BitsPerPointer))&BitsMask;
			(*mask)[i/PtrSize] = bits;
		}
		return;
	}
	// heap
	if(runtime·mlookup(p, &base, &n, nil)) {
		*len = n/PtrSize;
		*mask = runtime·mallocgc(*len, nil, 0);
		for(i = 0; i < n; i += PtrSize) {
			off = (uintptr*)(base+i) - (uintptr*)runtime·mheap.arena_start;
			b = runtime·mheap.arena_start - off/wordsPerBitmapByte - 1;
			shift = (off % wordsPerBitmapByte) * gcBits;
			bits = (*b >> (shift+2))&BitsMask;
			(*mask)[i/PtrSize] = bits;
		}
		return;
	}
	// stack
	frame.fn = nil;
	frame.sp = (uintptr)p;
	runtime·gentraceback((uintptr)runtime·getcallerpc(&p), (uintptr)runtime·getcallersp(&p), 0, g, 0, nil, 1000, getgcmaskcb, &frame, false);
	if(frame.fn != nil) {
		Func *f;
		StackMap *stackmap;
		BitVector bv;
		uintptr size;
		uintptr targetpc;
		int32 pcdata;

		f = frame.fn;
		targetpc = frame.continpc;
		if(targetpc == 0)
			return;
		if(targetpc != f->entry)
			targetpc--;
		pcdata = runtime·pcdatavalue(f, PCDATA_StackMapIndex, targetpc);
		if(pcdata == -1)
			return;
		stackmap = runtime·funcdata(f, FUNCDATA_LocalsPointerMaps);
		if(stackmap == nil || stackmap->n <= 0)
			return;
		bv = runtime·stackmapdata(stackmap, pcdata);
		size = bv.n/BitsPerPointer*PtrSize;
		n = ((PtrType*)t)->elem->size;
		*len = n/PtrSize;
		*mask = runtime·mallocgc(*len, nil, 0);
		for(i = 0; i < n; i += PtrSize) {
			off = (p+i-frame.varp+size)/PtrSize;
			bits = (bv.data[off*BitsPerPointer/32] >> ((off*BitsPerPointer)%32))&BitsMask;
			(*mask)[i/PtrSize] = bits;
		}
	}
}
