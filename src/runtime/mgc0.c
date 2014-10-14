// Copyright 2009 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// Garbage collector (GC).
//
// The GC runs concurrently with mutator threads, is type accurate (aka precise), allows multiple GC 
// thread to run in parallel. It is a concurrent mark and sweep that uses a write barrier. It is 
// non-generational and non-compacting. Allocation is done using size segregated per P allocation 
// areas to minimize fragmentation while eliminating locks in the common case. 
//
// The algorithm decomposes into several steps.
// This is a high level description of the algorithm being used. For an overview of GC a good
// place to start is Richard Jones' gchandbook.org.
// 
// The algorithm's intellectual heritage includes Dijkstra's on-the-fly algorithm, see
// Edsger W. Dijkstra, Leslie Lamport, A. J. Martin, C. S. Scholten, and E. F. M. Steffens. 1978. 
// On-the-fly garbage collection: an exercise in cooperation. Commun. ACM 21, 11 (November 1978), 966-975.
// For journal quality proofs that these steps are complete, correct, and terminate see
// Hudson, R., and Moss, J.E.B. Copying Garbage Collection without stopping the world. 
// Concurrency and Computation: Practice and Experience 15(3-5), 2003. 
//
//  0. Set phase = GCscan from GCoff.
//  1. Wait for all P's to acknowledge phase change.
//         At this point all goroutines have passed through a GC safepoint and
//         know we are in the GCscan phase.
//  2. GC scans all goroutine stacks, mark and enqueues all encountered pointers
//       (marking avoids most duplicate enqueuing but races may produce duplication which is benign).
//       Preempted goroutines are scanned before P schedules next goroutine.
//  3. Set phase = GCmark.
//  4. Wait for all P's to acknowledge phase change.
//  5. Now write barrier marks and enqueues black or grey to white pointers. If a pointer is
//       stored into a white slot, such pointer is not marked.
//       Malloc still allocates white (non-marked) objects.
//  6. Meanwhile GC transitively walks the heap marking reachable objects.
//  7. When GC finishes marking heap, it preempts P's one-by-one and
//       retakes partial wbufs (filled by write barrier or during a stack scan of the goroutine
//       currently scheduled on the P).
//  8. Once the GC has exhausted all available marking work it sets phase = marktermination.
//  9. Wait for all P's to acknowledge phase change.
// 10. Malloc now allocates black objects, so number of unmarked reachable objects
//        monotonically decreases.
// 11. GC preempts P's one-by-one taking partial wbufs and marks all unmarked yet reachable objects.
// 12. When GC completes a full cycle over P's and discovers no new grey
//         objects, (which means all reachable objects are marked) set phase = GCsweep.
// 13. Wait for all P's to acknowledge phase change.
// 14. Now malloc allocates white (but sweeps spans before use).
//         Write barrier becomes nop.
// 15. GC does background sweeping, see description below.
// 16. When sweeping is complete set phase to GCoff.
// 17. When sufficient allocation has taken place replay the sequence starting at 0 above, 
//         see discussion of GC rate below.

// Changing phases.
// Phases are changed by setting the gcphase to the next phase and call ackgcphase.
// All phase action must be benign in the presence of a change.
// Starting with GCoff
// GCoff to GCscan
//     GSscan scans stacks and globals greying them and never marks an object black.
//     Once all the P's are aware of the new phase they will scan gs on preemption.
//     This means that the scanning of preempted gs can't start until all the Ps
//     have acknowledged.
// GCscan to GCmark
//     GCMark turns on the write barrier which also only greys objects. No scanning
//     of objects (making them black) can happen until all the Ps have acknowledged 
//     the phase change.
// GCmark to GCmarktermination
//     The only change here is that we start allocating black so the Ps must acknowledge
//     the change before we begin the termination algorithm
// GCmarktermination to GSsweep
//     Object currently on the freelist must be marked black for this to work. 
//     Are things on the free lists black or white? How does the sweep phase work?

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

// GC rate.
// Next GC is after we've allocated an extra amount of memory proportional to
// the amount already in use. The proportion is controlled by GOGC environment variable
// (100 by default). If GOGC=100 and we're using 4M, we'll GC again when we get to 8M
// (this mark is tracked in next_gc variable). This keeps the GC cost in linear 
// proportion to the allocation cost. Adjusting GOGC just changes the linear constant	
// (and also the amount of extra memory used).

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
#include "textflag.h"

enum {
	Debug		= 0,
	ConcurrentSweep	= 1,

	FinBlockSize	= 4*1024,
	RootData	= 0,
	RootBss		= 1,
	RootFinalizers	= 2,
	RootSpans	= 3,
	RootFlushCaches = 4,
	RootCount	= 5,
};

// ptrmask for an allocation containing a single pointer.
static byte oneptr[] = {BitsPointer};

// Initialized from $GOGC.  GOGC=off means no gc.
extern int32 runtime·gcpercent;

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

typedef struct Markbits Markbits;
struct Markbits {
	byte *bitp; // pointer to the byte holding xbits
 	byte shift; // bits xbits needs to be shifted to get bits
	byte xbits; // byte holding all the bits from *bitp
	byte bits;  // bits relevant to corresponding slot.
};

extern byte runtime·data[];
extern byte runtime·edata[];
extern byte runtime·bss[];
extern byte runtime·ebss[];

extern byte runtime·gcdata[];
extern byte runtime·gcbss[];

Mutex	runtime·finlock;	// protects the following variables
G*	runtime·fing;		// goroutine that runs finalizers
FinBlock*	runtime·finq;	// list of finalizers that are to be executed
FinBlock*	runtime·finc;	// cache of free blocks
static byte finptrmask[FinBlockSize/PtrSize/PointersPerByte];
bool	runtime·fingwait;
bool	runtime·fingwake;
FinBlock	*runtime·allfin;	// list of all blocks

BitVector	runtime·gcdatamask;
BitVector	runtime·gcbssmask;

Mutex	runtime·gclock;

static Workbuf* getpartial(void);
static void	putpartial(Workbuf*);
static Workbuf* getempty(Workbuf*);
static Workbuf* getfull(Workbuf*);
static void	putempty(Workbuf*);
static Workbuf* handoff(Workbuf*);
static void	gchelperstart(void);
static void	flushallmcaches(void);
static bool	scanframe(Stkframe*, void*);
static void	scanstack(G*);
static BitVector	unrollglobgcprog(byte*, uintptr);
static void     scanblock(byte*, uintptr, byte*);
static byte*    objectstart(byte*, Markbits*);
static Workbuf*	greyobject(byte*, Markbits*, Workbuf*);
static bool     inheap(byte*);
static bool     shaded(byte*);
static void     shade(byte*);
static void	slottombits(byte*, Markbits*);

void runtime·bgsweep(void);
static FuncVal bgsweepv = {runtime·bgsweep};

typedef struct WorkData WorkData;
struct WorkData {
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
};
WorkData runtime·work;

// Is address b in the known heap. If it doesn't have a valid gcmap
// returns false. For example pointers into stacks will return false.
static bool
inheap(byte *b)
{
	MSpan *s;
	pageID k;
	uintptr x;

	if(b == nil || b < runtime·mheap.arena_start || b >= runtime·mheap.arena_used)
		return false;
	// Not a beginning of a block, consult span table to find the block beginning.
	k = (uintptr)b>>PageShift;
	x = k;
	x -= (uintptr)runtime·mheap.arena_start>>PageShift;
	s = runtime·mheap.spans[x];
	if(s == nil || k < s->start || b >= s->limit || s->state != MSpanInUse)
		return false;
	return true;
}

// Given an address in the heap return the relevant byte from the gcmap. This routine
// can be used on addresses to the start of an object or to the interior of the an object.
static void
slottombits(byte *obj, Markbits *mbits)
{
	uintptr off;

	off = (uintptr*)((uintptr)obj&~(PtrSize-1)) - (uintptr*)runtime·mheap.arena_start;
	mbits->bitp = runtime·mheap.arena_start - off/wordsPerBitmapByte - 1;
	mbits->shift = (off % wordsPerBitmapByte) * gcBits;
	mbits->xbits = *mbits->bitp;
	mbits->bits = (mbits->xbits >> mbits->shift) & bitMask;
}

// b is a pointer into the heap.
// Find the start of the object refered to by b.
// Set mbits to the associated bits from the bit map.
static byte*
objectstart(byte *b, Markbits *mbits)
{
	byte *obj, *p;
	MSpan *s;
	pageID k;
	uintptr x, size, idx;

	obj = (byte*)((uintptr)b&~(PtrSize-1));
	for(;;) {
		slottombits(obj, mbits);
		if(mbits->bits&bitBoundary == bitBoundary)
			break;
		
		// Not a beginning of a block, consult span table to find the block beginning.
		k = (uintptr)obj>>PageShift;
		x = k;
		x -= (uintptr)runtime·mheap.arena_start>>PageShift;
		s = runtime·mheap.spans[x];
		if(s == nil || k < s->start || obj >= s->limit || s->state != MSpanInUse){
			if(s->state == MSpanStack)
				break; // This is legit.

			// The following is catching some bugs left over from
			// us not being rigerous about what data structures are
			// hold valid pointers and different parts of the system
			// considering different structures as roots. For example
			// if there is a pointer into a stack that is left in 
			// a global data structure but that part of the runtime knows that 
			// those structures will be reinitialized before they are 
			// reused. Unfortunately the GC believes these roots are valid.
			// Typically a stack gets moved and only the structures that part of
			// the system knows are alive are updated. The span is freed
			// after the stack copy and the pointer is still alive. This 
			// check is catching that bug but for now we will not throw, 
			// instead we will simply break out of this routine and depend
			// on the caller to recognize that this pointer is not a valid 
			// heap pointer. I leave the code that catches the bug so that once
			// resolved we can turn this check back on and throw.

			//runtime·printf("Runtime: Span weird: obj=%p, k=%p", obj, k);
			//if (s == nil)
			//	runtime·printf(" s=nil\n");
			//else
			//	runtime·printf(" s->start=%p s->limit=%p, s->state=%d\n", s->start*PageSize, s->limit, s->state);
			//runtime·throw("Blowup on weird span");
			break; // We are not in a real block throw??
		}
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
	}
	// if size(obj.firstfield) < PtrSize, the &obj.secondfield could map to the boundary bit
	// Clear any low bits to get to the start of the object.
	// greyobject depends on this.
	return obj;
}

// obj is the start of an object with mark mbits.
// If it isn't already marked, mark it and enqueue into workbuf.
// Return possibly new workbuf to use.
static Workbuf*
greyobject(byte *obj, Markbits *mbits, Workbuf *wbuf) 
{
	// obj should be start of allocation, and so must be at least pointer-aligned.
	if(((uintptr)obj & (PtrSize-1)) != 0)
		runtime·throw("greyobject: obj not pointer-aligned");

	// If marked we have nothing to do.
	if((mbits->bits&bitMarked) != 0)
		return wbuf;

	// Each byte of GC bitmap holds info for two words.
	// If the current object is larger than two words, or if the object is one word
	// but the object it shares the byte with is already marked,
	// then all the possible concurrent updates are trying to set the same bit,
	// so we can use a non-atomic update.
	if((mbits->xbits&(bitMask|(bitMask<<gcBits))) != (bitBoundary|(bitBoundary<<gcBits)) || runtime·work.nproc == 1)
		*mbits->bitp = mbits->xbits | (bitMarked<<mbits->shift);
	else
		runtime·atomicor8(mbits->bitp, bitMarked<<mbits->shift);
	
	if(((mbits->xbits>>(mbits->shift+2))&BitsMask) == BitsDead)
		return wbuf;  // noscan object

	// Queue the obj for scanning. The PREFETCH(obj) logic has been removed but
	// seems like a nice optimization that can be added back in.
	// There needs to be time between the PREFETCH and the use.
	// Previously we put the obj in an 8 element buffer that is drained at a rate
	// to give the PREFETCH time to do its work.
	// Use of PREFETCHNTA might be more appropriate than PREFETCH

	// If workbuf is full, obtain an empty one.
	if(wbuf->nobj >= nelem(wbuf->obj)) {
		wbuf = getempty(wbuf);
	}

	wbuf->obj[wbuf->nobj] = obj;
	wbuf->nobj++;
	return wbuf;                    
}

// Scan the object b of size n, adding pointers to wbuf.
// Return possibly new wbuf to use.
// If ptrmask != nil, it specifies where pointers are in b.
// If ptrmask == nil, the GC bitmap should be consulted.
// In this case, n may be an overestimate of the size; the GC bitmap
// must also be used to make sure the scan stops at the end of b.
static Workbuf*
scanobject(byte *b, uintptr n, byte *ptrmask, Workbuf *wbuf)
{
	byte *obj, *arena_start, *arena_used, *ptrbitp, bits, cshift, cached;
	uintptr i;
	intptr ncached;
	Markbits mbits;

	arena_start = (byte*)runtime·mheap.arena_start;
	arena_used = runtime·mheap.arena_used;
	ptrbitp = nil;
	cached = 0;
	ncached = 0;

	// Find bits of the beginning of the object.
	if(ptrmask == nil) {
		b = objectstart(b, &mbits);
		ptrbitp = mbits.bitp; //arena_start - off/wordsPerBitmapByte - 1;
		cshift = mbits.shift; //(off % wordsPerBitmapByte) * gcBits;
		cached = *ptrbitp >> cshift;
		cached &= ~bitBoundary;
		ncached = (8 - cshift)/gcBits;
	}
	for(i = 0; i < n; i += PtrSize) {
		// Find bits for this word.
		if(ptrmask != nil) {
			// dense mask (stack or data)
			bits = (ptrmask[(i/PtrSize)/4]>>(((i/PtrSize)%4)*BitsPerPointer))&BitsMask;
		} else {
			// Check if we have reached end of span.
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
		} 

		if(bits == BitsScalar || bits == BitsDead)
			continue;
		if(bits != BitsPointer)
			runtime·throw("unexpected garbage collection bits");

		obj = *(byte**)(b+i);
		// At this point we have extracted the next potential pointer.
		// Check if it points into heap.
		if(obj == nil || obj < arena_start || obj >= arena_used)
			continue;
		// Mark the object. return some important bits.
		// We we combine the following two rotines we don't have to pass mbits or obj around.
		obj = objectstart(obj, &mbits);
		wbuf = greyobject(obj, &mbits, wbuf);
	}
	return wbuf;
}

// scanblock starts by scanning b as scanobject would.
// If the gcphase is GCscan, that's all scanblock does.
// Otherwise it traverses some fraction of the pointers it found in b, recursively.
// As a special case, scanblock(nil, 0, nil) means to scan previously queued work,
// stopping only when no work is left in the system.
static void
scanblock(byte *b, uintptr n, byte *ptrmask)
{
	Workbuf *wbuf;
	bool keepworking;

	wbuf = getpartial();
	if(b != nil) {
		wbuf = scanobject(b, n, ptrmask, wbuf);
		if(runtime·gcphase == GCscan) {
			putpartial(wbuf);
			return;
		}
	}

	keepworking = b == nil;

	// ptrmask can have 2 possible values:
	// 1. nil - obtain pointer mask from GC bitmap.
	// 2. pointer to a compact mask (for stacks and data).
	for(;;) {
		if(wbuf->nobj == 0) {
			if(!keepworking) {
				putempty(wbuf);
				return;
			}
			// Refill workbuf from global queue.
			wbuf = getfull(wbuf);
			if(wbuf == nil) // nil means out of work barrier reached
				return;
		}

		// If another proc wants a pointer, give it some.
		if(runtime·work.nwait > 0 && wbuf->nobj > 4 && runtime·work.full == 0) {
			wbuf = handoff(wbuf);
		}

		// This might be a good place to add prefetch code...
		// if(wbuf->nobj > 4) {
		//         PREFETCH(wbuf->obj[wbuf->nobj - 3];
		//  }
		--wbuf->nobj;
		b = wbuf->obj[wbuf->nobj];
		wbuf = scanobject(b, runtime·mheap.arena_used - b, nil, wbuf);
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
	uint32 status;
	bool restart;

	USED(&desc);
	// Note: if you add a case here, please also update heapdump.c:dumproots.
	switch(i) {
	case RootData:
		scanblock(runtime·data, runtime·edata - runtime·data, runtime·gcdatamask.bytedata);
		break;

	case RootBss:
		scanblock(runtime·bss, runtime·ebss - runtime·bss, runtime·gcbssmask.bytedata);
		break;

	case RootFinalizers:
		for(fb=runtime·allfin; fb; fb=fb->alllink)
			scanblock((byte*)fb->fin, fb->cnt*sizeof(fb->fin[0]), finptrmask);
		break;

	case RootSpans:
		// mark MSpan.specials
		sg = runtime·mheap.sweepgen;
		for(spanidx=0; spanidx<runtime·work.nspan; spanidx++) {
			Special *sp;
			SpecialFinalizer *spf;

			s = runtime·work.spans[spanidx];
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
				if(runtime·gcphase != GCscan)
					scanblock(p, s->elemsize, nil); // Scanned during mark phase
				scanblock((void*)&spf->fn, PtrSize, oneptr);
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
		status = runtime·readgstatus(gp); // We are not in a scan state
		if((status == Gwaiting || status == Gsyscall) && gp->waitsince == 0)
			gp->waitsince = runtime·work.tstart;
		// Shrink a stack if not much of it is being used.
		runtime·shrinkstack(gp);
		if(runtime·readgstatus(gp) == Gdead) 
			gp->gcworkdone = true;
		else 
			gp->gcworkdone = false; 
		restart = runtime·stopg(gp);

		// goroutine will scan its own stack when it stops running.
		// Wait until it has.
		while((status = runtime·readgstatus(gp)) == Grunning && !gp->gcworkdone) {
			if(status == Gdead) {
				// TBD you need to explain why Gdead without gp->gcworkdone
				// being true. If there is a race then it needs to be
				// explained here.
				gp->gcworkdone = true; // scan is a noop
				break;
				//do nothing, scan not needed. 
			}
			// try again
		}

		//		scanstack(gp); now done as part of gcphasework
		// But to make sure we finished we need to make sure that
		// the stack traps have all responded so drop into
		// this while loop until they respond.
		if(!gp->gcworkdone)
			// For some reason a G has not completed its work. This is  a bug that
			// needs to be investigated. For now I'll just print this message in
			// case the bug is benign.
			runtime·printf("runtime:markroot: post stack scan work not done gp=%p has status %x\n", gp, status);

		if(restart)
			runtime·restartg(gp);
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
		runtime·lfstackpush(&runtime·work.full, &b->node);
	b = nil;
	c = g->m->mcache;
	if(c->gcworkbuf != nil) {
		b = c->gcworkbuf;
		c->gcworkbuf = nil;
	}
	if(b == nil)
		b = (Workbuf*)runtime·lfstackpop(&runtime·work.empty);
	if(b == nil) {
		b = runtime·persistentalloc(sizeof(*b), CacheLineSize, &mstats.gc_sys);
		b->nobj = 0;
	}
	if(b->nobj != 0) 
		runtime·throw("getempty: b->nobj not 0/n");
	b->nobj = 0;
	return b;
}

static void
putempty(Workbuf *b)
{
	MCache *c;

	if(b->nobj != 0) 
		runtime·throw("putempty: b->nobj=%D not 0\n");
	c = g->m->mcache;
	if(c->gcworkbuf == nil) {
		c->gcworkbuf = b;
		return;
	}
	runtime·lfstackpush(&runtime·work.empty, &b->node);
}

// Get an partially empty work buffer from the mcache structure
// and if non is available get an empty one.
static Workbuf*
getpartial(void)
{
	MCache *c;
	Workbuf *b;

	c = g->m->mcache;
	if(c->gcworkbuf != nil) {
		b = c->gcworkbuf;
		c->gcworkbuf = nil;
	} else {
		b = getempty(nil);
	}
	return b;
}

static void
putpartial(Workbuf *b)
{
	MCache *c;

	c = g->m->mcache;
	if(c->gcworkbuf == nil) {
		c->gcworkbuf = b;
		return;
	}

	runtime·throw("putpartial: c->gcworkbuf is not nil\n");
	
	runtime·lfstackpush(&runtime·work.full, &b->node);
}

void
runtime·gcworkbuffree(Workbuf *b)
{
	if(b != nil) {
		if(b->nobj != 0) 
			runtime·throw("gcworkbufferfree: b->nobj not 0\n");
		putempty(b);
	}
}

// Get a full work buffer off the work.full list, or return nil.
// getfull acts as a barrier for work.nproc helpers. As long as one
// gchelper is actively marking objects it
// may create a workbuffer that the other helpers can work on.
// The for loop either exits when a work buffer is found
// or when _all_ of the work.nproc GC helpers are in the loop 
// looking for work and thus not capable of creating new work.
// This is in fact the termination condition for the STW mark 
// phase.
static Workbuf*
getfull(Workbuf *b)
{
	int32 i;

	if(b != nil) {
		if(b->nobj != 0) 
			runtime·printf("runtime:getfull: b->nobj=%D not 0.", b->nobj);
		runtime·lfstackpush(&runtime·work.empty, &b->node);
	}
	b = (Workbuf*)runtime·lfstackpop(&runtime·work.full);
	if(b != nil || runtime·work.nproc == 1)
		return b;

	runtime·xadd(&runtime·work.nwait, +1);
	for(i=0;; i++) {
		if(runtime·work.full != 0) {
			runtime·xadd(&runtime·work.nwait, -1);
			b = (Workbuf*)runtime·lfstackpop(&runtime·work.full);
			if(b != nil)
				return b;
			runtime·xadd(&runtime·work.nwait, +1);
		}
		if(runtime·work.nwait == runtime·work.nproc)
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
	runtime·lfstackpush(&runtime·work.full, &b->node);
	return b1;
}

BitVector
runtime·stackmapdata(StackMap *stackmap, int32 n)
{
	if(n < 0 || n >= stackmap->n)
		runtime·throw("stackmapdata: index out of range");
	return (BitVector){stackmap->nbit, stackmap->bytedata + n*((stackmap->nbit+31)/32*4)};
}

// Scan a stack frame: local variables and function arguments/results.
static bool
scanframe(Stkframe *frame, void *unused)
{
	Func *f;
	StackMap *stackmap;
	BitVector bv;
	uintptr size, minsize;
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
	size = frame->varp - frame->sp;
	if(thechar != '6' && thechar != '8')
		minsize = sizeof(uintptr);
	else
		minsize = 0;
	if(size > minsize) {
		stackmap = runtime·funcdata(f, FUNCDATA_LocalsPointerMaps);
		if(stackmap == nil || stackmap->n <= 0) {
			runtime·printf("runtime: frame %s untyped locals %p+%p\n", runtime·funcname(f), (byte*)(frame->varp-size), size);
			runtime·throw("missing stackmap");
		}

		// Locals bitmap information, scan just the pointers in locals.
		if(pcdata < 0 || pcdata >= stackmap->n) {
			// don't know where we are
			runtime·printf("runtime: pcdata is %d and %d locals stack map entries for %s (targetpc=%p)\n",
				pcdata, stackmap->n, runtime·funcname(f), targetpc);
			runtime·throw("scanframe: bad symbol table");
		}
		bv = runtime·stackmapdata(stackmap, pcdata);
		size = (bv.n * PtrSize) / BitsPerPointer;
		scanblock((byte*)(frame->varp - size), bv.n/BitsPerPointer*PtrSize, bv.bytedata);
	}

	// Scan arguments.
	if(frame->arglen > 0) {
		if(frame->argmap != nil)
			bv = *frame->argmap;
		else {
			stackmap = runtime·funcdata(f, FUNCDATA_ArgsPointerMaps);
			if(stackmap == nil || stackmap->n <= 0) {
				runtime·printf("runtime: frame %s untyped args %p+%p\n", runtime·funcname(f), frame->argp, (uintptr)frame->arglen);
				runtime·throw("missing stackmap");
			}
			if(pcdata < 0 || pcdata >= stackmap->n) {
				// don't know where we are
				runtime·printf("runtime: pcdata is %d and %d args stack map entries for %s (targetpc=%p)\n",
					pcdata, stackmap->n, runtime·funcname(f), targetpc);
				runtime·throw("scanframe: bad symbol table");
			}
 			bv = runtime·stackmapdata(stackmap, pcdata);
		}
		scanblock((byte*)frame->argp, bv.n/BitsPerPointer*PtrSize, bv.bytedata);
 	}
 	return true;
}

static void
scanstack(G *gp)
{
	M *mp;
	bool (*fn)(Stkframe*, void*);

	if(runtime·readgstatus(gp)&Gscan == 0) {
		runtime·printf("runtime: gp=%p, goid=%D, gp->atomicstatus=%d\n", gp, gp->goid, runtime·readgstatus(gp));
		runtime·throw("mark - bad status");
	}

	switch(runtime·readgstatus(gp)&~Gscan) {
	default:
		runtime·printf("runtime: gp=%p, goid=%D, gp->atomicstatus=%d\n", gp, gp->goid, runtime·readgstatus(gp));
		runtime·throw("mark - bad status");
	case Gdead:
		return;
	case Grunning:
		runtime·printf("runtime: gp=%p, goid=%D, gp->atomicstatus=%d\n", gp, gp->goid, runtime·readgstatus(gp));
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

	fn = scanframe;
	runtime·gentraceback(~(uintptr)0, ~(uintptr)0, 0, gp, 0, nil, 0x7fffffff, &fn, nil, false);
	runtime·tracebackdefers(gp, &fn, nil);
}

// If the slot is grey or black return true, if white return false.
// If the slot is not in the known heap and thus does not have a valid GC bitmap then
// it is considered grey. Globals and stacks can hold such slots.
// The slot is grey if its mark bit is set and it is enqueued to be scanned.
// The slot is black if it has already been scanned.
// It is white if it has a valid mark bit and the bit is not set. 
static bool
shaded(byte *slot)
{
	Markbits mbits;

	if(!inheap(slot)) // non-heap slots considered grey
		return true;

	objectstart(slot, &mbits);
	return (mbits.bits&bitMarked) != 0;
}

// Shade the object if it isn't already.
// The object is not nil and known to be in the heap.
static void
shade(byte *b)
{
	byte *obj;
	Workbuf *wbuf;
	Markbits mbits;
	
	if(!inheap(b))
		runtime·throw("shade: passed an address not in the heap");
	
	wbuf = getpartial();
	// Mark the object, return some important bits.
	// If we combine the following two rotines we don't have to pass mbits or obj around.
	obj = objectstart(b, &mbits);
	wbuf = greyobject(obj, &mbits, wbuf); // augments the wbuf
	putpartial(wbuf);
	return;
}

// This is the Dijkstra barrier coarsened to shade grey to white whereas
// the original Dijkstra barrier only shaded black to white.
//
// Shade indicates that it has seen a white pointer by adding the referent
// to wbuf.
void
runtime·markwb(void **slot, void *ptr)
{
	// initial nil check avoids some needlesss loads
	if(ptr != nil && inheap(ptr) && shaded((void*)slot))
		shade(ptr);
	*slot = ptr;
}

// The gp has been moved to a gc safepoint. If there is gcphase specific
// work it is done here. 
void
runtime·gcphasework(G *gp)
{
	switch(runtime·gcphase) {
	default:
		runtime·throw("gcphasework in bad gcphase");
	case GCoff:
	case GCquiesce:
	case GCstw:
	case GCsweep:
		// No work.
		break;
	case GCscan:
		// scan the stack, mark the objects, put pointers in work buffers
		// hanging off the P where this is being run.
		scanstack(gp);
		break;
	case GCmark:
	case GCmarktermination:
		//
		// Disabled until concurrent GC is implemented
		// but indicate the scan has been done. 
		scanstack(gp);
		// scanstack will call shade which will populate
		// the Workbuf.
		// emptywbuf(gp) will empty it before returning
		// 
		break;
	}
	gp->gcworkdone = true;
}

#pragma dataflag NOPTR
static byte finalizer1[] = {
	// Each Finalizer is 5 words, ptr ptr uintptr ptr ptr.
	// Each byte describes 4 words.
	// Need 4 Finalizers described by 5 bytes before pattern repeats:
	//	ptr ptr uintptr ptr ptr
	//	ptr ptr uintptr ptr ptr
	//	ptr ptr uintptr ptr ptr
	//	ptr ptr uintptr ptr ptr
	// aka
	//	ptr ptr uintptr ptr
	//	ptr ptr ptr uintptr
	//	ptr ptr ptr ptr
	//	uintptr ptr ptr ptr
	//	ptr uintptr ptr ptr
	// Assumptions about Finalizer layout checked below.
	BitsPointer | BitsPointer<<2 | BitsScalar<<4 | BitsPointer<<6,
	BitsPointer | BitsPointer<<2 | BitsPointer<<4 | BitsScalar<<6,
	BitsPointer | BitsPointer<<2 | BitsPointer<<4 | BitsPointer<<6,
	BitsScalar | BitsPointer<<2 | BitsPointer<<4 | BitsPointer<<6,
	BitsPointer | BitsScalar<<2 | BitsPointer<<4 | BitsPointer<<6,
};

void
runtime·queuefinalizer(byte *p, FuncVal *fn, uintptr nret, Type *fint, PtrType *ot)
{
	FinBlock *block;
	Finalizer *f;
	int32 i;

	runtime·lock(&runtime·finlock);
	if(runtime·finq == nil || runtime·finq->cnt == runtime·finq->cap) {
		if(runtime·finc == nil) {
			runtime·finc = runtime·persistentalloc(FinBlockSize, 0, &mstats.gc_sys);
			runtime·finc->cap = (FinBlockSize - sizeof(FinBlock)) / sizeof(Finalizer) + 1;
			runtime·finc->alllink = runtime·allfin;
			runtime·allfin = runtime·finc;
			if(finptrmask[0] == 0) {
				// Build pointer mask for Finalizer array in block.
				// Check assumptions made in finalizer1 array above.
				if(sizeof(Finalizer) != 5*PtrSize ||
					offsetof(Finalizer, fn) != 0 ||
					offsetof(Finalizer, arg) != PtrSize ||
					offsetof(Finalizer, nret) != 2*PtrSize ||
					offsetof(Finalizer, fint) != 3*PtrSize ||
					offsetof(Finalizer, ot) != 4*PtrSize ||
					BitsPerPointer != 2) {
					runtime·throw("finalizer out of sync");
				}
				for(i=0; i<nelem(finptrmask); i++)
					finptrmask[i] = finalizer1[i%nelem(finalizer1)];
			}
		}
		block = runtime·finc;
		runtime·finc = block->next;
		block->next = runtime·finq;
		runtime·finq = block;
	}
	f = &runtime·finq->fin[runtime·finq->cnt];
	runtime·finq->cnt++;
	f->fn = fn;
	f->nret = nret;
	f->fint = fint;
	f->ot = ot;
	f->arg = p;
	runtime·fingwake = true;
	runtime·unlock(&runtime·finlock);
}

void
runtime·iterate_finq(void (*callback)(FuncVal*, byte*, uintptr, Type*, PtrType*))
{
	FinBlock *fb;
	Finalizer *f;
	uintptr i;

	for(fb = runtime·allfin; fb; fb = fb->alllink) {
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

// State of background runtime·sweep.
// Protected by runtime·gclock.
typedef struct SweepData SweepData;
struct SweepData
{
	G*	g;
	bool	parked;

	uint32	spanidx;	// background sweeper position

	uint32	nbgsweep;
	uint32	npausesweep;
};
SweepData runtime·sweep;

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
		idx = runtime·xadd(&runtime·sweep.spanidx, 1) - 1;
		if(idx >= runtime·work.nspan) {
			runtime·mheap.sweepdone = true;
			g->m->locks--;
			return -1;
		}
		s = runtime·work.spans[idx];
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

static void
sweepone_m(void)
{
	g->m->scalararg[0] = runtime·sweepone();
}

#pragma textflag NOSPLIT
uintptr
runtime·gosweepone(void)
{
	void (*fn)(void);
	
	fn = sweepone_m;
	runtime·onM(&fn);
	return g->m->scalararg[0];
}

#pragma textflag NOSPLIT
bool
runtime·gosweepdone(void)
{
	return runtime·mheap.sweepdone;
}


void
runtime·gchelper(void)
{
	uint32 nproc;

	g->m->traceback = 2;
	gchelperstart();

	// parallel mark for over gc roots
	runtime·parfordo(runtime·work.markfor);
	if(runtime·gcphase != GCscan) 
		scanblock(nil, 0, nil); // blocks in getfull
	nproc = runtime·work.nproc;  // work.nproc can change right after we increment work.ndone
	if(runtime·xadd(&runtime·work.ndone, +1) == nproc-1)
		runtime·notewakeup(&runtime·work.alldone);
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
	void (*fn)(G*);

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
	else {
		fn = flushallmcaches_m;
		runtime·mcall(&fn);
	}

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
	mstats.nfree += mstats.tinyallocs;
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

	runtime·work.markfor = runtime·parforalloc(MaxGcproc);
	runtime·gcpercent = runtime·readgogc();
	runtime·gcdatamask = unrollglobgcprog(runtime·gcdata, runtime·edata - runtime·data);
	runtime·gcbssmask = unrollglobgcprog(runtime·gcbss, runtime·ebss - runtime·bss);
}

// Called from malloc.go using onM, stopping and starting the world handled in caller.
void
runtime·gc_m(void)
{
	struct gc_args a;
	G *gp;

	gp = g->m->curg;
	runtime·casgstatus(gp, Grunning, Gwaiting);
	gp->waitreason = runtime·gostringnocopy((byte*)"garbage collection");

	a.start_time = (uint64)(g->m->scalararg[0]) | ((uint64)(g->m->scalararg[1]) << 32);
	a.eagersweep = g->m->scalararg[2];
	gc(&a);

	runtime·casgstatus(gp, Gwaiting, Grunning);
}

static void
gc(struct gc_args *args)
{
	int64 t0, t1, t2, t3, t4;
	uint64 heap0, heap1, obj;
	GCStats stats;
	uint32 oldphase;
	
	if(runtime·debug.allocfreetrace)
		runtime·tracegc();

	g->m->traceback = 2;
	t0 = args->start_time;
	runtime·work.tstart = args->start_time; 

	t1 = 0;
	if(runtime·debug.gctrace)
		t1 = runtime·nanotime();

	// Sweep what is not sweeped by bgsweep.
	while(runtime·sweepone() != -1)
		runtime·sweep.npausesweep++;

	// Cache runtime·mheap.allspans in work.spans to avoid conflicts with
	// resizing/freeing allspans.
	// New spans can be created while GC progresses, but they are not garbage for
	// this round:
	//  - new stack spans can be created even while the world is stopped.
	//  - new malloc spans can be created during the concurrent sweep

	// Even if this is stop-the-world, a concurrent exitsyscall can allocate a stack from heap.
	runtime·lock(&runtime·mheap.lock);
	// Free the old cached sweep array if necessary.
	if(runtime·work.spans != nil && runtime·work.spans != runtime·mheap.allspans)
		runtime·SysFree(runtime·work.spans, runtime·work.nspan*sizeof(runtime·work.spans[0]), &mstats.other_sys);
	// Cache the current array for marking.
	runtime·mheap.gcspans = runtime·mheap.allspans;
	runtime·work.spans = runtime·mheap.allspans;
	runtime·work.nspan = runtime·mheap.nspan;
	runtime·unlock(&runtime·mheap.lock);
	oldphase = runtime·gcphase;

	runtime·work.nwait = 0;
	runtime·work.ndone = 0;
	runtime·work.nproc = runtime·gcprocs(); 
	runtime·gcphase = GCmark;              //^^  vv

	runtime·parforsetup(runtime·work.markfor, runtime·work.nproc, RootCount + runtime·allglen, nil, false, markroot);
	if(runtime·work.nproc > 1) {
		runtime·noteclear(&runtime·work.alldone);
		runtime·helpgc(runtime·work.nproc);
	}

	t2 = 0;
	if(runtime·debug.gctrace)
		t2 = runtime·nanotime();

	gchelperstart();
	runtime·parfordo(runtime·work.markfor);

	scanblock(nil, 0, nil);
	runtime·gcphase = oldphase;            //^^  vv
	t3 = 0;
	if(runtime·debug.gctrace)
		t3 = runtime·nanotime();

	if(runtime·work.nproc > 1)
		runtime·notesleep(&runtime·work.alldone);

	cachestats();
	// next_gc calculation is tricky with concurrent sweep since we don't know size of live heap
	// estimate what was live heap size after previous GC (for tracing only)
	heap0 = mstats.next_gc*100/(runtime·gcpercent+100);
	// conservatively set next_gc to high value assuming that everything is live
	// concurrent/lazy sweep will reduce this number while discovering new garbage
	mstats.next_gc = mstats.heap_alloc+mstats.heap_alloc*runtime·gcpercent/100;

	t4 = runtime·nanotime();
	runtime·atomicstore64(&mstats.last_gc, runtime·unixnanotime());  // must be Unix time to make sense to user
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

		stats.nprocyield += runtime·work.markfor->nprocyield;
		stats.nosyield += runtime·work.markfor->nosyield;
		stats.nsleep += runtime·work.markfor->nsleep;

		runtime·printf("gc%d(%d): %D+%D+%D+%D us, %D -> %D MB, %D (%D-%D) objects,"
				" %d goroutines,"
				" %d/%d/%d sweeps,"
				" %D(%D) handoff, %D(%D) steal, %D/%D/%D yields\n",
			mstats.numgc, runtime·work.nproc, (t1-t0)/1000, (t2-t1)/1000, (t3-t2)/1000, (t4-t3)/1000,
			heap0>>20, heap1>>20, obj,
			mstats.nmalloc, mstats.nfree,
			runtime·gcount(),
			runtime·work.nspan, runtime·sweep.nbgsweep, runtime·sweep.npausesweep,
			stats.nhandoff, stats.nhandoffcnt,
			runtime·work.markfor->nsteal, runtime·work.markfor->nstealcnt,
			stats.nprocyield, stats.nosyield, stats.nsleep);
		runtime·sweep.nbgsweep = runtime·sweep.npausesweep = 0;
	}

	// See the comment in the beginning of this function as to why we need the following.
	// Even if this is still stop-the-world, a concurrent exitsyscall can allocate a stack from heap.
	runtime·lock(&runtime·mheap.lock);
	// Free the old cached mark array if necessary.
	if(runtime·work.spans != nil && runtime·work.spans != runtime·mheap.allspans)
		runtime·SysFree(runtime·work.spans, runtime·work.nspan*sizeof(runtime·work.spans[0]), &mstats.other_sys);
	// Cache the current array for sweeping.
	runtime·mheap.gcspans = runtime·mheap.allspans;
	runtime·mheap.sweepgen += 2;
	runtime·mheap.sweepdone = false;
	runtime·work.spans = runtime·mheap.allspans;
	runtime·work.nspan = runtime·mheap.nspan;
	runtime·sweep.spanidx = 0;
	runtime·unlock(&runtime·mheap.lock);

	// Temporary disable concurrent sweep, because we see failures on builders.
	if(ConcurrentSweep && !args->eagersweep) {
		runtime·lock(&runtime·gclock);
		if(runtime·sweep.g == nil)
			runtime·sweep.g = runtime·newproc1(&bgsweepv, nil, 0, 0, gc);
		else if(runtime·sweep.parked) {
			runtime·sweep.parked = false;
			runtime·ready(runtime·sweep.g);
		}
		runtime·unlock(&runtime·gclock);
	} else {
		// Sweep all spans eagerly.
		while(runtime·sweepone() != -1)
			runtime·sweep.npausesweep++;
	}

	runtime·mProf_GC();
	g->m->traceback = 0;
}

extern uintptr runtime·sizeof_C_MStats;

static void readmemstats_m(void);

void
runtime·readmemstats_m(void)
{
	MStats *stats;
	
	stats = g->m->ptrarg[0];
	g->m->ptrarg[0] = nil;

	runtime·updatememstats(nil);
	// Size of the trailing by_size array differs between Go and C,
	// NumSizeClasses was changed, but we can not change Go struct because of backward compatibility.
	runtime·memmove(stats, &mstats, runtime·sizeof_C_MStats);

	// Stack numbers are part of the heap numbers, separate those out for user consumption
	stats->stacks_sys = stats->stacks_inuse;
	stats->heap_inuse -= stats->stacks_inuse;
	stats->heap_sys -= stats->stacks_inuse;
}

static void readgcstats_m(void);

#pragma textflag NOSPLIT
void
runtime∕debug·readGCStats(Slice *pauses)
{
	void (*fn)(void);
	
	g->m->ptrarg[0] = pauses;
	fn = readgcstats_m;
	runtime·onM(&fn);
}

static void
readgcstats_m(void)
{
	Slice *pauses;	
	uint64 *p;
	uint32 i, n;
	
	pauses = g->m->ptrarg[0];
	g->m->ptrarg[0] = nil;

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
runtime·setgcpercent_m(void)
{
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

G*
runtime·wakefing(void)
{
	G *res;

	res = nil;
	runtime·lock(&runtime·finlock);
	if(runtime·fingwait && runtime·fingwake) {
		runtime·fingwait = false;
		runtime·fingwake = false;
		res = runtime·fing;
	}
	runtime·unlock(&runtime·finlock);
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
static BitVector
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
	return (BitVector){masksize*8, mask};
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
	static Mutex lock;
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
	if(frame->sp <= frame0->sp && frame0->sp < frame->varp) {
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
	bool (*cb)(Stkframe*, void*);

	*mask = nil;
	*len = 0;

	// data
	if(p >= runtime·data && p < runtime·edata) {
		n = ((PtrType*)t)->elem->size;
		*len = n/PtrSize;
		*mask = runtime·mallocgc(*len, nil, FlagNoScan);
		for(i = 0; i < n; i += PtrSize) {
			off = (p+i-runtime·data)/PtrSize;
			bits = (runtime·gcdatamask.bytedata[off/PointersPerByte] >> ((off%PointersPerByte)*BitsPerPointer))&BitsMask;
			(*mask)[i/PtrSize] = bits;
		}
		return;
	}
	// bss
	if(p >= runtime·bss && p < runtime·ebss) {
		n = ((PtrType*)t)->elem->size;
		*len = n/PtrSize;
		*mask = runtime·mallocgc(*len, nil, FlagNoScan);
		for(i = 0; i < n; i += PtrSize) {
			off = (p+i-runtime·bss)/PtrSize;
			bits = (runtime·gcbssmask.bytedata[off/PointersPerByte] >> ((off%PointersPerByte)*BitsPerPointer))&BitsMask;
			(*mask)[i/PtrSize] = bits;
		}
		return;
	}
	// heap
	if(runtime·mlookup(p, &base, &n, nil)) {
		*len = n/PtrSize;
		*mask = runtime·mallocgc(*len, nil, FlagNoScan);
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
	cb = getgcmaskcb;
	runtime·gentraceback(g->m->curg->sched.pc, g->m->curg->sched.sp, 0, g->m->curg, 0, nil, 1000, &cb, &frame, false);
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
		*mask = runtime·mallocgc(*len, nil, FlagNoScan);
		for(i = 0; i < n; i += PtrSize) {
			off = (p+i-(byte*)frame.varp+size)/PtrSize;
			bits = (bv.bytedata[off*BitsPerPointer/8] >> ((off*BitsPerPointer)%8))&BitsMask;
			(*mask)[i/PtrSize] = bits;
		}
	}
}

void runtime·gc_unixnanotime(int64 *now);

int64
runtime·unixnanotime(void)
{
	int64 now;

	runtime·gc_unixnanotime(&now);
	return now;
}
