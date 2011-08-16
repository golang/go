// Copyright 2009 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// Garbage collector.

#include "runtime.h"
#include "malloc.h"
#include "stack.h"

enum {
	Debug = 0,
	UseCas = 1,
	PtrSize = sizeof(void*),
	
	// Four bits per word (see #defines below).
	wordsPerBitmapWord = sizeof(void*)*8/4,
	bitShift = sizeof(void*)*8/4,
};

// Bits in per-word bitmap.
// #defines because enum might not be able to hold the values.
//
// Each word in the bitmap describes wordsPerBitmapWord words
// of heap memory.  There are 4 bitmap bits dedicated to each heap word,
// so on a 64-bit system there is one bitmap word per 16 heap words.
// The bits in the word are packed together by type first, then by
// heap location, so each 64-bit bitmap word consists of, from top to bottom,
// the 16 bitSpecial bits for the corresponding heap words, then the 16 bitMarked bits,
// then the 16 bitNoPointers/bitBlockBoundary bits, then the 16 bitAllocated bits.
// This layout makes it easier to iterate over the bits of a given type.
//
// The bitmap starts at mheap.arena_start and extends *backward* from
// there.  On a 64-bit system the off'th word in the arena is tracked by
// the off/16+1'th word before mheap.arena_start.  (On a 32-bit system,
// the only difference is that the divisor is 8.)
//
// To pull out the bits corresponding to a given pointer p, we use:
//
//	off = p - (uintptr*)mheap.arena_start;  // word offset
//	b = (uintptr*)mheap.arena_start - off/wordsPerBitmapWord - 1;
//	shift = off % wordsPerBitmapWord
//	bits = *b >> shift;
//	/* then test bits & bitAllocated, bits & bitMarked, etc. */
//
#define bitAllocated		((uintptr)1<<(bitShift*0))
#define bitNoPointers		((uintptr)1<<(bitShift*1))	/* when bitAllocated is set */
#define bitMarked		((uintptr)1<<(bitShift*2))	/* when bitAllocated is set */
#define bitSpecial		((uintptr)1<<(bitShift*3))	/* when bitAllocated is set - has finalizer or being profiled */
#define bitBlockBoundary	((uintptr)1<<(bitShift*1))	/* when bitAllocated is NOT set */

#define bitMask (bitBlockBoundary | bitAllocated | bitMarked | bitSpecial)

static uint64 nlookup;
static uint64 nsizelookup;
static uint64 naddrlookup;
static int32 gctrace;

typedef struct Workbuf Workbuf;
struct Workbuf
{
	Workbuf *next;
	uintptr nw;
	byte *w[2048-2];
};

extern byte data[];
extern byte etext[];
extern byte end[];

static G *fing;
static Finalizer *finq;
static int32 fingwait;

static void runfinq(void);
static Workbuf* getempty(Workbuf*);
static Workbuf* getfull(Workbuf*);

// scanblock scans a block of n bytes starting at pointer b for references
// to other objects, scanning any it finds recursively until there are no
// unscanned objects left.  Instead of using an explicit recursion, it keeps
// a work list in the Workbuf* structures and loops in the main function
// body.  Keeping an explicit work list is easier on the stack allocator and
// more efficient.
static void
scanblock(byte *b, int64 n)
{
	byte *obj, *arena_start, *p;
	void **vp;
	uintptr size, *bitp, bits, shift, i, j, x, xbits, off;
	MSpan *s;
	PageID k;
	void **bw, **w, **ew;
	Workbuf *wbuf;

	if((int64)(uintptr)n != n || n < 0) {
		runtime·printf("scanblock %p %D\n", b, n);
		runtime·throw("scanblock");
	}

	// Memory arena parameters.
	arena_start = runtime·mheap.arena_start;
	
	wbuf = nil;  // current work buffer
	ew = nil;  // end of work buffer
	bw = nil;  // beginning of work buffer
	w = nil;  // current pointer into work buffer

	// Align b to a word boundary.
	off = (uintptr)b & (PtrSize-1);
	if(off != 0) {
		b += PtrSize - off;
		n -= PtrSize - off;
	}

	for(;;) {
		// Each iteration scans the block b of length n, queueing pointers in
		// the work buffer.
		if(Debug > 1)
			runtime·printf("scanblock %p %D\n", b, n);

		vp = (void**)b;
		n /= PtrSize;
		for(i=0; i<n; i++) {
			obj = (byte*)vp[i];
			
			// Words outside the arena cannot be pointers.
			if((byte*)obj < arena_start || (byte*)obj >= runtime·mheap.arena_used)
				continue;
			
			// obj may be a pointer to a live object.
			// Try to find the beginning of the object.
			
			// Round down to word boundary.
			obj = (void*)((uintptr)obj & ~((uintptr)PtrSize-1));

			// Find bits for this word.
			off = (uintptr*)obj - (uintptr*)arena_start;
			bitp = (uintptr*)arena_start - off/wordsPerBitmapWord - 1;
			shift = off % wordsPerBitmapWord;
			xbits = *bitp;
			bits = xbits >> shift;

			// Pointing at the beginning of a block?
			if((bits & (bitAllocated|bitBlockBoundary)) != 0)
				goto found;

			// Pointing just past the beginning?
			// Scan backward a little to find a block boundary.
			for(j=shift; j-->0; ) {
				if(((xbits>>j) & (bitAllocated|bitBlockBoundary)) != 0) {
					obj = (byte*)obj - (shift-j)*PtrSize;
					shift = j;
					bits = xbits>>shift;
					goto found;
				}
			}

			// Otherwise consult span table to find beginning.
			// (Manually inlined copy of MHeap_LookupMaybe.)
			nlookup++;
			naddrlookup++;
			k = (uintptr)obj>>PageShift;
			x = k;
			if(sizeof(void*) == 8)
				x -= (uintptr)arena_start>>PageShift;
			s = runtime·mheap.map[x];
			if(s == nil || k < s->start || k - s->start >= s->npages || s->state != MSpanInUse)
				continue;
			p =  (byte*)((uintptr)s->start<<PageShift);
			if(s->sizeclass == 0) {
				obj = p;
			} else {
				if((byte*)obj >= (byte*)s->limit)
					continue;
				size = runtime·class_to_size[s->sizeclass];
				int32 i = ((byte*)obj - p)/size;
				obj = p+i*size;
			}

			// Now that we know the object header, reload bits.
			off = (uintptr*)obj - (uintptr*)arena_start;
			bitp = (uintptr*)arena_start - off/wordsPerBitmapWord - 1;
			shift = off % wordsPerBitmapWord;
			xbits = *bitp;
			bits = xbits >> shift;

		found:
			// Now we have bits, bitp, and shift correct for
			// obj pointing at the base of the object.
			// If not allocated or already marked, done.
			if((bits & bitAllocated) == 0 || (bits & bitMarked) != 0)
				continue;
			*bitp |= bitMarked<<shift;

			// If object has no pointers, don't need to scan further.
			if((bits & bitNoPointers) != 0)
				continue;

			// If buffer is full, get a new one.
			if(w >= ew) {
				wbuf = getempty(wbuf);
				bw = wbuf->w;
				w = bw;
				ew = bw + nelem(wbuf->w);
			}
			*w++ = obj;
		}
		
		// Done scanning [b, b+n).  Prepare for the next iteration of
		// the loop by setting b and n to the parameters for the next block.

		// Fetch b from the work buffers.
		if(w <= bw) {
			// Emptied our buffer: refill.
			wbuf = getfull(wbuf);
			if(wbuf == nil)
				break;
			bw = wbuf->w;
			ew = wbuf->w + nelem(wbuf->w);
			w = bw+wbuf->nw;
		}
		b = *--w;
	
		// Figure out n = size of b.  Start by loading bits for b.
		off = (uintptr*)b - (uintptr*)arena_start;
		bitp = (uintptr*)arena_start - off/wordsPerBitmapWord - 1;
		shift = off % wordsPerBitmapWord;
		xbits = *bitp;
		bits = xbits >> shift;
		
		// Might be small; look for nearby block boundary.
		// A block boundary is marked by either bitBlockBoundary
		// or bitAllocated being set (see notes near their definition).
		enum {
			boundary = bitBlockBoundary|bitAllocated
		};
		// Look for a block boundary both after and before b
		// in the same bitmap word.
		//
		// A block boundary j words after b is indicated by
		//	bits>>j & boundary
		// assuming shift+j < bitShift.  (If shift+j >= bitShift then
		// we'll be bleeding other bit types like bitMarked into our test.)
		// Instead of inserting the conditional shift+j < bitShift into the loop,
		// we can let j range from 1 to bitShift as long as we first
		// apply a mask to keep only the bits corresponding
		// to shift+j < bitShift aka j < bitShift-shift.
		bits &= (boundary<<(bitShift-shift)) - boundary;
		
		// A block boundary j words before b is indicated by
		//	xbits>>(shift-j) & boundary
		// (assuming shift >= j).  There is no cleverness here
		// avoid the test, because when j gets too large the shift
		// turns negative, which is undefined in C.		

		for(j=1; j<bitShift; j++) {
			if(((bits>>j)&boundary) != 0 || shift>=j && ((xbits>>(shift-j))&boundary) != 0) {
				n = j*PtrSize;
				goto scan;
			}
		}
		
		// Fall back to asking span about size class.
		// (Manually inlined copy of MHeap_Lookup.)
		nlookup++;
		nsizelookup++;
		x = (uintptr)b>>PageShift;
		if(sizeof(void*) == 8)
			x -= (uintptr)arena_start>>PageShift;
		s = runtime·mheap.map[x];
		if(s->sizeclass == 0)
			n = s->npages<<PageShift;
		else
			n = runtime·class_to_size[s->sizeclass];
	scan:;
	}
}

static struct {
	Workbuf	*full;
	Workbuf	*empty;
	byte	*chunk;
	uintptr	nchunk;
} work;

// Get an empty work buffer off the work.empty list,
// allocating new buffers as needed.
static Workbuf*
getempty(Workbuf *b)
{
	if(b != nil) {
		b->nw = nelem(b->w);
		b->next = work.full;
		work.full = b;
	}
	b = work.empty;
	if(b != nil) {
		work.empty = b->next;
		return b;
	}
	
	if(work.nchunk < sizeof *b) {
		work.nchunk = 1<<20;
		work.chunk = runtime·SysAlloc(work.nchunk);
	}
	b = (Workbuf*)work.chunk;
	work.chunk += sizeof *b;
	work.nchunk -= sizeof *b;
	return b;
}

// Get a full work buffer off the work.full list, or return nil.
static Workbuf*
getfull(Workbuf *b)
{
	if(b != nil) {
		b->nw = 0;
		b->next = work.empty;
		work.empty = b;
	}
	b = work.full;
	if(b != nil)
		work.full = b->next;
	return b;
}

// Scanstack calls scanblock on each of gp's stack segments.
static void
scanstack(G *gp)
{
	int32 n;
	Stktop *stk;
	byte *sp, *guard;

	stk = (Stktop*)gp->stackbase;
	guard = gp->stackguard;

	if(gp == g) {
		// Scanning our own stack: start at &gp.
		sp = (byte*)&gp;
	} else {
		// Scanning another goroutine's stack.
		// The goroutine is usually asleep (the world is stopped).
		sp = gp->sched.sp;

		// The exception is that if the goroutine is about to enter or might
		// have just exited a system call, it may be executing code such
		// as schedlock and may have needed to start a new stack segment.
		// Use the stack segment and stack pointer at the time of
		// the system call instead, since that won't change underfoot.
		if(gp->gcstack != nil) {
			stk = (Stktop*)gp->gcstack;
			sp = gp->gcsp;
			guard = gp->gcguard;
		}
	}

	if(Debug > 1)
		runtime·printf("scanstack %d %p\n", gp->goid, sp);
	n = 0;
	while(stk) {
		if(sp < guard-StackGuard || (byte*)stk < sp) {
			runtime·printf("scanstack inconsistent: g%d#%d sp=%p not in [%p,%p]\n", gp->goid, n, sp, guard-StackGuard, stk);
			runtime·throw("scanstack");
		}
		scanblock(sp, (byte*)stk - sp);
		sp = stk->gobuf.sp;
		guard = stk->stackguard;
		stk = (Stktop*)stk->stackbase;
		n++;
	}
}

// Markfin calls scanblock on the blocks that have finalizers:
// the things pointed at cannot be freed until the finalizers have run.
static void
markfin(void *v)
{
	uintptr size;

	size = 0;
	if(!runtime·mlookup(v, &v, &size, nil) || !runtime·blockspecial(v))
		runtime·throw("mark - finalizer inconsistency");

	// do not mark the finalizer block itself.  just mark the things it points at.
	scanblock(v, size);
}

// Mark 
static void
mark(void)
{
	G *gp;

	// mark data+bss.
	// skip runtime·mheap itself, which has no interesting pointers
	// and is mostly zeroed and would not otherwise be paged in.
	scanblock(data, (byte*)&runtime·mheap - data);
	scanblock((byte*)(&runtime·mheap+1), end - (byte*)(&runtime·mheap+1));

	// mark stacks
	for(gp=runtime·allg; gp!=nil; gp=gp->alllink) {
		switch(gp->status){
		default:
			runtime·printf("unexpected G.status %d\n", gp->status);
			runtime·throw("mark - bad status");
		case Gdead:
			break;
		case Grunning:
			if(gp != g)
				runtime·throw("mark - world not stopped");
			scanstack(gp);
			break;
		case Grunnable:
		case Gsyscall:
		case Gwaiting:
			scanstack(gp);
			break;
		}
	}

	// mark things pointed at by objects with finalizers
	runtime·walkfintab(markfin);
}

// Sweep frees or calls finalizers for blocks not marked in the mark phase.
// It clears the mark bits in preparation for the next GC round.
static void
sweep(void)
{
	MSpan *s;
	int32 cl, n, npages;
	uintptr size;
	byte *p;
	MCache *c;
	Finalizer *f;

	for(s = runtime·mheap.allspans; s != nil; s = s->allnext) {
		if(s->state != MSpanInUse)
			continue;

		p = (byte*)(s->start << PageShift);
		cl = s->sizeclass;
		if(cl == 0) {
			size = s->npages<<PageShift;
			n = 1;
		} else {
			// Chunk full of small blocks.
			size = runtime·class_to_size[cl];
			npages = runtime·class_to_allocnpages[cl];
			n = (npages << PageShift) / size;
		}
	
		// sweep through n objects of given size starting at p.
		for(; n > 0; n--, p += size) {
			uintptr off, *bitp, shift, bits;

			off = (uintptr*)p - (uintptr*)runtime·mheap.arena_start;
			bitp = (uintptr*)runtime·mheap.arena_start - off/wordsPerBitmapWord - 1;
			shift = off % wordsPerBitmapWord;
			bits = *bitp>>shift;

			if((bits & bitAllocated) == 0)
				continue;

			if((bits & bitMarked) != 0) {
				*bitp &= ~(bitMarked<<shift);
				continue;
			}

			if((bits & bitSpecial) != 0) {
				// Special means it has a finalizer or is being profiled.
				f = runtime·getfinalizer(p, 1);
				if(f != nil) {
					f->arg = p;
					f->next = finq;
					finq = f;
					continue;
				}
				runtime·MProf_Free(p, size);
			}

			// Mark freed; restore block boundary bit.
			*bitp = (*bitp & ~(bitMask<<shift)) | (bitBlockBoundary<<shift);

			c = m->mcache;
			if(s->sizeclass == 0) {
				// Free large span.
				runtime·unmarkspan(p, 1<<PageShift);
				*(uintptr*)p = 1;	// needs zeroing
				runtime·MHeap_Free(&runtime·mheap, s, 1);
			} else {
				// Free small object.
				if(size > sizeof(uintptr))
					((uintptr*)p)[1] = 1;	// mark as "needs to be zeroed"
				c->local_by_size[s->sizeclass].nfree++;
				runtime·MCache_Free(c, p, s->sizeclass, size);
			}
			c->local_alloc -= size;
			c->local_nfree++;
		}
	}
}

// Semaphore, not Lock, so that the goroutine
// reschedules when there is contention rather
// than spinning.
static uint32 gcsema = 1;

// Initialized from $GOGC.  GOGC=off means no gc.
//
// Next gc is after we've allocated an extra amount of
// memory proportional to the amount already in use.
// If gcpercent=100 and we're using 4M, we'll gc again
// when we get to 8M.  This keeps the gc cost in linear
// proportion to the allocation cost.  Adjusting gcpercent
// just changes the linear constant (and also the amount of
// extra memory used).
static int32 gcpercent = -2;

static void
stealcache(void)
{
	M *m;
	
	for(m=runtime·allm; m; m=m->alllink)
		runtime·MCache_ReleaseAll(m->mcache);
}

static void
cachestats(void)
{
	M *m;
	MCache *c;
	int32 i;
	uint64 stacks_inuse;
	uint64 stacks_sys;

	stacks_inuse = 0;
	stacks_sys = 0;
	for(m=runtime·allm; m; m=m->alllink) {
		runtime·purgecachedstats(m);
		stacks_inuse += m->stackalloc->inuse;
		stacks_sys += m->stackalloc->sys;
		c = m->mcache;
		for(i=0; i<nelem(c->local_by_size); i++) {
			mstats.by_size[i].nmalloc += c->local_by_size[i].nmalloc;
			c->local_by_size[i].nmalloc = 0;
			mstats.by_size[i].nfree += c->local_by_size[i].nfree;
			c->local_by_size[i].nfree = 0;
		}
	}
	mstats.stacks_inuse = stacks_inuse;
	mstats.stacks_sys = stacks_sys;
}

void
runtime·gc(int32 force)
{
	int64 t0, t1, t2, t3;
	uint64 heap0, heap1, obj0, obj1;
	byte *p;
	Finalizer *fp;

	// The gc is turned off (via enablegc) until
	// the bootstrap has completed.
	// Also, malloc gets called in the guts
	// of a number of libraries that might be
	// holding locks.  To avoid priority inversion
	// problems, don't bother trying to run gc
	// while holding a lock.  The next mallocgc
	// without a lock will do the gc instead.
	if(!mstats.enablegc || m->locks > 0 || runtime·panicking)
		return;

	if(gcpercent == -2) {	// first time through
		p = runtime·getenv("GOGC");
		if(p == nil || p[0] == '\0')
			gcpercent = 100;
		else if(runtime·strcmp(p, (byte*)"off") == 0)
			gcpercent = -1;
		else
			gcpercent = runtime·atoi(p);
		
		p = runtime·getenv("GOGCTRACE");
		if(p != nil)
			gctrace = runtime·atoi(p);
	}
	if(gcpercent < 0)
		return;

	runtime·semacquire(&gcsema);
	if(!force && mstats.heap_alloc < mstats.next_gc) {
		runtime·semrelease(&gcsema);
		return;
	}

	t0 = runtime·nanotime();
	nlookup = 0;
	nsizelookup = 0;
	naddrlookup = 0;

	m->gcing = 1;
	runtime·stoptheworld();
	if(runtime·mheap.Lock.key != 0)
		runtime·throw("runtime·mheap locked during gc");

	cachestats();
	heap0 = mstats.heap_alloc;
	obj0 = mstats.nmalloc - mstats.nfree;

	mark();
	t1 = runtime·nanotime();
	sweep();
	t2 = runtime·nanotime();
	stealcache();
	cachestats();

	mstats.next_gc = mstats.heap_alloc+mstats.heap_alloc*gcpercent/100;
	m->gcing = 0;

	m->locks++;	// disable gc during the mallocs in newproc
	fp = finq;
	if(fp != nil) {
		// kick off or wake up goroutine to run queued finalizers
		if(fing == nil)
			fing = runtime·newproc1((byte*)runfinq, nil, 0, 0, runtime·gc);
		else if(fingwait) {
			fingwait = 0;
			runtime·ready(fing);
		}
	}
	m->locks--;

	cachestats();
	heap1 = mstats.heap_alloc;
	obj1 = mstats.nmalloc - mstats.nfree;

	t3 = runtime·nanotime();
	mstats.pause_ns[mstats.numgc%nelem(mstats.pause_ns)] = t3 - t0;
	mstats.pause_total_ns += t3 - t0;
	mstats.numgc++;
	if(mstats.debuggc)
		runtime·printf("pause %D\n", t3-t0);
	
	if(gctrace) {
		runtime·printf("gc%d: %D+%D+%D ms %D -> %D MB %D -> %D (%D-%D) objects %D pointer lookups (%D size, %D addr)\n",
			mstats.numgc, (t1-t0)/1000000, (t2-t1)/1000000, (t3-t2)/1000000,
			heap0>>20, heap1>>20, obj0, obj1,
			mstats.nmalloc, mstats.nfree,
			nlookup, nsizelookup, naddrlookup);
	}

	runtime·semrelease(&gcsema);
	runtime·starttheworld();
	
	// give the queued finalizers, if any, a chance to run
	if(fp != nil)
		runtime·gosched();
	
	if(gctrace > 1 && !force)
		runtime·gc(1);
}

void
runtime·UpdateMemStats(void)
{
	// Have to acquire gcsema to stop the world,
	// because stoptheworld can only be used by
	// one goroutine at a time, and there might be
	// a pending garbage collection already calling it.
	runtime·semacquire(&gcsema);
	m->gcing = 1;
	runtime·stoptheworld();
	cachestats();
	m->gcing = 0;
	runtime·semrelease(&gcsema);
	runtime·starttheworld();
}

static void
runfinq(void)
{
	Finalizer *f, *next;
	byte *frame;

	for(;;) {
		// There's no need for a lock in this section
		// because it only conflicts with the garbage
		// collector, and the garbage collector only
		// runs when everyone else is stopped, and
		// runfinq only stops at the gosched() or
		// during the calls in the for loop.
		f = finq;
		finq = nil;
		if(f == nil) {
			fingwait = 1;
			g->status = Gwaiting;
			runtime·gosched();
			continue;
		}
		for(; f; f=next) {
			next = f->next;
			frame = runtime·mal(sizeof(uintptr) + f->nret);
			*(void**)frame = f->arg;
			reflect·call((byte*)f->fn, frame, sizeof(uintptr) + f->nret);
			runtime·free(frame);
			f->fn = nil;
			f->arg = nil;
			f->next = nil;
			runtime·free(f);
		}
		runtime·gc(1);	// trigger another gc to clean up the finalized objects, if possible
	}
}

// mark the block at v of size n as allocated.
// If noptr is true, mark it as having no pointers.
void
runtime·markallocated(void *v, uintptr n, bool noptr)
{
	uintptr *b, obits, bits, off, shift;

	if(0)
		runtime·printf("markallocated %p+%p\n", v, n);

	if((byte*)v+n > (byte*)runtime·mheap.arena_used || (byte*)v < runtime·mheap.arena_start)
		runtime·throw("markallocated: bad pointer");

	off = (uintptr*)v - (uintptr*)runtime·mheap.arena_start;  // word offset
	b = (uintptr*)runtime·mheap.arena_start - off/wordsPerBitmapWord - 1;
	shift = off % wordsPerBitmapWord;

	for(;;) {
		obits = *b;
		bits = (obits & ~(bitMask<<shift)) | (bitAllocated<<shift);
		if(noptr)
			bits |= bitNoPointers<<shift;
		if(runtime·singleproc) {
			*b = bits;
			break;
		} else {
			// more than one goroutine is potentially running: use atomic op
			if(runtime·casp((void**)b, (void*)obits, (void*)bits))
				break;
		}
	}
}

// mark the block at v of size n as freed.
void
runtime·markfreed(void *v, uintptr n)
{
	uintptr *b, obits, bits, off, shift;

	if(0)
		runtime·printf("markallocated %p+%p\n", v, n);

	if((byte*)v+n > (byte*)runtime·mheap.arena_used || (byte*)v < runtime·mheap.arena_start)
		runtime·throw("markallocated: bad pointer");

	off = (uintptr*)v - (uintptr*)runtime·mheap.arena_start;  // word offset
	b = (uintptr*)runtime·mheap.arena_start - off/wordsPerBitmapWord - 1;
	shift = off % wordsPerBitmapWord;

	for(;;) {
		obits = *b;
		bits = (obits & ~(bitMask<<shift)) | (bitBlockBoundary<<shift);
		if(runtime·singleproc) {
			*b = bits;
			break;
		} else {
			// more than one goroutine is potentially running: use atomic op
			if(runtime·casp((void**)b, (void*)obits, (void*)bits))
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
	uintptr *b, off, shift;
	byte *p;

	if((byte*)v+size*n > (byte*)runtime·mheap.arena_used || (byte*)v < runtime·mheap.arena_start)
		runtime·throw("markspan: bad pointer");

	p = v;
	if(leftover)	// mark a boundary just past end of last block too
		n++;
	for(; n-- > 0; p += size) {
		// Okay to use non-atomic ops here, because we control
		// the entire span, and each bitmap word has bits for only
		// one span, so no other goroutines are changing these
		// bitmap words.
		off = (uintptr*)p - (uintptr*)runtime·mheap.arena_start;  // word offset
		b = (uintptr*)runtime·mheap.arena_start - off/wordsPerBitmapWord - 1;
		shift = off % wordsPerBitmapWord;
		*b = (*b & ~(bitMask<<shift)) | (bitBlockBoundary<<shift);
	}
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

bool
runtime·blockspecial(void *v)
{
	uintptr *b, off, shift;

	off = (uintptr*)v - (uintptr*)runtime·mheap.arena_start;
	b = (uintptr*)runtime·mheap.arena_start - off/wordsPerBitmapWord - 1;
	shift = off % wordsPerBitmapWord;

	return (*b & (bitSpecial<<shift)) != 0;
}

void
runtime·setblockspecial(void *v)
{
	uintptr *b, off, shift, bits, obits;

	off = (uintptr*)v - (uintptr*)runtime·mheap.arena_start;
	b = (uintptr*)runtime·mheap.arena_start - off/wordsPerBitmapWord - 1;
	shift = off % wordsPerBitmapWord;

	for(;;) {
		obits = *b;
		bits = obits | (bitSpecial<<shift);
		if(runtime·singleproc) {
			*b = bits;
			break;
		} else {
			// more than one goroutine is potentially running: use atomic op
			if(runtime·casp((void**)b, (void*)obits, (void*)bits))
				break;
		}
	}
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
	n = (n+bitmapChunk-1) & ~(bitmapChunk-1);
	if(h->bitmap_mapped >= n)
		return;

	runtime·SysMap(h->arena_start - n, n - h->bitmap_mapped);
	h->bitmap_mapped = n;
}
