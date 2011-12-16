// Copyright 2009 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// Garbage collector.

#include "runtime.h"
#include "arch_GOARCH.h"
#include "malloc.h"
#include "stack.h"

enum {
	Debug = 0,
	PtrSize = sizeof(void*),
	DebugMark = 0,  // run second pass to check mark

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

// TODO: Make these per-M.
static uint64 nlookup;
static uint64 nsizelookup;
static uint64 naddrlookup;
static uint64 nhandoff;

static int32 gctrace;

typedef struct Workbuf Workbuf;
struct Workbuf
{
	Workbuf *next;
	uintptr nobj;
	byte *obj[512-2];
};

typedef struct Finalizer Finalizer;
struct Finalizer
{
	void (*fn)(void*);
	void *arg;
	int32 nret;
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
extern byte etext[];
extern byte end[];

static G *fing;
static FinBlock *finq; // list of finalizers that are to be executed
static FinBlock *finc; // cache of free blocks
static FinBlock *allfin; // list of all blocks
static Lock finlock;
static int32 fingwait;

static void runfinq(void);
static Workbuf* getempty(Workbuf*);
static Workbuf* getfull(Workbuf*);
static void	putempty(Workbuf*);
static Workbuf* handoff(Workbuf*);

static struct {
	Lock fmu;
	Workbuf	*full;
	Lock emu;
	Workbuf	*empty;
	uint32	nproc;
	volatile uint32	nwait;
	volatile uint32	ndone;
	Note	alldone;
	Lock	markgate;
	Lock	sweepgate;
	MSpan	*spans;

	Lock;
	byte	*chunk;
	uintptr	nchunk;
} work;

// scanblock scans a block of n bytes starting at pointer b for references
// to other objects, scanning any it finds recursively until there are no
// unscanned objects left.  Instead of using an explicit recursion, it keeps
// a work list in the Workbuf* structures and loops in the main function
// body.  Keeping an explicit work list is easier on the stack allocator and
// more efficient.
static void
scanblock(byte *b, int64 n)
{
	byte *obj, *arena_start, *arena_used, *p;
	void **vp;
	uintptr size, *bitp, bits, shift, i, j, x, xbits, off, nobj, nproc;
	MSpan *s;
	PageID k;
	void **wp;
	Workbuf *wbuf;
	bool keepworking;

	if((int64)(uintptr)n != n || n < 0) {
		runtime·printf("scanblock %p %D\n", b, n);
		runtime·throw("scanblock");
	}

	// Memory arena parameters.
	arena_start = runtime·mheap.arena_start;
	arena_used = runtime·mheap.arena_used;
	nproc = work.nproc;

	wbuf = nil;  // current work buffer
	wp = nil;  // storage for next queued pointer (write pointer)
	nobj = 0;  // number of queued objects

	// Scanblock helpers pass b==nil.
	// The main proc needs to return to make more
	// calls to scanblock.  But if work.nproc==1 then
	// might as well process blocks as soon as we
	// have them.
	keepworking = b == nil || work.nproc == 1;

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
		n >>= (2+PtrSize/8);  /* n /= PtrSize (4 or 8) */
		for(i=0; i<n; i++) {
			obj = (byte*)vp[i];

			// Words outside the arena cannot be pointers.
			if((byte*)obj < arena_start || (byte*)obj >= arena_used)
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
			// Only care about allocated and not marked.
			if((bits & (bitAllocated|bitMarked)) != bitAllocated)
				continue;
			if(nproc == 1)
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
			if((bits & bitNoPointers) != 0)
				continue;

			// If another proc wants a pointer, give it some.
			if(nobj > 4 && work.nwait > 0 && work.full == nil) {
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
			*wp++ = obj;
			nobj++;
		continue_obj:;
		}

		// Done scanning [b, b+n).  Prepare for the next iteration of
		// the loop by setting b and n to the parameters for the next block.

		// Fetch b from the work buffer.
		if(nobj == 0) {
			if(!keepworking) {
				putempty(wbuf);
				return;
			}
			// Emptied our buffer: refill.
			wbuf = getfull(wbuf);
			if(wbuf == nil)
				return;
			nobj = wbuf->nobj;
			wp = wbuf->obj + wbuf->nobj;
		}
		b = *--wp;
		nobj--;

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

// debug_scanblock is the debug copy of scanblock.
// it is simpler, slower, single-threaded, recursive,
// and uses bitSpecial as the mark bit.
static void
debug_scanblock(byte *b, int64 n)
{
	byte *obj, *p;
	void **vp;
	uintptr size, *bitp, bits, shift, i, xbits, off;
	MSpan *s;

	if(!DebugMark)
		runtime·throw("debug_scanblock without DebugMark");

	if((int64)(uintptr)n != n || n < 0) {
		runtime·printf("debug_scanblock %p %D\n", b, n);
		runtime·throw("debug_scanblock");
	}

	// Align b to a word boundary.
	off = (uintptr)b & (PtrSize-1);
	if(off != 0) {
		b += PtrSize - off;
		n -= PtrSize - off;
	}

	vp = (void**)b;
	n /= PtrSize;
	for(i=0; i<n; i++) {
		obj = (byte*)vp[i];

		// Words outside the arena cannot be pointers.
		if((byte*)obj < runtime·mheap.arena_start || (byte*)obj >= runtime·mheap.arena_used)
			continue;

		// Round down to word boundary.
		obj = (void*)((uintptr)obj & ~((uintptr)PtrSize-1));

		// Consult span table to find beginning.
		s = runtime·MHeap_LookupMaybe(&runtime·mheap, obj);
		if(s == nil)
			continue;


		p =  (byte*)((uintptr)s->start<<PageShift);
		if(s->sizeclass == 0) {
			obj = p;
			size = (uintptr)s->npages<<PageShift;
		} else {
			if((byte*)obj >= (byte*)s->limit)
				continue;
			size = runtime·class_to_size[s->sizeclass];
			int32 i = ((byte*)obj - p)/size;
			obj = p+i*size;
		}

		// Now that we know the object header, reload bits.
		off = (uintptr*)obj - (uintptr*)runtime·mheap.arena_start;
		bitp = (uintptr*)runtime·mheap.arena_start - off/wordsPerBitmapWord - 1;
		shift = off % wordsPerBitmapWord;
		xbits = *bitp;
		bits = xbits >> shift;

		// Now we have bits, bitp, and shift correct for
		// obj pointing at the base of the object.
		// If not allocated or already marked, done.
		if((bits & bitAllocated) == 0 || (bits & bitSpecial) != 0)  // NOTE: bitSpecial not bitMarked
			continue;
		*bitp |= bitSpecial<<shift;
		if(!(bits & bitMarked))
			runtime·printf("found unmarked block %p in %p\n", obj, vp+i);

		// If object has no pointers, don't need to scan further.
		if((bits & bitNoPointers) != 0)
			continue;

		debug_scanblock(obj, size);
	}
}

// Get an empty work buffer off the work.empty list,
// allocating new buffers as needed.
static Workbuf*
getempty(Workbuf *b)
{
	if(work.nproc == 1) {
		// Put b on full list.
		if(b != nil) {
			b->next = work.full;
			work.full = b;
		}
		// Grab from empty list if possible.
		b = work.empty;
		if(b != nil) {
			work.empty = b->next;
			goto haveb;
		}
	} else {
		// Put b on full list.
		if(b != nil) {
			runtime·lock(&work.fmu);
			b->next = work.full;
			work.full = b;
			runtime·unlock(&work.fmu);
		}
		// Grab from empty list if possible.
		runtime·lock(&work.emu);
		b = work.empty;
		if(b != nil)
			work.empty = b->next;
		runtime·unlock(&work.emu);
		if(b != nil)
			goto haveb;
	}

	// Need to allocate.
	runtime·lock(&work);
	if(work.nchunk < sizeof *b) {
		work.nchunk = 1<<20;
		work.chunk = runtime·SysAlloc(work.nchunk);
	}
	b = (Workbuf*)work.chunk;
	work.chunk += sizeof *b;
	work.nchunk -= sizeof *b;
	runtime·unlock(&work);

haveb:
	b->nobj = 0;
	return b;
}

static void
putempty(Workbuf *b)
{
	if(b == nil)
		return;

	if(work.nproc == 1) {
		b->next = work.empty;
		work.empty = b;
		return;
	}

	runtime·lock(&work.emu);
	b->next = work.empty;
	work.empty = b;
	runtime·unlock(&work.emu);
}

// Get a full work buffer off the work.full list, or return nil.
static Workbuf*
getfull(Workbuf *b)
{
	int32 i;
	Workbuf *b1;

	if(work.nproc == 1) {
		// Put b on empty list.
		if(b != nil) {
			b->next = work.empty;
			work.empty = b;
		}
		// Grab from full list if possible.
		// Since work.nproc==1, no one else is
		// going to give us work.
		b = work.full;
		if(b != nil)
			work.full = b->next;
		return b;
	}

	putempty(b);

	// Grab buffer from full list if possible.
	for(;;) {
		b1 = work.full;
		if(b1 == nil)
			break;
		runtime·lock(&work.fmu);
		if(work.full != nil) {
			b1 = work.full;
			work.full = b1->next;
			runtime·unlock(&work.fmu);
			return b1;
		}
		runtime·unlock(&work.fmu);
	}

	runtime·xadd(&work.nwait, +1);
	for(i=0;; i++) {
		b1 = work.full;
		if(b1 != nil) {
			runtime·lock(&work.fmu);
			if(work.full != nil) {
				runtime·xadd(&work.nwait, -1);
				b1 = work.full;
				work.full = b1->next;
				runtime·unlock(&work.fmu);
				return b1;
			}
			runtime·unlock(&work.fmu);
			continue;
		}
		if(work.nwait == work.nproc)
			return nil;
		if(i < 10)
			runtime·procyield(20);
		else if(i < 20)
			runtime·osyield();
		else
			runtime·usleep(100);
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
	nhandoff += n;

	// Put b on full list - let first half of b get stolen.
	runtime·lock(&work.fmu);
	b->next = work.full;
	work.full = b;
	runtime·unlock(&work.fmu);

	return b1;
}

// Scanstack calls scanblock on each of gp's stack segments.
static void
scanstack(void (*scanblock)(byte*, int64), G *gp)
{
	M *mp;
	int32 n;
	Stktop *stk;
	byte *sp, *guard;

	stk = (Stktop*)gp->stackbase;
	guard = gp->stackguard;

	if(gp == g) {
		// Scanning our own stack: start at &gp.
		sp = (byte*)&gp;
	} else if((mp = gp->m) != nil && mp->helpgc) {
		// gchelper's stack is in active use and has no interesting pointers.
		return;
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

static void
debug_markfin(void *v)
{
	uintptr size;

	if(!runtime·mlookup(v, &v, &size, nil))
		runtime·throw("debug_mark - finalizer inconsistency");
	debug_scanblock(v, size);
}

// Mark
static void
mark(void (*scan)(byte*, int64))
{
	G *gp;
	FinBlock *fb;

	// mark data+bss.
	// skip runtime·mheap itself, which has no interesting pointers
	// and is mostly zeroed and would not otherwise be paged in.
	scan(data, (byte*)&runtime·mheap - data);
	scan((byte*)(&runtime·mheap+1), end - (byte*)(&runtime·mheap+1));

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
			scanstack(scan, gp);
			break;
		case Grunnable:
		case Gsyscall:
		case Gwaiting:
			scanstack(scan, gp);
			break;
		}
	}

	// mark things pointed at by objects with finalizers
	if(scan == debug_scanblock)
		runtime·walkfintab(debug_markfin);
	else
		runtime·walkfintab(markfin);

	for(fb=allfin; fb; fb=fb->alllink)
		scanblock((byte*)fb->fin, fb->cnt*sizeof(fb->fin[0]));

	// in multiproc mode, join in the queued work.
	scan(nil, 0);
}

static bool
handlespecial(byte *p, uintptr size)
{
	void (*fn)(void*);
	int32 nret;
	FinBlock *block;
	Finalizer *f;
	
	if(!runtime·getfinalizer(p, true, &fn, &nret)) {
		runtime·setblockspecial(p, false);
		runtime·MProf_Free(p, size);
		return false;
	}

	runtime·lock(&finlock);
	if(finq == nil || finq->cnt == finq->cap) {
		if(finc == nil) {
			finc = runtime·SysAlloc(PageSize);
			finc->cap = (PageSize - sizeof(FinBlock)) / sizeof(Finalizer) + 1;
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
	f->arg = p;
	runtime·unlock(&finlock); 
	return true;
}

// Sweep frees or collects finalizers for blocks not marked in the mark phase.
// It clears the mark bits in preparation for the next GC round.
static void
sweep(void)
{
	MSpan *s;
	int32 cl, n, npages;
	uintptr size;
	byte *p;
	MCache *c;
	byte *arena_start;

	arena_start = runtime·mheap.arena_start;

	for(;;) {
		s = work.spans;
		if(s == nil)
			break;
		if(!runtime·casp(&work.spans, s, s->allnext))
			continue;

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

		// Sweep through n objects of given size starting at p.
		// This thread owns the span now, so it can manipulate
		// the block bitmap without atomic operations.
		for(; n > 0; n--, p += size) {
			uintptr off, *bitp, shift, bits;

			off = (uintptr*)p - (uintptr*)arena_start;
			bitp = (uintptr*)arena_start - off/wordsPerBitmapWord - 1;
			shift = off % wordsPerBitmapWord;
			bits = *bitp>>shift;

			if((bits & bitAllocated) == 0)
				continue;

			if((bits & bitMarked) != 0) {
				if(DebugMark) {
					if(!(bits & bitSpecial))
						runtime·printf("found spurious mark on %p\n", p);
					*bitp &= ~(bitSpecial<<shift);
				}
				*bitp &= ~(bitMarked<<shift);
				continue;
			}

			// Special means it has a finalizer or is being profiled.
			// In DebugMark mode, the bit has been coopted so
			// we have to assume all blocks are special.
			if(DebugMark || (bits & bitSpecial) != 0) {
				if(handlespecial(p, size))
					continue;
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

void
runtime·gchelper(void)
{
	// Wait until main proc is ready for mark help.
	runtime·lock(&work.markgate);
	runtime·unlock(&work.markgate);
	scanblock(nil, 0);

	// Wait until main proc is ready for sweep help.
	runtime·lock(&work.sweepgate);
	runtime·unlock(&work.sweepgate);
	sweep();

	if(runtime·xadd(&work.ndone, +1) == work.nproc-1)
		runtime·notewakeup(&work.alldone);
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
	bool extra;

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
	nhandoff = 0;

	m->gcing = 1;
	runtime·stoptheworld();

	cachestats();
	heap0 = mstats.heap_alloc;
	obj0 = mstats.nmalloc - mstats.nfree;

	runtime·lock(&work.markgate);
	runtime·lock(&work.sweepgate);

	extra = false;
	work.nproc = 1;
	if(runtime·gomaxprocs > 1 && runtime·ncpu > 1) {
		runtime·noteclear(&work.alldone);
		work.nproc += runtime·helpgc(&extra);
	}
	work.nwait = 0;
	work.ndone = 0;

	runtime·unlock(&work.markgate);  // let the helpers in
	mark(scanblock);
	if(DebugMark)
		mark(debug_scanblock);
	t1 = runtime·nanotime();

	work.spans = runtime·mheap.allspans;
	runtime·unlock(&work.sweepgate);  // let the helpers in
	sweep();
	if(work.nproc > 1)
		runtime·notesleep(&work.alldone);
	t2 = runtime·nanotime();

	stealcache();
	cachestats();

	mstats.next_gc = mstats.heap_alloc+mstats.heap_alloc*gcpercent/100;
	m->gcing = 0;

	m->locks++;	// disable gc during the mallocs in newproc
	if(finq != nil) {
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
		runtime·printf("gc%d(%d): %D+%D+%D ms %D -> %D MB %D -> %D (%D-%D) objects %D pointer lookups (%D size, %D addr) %D handoff\n",
			mstats.numgc, work.nproc, (t1-t0)/1000000, (t2-t1)/1000000, (t3-t2)/1000000,
			heap0>>20, heap1>>20, obj0, obj1,
			mstats.nmalloc, mstats.nfree,
			nlookup, nsizelookup, naddrlookup, nhandoff);
	}

	runtime·semrelease(&gcsema);

	// If we could have used another helper proc, start one now,
	// in the hope that it will be available next time.
	// It would have been even better to start it before the collection,
	// but doing so requires allocating memory, so it's tricky to
	// coordinate.  This lazy approach works out in practice:
	// we don't mind if the first couple gc rounds don't have quite
	// the maximum number of procs.
	runtime·starttheworld(extra);

	// give the queued finalizers, if any, a chance to run	
	if(finq != nil)	
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
	runtime·starttheworld(false);
}

static void
runfinq(void)
{
	Finalizer *f;
	FinBlock *fb, *next;
	byte *frame;
	uint32 framesz, framecap, i;

	frame = nil;
	framecap = 0;
	for(;;) {
		// There's no need for a lock in this section
		// because it only conflicts with the garbage
		// collector, and the garbage collector only
		// runs when everyone else is stopped, and
		// runfinq only stops at the gosched() or
		// during the calls in the for loop.
		fb = finq;
		finq = nil;
		if(fb == nil) {
			fingwait = 1;
			g->status = Gwaiting;
			g->waitreason = "finalizer wait";
			runtime·gosched();
			continue;
		}
		for(; fb; fb=next) {
			next = fb->next;
			for(i=0; i<fb->cnt; i++) {
				f = &fb->fin[i];
				framesz = sizeof(uintptr) + f->nret;
				if(framecap < framesz) {
					runtime·free(frame);
					frame = runtime·mal(framesz);
					framecap = framesz;
				}
				*(void**)frame = f->arg;
				runtime·setblockspecial(f->arg, false);
				reflect·call((byte*)f->fn, frame, sizeof(uintptr) + f->nret);
				f->fn = nil;
				f->arg = nil;
			}
			fb->cnt = 0;
			fb->next = finc;
			finc = fb;
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

	if(DebugMark)
		return true;

	off = (uintptr*)v - (uintptr*)runtime·mheap.arena_start;
	b = (uintptr*)runtime·mheap.arena_start - off/wordsPerBitmapWord - 1;
	shift = off % wordsPerBitmapWord;

	return (*b & (bitSpecial<<shift)) != 0;
}

void
runtime·setblockspecial(void *v, bool s)
{
	uintptr *b, off, shift, bits, obits;

	if(DebugMark)
		return;

	off = (uintptr*)v - (uintptr*)runtime·mheap.arena_start;
	b = (uintptr*)runtime·mheap.arena_start - off/wordsPerBitmapWord - 1;
	shift = off % wordsPerBitmapWord;

	for(;;) {
		obits = *b;
		if(s)
			bits = obits | (bitSpecial<<shift);
		else
			bits = obits & ~(bitSpecial<<shift);
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
