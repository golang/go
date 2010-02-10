// Copyright 2009 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// Garbage collector -- step 0.
//
// Stop the world, mark and sweep garbage collector.
// NOT INTENDED FOR PRODUCTION USE.
//
// A mark and sweep collector provides a way to exercise
// and test the memory allocator and the stack walking machinery
// without also needing to get reference counting
// exactly right.

#include "runtime.h"
#include "malloc.h"

enum {
	Debug = 0
};

extern byte data[];
extern byte etext[];
extern byte end[];

typedef struct Finq Finq;
struct Finq
{
	void (*fn)(void*);
	void *p;
	int32 nret;
};

static Finq finq[128];	// finalizer queue - two elements per entry
static Finq *pfinq = finq;
static Finq *efinq = finq+nelem(finq);

static void sweepblock(byte*, int64, uint32*, int32);

enum {
	PtrSize = sizeof(void*)
};

static void
scanblock(int32 depth, byte *b, int64 n)
{
	int32 off;
	void *obj;
	uintptr size;
	uint32 *refp, ref;
	void **vp;
	int64 i;

	if(Debug > 1)
		printf("%d scanblock %p %D\n", depth, b, n);
	off = (uint32)(uintptr)b & (PtrSize-1);
	if(off) {
		b += PtrSize - off;
		n -= PtrSize - off;
	}

	vp = (void**)b;
	n /= PtrSize;
	for(i=0; i<n; i++) {
		obj = vp[i];
		if(obj == nil || (byte*)obj < mheap.min || (byte*)obj >= mheap.max)
			continue;
		if(mlookup(obj, &obj, &size, &refp)) {
			ref = *refp;
			switch(ref & ~(RefNoPointers|RefHasFinalizer)) {
			case RefFinalize:
				// If marked for finalization already, some other finalization-ready
				// object has a pointer: turn off finalization until that object is gone.
				// This means that cyclic finalizer loops never get collected,
				// so don't do that.
				/* fall through */
			case RefNone:
				if(Debug > 1)
					printf("%d found at %p: ", depth, &vp[i]);
				*refp = RefSome | (ref & (RefNoPointers|RefHasFinalizer));
				if(!(ref & RefNoPointers))
					scanblock(depth+1, obj, size);
				break;
			}
		}
	}
}

static void
scanstack(G *gp)
{
	Stktop *stk;
	byte *sp;

	if(gp == g)
		sp = (byte*)&gp;
	else
		sp = gp->sched.sp;
	if(Debug > 1)
		printf("scanstack %d %p\n", gp->goid, sp);
	stk = (Stktop*)gp->stackbase;
	while(stk) {
		scanblock(0, sp, (byte*)stk - sp);
		sp = stk->gobuf.sp;
		stk = (Stktop*)stk->stackbase;
	}
}

static void
mark(void)
{
	G *gp;

	// mark data+bss.
	// skip mheap itself, which has no interesting pointers
	// and is mostly zeroed and would not otherwise be paged in.
	scanblock(0, data, (byte*)&mheap - data);
	scanblock(0, (byte*)(&mheap+1), end - (byte*)(&mheap+1));

	// mark stacks
	for(gp=allg; gp!=nil; gp=gp->alllink) {
		switch(gp->status){
		default:
			printf("unexpected G.status %d\n", gp->status);
			throw("mark - bad status");
		case Gdead:
			break;
		case Grunning:
			if(gp != g)
				throw("mark - world not stopped");
			scanstack(gp);
			break;
		case Grunnable:
		case Gsyscall:
		case Gwaiting:
			scanstack(gp);
			break;
		}
	}
}

// pass 0: mark RefNone with finalizer as RefFinalize and trace
static void
sweepspan0(MSpan *s)
{
	byte *p;
	uint32 ref, *gcrefp, *gcrefep;
	int32 n, size, npages;

	p = (byte*)(s->start << PageShift);
	if(s->sizeclass == 0) {
		// Large block.
		ref = s->gcref0;
		if((ref&~RefNoPointers) == (RefNone|RefHasFinalizer)) {
			// Mark as finalizable.
			s->gcref0 = RefFinalize | RefHasFinalizer | (ref&RefNoPointers);
			if(!(ref & RefNoPointers))
				scanblock(100, p, s->npages<<PageShift);
		}
		return;
	}

	// Chunk full of small blocks.
	MGetSizeClassInfo(s->sizeclass, &size, &npages, &n);
	gcrefp = s->gcref;
	gcrefep = s->gcref + n;
	for(; gcrefp < gcrefep; gcrefp++) {
		ref = *gcrefp;
		if((ref&~RefNoPointers) == (RefNone|RefHasFinalizer)) {
			// Mark as finalizable.
			*gcrefp = RefFinalize | RefHasFinalizer | (ref&RefNoPointers);
			if(!(ref & RefNoPointers))
				scanblock(100, p+(gcrefp-s->gcref)*size, size);
		}
	}
}	

// pass 1: free RefNone, queue RefFinalize, reset RefSome
static void
sweepspan1(MSpan *s)
{
	int32 n, npages, size;
	byte *p;
	uint32 ref, *gcrefp, *gcrefep;
	MCache *c;

	p = (byte*)(s->start << PageShift);
	if(s->sizeclass == 0) {
		// Large block.
		ref = s->gcref0;
		switch(ref & ~(RefNoPointers|RefHasFinalizer)) {
		case RefNone:
			// Free large object.
			mstats.alloc -= s->npages<<PageShift;
			runtime_memclr(p, s->npages<<PageShift);
			s->gcref0 = RefFree;
			MHeap_Free(&mheap, s);
			break;
		case RefFinalize:
			if(pfinq < efinq) {
				pfinq->p = p;
				pfinq->nret = 0;
				pfinq->fn = getfinalizer(p, 1, &pfinq->nret);
				ref &= ~RefHasFinalizer;
				if(pfinq->fn == nil)
					throw("finalizer inconsistency");
				pfinq++;
			}
			// fall through
		case RefSome:
			s->gcref0 = RefNone | (ref&(RefNoPointers|RefHasFinalizer));
			break;
		}
		return;
	}

	// Chunk full of small blocks.
	MGetSizeClassInfo(s->sizeclass, &size, &npages, &n);
	gcrefp = s->gcref;
	gcrefep = s->gcref + n;
	for(; gcrefp < gcrefep; gcrefp++, p += size) {
		ref = *gcrefp;
		if(ref < RefNone)	// RefFree or RefStack
			continue;
		switch(ref & ~(RefNoPointers|RefHasFinalizer)) {
		case RefNone:
			// Free small object.
			*gcrefp = RefFree;
			c = m->mcache;
			if(size > sizeof(uintptr))
				((uintptr*)p)[1] = 1;	// mark as "needs to be zeroed"
			mstats.alloc -= size;
			mstats.by_size[s->sizeclass].nfree++;
			MCache_Free(c, p, s->sizeclass, size);
			break;
		case RefFinalize:
			if(pfinq < efinq) {
				pfinq->p = p;
				pfinq->nret = 0;
				pfinq->fn = getfinalizer(p, 1, &pfinq->nret);
				ref &= ~RefHasFinalizer;
				if(pfinq->fn == nil)	
					throw("finalizer inconsistency");
				pfinq++;
			}
			// fall through
		case RefSome:
			*gcrefp = RefNone | (ref&(RefNoPointers|RefHasFinalizer));
			break;
		}
	}
}

static void
sweep(void)
{
	MSpan *s;

	// Sweep all the spans marking blocks to be finalized.
	for(s = mheap.allspans; s != nil; s = s->allnext)
		if(s->state == MSpanInUse)
			sweepspan0(s);

	// Sweep again queueing finalizers and freeing the others.
	for(s = mheap.allspans; s != nil; s = s->allnext)
		if(s->state == MSpanInUse)
			sweepspan1(s);
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

void
gc(int32 force)
{
	int64 t0, t1;
	byte *p;
	Finq *fp;

	// The gc is turned off (via enablegc) until
	// the bootstrap has completed.
	// Also, malloc gets called in the guts
	// of a number of libraries that might be
	// holding locks.  To avoid priority inversion
	// problems, don't bother trying to run gc
	// while holding a lock.  The next mallocgc
	// without a lock will do the gc instead.
	if(!mstats.enablegc || m->locks > 0 || panicking)
		return;

	if(gcpercent == -2) {	// first time through
		p = getenv("GOGC");
		if(p == nil || p[0] == '\0')
			gcpercent = 100;
		else if(strcmp(p, (byte*)"off") == 0)
			gcpercent = -1;
		else
			gcpercent = atoi(p);
	}
	if(gcpercent < 0)
		return;

//printf("gc...\n");
	semacquire(&gcsema);
	t0 = nanotime();
	m->gcing = 1;
	stoptheworld();
	if(mheap.Lock.key != 0)
		throw("mheap locked during gc");
	if(force || mstats.inuse_pages >= mstats.next_gc) {
		mark();
		sweep();
		mstats.next_gc = mstats.inuse_pages+mstats.inuse_pages*gcpercent/100;
	}
	m->gcing = 0;

	// kick off goroutines to run queued finalizers
	m->locks++;	// disable gc during the mallocs in newproc
	for(fp=finq; fp<pfinq; fp++) {
		newproc1((byte*)fp->fn, (byte*)&fp->p, sizeof(fp->p), fp->nret);
		fp->fn = nil;
		fp->p = nil;
	}
	pfinq = finq;
	m->locks--;

	t1 = nanotime();
	mstats.numgc++;
	mstats.pause_ns += t1 - t0;
	if(mstats.debuggc)
		printf("pause %D\n", t1-t0);
	semrelease(&gcsema);
	starttheworld();
}
