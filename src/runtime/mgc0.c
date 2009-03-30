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

extern byte etext[];
extern byte end[];

static void
scanblock(int32 depth, byte *b, int64 n)
{
	int32 off;
	void *obj;
	uintptr size;
	uint32 *ref;
	void **vp;
	int64 i;

	if(Debug)
		printf("%d scanblock %p %D\n", depth, b, n);
	off = (uint32)(uintptr)b & 7;
	if(off) {
		b += 8 - off;
		n -= 8 - off;
	}

	vp = (void**)b;
	n /= 8;
	for(i=0; i<n; i++) {
		if(mlookup(vp[i], &obj, &size, &ref)) {
			if(*ref == RefFree || *ref == RefStack)
				continue;
			if(*ref == RefNone) {
				if(Debug)
					printf("%d found at %p: ", depth, &vp[i]);
				*ref = RefSome;
				scanblock(depth+1, obj, size);
			}
		}
	}
}

static void
scanstack(G *g)
{
	Stktop *stk;
	byte *sp;

	sp = g->sched.SP;
	stk = (Stktop*)g->stackbase;
	while(stk) {
		scanblock(0, sp, (byte*)stk - sp);
		sp = stk->oldsp;
		stk = (Stktop*)stk->oldbase;
	}
}

static void
mark(void)
{
	G *gp;

	// mark data+bss
	scanblock(0, etext, end - etext);

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

static void
sweepspan(MSpan *s)
{
	int32 i, n, npages, size;
	byte *p;

	if(s->state != MSpanInUse)
		return;

	p = (byte*)(s->start << PageShift);
	if(s->sizeclass == 0) {
		// Large block.
		switch(s->gcref0) {
		default:
			throw("bad 'ref count'");
		case RefFree:
		case RefManual:
		case RefStack:
			break;
		case RefNone:
			if(Debug)
				printf("free %D at %p\n", (uint64)s->npages<<PageShift, p);
			free(p);
			break;
		case RefSome:
			s->gcref0 = RefNone;	// set up for next mark phase
			break;
		}
		return;
	}

	// Chunk full of small blocks.
	// Must match computation in MCentral_Grow.
	size = class_to_size[s->sizeclass];
	npages = class_to_allocnpages[s->sizeclass];
	n = (npages << PageShift) / (size + RefcountOverhead);
	for(i=0; i<n; i++) {
		switch(s->gcref[i]) {
		default:
			throw("bad 'ref count'");
		case RefFree:
		case RefManual:
		case RefStack:
			break;
		case RefNone:
			if(Debug)
				printf("free %d at %p\n", size, p+i*size);
			free(p + i*size);
			break;
		case RefSome:
			s->gcref[i] = RefNone;	// set up for next mark phase
			break;
		}
	}
}

static void
sweep(void)
{
	MSpan *s;

	// Sweep all the spans.
	for(s = mheap.allspans; s != nil; s = s->allnext)
		sweepspan(s);
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
	byte *p;

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

	semacquire(&gcsema);
	gosave(&g->sched);	// update g's stack pointer for scanstack
	stoptheworld();
	if(mheap.Lock.key != 0)
		throw("mheap locked during gc");
	if(force || mstats.inuse_pages >= mstats.next_gc) {
		mark();
		sweep();
		mstats.next_gc = mstats.inuse_pages+mstats.inuse_pages*gcpercent/100;
	}
	starttheworld();
	gosave(&g->sched);	// update g's stack pointer for debugging
	semrelease(&gcsema);
}
