// Copyright 2009 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// Page heap.
//
// See malloc.h for overview.
//
// When a MSpan is in the heap free list, state == MSpanFree
// and heapmap(s->start) == span, heapmap(s->start+s->npages-1) == span.
//
// When a MSpan is allocated, state == MSpanInUse or MSpanStack
// and heapmap(i) == span for all s->start <= i < s->start+s->npages.

#include "runtime.h"
#include "arch_GOARCH.h"
#include "malloc.h"

static MSpan *MHeap_AllocSpanLocked(MHeap*, uintptr);
static void MHeap_FreeSpanLocked(MHeap*, MSpan*, bool, bool);
static bool MHeap_Grow(MHeap*, uintptr);
static MSpan *MHeap_AllocLarge(MHeap*, uintptr);
static MSpan *BestFit(MSpan*, uintptr, MSpan*);

static void
RecordSpan(void *vh, byte *p)
{
	MHeap *h;
	MSpan *s;
	MSpan **all;
	uint32 cap;

	h = vh;
	s = (MSpan*)p;
	if(h->nspan >= h->nspancap) {
		cap = 64*1024/sizeof(all[0]);
		if(cap < h->nspancap*3/2)
			cap = h->nspancap*3/2;
		all = (MSpan**)runtime·sysAlloc(cap*sizeof(all[0]), &mstats.other_sys);
		if(all == nil)
			runtime·throw("runtime: cannot allocate memory");
		if(h->allspans) {
			runtime·memmove(all, h->allspans, h->nspancap*sizeof(all[0]));
			// Don't free the old array if it's referenced by sweep.
			// See the comment in mgc0.c.
			if(h->allspans != runtime·mheap.gcspans)
				runtime·SysFree(h->allspans, h->nspancap*sizeof(all[0]), &mstats.other_sys);
		}
		h->allspans = all;
		h->nspancap = cap;
	}
	h->allspans[h->nspan++] = s;
}

// Initialize the heap; fetch memory using alloc.
void
runtime·MHeap_Init(MHeap *h)
{
	uint32 i;

	runtime·FixAlloc_Init(&h->spanalloc, sizeof(MSpan), RecordSpan, h, &mstats.mspan_sys);
	runtime·FixAlloc_Init(&h->cachealloc, sizeof(MCache), nil, nil, &mstats.mcache_sys);
	runtime·FixAlloc_Init(&h->specialfinalizeralloc, sizeof(SpecialFinalizer), nil, nil, &mstats.other_sys);
	runtime·FixAlloc_Init(&h->specialprofilealloc, sizeof(SpecialProfile), nil, nil, &mstats.other_sys);
	// h->mapcache needs no init
	for(i=0; i<nelem(h->free); i++) {
		runtime·MSpanList_Init(&h->free[i]);
		runtime·MSpanList_Init(&h->busy[i]);
	}
	runtime·MSpanList_Init(&h->freelarge);
	runtime·MSpanList_Init(&h->busylarge);
	for(i=0; i<nelem(h->central); i++)
		runtime·MCentral_Init(&h->central[i].mcentral, i);
}

void
runtime·MHeap_MapSpans(MHeap *h)
{
	uintptr n;

	// Map spans array, PageSize at a time.
	n = (uintptr)h->arena_used;
	n -= (uintptr)h->arena_start;
	n = n / PageSize * sizeof(h->spans[0]);
	n = ROUND(n, PhysPageSize);
	if(h->spans_mapped >= n)
		return;
	runtime·SysMap((byte*)h->spans + h->spans_mapped, n - h->spans_mapped, h->arena_reserved, &mstats.other_sys);
	h->spans_mapped = n;
}

// Sweeps spans in list until reclaims at least npages into heap.
// Returns the actual number of pages reclaimed.
static uintptr
MHeap_ReclaimList(MHeap *h, MSpan *list, uintptr npages)
{
	MSpan *s;
	uintptr n;
	uint32 sg;

	n = 0;
	sg = runtime·mheap.sweepgen;
retry:
	for(s = list->next; s != list; s = s->next) {
		if(s->sweepgen == sg-2 && runtime·cas(&s->sweepgen, sg-2, sg-1)) {
			runtime·MSpanList_Remove(s);
			// swept spans are at the end of the list
			runtime·MSpanList_InsertBack(list, s);
			runtime·unlock(&h->lock);
			n += runtime·MSpan_Sweep(s, false);
			runtime·lock(&h->lock);
			if(n >= npages)
				return n;
			// the span could have been moved elsewhere
			goto retry;
		}
		if(s->sweepgen == sg-1) {
			// the span is being sweept by background sweeper, skip
			continue;
		}
		// already swept empty span,
		// all subsequent ones must also be either swept or in process of sweeping
		break;
	}
	return n;
}

// Sweeps and reclaims at least npage pages into heap.
// Called before allocating npage pages.
static void
MHeap_Reclaim(MHeap *h, uintptr npage)
{
	uintptr reclaimed, n;

	// First try to sweep busy spans with large objects of size >= npage,
	// this has good chances of reclaiming the necessary space.
	for(n=npage; n < nelem(h->busy); n++) {
		if(MHeap_ReclaimList(h, &h->busy[n], npage))
			return;  // Bingo!
	}

	// Then -- even larger objects.
	if(MHeap_ReclaimList(h, &h->busylarge, npage))
		return;  // Bingo!

	// Now try smaller objects.
	// One such object is not enough, so we need to reclaim several of them.
	reclaimed = 0;
	for(n=0; n < npage && n < nelem(h->busy); n++) {
		reclaimed += MHeap_ReclaimList(h, &h->busy[n], npage-reclaimed);
		if(reclaimed >= npage)
			return;
	}

	// Now sweep everything that is not yet swept.
	runtime·unlock(&h->lock);
	for(;;) {
		n = runtime·sweepone();
		if(n == -1)  // all spans are swept
			break;
		reclaimed += n;
		if(reclaimed >= npage)
			break;
	}
	runtime·lock(&h->lock);
}

// Allocate a new span of npage pages from the heap for GC'd memory
// and record its size class in the HeapMap and HeapMapCache.
static MSpan*
mheap_alloc(MHeap *h, uintptr npage, int32 sizeclass, bool large)
{
	MSpan *s;

	if(g != g->m->g0)
		runtime·throw("mheap_alloc not on M stack");
	runtime·lock(&h->lock);

	// To prevent excessive heap growth, before allocating n pages
	// we need to sweep and reclaim at least n pages.
	if(!h->sweepdone)
		MHeap_Reclaim(h, npage);

	// transfer stats from cache to global
	mstats.heap_alloc += g->m->mcache->local_cachealloc;
	g->m->mcache->local_cachealloc = 0;
	mstats.tinyallocs += g->m->mcache->local_tinyallocs;
	g->m->mcache->local_tinyallocs = 0;

	s = MHeap_AllocSpanLocked(h, npage);
	if(s != nil) {
		// Record span info, because gc needs to be
		// able to map interior pointer to containing span.
		runtime·atomicstore(&s->sweepgen, h->sweepgen);
		s->state = MSpanInUse;
		s->freelist = nil;
		s->ref = 0;
		s->sizeclass = sizeclass;
		s->elemsize = (sizeclass==0 ? s->npages<<PageShift : runtime·class_to_size[sizeclass]);

		// update stats, sweep lists
		if(large) {
			mstats.heap_objects++;
			mstats.heap_alloc += npage<<PageShift;
			// Swept spans are at the end of lists.
			if(s->npages < nelem(h->free))
				runtime·MSpanList_InsertBack(&h->busy[s->npages], s);
			else
				runtime·MSpanList_InsertBack(&h->busylarge, s);
		}
	}
	runtime·unlock(&h->lock);
	return s;
}

static void
mheap_alloc_m(G *gp)
{
	MHeap *h;
	MSpan *s;

	h = g->m->ptrarg[0];
	g->m->ptrarg[0] = nil;
	s = mheap_alloc(h, g->m->scalararg[0], g->m->scalararg[1], g->m->scalararg[2]);
	g->m->ptrarg[0] = s;

	runtime·gogo(&gp->sched);
}

MSpan*
runtime·MHeap_Alloc(MHeap *h, uintptr npage, int32 sizeclass, bool large, bool needzero)
{
	MSpan *s;
	void (*fn)(G*);

	// Don't do any operations that lock the heap on the G stack.
	// It might trigger stack growth, and the stack growth code needs
	// to be able to allocate heap.
	if(g == g->m->g0) {
		s = mheap_alloc(h, npage, sizeclass, large);
	} else {
		g->m->ptrarg[0] = h;
		g->m->scalararg[0] = npage;
		g->m->scalararg[1] = sizeclass;
		g->m->scalararg[2] = large;
		fn = mheap_alloc_m;
		runtime·mcall(&fn);
		s = g->m->ptrarg[0];
		g->m->ptrarg[0] = nil;
	}
	if(s != nil) {
		if(needzero && s->needzero)
			runtime·memclr((byte*)(s->start<<PageShift), s->npages<<PageShift);
		s->needzero = 0;
	}
	return s;
}

MSpan*
runtime·MHeap_AllocStack(MHeap *h, uintptr npage)
{
	MSpan *s;

	if(g != g->m->g0)
		runtime·throw("mheap_allocstack not on M stack");
	runtime·lock(&h->lock);
	s = MHeap_AllocSpanLocked(h, npage);
	if(s != nil) {
		s->state = MSpanStack;
		s->freelist = nil;
		s->ref = 0;
		mstats.stacks_inuse += s->npages<<PageShift;
	}
	runtime·unlock(&h->lock);
	return s;
}

// Allocates a span of the given size.  h must be locked.
// The returned span has been removed from the
// free list, but its state is still MSpanFree.
static MSpan*
MHeap_AllocSpanLocked(MHeap *h, uintptr npage)
{
	uintptr n;
	MSpan *s, *t;
	pageID p;

	// Try in fixed-size lists up to max.
	for(n=npage; n < nelem(h->free); n++) {
		if(!runtime·MSpanList_IsEmpty(&h->free[n])) {
			s = h->free[n].next;
			goto HaveSpan;
		}
	}

	// Best fit in list of large spans.
	if((s = MHeap_AllocLarge(h, npage)) == nil) {
		if(!MHeap_Grow(h, npage))
			return nil;
		if((s = MHeap_AllocLarge(h, npage)) == nil)
			return nil;
	}

HaveSpan:
	// Mark span in use.
	if(s->state != MSpanFree)
		runtime·throw("MHeap_AllocLocked - MSpan not free");
	if(s->npages < npage)
		runtime·throw("MHeap_AllocLocked - bad npages");
	runtime·MSpanList_Remove(s);
	if(s->next != nil || s->prev != nil)
		runtime·throw("still in list");
	if(s->npreleased > 0) {
		runtime·SysUsed((void*)(s->start<<PageShift), s->npages<<PageShift);
		mstats.heap_released -= s->npreleased<<PageShift;
		s->npreleased = 0;
	}

	if(s->npages > npage) {
		// Trim extra and put it back in the heap.
		t = runtime·FixAlloc_Alloc(&h->spanalloc);
		runtime·MSpan_Init(t, s->start + npage, s->npages - npage);
		s->npages = npage;
		p = t->start;
		p -= ((uintptr)h->arena_start>>PageShift);
		if(p > 0)
			h->spans[p-1] = s;
		h->spans[p] = t;
		h->spans[p+t->npages-1] = t;
		t->needzero = s->needzero;
		s->state = MSpanStack; // prevent coalescing with s
		t->state = MSpanStack;
		MHeap_FreeSpanLocked(h, t, false, false);
		t->unusedsince = s->unusedsince; // preserve age (TODO: wrong: t is possibly merged and/or deallocated at this point)
		s->state = MSpanFree;
	}
	s->unusedsince = 0;

	p = s->start;
	p -= ((uintptr)h->arena_start>>PageShift);
	for(n=0; n<npage; n++)
		h->spans[p+n] = s;

	mstats.heap_inuse += npage<<PageShift;
	mstats.heap_idle -= npage<<PageShift;

	//runtime·printf("spanalloc %p\n", s->start << PageShift);
	if(s->next != nil || s->prev != nil)
		runtime·throw("still in list");
	return s;
}

// Allocate a span of exactly npage pages from the list of large spans.
static MSpan*
MHeap_AllocLarge(MHeap *h, uintptr npage)
{
	return BestFit(&h->freelarge, npage, nil);
}

// Search list for smallest span with >= npage pages.
// If there are multiple smallest spans, take the one
// with the earliest starting address.
static MSpan*
BestFit(MSpan *list, uintptr npage, MSpan *best)
{
	MSpan *s;

	for(s=list->next; s != list; s=s->next) {
		if(s->npages < npage)
			continue;
		if(best == nil
		|| s->npages < best->npages
		|| (s->npages == best->npages && s->start < best->start))
			best = s;
	}
	return best;
}

// Try to add at least npage pages of memory to the heap,
// returning whether it worked.
static bool
MHeap_Grow(MHeap *h, uintptr npage)
{
	uintptr ask;
	void *v;
	MSpan *s;
	pageID p;

	// Ask for a big chunk, to reduce the number of mappings
	// the operating system needs to track; also amortizes
	// the overhead of an operating system mapping.
	// Allocate a multiple of 64kB.
	npage = ROUND(npage, (64<<10)/PageSize);
	ask = npage<<PageShift;
	if(ask < HeapAllocChunk)
		ask = HeapAllocChunk;

	v = runtime·MHeap_SysAlloc(h, ask);
	if(v == nil) {
		if(ask > (npage<<PageShift)) {
			ask = npage<<PageShift;
			v = runtime·MHeap_SysAlloc(h, ask);
		}
		if(v == nil) {
			runtime·printf("runtime: out of memory: cannot allocate %D-byte block (%D in use)\n", (uint64)ask, mstats.heap_sys);
			return false;
		}
	}

	// Create a fake "in use" span and free it, so that the
	// right coalescing happens.
	s = runtime·FixAlloc_Alloc(&h->spanalloc);
	runtime·MSpan_Init(s, (uintptr)v>>PageShift, ask>>PageShift);
	p = s->start;
	p -= ((uintptr)h->arena_start>>PageShift);
	h->spans[p] = s;
	h->spans[p + s->npages - 1] = s;
	runtime·atomicstore(&s->sweepgen, h->sweepgen);
	s->state = MSpanInUse;
	MHeap_FreeSpanLocked(h, s, false, true);
	return true;
}

// Look up the span at the given address.
// Address is guaranteed to be in map
// and is guaranteed to be start or end of span.
MSpan*
runtime·MHeap_Lookup(MHeap *h, void *v)
{
	uintptr p;
	
	p = (uintptr)v;
	p -= (uintptr)h->arena_start;
	return h->spans[p >> PageShift];
}

// Look up the span at the given address.
// Address is *not* guaranteed to be in map
// and may be anywhere in the span.
// Map entries for the middle of a span are only
// valid for allocated spans.  Free spans may have
// other garbage in their middles, so we have to
// check for that.
MSpan*
runtime·MHeap_LookupMaybe(MHeap *h, void *v)
{
	MSpan *s;
	pageID p, q;

	if((byte*)v < h->arena_start || (byte*)v >= h->arena_used)
		return nil;
	p = (uintptr)v>>PageShift;
	q = p;
	q -= (uintptr)h->arena_start >> PageShift;
	s = h->spans[q];
	if(s == nil || p < s->start || v >= s->limit || s->state != MSpanInUse)
		return nil;
	return s;
}

// Free the span back into the heap.
static void
mheap_free(MHeap *h, MSpan *s, int32 acct)
{
	if(g != g->m->g0)
		runtime·throw("mheap_free not on M stack");
	runtime·lock(&h->lock);
	mstats.heap_alloc += g->m->mcache->local_cachealloc;
	g->m->mcache->local_cachealloc = 0;
	mstats.tinyallocs += g->m->mcache->local_tinyallocs;
	g->m->mcache->local_tinyallocs = 0;
	if(acct) {
		mstats.heap_alloc -= s->npages<<PageShift;
		mstats.heap_objects--;
	}
	MHeap_FreeSpanLocked(h, s, true, true);
	runtime·unlock(&h->lock);
}

static void
mheap_free_m(G *gp)
{
	MHeap *h;
	MSpan *s;
	
	h = g->m->ptrarg[0];
	s = g->m->ptrarg[1];
	g->m->ptrarg[0] = nil;
	g->m->ptrarg[1] = nil;
	mheap_free(h, s, g->m->scalararg[0]);
	runtime·gogo(&gp->sched);
}

void
runtime·MHeap_Free(MHeap *h, MSpan *s, int32 acct)
{
	void (*fn)(G*);

	if(g == g->m->g0) {
		mheap_free(h, s, acct);
	} else {
		g->m->ptrarg[0] = h;
		g->m->ptrarg[1] = s;
		g->m->scalararg[0] = acct;
		fn = mheap_free_m;
		runtime·mcall(&fn);
	}
}

void
runtime·MHeap_FreeStack(MHeap *h, MSpan *s)
{
	if(g != g->m->g0)
		runtime·throw("mheap_freestack not on M stack");
	s->needzero = 1;
	runtime·lock(&h->lock);
	mstats.stacks_inuse -= s->npages<<PageShift;
	MHeap_FreeSpanLocked(h, s, true, true);
	runtime·unlock(&h->lock);
}

static void
MHeap_FreeSpanLocked(MHeap *h, MSpan *s, bool acctinuse, bool acctidle)
{
	MSpan *t;
	pageID p;

	switch(s->state) {
	case MSpanStack:
		if(s->ref != 0)
			runtime·throw("MHeap_FreeSpanLocked - invalid stack free");
		break;
	case MSpanInUse:
		if(s->ref != 0 || s->sweepgen != h->sweepgen) {
			runtime·printf("MHeap_FreeSpanLocked - span %p ptr %p ref %d sweepgen %d/%d\n",
				       s, s->start<<PageShift, s->ref, s->sweepgen, h->sweepgen);
			runtime·throw("MHeap_FreeSpanLocked - invalid free");
		}
		break;
	default:
		runtime·throw("MHeap_FreeSpanLocked - invalid span state");
		break;
	}
	if(acctinuse)
		mstats.heap_inuse -= s->npages<<PageShift;
	if(acctidle)
		mstats.heap_idle += s->npages<<PageShift;
	s->state = MSpanFree;
	runtime·MSpanList_Remove(s);
	// Stamp newly unused spans. The scavenger will use that
	// info to potentially give back some pages to the OS.
	s->unusedsince = runtime·nanotime();
	s->npreleased = 0;

	// Coalesce with earlier, later spans.
	p = s->start;
	p -= (uintptr)h->arena_start >> PageShift;
	if(p > 0 && (t = h->spans[p-1]) != nil && t->state != MSpanInUse && t->state != MSpanStack) {
		s->start = t->start;
		s->npages += t->npages;
		s->npreleased = t->npreleased; // absorb released pages
		s->needzero |= t->needzero;
		p -= t->npages;
		h->spans[p] = s;
		runtime·MSpanList_Remove(t);
		t->state = MSpanDead;
		runtime·FixAlloc_Free(&h->spanalloc, t);
	}
	if((p+s->npages)*sizeof(h->spans[0]) < h->spans_mapped && (t = h->spans[p+s->npages]) != nil && t->state != MSpanInUse && t->state != MSpanStack) {
		s->npages += t->npages;
		s->npreleased += t->npreleased;
		s->needzero |= t->needzero;
		h->spans[p + s->npages - 1] = s;
		runtime·MSpanList_Remove(t);
		t->state = MSpanDead;
		runtime·FixAlloc_Free(&h->spanalloc, t);
	}

	// Insert s into appropriate list.
	if(s->npages < nelem(h->free))
		runtime·MSpanList_Insert(&h->free[s->npages], s);
	else
		runtime·MSpanList_Insert(&h->freelarge, s);
}

static uintptr
scavengelist(MSpan *list, uint64 now, uint64 limit)
{
	uintptr released, sumreleased;
	MSpan *s;

	if(runtime·MSpanList_IsEmpty(list))
		return 0;

	sumreleased = 0;
	for(s=list->next; s != list; s=s->next) {
		if((now - s->unusedsince) > limit && s->npreleased != s->npages) {
			released = (s->npages - s->npreleased) << PageShift;
			mstats.heap_released += released;
			sumreleased += released;
			s->npreleased = s->npages;
			runtime·SysUnused((void*)(s->start << PageShift), s->npages << PageShift);
		}
	}
	return sumreleased;
}

void
runtime·MHeap_Scavenge(int32 k, uint64 now, uint64 limit)
{
	uint32 i;
	uintptr sumreleased;
	MHeap *h;
	
	h = &runtime·mheap;
	runtime·lock(&h->lock);
	sumreleased = 0;
	for(i=0; i < nelem(h->free); i++)
		sumreleased += scavengelist(&h->free[i], now, limit);
	sumreleased += scavengelist(&h->freelarge, now, limit);
	runtime·unlock(&h->lock);

	if(runtime·debug.gctrace > 0) {
		if(sumreleased > 0)
			runtime·printf("scvg%d: %D MB released\n", k, (uint64)sumreleased>>20);
		// TODO(dvyukov): these stats are incorrect as we don't subtract stack usage from heap.
		// But we can't call ReadMemStats on g0 holding locks.
		runtime·printf("scvg%d: inuse: %D, idle: %D, sys: %D, released: %D, consumed: %D (MB)\n",
			k, mstats.heap_inuse>>20, mstats.heap_idle>>20, mstats.heap_sys>>20,
			mstats.heap_released>>20, (mstats.heap_sys - mstats.heap_released)>>20);
	}
}

void
runtime·scavenge_m(void)
{
	runtime·MHeap_Scavenge(-1, ~(uintptr)0, 0);
}

// Initialize a new span with the given start and npages.
void
runtime·MSpan_Init(MSpan *span, pageID start, uintptr npages)
{
	span->next = nil;
	span->prev = nil;
	span->start = start;
	span->npages = npages;
	span->freelist = nil;
	span->ref = 0;
	span->sizeclass = 0;
	span->incache = false;
	span->elemsize = 0;
	span->state = MSpanDead;
	span->unusedsince = 0;
	span->npreleased = 0;
	span->specialLock.key = 0;
	span->specials = nil;
	span->needzero = 0;
}

// Initialize an empty doubly-linked list.
void
runtime·MSpanList_Init(MSpan *list)
{
	list->state = MSpanListHead;
	list->next = list;
	list->prev = list;
}

void
runtime·MSpanList_Remove(MSpan *span)
{
	if(span->prev == nil && span->next == nil)
		return;
	span->prev->next = span->next;
	span->next->prev = span->prev;
	span->prev = nil;
	span->next = nil;
}

bool
runtime·MSpanList_IsEmpty(MSpan *list)
{
	return list->next == list;
}

void
runtime·MSpanList_Insert(MSpan *list, MSpan *span)
{
	if(span->next != nil || span->prev != nil) {
		runtime·printf("failed MSpanList_Insert %p %p %p\n", span, span->next, span->prev);
		runtime·throw("MSpanList_Insert");
	}
	span->next = list->next;
	span->prev = list;
	span->next->prev = span;
	span->prev->next = span;
}

void
runtime·MSpanList_InsertBack(MSpan *list, MSpan *span)
{
	if(span->next != nil || span->prev != nil) {
		runtime·printf("failed MSpanList_Insert %p %p %p\n", span, span->next, span->prev);
		runtime·throw("MSpanList_Insert");
	}
	span->next = list;
	span->prev = list->prev;
	span->next->prev = span;
	span->prev->next = span;
}

// Adds the special record s to the list of special records for
// the object p.  All fields of s should be filled in except for
// offset & next, which this routine will fill in.
// Returns true if the special was successfully added, false otherwise.
// (The add will fail only if a record with the same p and s->kind
//  already exists.)
static bool
addspecial(void *p, Special *s)
{
	MSpan *span;
	Special **t, *x;
	uintptr offset;
	byte kind;

	span = runtime·MHeap_LookupMaybe(&runtime·mheap, p);
	if(span == nil)
		runtime·throw("addspecial on invalid pointer");

	// Ensure that the span is swept.
	// GC accesses specials list w/o locks. And it's just much safer.
	g->m->locks++;
	runtime·MSpan_EnsureSwept(span);

	offset = (uintptr)p - (span->start << PageShift);
	kind = s->kind;

	runtime·lock(&span->specialLock);

	// Find splice point, check for existing record.
	t = &span->specials;
	while((x = *t) != nil) {
		if(offset == x->offset && kind == x->kind) {
			runtime·unlock(&span->specialLock);
			g->m->locks--;
			return false; // already exists
		}
		if(offset < x->offset || (offset == x->offset && kind < x->kind))
			break;
		t = &x->next;
	}
	// Splice in record, fill in offset.
	s->offset = offset;
	s->next = x;
	*t = s;
	runtime·unlock(&span->specialLock);
	g->m->locks--;
	return true;
}

// Removes the Special record of the given kind for the object p.
// Returns the record if the record existed, nil otherwise.
// The caller must FixAlloc_Free the result.
static Special*
removespecial(void *p, byte kind)
{
	MSpan *span;
	Special *s, **t;
	uintptr offset;

	span = runtime·MHeap_LookupMaybe(&runtime·mheap, p);
	if(span == nil)
		runtime·throw("removespecial on invalid pointer");

	// Ensure that the span is swept.
	// GC accesses specials list w/o locks. And it's just much safer.
	g->m->locks++;
	runtime·MSpan_EnsureSwept(span);

	offset = (uintptr)p - (span->start << PageShift);

	runtime·lock(&span->specialLock);
	t = &span->specials;
	while((s = *t) != nil) {
		// This function is used for finalizers only, so we don't check for
		// "interior" specials (p must be exactly equal to s->offset).
		if(offset == s->offset && kind == s->kind) {
			*t = s->next;
			runtime·unlock(&span->specialLock);
			g->m->locks--;
			return s;
		}
		t = &s->next;
	}
	runtime·unlock(&span->specialLock);
	g->m->locks--;
	return nil;
}

// Adds a finalizer to the object p.  Returns true if it succeeded.
bool
runtime·addfinalizer(void *p, FuncVal *f, uintptr nret, Type *fint, PtrType *ot)
{
	SpecialFinalizer *s;

	runtime·lock(&runtime·mheap.speciallock);
	s = runtime·FixAlloc_Alloc(&runtime·mheap.specialfinalizeralloc);
	runtime·unlock(&runtime·mheap.speciallock);
	s->special.kind = KindSpecialFinalizer;
	s->fn = f;
	s->nret = nret;
	s->fint = fint;
	s->ot = ot;
	if(addspecial(p, &s->special))
		return true;

	// There was an old finalizer
	runtime·lock(&runtime·mheap.speciallock);
	runtime·FixAlloc_Free(&runtime·mheap.specialfinalizeralloc, s);
	runtime·unlock(&runtime·mheap.speciallock);
	return false;
}

// Removes the finalizer (if any) from the object p.
void
runtime·removefinalizer(void *p)
{
	SpecialFinalizer *s;

	s = (SpecialFinalizer*)removespecial(p, KindSpecialFinalizer);
	if(s == nil)
		return; // there wasn't a finalizer to remove
	runtime·lock(&runtime·mheap.speciallock);
	runtime·FixAlloc_Free(&runtime·mheap.specialfinalizeralloc, s);
	runtime·unlock(&runtime·mheap.speciallock);
}

// Set the heap profile bucket associated with addr to b.
void
runtime·setprofilebucket_m(void)
{	
	void *p;
	Bucket *b;
	SpecialProfile *s;
	
	p = g->m->ptrarg[0];
	b = g->m->ptrarg[1];
	g->m->ptrarg[0] = nil;
	g->m->ptrarg[1] = nil;

	runtime·lock(&runtime·mheap.speciallock);
	s = runtime·FixAlloc_Alloc(&runtime·mheap.specialprofilealloc);
	runtime·unlock(&runtime·mheap.speciallock);
	s->special.kind = KindSpecialProfile;
	s->b = b;
	if(!addspecial(p, &s->special))
		runtime·throw("setprofilebucket: profile already set");
}

// Do whatever cleanup needs to be done to deallocate s.  It has
// already been unlinked from the MSpan specials list.
// Returns true if we should keep working on deallocating p.
bool
runtime·freespecial(Special *s, void *p, uintptr size, bool freed)
{
	SpecialFinalizer *sf;
	SpecialProfile *sp;

	switch(s->kind) {
	case KindSpecialFinalizer:
		sf = (SpecialFinalizer*)s;
		runtime·queuefinalizer(p, sf->fn, sf->nret, sf->fint, sf->ot);
		runtime·lock(&runtime·mheap.speciallock);
		runtime·FixAlloc_Free(&runtime·mheap.specialfinalizeralloc, sf);
		runtime·unlock(&runtime·mheap.speciallock);
		return false; // don't free p until finalizer is done
	case KindSpecialProfile:
		sp = (SpecialProfile*)s;
		runtime·mProf_Free(sp->b, size, freed);
		runtime·lock(&runtime·mheap.speciallock);
		runtime·FixAlloc_Free(&runtime·mheap.specialprofilealloc, sp);
		runtime·unlock(&runtime·mheap.speciallock);
		return true;
	default:
		runtime·throw("bad special kind");
		return true;
	}
}
