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
// When a MSpan is allocated, state == MSpanInUse
// and heapmap(i) == span for all s->start <= i < s->start+s->npages.

#include "runtime.h"
#include "arch_GOARCH.h"
#include "malloc.h"

static MSpan *MHeap_AllocLocked(MHeap*, uintptr, int32);
static bool MHeap_Grow(MHeap*, uintptr);
static void MHeap_FreeLocked(MHeap*, MSpan*);
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
		all = (MSpan**)runtime·SysAlloc(cap*sizeof(all[0]));
		if(all == nil)
			runtime·throw("runtime: cannot allocate memory");
		if(h->allspans) {
			runtime·memmove(all, h->allspans, h->nspancap*sizeof(all[0]));
			runtime·SysFree(h->allspans, h->nspancap*sizeof(all[0]));
		}
		h->allspans = all;
		h->nspancap = cap;
	}
	h->allspans[h->nspan++] = s;
}

// Initialize the heap; fetch memory using alloc.
void
runtime·MHeap_Init(MHeap *h, void *(*alloc)(uintptr))
{
	uint32 i;

	runtime·FixAlloc_Init(&h->spanalloc, sizeof(MSpan), alloc, RecordSpan, h);
	runtime·FixAlloc_Init(&h->cachealloc, sizeof(MCache), alloc, nil, nil);
	// h->mapcache needs no init
	for(i=0; i<nelem(h->free); i++)
		runtime·MSpanList_Init(&h->free[i]);
	runtime·MSpanList_Init(&h->large);
	for(i=0; i<nelem(h->central); i++)
		runtime·MCentral_Init(&h->central[i], i);
}

// Allocate a new span of npage pages from the heap
// and record its size class in the HeapMap and HeapMapCache.
MSpan*
runtime·MHeap_Alloc(MHeap *h, uintptr npage, int32 sizeclass, int32 acct, int32 zeroed)
{
	MSpan *s;

	runtime·lock(h);
	runtime·purgecachedstats(m->mcache);
	s = MHeap_AllocLocked(h, npage, sizeclass);
	if(s != nil) {
		mstats.heap_inuse += npage<<PageShift;
		if(acct) {
			mstats.heap_objects++;
			mstats.heap_alloc += npage<<PageShift;
		}
	}
	runtime·unlock(h);
	if(s != nil && *(uintptr*)(s->start<<PageShift) != 0 && zeroed)
		runtime·memclr((byte*)(s->start<<PageShift), s->npages<<PageShift);
	return s;
}

static MSpan*
MHeap_AllocLocked(MHeap *h, uintptr npage, int32 sizeclass)
{
	uintptr n;
	MSpan *s, *t;
	PageID p;

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
	s->state = MSpanInUse;
	mstats.heap_idle -= s->npages<<PageShift;
	mstats.heap_released -= s->npreleased<<PageShift;
	if(s->npreleased > 0) {
		// We have called runtime·SysUnused with these pages, and on
		// Unix systems it called madvise.  At this point at least
		// some BSD-based kernels will return these pages either as
		// zeros or with the old data.  For our caller, the first word
		// in the page indicates whether the span contains zeros or
		// not (this word was set when the span was freed by
		// MCentral_Free or runtime·MCentral_FreeSpan).  If the first
		// page in the span is returned as zeros, and some subsequent
		// page is returned with the old data, then we will be
		// returning a span that is assumed to be all zeros, but the
		// actual data will not be all zeros.  Avoid that problem by
		// explicitly marking the span as not being zeroed, just in
		// case.  The beadbead constant we use here means nothing, it
		// is just a unique constant not seen elsewhere in the
		// runtime, as a clue in case it turns up unexpectedly in
		// memory or in a stack trace.
		*(uintptr*)(s->start<<PageShift) = (uintptr)0xbeadbeadbeadbeadULL;
	}
	s->npreleased = 0;

	if(s->npages > npage) {
		// Trim extra and put it back in the heap.
		t = runtime·FixAlloc_Alloc(&h->spanalloc);
		mstats.mspan_inuse = h->spanalloc.inuse;
		mstats.mspan_sys = h->spanalloc.sys;
		runtime·MSpan_Init(t, s->start + npage, s->npages - npage);
		s->npages = npage;
		p = t->start;
		if(sizeof(void*) == 8)
			p -= ((uintptr)h->arena_start>>PageShift);
		if(p > 0)
			h->map[p-1] = s;
		h->map[p] = t;
		h->map[p+t->npages-1] = t;
		*(uintptr*)(t->start<<PageShift) = *(uintptr*)(s->start<<PageShift);  // copy "needs zeroing" mark
		t->state = MSpanInUse;
		MHeap_FreeLocked(h, t);
		t->unusedsince = s->unusedsince; // preserve age
	}
	s->unusedsince = 0;

	// Record span info, because gc needs to be
	// able to map interior pointer to containing span.
	s->sizeclass = sizeclass;
	s->elemsize = (sizeclass==0 ? s->npages<<PageShift : runtime·class_to_size[sizeclass]);
	s->types.compression = MTypes_Empty;
	p = s->start;
	if(sizeof(void*) == 8)
		p -= ((uintptr)h->arena_start>>PageShift);
	for(n=0; n<npage; n++)
		h->map[p+n] = s;
	return s;
}

// Allocate a span of exactly npage pages from the list of large spans.
static MSpan*
MHeap_AllocLarge(MHeap *h, uintptr npage)
{
	return BestFit(&h->large, npage, nil);
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
	PageID p;

	// Ask for a big chunk, to reduce the number of mappings
	// the operating system needs to track; also amortizes
	// the overhead of an operating system mapping.
	// Allocate a multiple of 64kB (16 pages).
	npage = (npage+15)&~15;
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
	mstats.heap_sys += ask;

	// Create a fake "in use" span and free it, so that the
	// right coalescing happens.
	s = runtime·FixAlloc_Alloc(&h->spanalloc);
	mstats.mspan_inuse = h->spanalloc.inuse;
	mstats.mspan_sys = h->spanalloc.sys;
	runtime·MSpan_Init(s, (uintptr)v>>PageShift, ask>>PageShift);
	p = s->start;
	if(sizeof(void*) == 8)
		p -= ((uintptr)h->arena_start>>PageShift);
	h->map[p] = s;
	h->map[p + s->npages - 1] = s;
	s->state = MSpanInUse;
	MHeap_FreeLocked(h, s);
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
	if(sizeof(void*) == 8)
		p -= (uintptr)h->arena_start;
	return h->map[p >> PageShift];
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
	PageID p, q;

	if((byte*)v < h->arena_start || (byte*)v >= h->arena_used)
		return nil;
	p = (uintptr)v>>PageShift;
	q = p;
	if(sizeof(void*) == 8)
		q -= (uintptr)h->arena_start >> PageShift;
	s = h->map[q];
	if(s == nil || p < s->start || p - s->start >= s->npages)
		return nil;
	if(s->state != MSpanInUse)
		return nil;
	return s;
}

// Free the span back into the heap.
void
runtime·MHeap_Free(MHeap *h, MSpan *s, int32 acct)
{
	runtime·lock(h);
	runtime·purgecachedstats(m->mcache);
	mstats.heap_inuse -= s->npages<<PageShift;
	if(acct) {
		mstats.heap_alloc -= s->npages<<PageShift;
		mstats.heap_objects--;
	}
	MHeap_FreeLocked(h, s);
	runtime·unlock(h);
}

static void
MHeap_FreeLocked(MHeap *h, MSpan *s)
{
	uintptr *sp, *tp;
	MSpan *t;
	PageID p;

	if(s->types.sysalloc)
		runtime·settype_sysfree(s);
	s->types.compression = MTypes_Empty;

	if(s->state != MSpanInUse || s->ref != 0) {
		runtime·printf("MHeap_FreeLocked - span %p ptr %p state %d ref %d\n", s, s->start<<PageShift, s->state, s->ref);
		runtime·throw("MHeap_FreeLocked - invalid free");
	}
	mstats.heap_idle += s->npages<<PageShift;
	s->state = MSpanFree;
	runtime·MSpanList_Remove(s);
	sp = (uintptr*)(s->start<<PageShift);
	// Stamp newly unused spans. The scavenger will use that
	// info to potentially give back some pages to the OS.
	s->unusedsince = runtime·nanotime();
	s->npreleased = 0;

	// Coalesce with earlier, later spans.
	p = s->start;
	if(sizeof(void*) == 8)
		p -= (uintptr)h->arena_start >> PageShift;
	if(p > 0 && (t = h->map[p-1]) != nil && t->state != MSpanInUse) {
		tp = (uintptr*)(t->start<<PageShift);
		*tp |= *sp;	// propagate "needs zeroing" mark
		s->start = t->start;
		s->npages += t->npages;
		s->npreleased = t->npreleased; // absorb released pages
		p -= t->npages;
		h->map[p] = s;
		runtime·MSpanList_Remove(t);
		t->state = MSpanDead;
		runtime·FixAlloc_Free(&h->spanalloc, t);
		mstats.mspan_inuse = h->spanalloc.inuse;
		mstats.mspan_sys = h->spanalloc.sys;
	}
	if(p+s->npages < nelem(h->map) && (t = h->map[p+s->npages]) != nil && t->state != MSpanInUse) {
		tp = (uintptr*)(t->start<<PageShift);
		*sp |= *tp;	// propagate "needs zeroing" mark
		s->npages += t->npages;
		s->npreleased += t->npreleased;
		h->map[p + s->npages - 1] = s;
		runtime·MSpanList_Remove(t);
		t->state = MSpanDead;
		runtime·FixAlloc_Free(&h->spanalloc, t);
		mstats.mspan_inuse = h->spanalloc.inuse;
		mstats.mspan_sys = h->spanalloc.sys;
	}

	// Insert s into appropriate list.
	if(s->npages < nelem(h->free))
		runtime·MSpanList_Insert(&h->free[s->npages], s);
	else
		runtime·MSpanList_Insert(&h->large, s);
}

static void
forcegchelper(Note *note)
{
	runtime·gc(1);
	runtime·notewakeup(note);
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
		if((now - s->unusedsince) > limit) {
			released = (s->npages - s->npreleased) << PageShift;
			mstats.heap_released += released;
			sumreleased += released;
			s->npreleased = s->npages;
			runtime·SysUnused((void*)(s->start << PageShift), s->npages << PageShift);
		}
	}
	return sumreleased;
}

static uintptr
scavenge(uint64 now, uint64 limit)
{
	uint32 i;
	uintptr sumreleased;
	MHeap *h;
	
	h = runtime·mheap;
	sumreleased = 0;
	for(i=0; i < nelem(h->free); i++)
		sumreleased += scavengelist(&h->free[i], now, limit);
	sumreleased += scavengelist(&h->large, now, limit);
	return sumreleased;
}

static FuncVal forcegchelperv = {(void(*)(void))forcegchelper};

// Release (part of) unused memory to OS.
// Goroutine created at startup.
// Loop forever.
void
runtime·MHeap_Scavenger(void)
{
	MHeap *h;
	uint64 tick, now, forcegc, limit;
	uint32 k;
	uintptr sumreleased;
	byte *env;
	bool trace;
	Note note, *notep;

	g->issystem = true;
	g->isbackground = true;

	// If we go two minutes without a garbage collection, force one to run.
	forcegc = 2*60*1e9;
	// If a span goes unused for 5 minutes after a garbage collection,
	// we hand it back to the operating system.
	limit = 5*60*1e9;
	// Make wake-up period small enough for the sampling to be correct.
	if(forcegc < limit)
		tick = forcegc/2;
	else
		tick = limit/2;

	trace = false;
	env = runtime·getenv("GOGCTRACE");
	if(env != nil)
		trace = runtime·atoi(env) > 0;

	h = runtime·mheap;
	for(k=0;; k++) {
		runtime·noteclear(&note);
		runtime·entersyscallblock();
		runtime·notetsleep(&note, tick);
		runtime·exitsyscall();

		runtime·lock(h);
		now = runtime·nanotime();
		if(now - mstats.last_gc > forcegc) {
			runtime·unlock(h);
			// The scavenger can not block other goroutines,
			// otherwise deadlock detector can fire spuriously.
			// GC blocks other goroutines via the runtime·worldsema.
			runtime·noteclear(&note);
			notep = &note;
			runtime·newproc1(&forcegchelperv, (byte*)&notep, sizeof(notep), 0, runtime·MHeap_Scavenger);
			runtime·entersyscallblock();
			runtime·notesleep(&note);
			runtime·exitsyscall();
			if(trace)
				runtime·printf("scvg%d: GC forced\n", k);
			runtime·lock(h);
			now = runtime·nanotime();
		}
		sumreleased = scavenge(now, limit);
		runtime·unlock(h);

		if(trace) {
			if(sumreleased > 0)
				runtime·printf("scvg%d: %p MB released\n", k, sumreleased>>20);
			runtime·printf("scvg%d: inuse: %D, idle: %D, sys: %D, released: %D, consumed: %D (MB)\n",
				k, mstats.heap_inuse>>20, mstats.heap_idle>>20, mstats.heap_sys>>20,
				mstats.heap_released>>20, (mstats.heap_sys - mstats.heap_released)>>20);
		}
	}
}

void
runtime∕debug·freeOSMemory(void)
{
	runtime·gc(1);
	runtime·lock(runtime·mheap);
	scavenge(~(uintptr)0, 0);
	runtime·unlock(runtime·mheap);
}

// Initialize a new span with the given start and npages.
void
runtime·MSpan_Init(MSpan *span, PageID start, uintptr npages)
{
	span->next = nil;
	span->prev = nil;
	span->start = start;
	span->npages = npages;
	span->freelist = nil;
	span->ref = 0;
	span->sizeclass = 0;
	span->elemsize = 0;
	span->state = 0;
	span->unusedsince = 0;
	span->npreleased = 0;
	span->types.compression = MTypes_Empty;
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


