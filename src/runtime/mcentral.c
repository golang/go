// Copyright 2009 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// Central free lists.
//
// See malloc.h for an overview.
//
// The MCentral doesn't actually contain the list of free objects; the MSpan does.
// Each MCentral is two lists of MSpans: those with free objects (c->nonempty)
// and those that are completely allocated (c->empty).

#include "runtime.h"
#include "arch_GOARCH.h"
#include "malloc.h"

static MSpan* MCentral_Grow(MCentral *c);

// Initialize a single central free list.
void
runtime·MCentral_Init(MCentral *c, int32 sizeclass)
{
	c->sizeclass = sizeclass;
	runtime·MSpanList_Init(&c->nonempty);
	runtime·MSpanList_Init(&c->empty);
}

// Allocate a span to use in an MCache.
MSpan*
runtime·MCentral_CacheSpan(MCentral *c)
{
	MSpan *s;
	int32 cap, n;
	uint32 sg;

	runtime·lock(&c->lock);
	sg = runtime·mheap.sweepgen;
retry:
	for(s = c->nonempty.next; s != &c->nonempty; s = s->next) {
		if(s->sweepgen == sg-2 && runtime·cas(&s->sweepgen, sg-2, sg-1)) {
			runtime·MSpanList_Remove(s);
			runtime·MSpanList_InsertBack(&c->empty, s);
			runtime·unlock(&c->lock);
			runtime·MSpan_Sweep(s, true);
			goto havespan;
		}
		if(s->sweepgen == sg-1) {
			// the span is being swept by background sweeper, skip
			continue;
		}
		// we have a nonempty span that does not require sweeping, allocate from it
		runtime·MSpanList_Remove(s);
		runtime·MSpanList_InsertBack(&c->empty, s);
		runtime·unlock(&c->lock);
		goto havespan;
	}

	for(s = c->empty.next; s != &c->empty; s = s->next) {
		if(s->sweepgen == sg-2 && runtime·cas(&s->sweepgen, sg-2, sg-1)) {
			// we have an empty span that requires sweeping,
			// sweep it and see if we can free some space in it
			runtime·MSpanList_Remove(s);
			// swept spans are at the end of the list
			runtime·MSpanList_InsertBack(&c->empty, s);
			runtime·unlock(&c->lock);
			runtime·MSpan_Sweep(s, true);
			if(s->freelist != nil)
				goto havespan;
			runtime·lock(&c->lock);
			// the span is still empty after sweep
			// it is already in the empty list, so just retry
			goto retry;
		}
		if(s->sweepgen == sg-1) {
			// the span is being swept by background sweeper, skip
			continue;
		}
		// already swept empty span,
		// all subsequent ones must also be either swept or in process of sweeping
		break;
	}
	runtime·unlock(&c->lock);

	// Replenish central list if empty.
	s = MCentral_Grow(c);
	if(s == nil)
		return nil;
	runtime·lock(&c->lock);
	runtime·MSpanList_InsertBack(&c->empty, s);
	runtime·unlock(&c->lock);

havespan:
	// At this point s is a non-empty span, queued at the end of the empty list,
	// c is unlocked.
	cap = (s->npages << PageShift) / s->elemsize;
	n = cap - s->ref;
	if(n == 0)
		runtime·throw("empty span");
	if(s->freelist == nil)
		runtime·throw("freelist empty");
	s->incache = true;
	return s;
}

// Return span from an MCache.
void
runtime·MCentral_UncacheSpan(MCentral *c, MSpan *s)
{
	int32 cap, n;

	runtime·lock(&c->lock);

	s->incache = false;

	if(s->ref == 0)
		runtime·throw("uncaching full span");

	cap = (s->npages << PageShift) / s->elemsize;
	n = cap - s->ref;
	if(n > 0) {
		runtime·MSpanList_Remove(s);
		runtime·MSpanList_Insert(&c->nonempty, s);
	}
	runtime·unlock(&c->lock);
}

// Free n objects from a span s back into the central free list c.
// Called during sweep.
// Returns true if the span was returned to heap.  Sets sweepgen to
// the latest generation.
// If preserve=true, don't return the span to heap nor relink in MCentral lists;
// caller takes care of it.
bool
runtime·MCentral_FreeSpan(MCentral *c, MSpan *s, int32 n, MLink *start, MLink *end, bool preserve)
{
	bool wasempty;

	if(s->incache)
		runtime·throw("freespan into cached span");

	// Add the objects back to s's free list.
	wasempty = s->freelist == nil;
	end->next = s->freelist;
	s->freelist = start;
	s->ref -= n;

	if(preserve) {
		// preserve is set only when called from MCentral_CacheSpan above,
		// the span must be in the empty list.
		if(s->next == nil)
			runtime·throw("can't preserve unlinked span");
		runtime·atomicstore(&s->sweepgen, runtime·mheap.sweepgen);
		return false;
	}

	runtime·lock(&c->lock);

	// Move to nonempty if necessary.
	if(wasempty) {
		runtime·MSpanList_Remove(s);
		runtime·MSpanList_Insert(&c->nonempty, s);
	}

	// delay updating sweepgen until here.  This is the signal that
	// the span may be used in an MCache, so it must come after the
	// linked list operations above (actually, just after the
	// lock of c above.)
	runtime·atomicstore(&s->sweepgen, runtime·mheap.sweepgen);

	if(s->ref != 0) {
		runtime·unlock(&c->lock);
		return false;
	}

	// s is completely freed, return it to the heap.
	runtime·MSpanList_Remove(s);
	s->needzero = 1;
	s->freelist = nil;
	runtime·unlock(&c->lock);
	runtime·unmarkspan((byte*)(s->start<<PageShift), s->npages<<PageShift);
	runtime·MHeap_Free(&runtime·mheap, s, 0);
	return true;
}

// Fetch a new span from the heap and carve into objects for the free list.
static MSpan*
MCentral_Grow(MCentral *c)
{
	uintptr size, npages, i, n;
	MLink **tailp, *v;
	byte *p;
	MSpan *s;

	npages = runtime·class_to_allocnpages[c->sizeclass];
	size = runtime·class_to_size[c->sizeclass];
	n = (npages << PageShift) / size;
	s = runtime·MHeap_Alloc(&runtime·mheap, npages, c->sizeclass, 0, 1);
	if(s == nil)
		return nil;

	// Carve span into sequence of blocks.
	tailp = &s->freelist;
	p = (byte*)(s->start << PageShift);
	s->limit = p + size*n;
	for(i=0; i<n; i++) {
		v = (MLink*)p;
		*tailp = v;
		tailp = &v->next;
		p += size;
	}
	*tailp = nil;
	runtime·markspan((byte*)(s->start<<PageShift), size, n, size*n < (s->npages<<PageShift));
	return s;
}
