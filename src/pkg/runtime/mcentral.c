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
//
// TODO(rsc): tcmalloc uses a "transfer cache" to split the list
// into sections of class_to_transfercount[sizeclass] objects
// so that it is faster to move those lists between MCaches and MCentrals.

#include "runtime.h"
#include "arch_GOARCH.h"
#include "malloc.h"

static bool MCentral_Grow(MCentral *c);

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
			runtime·unlock(&c->lock);
			runtime·MSpan_Sweep(s);
			runtime·lock(&c->lock);
			// the span could have been moved to heap, retry
			goto retry;
		}
		if(s->sweepgen == sg-1) {
			// the span is being swept by background sweeper, skip
			continue;
		}
		// we have a nonempty span that does not require sweeping, allocate from it
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
			runtime·MSpan_Sweep(s);
			runtime·lock(&c->lock);
			// the span could be moved to nonempty or heap, retry
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

	// Replenish central list if empty.
	if(!MCentral_Grow(c)) {
		runtime·unlock(&c->lock);
		return nil;
	}
	goto retry;

havespan:
	cap = (s->npages << PageShift) / s->elemsize;
	n = cap - s->ref;
	if(n == 0)
		runtime·throw("empty span");
	if(s->freelist == nil)
		runtime·throw("freelist empty");
	runtime·MSpanList_Remove(s);
	runtime·MSpanList_InsertBack(&c->empty, s);
	s->incache = true;
	runtime·unlock(&c->lock);
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
bool
runtime·MCentral_FreeSpan(MCentral *c, MSpan *s, int32 n, MLink *start, MLink *end)
{
	if(s->incache)
		runtime·throw("freespan into cached span");
	runtime·lock(&c->lock);

	// Move to nonempty if necessary.
	if(s->freelist == nil) {
		runtime·MSpanList_Remove(s);
		runtime·MSpanList_Insert(&c->nonempty, s);
	}

	// Add the objects back to s's free list.
	end->next = s->freelist;
	s->freelist = start;
	s->ref -= n;
	
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

// Fetch a new span from the heap and
// carve into objects for the free list.
static bool
MCentral_Grow(MCentral *c)
{
	uintptr size, npages, i, n;
	MLink **tailp, *v;
	byte *p;
	MSpan *s;

	runtime·unlock(&c->lock);
	npages = runtime·class_to_allocnpages[c->sizeclass];
	size = runtime·class_to_size[c->sizeclass];
	n = (npages << PageShift) / size;
	s = runtime·MHeap_Alloc(&runtime·mheap, npages, c->sizeclass, 0, 1);
	if(s == nil) {
		// TODO(rsc): Log out of memory
		runtime·lock(&c->lock);
		return false;
	}

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

	runtime·lock(&c->lock);
	runtime·MSpanList_Insert(&c->nonempty, s);
	return true;
}
