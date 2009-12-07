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
#include "malloc.h"

static bool MCentral_Grow(MCentral *c);
static void* MCentral_Alloc(MCentral *c);
static void MCentral_Free(MCentral *c, void *v);

// Initialize a single central free list.
void
MCentral_Init(MCentral *c, int32 sizeclass)
{
	c->sizeclass = sizeclass;
	MSpanList_Init(&c->nonempty);
	MSpanList_Init(&c->empty);
}

// Allocate up to n objects from the central free list.
// Return the number of objects allocated.
// The objects are linked together by their first words.
// On return, *pstart points at the first object and *pend at the last.
int32
MCentral_AllocList(MCentral *c, int32 n, MLink **pfirst)
{
	MLink *first, *last, *v;
	int32 i;

	lock(c);
	// Replenish central list if empty.
	if(MSpanList_IsEmpty(&c->nonempty)) {
		if(!MCentral_Grow(c)) {
			unlock(c);
			*pfirst = nil;
			return 0;
		}
	}

	// Copy from list, up to n.
	// First one is guaranteed to work, because we just grew the list.
	first = MCentral_Alloc(c);
	last = first;
	for(i=1; i<n && (v = MCentral_Alloc(c)) != nil; i++) {
		last->next = v;
		last = v;
	}
	last->next = nil;
	c->nfree -= i;

	unlock(c);
	*pfirst = first;
	return i;
}

// Helper: allocate one object from the central free list.
static void*
MCentral_Alloc(MCentral *c)
{
	MSpan *s;
	MLink *v;

	if(MSpanList_IsEmpty(&c->nonempty))
		return nil;
	s = c->nonempty.next;
	s->ref++;
	v = s->freelist;
	s->freelist = v->next;
	if(s->freelist == nil) {
		MSpanList_Remove(s);
		MSpanList_Insert(&c->empty, s);
	}
	return v;
}

// Free n objects back into the central free list.
// Return the number of objects allocated.
// The objects are linked together by their first words.
// On return, *pstart points at the first object and *pend at the last.
void
MCentral_FreeList(MCentral *c, int32 n, MLink *start)
{
	MLink *v, *next;

	// Assume next == nil marks end of list.
	// n and end would be useful if we implemented
	// the transfer cache optimization in the TODO above.
	USED(n);

	lock(c);
	for(v=start; v; v=next) {
		next = v->next;
		MCentral_Free(c, v);
	}
	unlock(c);
}

// Helper: free one object back into the central free list.
static void
MCentral_Free(MCentral *c, void *v)
{
	MSpan *s;
	PageID page;
	MLink *p, *next;

	// Find span for v.
	page = (uintptr)v >> PageShift;
	s = MHeap_Lookup(&mheap, page);
	if(s == nil || s->ref == 0)
		throw("invalid free");

	// Move to nonempty if necessary.
	if(s->freelist == nil) {
		MSpanList_Remove(s);
		MSpanList_Insert(&c->nonempty, s);
	}

	// Add v back to s's free list.
	p = v;
	p->next = s->freelist;
	s->freelist = p;
	c->nfree++;

	// If s is completely freed, return it to the heap.
	if(--s->ref == 0) {
		MSpanList_Remove(s);
		// Freed blocks are zeroed except for the link pointer.
		// Zero the link pointers so that the page is all zero.
		for(p=s->freelist; p; p=next) {
			next = p->next;
			p->next = nil;
		}
		s->freelist = nil;
		c->nfree -= (s->npages << PageShift) / class_to_size[c->sizeclass];
		unlock(c);
		MHeap_Free(&mheap, s);
		lock(c);
	}
}

// Fetch a new span from the heap and
// carve into objects for the free list.
static bool
MCentral_Grow(MCentral *c)
{
	int32 i, n, npages, size;
	MLink **tailp, *v;
	byte *p;
	MSpan *s;

	unlock(c);
	npages = class_to_allocnpages[c->sizeclass];
	s = MHeap_Alloc(&mheap, npages, c->sizeclass);
	if(s == nil) {
		// TODO(rsc): Log out of memory
		lock(c);
		return false;
	}

	// Carve span into sequence of blocks.
	tailp = &s->freelist;
	p = (byte*)(s->start << PageShift);
	size = class_to_size[c->sizeclass];
	n = (npages << PageShift) / (size + RefcountOverhead);
	s->gcref = (uint32*)(p + size*n);
	for(i=0; i<n; i++) {
		v = (MLink*)p;
		*tailp = v;
		tailp = &v->next;
		p += size;
	}
	*tailp = nil;

	lock(c);
	c->nfree += n;
	MSpanList_Insert(&c->nonempty, s);
	return true;
}
