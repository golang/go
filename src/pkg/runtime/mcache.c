// Copyright 2009 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// Per-P malloc cache for small objects.
//
// See malloc.h for an overview.

#include "runtime.h"
#include "arch_GOARCH.h"
#include "malloc.h"

void
runtime·MCache_Refill(MCache *c, int32 sizeclass)
{
	MCacheList *l;

	// Replenish using central lists.
	l = &c->list[sizeclass];
	if(l->list)
		runtime·throw("MCache_Refill: the list is not empty");
	l->nlist = runtime·MCentral_AllocList(&runtime·mheap->central[sizeclass], &l->list);
	if(l->list == nil)
		runtime·throw("out of memory");
}

// Take n elements off l and return them to the central free list.
static void
ReleaseN(MCacheList *l, int32 n, int32 sizeclass)
{
	MLink *first, **lp;
	int32 i;

	// Cut off first n elements.
	first = l->list;
	lp = &l->list;
	for(i=0; i<n; i++)
		lp = &(*lp)->next;
	l->list = *lp;
	*lp = nil;
	l->nlist -= n;

	// Return them to central free list.
	runtime·MCentral_FreeList(&runtime·mheap->central[sizeclass], first);
}

void
runtime·MCache_Free(MCache *c, void *v, int32 sizeclass, uintptr size)
{
	MCacheList *l;
	MLink *p;

	// Put back on list.
	l = &c->list[sizeclass];
	p = v;
	p->next = l->list;
	l->list = p;
	l->nlist++;
	c->local_cachealloc -= size;
	c->local_objects--;

	// We transfer span at a time from MCentral to MCache,
	// if we have 2 times more than that, release a half back.
	if(l->nlist >= 2*(runtime·class_to_allocnpages[sizeclass]<<PageShift)/size)
		ReleaseN(l, l->nlist/2, sizeclass);
}

void
runtime·MCache_ReleaseAll(MCache *c)
{
	int32 i;
	MCacheList *l;

	for(i=0; i<NumSizeClasses; i++) {
		l = &c->list[i];
		if(l->list) {
			runtime·MCentral_FreeList(&runtime·mheap->central[i], l->list);
			l->list = nil;
			l->nlist = 0;
		}
	}
}
