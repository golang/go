// Copyright 2009 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// Per-thread (in Go, per-M) malloc cache for small objects.
//
// See malloc.h for an overview.

#include "runtime.h"
#include "arch_GOARCH.h"
#include "malloc.h"

void*
runtime·MCache_Alloc(MCache *c, int32 sizeclass, uintptr size, int32 zeroed)
{
	MCacheList *l;
	MLink *first, *v;
	int32 n;

	// Allocate from list.
	l = &c->list[sizeclass];
	if(l->list == nil) {
		// Replenish using central lists.
		n = runtime·MCentral_AllocList(&runtime·mheap->central[sizeclass],
			runtime·class_to_transfercount[sizeclass], &first);
		if(n == 0)
			runtime·throw("out of memory");
		l->list = first;
		l->nlist = n;
		c->size += n*size;
	}
	v = l->list;
	l->list = v->next;
	l->nlist--;
	if(l->nlist < l->nlistmin)
		l->nlistmin = l->nlist;
	c->size -= size;

	// v is zeroed except for the link pointer
	// that we used above; zero that.
	v->next = nil;
	if(zeroed) {
		// block is zeroed iff second word is zero ...
		if(size > sizeof(uintptr) && ((uintptr*)v)[1] != 0)
			runtime·memclr((byte*)v, size);
	}
	c->local_cachealloc += size;
	c->local_objects++;
	return v;
}

// Take n elements off l and return them to the central free list.
static void
ReleaseN(MCache *c, MCacheList *l, int32 n, int32 sizeclass)
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
	if(l->nlist < l->nlistmin)
		l->nlistmin = l->nlist;
	c->size -= n*runtime·class_to_size[sizeclass];

	// Return them to central free list.
	runtime·MCentral_FreeList(&runtime·mheap->central[sizeclass], n, first);
}

void
runtime·MCache_Free(MCache *c, void *v, int32 sizeclass, uintptr size)
{
	int32 i, n;
	MCacheList *l;
	MLink *p;

	// Put back on list.
	l = &c->list[sizeclass];
	p = v;
	p->next = l->list;
	l->list = p;
	l->nlist++;
	c->size += size;
	c->local_cachealloc -= size;
	c->local_objects--;

	if(l->nlist >= MaxMCacheListLen) {
		// Release a chunk back.
		ReleaseN(c, l, runtime·class_to_transfercount[sizeclass], sizeclass);
	}

	if(c->size >= MaxMCacheSize) {
		// Scavenge.
		for(i=0; i<NumSizeClasses; i++) {
			l = &c->list[i];
			n = l->nlistmin;

			// n is the minimum number of elements we've seen on
			// the list since the last scavenge.  If n > 0, it means that
			// we could have gotten by with n fewer elements
			// without needing to consult the central free list.
			// Move toward that situation by releasing n/2 of them.
			if(n > 0) {
				if(n > 1)
					n /= 2;
				ReleaseN(c, l, n, i);
			}
			l->nlistmin = l->nlist;
		}
	}
}

void
runtime·MCache_ReleaseAll(MCache *c)
{
	int32 i;
	MCacheList *l;

	for(i=0; i<NumSizeClasses; i++) {
		l = &c->list[i];
		ReleaseN(c, l, l->nlist, i);
		l->nlistmin = 0;
	}
}
