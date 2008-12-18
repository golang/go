// Copyright 2009 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// Per-thread (in Go, per-M) malloc cache for small objects.
//
// See malloc.h for an overview.

#include "runtime.h"
#include "malloc.h"

void*
MCache_Alloc(MCache *c, int32 sizeclass, uintptr size)
{
	MCacheList *l;
	void *v, *start, *end;
	int32 n;

	// Allocate from list.
	l = &c->list[sizeclass];
	if(l->list == nil) {
		// Replenish using central lists.
		n = MCentral_AllocList(&mheap.central[sizeclass],
			class_to_transfercount[sizeclass], &start, &end);
		if(n == 0)
			return nil;
		l->list = start;
		l->nlist = n;
		c->size += n*size;
	}
	v = l->list;
	l->list = *(void**)v;
	l->nlist--;
	c->size -= size;

	// v is zeroed except for the link pointer
	// that we used above; zero that.
	*(void**)v = nil;
	return v;
}

void
MCache_Free(MCache *c, void *p, int32 sizeclass, uintptr size)
{
	MCacheList *l;

	// Put back on list.
	l = &c->list[sizeclass];
	*(void**)p = l->list;
	l->list = p;
	l->nlist++;
	c->size += size;

	if(l->nlist >= MaxMCacheListLen) {
		// TODO(rsc): Release to central cache.
	}
	if(c->size >= MaxMCacheSize) {
		// TODO(rsc): Scavenge.
	}
}

