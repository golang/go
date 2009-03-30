// Copyright 2009 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// Heap map, 32-bit version
// See malloc.h and mheap.c for overview.

#include "runtime.h"
#include "malloc.h"

// 3-level radix tree mapping page ids to Span*.
void
MHeapMap_Init(MHeapMap *m, void *(*allocator)(size_t))
{
	m->allocator = allocator;
}

MSpan*
MHeapMap_Get(MHeapMap *m, PageID k)
{
	int32 i1, i2;

	i2 = k & MHeapMap_Level2Mask;
	k >>= MHeapMap_Level2Bits;
	i1 = k & MHeapMap_Level1Mask;
	k >>= MHeapMap_Level1Bits;
	if(k != 0)
		throw("MHeapMap_Get");

	return m->p[i1]->s[i2];
}

MSpan*
MHeapMap_GetMaybe(MHeapMap *m, PageID k)
{
	int32 i1, i2;
	MHeapMapNode2 *p2;

	i2 = k & MHeapMap_Level2Mask;
	k >>= MHeapMap_Level2Bits;
	i1 = k & MHeapMap_Level1Mask;
	k >>= MHeapMap_Level1Bits;
	if(k != 0)
		throw("MHeapMap_Get");

	p2 = m->p[i1];
	if(p2 == nil)
		return nil;
	return p2->s[i2];
}

void
MHeapMap_Set(MHeapMap *m, PageID k, MSpan *s)
{
	int32 i1, i2;

	i2 = k & MHeapMap_Level2Mask;
	k >>= MHeapMap_Level2Bits;
	i1 = k & MHeapMap_Level1Mask;
	k >>= MHeapMap_Level1Bits;
	if(k != 0)
		throw("MHeapMap_Set");

	m->p[i1]->s[i2] = s;
}

// Allocate the storage required for entries [k, k+1, ..., k+len-1]
// so that Get and Set calls need not check for nil pointers.
bool
MHeapMap_Preallocate(MHeapMap *m, PageID k, uintptr len)
{
	uintptr end;
	int32 i1;
	MHeapMapNode2 *p2;

	end = k+len;
	while(k < end) {
		if((k >> MHeapMap_TotalBits) != 0)
			return false;
		i1 = (k >> MHeapMap_Level2Bits) & MHeapMap_Level1Mask;

		// first-level pointer
		if(m->p[i1] == nil) {
			p2 = m->allocator(sizeof *p2);
			if(p2 == nil)
				return false;
			sys_memclr((byte*)p2, sizeof *p2);
			m->p[i1] = p2;
		}

		// advance key past this leaf node
		k = ((k >> MHeapMap_Level2Bits) + 1) << MHeapMap_Level2Bits;
	}
	return true;
}

