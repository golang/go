// Copyright 2009 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// Heap map, 64-bit version 
// See malloc.h and mheap.c for overview.

#include "runtime.h"
#include "malloc.h"

// 3-level radix tree mapping page ids to Span*.
void
MHeapMap_Init(MHeapMap *m, void *(*allocator)(uintptr))
{
	m->allocator = allocator;
}

MSpan*
MHeapMap_Get(MHeapMap *m, PageID k)
{
	int32 i1, i2, i3;

	i3 = k & MHeapMap_Level3Mask;
	k >>= MHeapMap_Level3Bits;
	i2 = k & MHeapMap_Level2Mask;
	k >>= MHeapMap_Level2Bits;
	i1 = k & MHeapMap_Level1Mask;
	k >>= MHeapMap_Level1Bits;
	if(k != 0)
		throw("MHeapMap_Get");

	return m->p[i1]->p[i2]->s[i3];
}

MSpan*
MHeapMap_GetMaybe(MHeapMap *m, PageID k)
{
	int32 i1, i2, i3;
	MHeapMapNode2 *p2;
	MHeapMapNode3 *p3;

	i3 = k & MHeapMap_Level3Mask;
	k >>= MHeapMap_Level3Bits;
	i2 = k & MHeapMap_Level2Mask;
	k >>= MHeapMap_Level2Bits;
	i1 = k & MHeapMap_Level1Mask;
	k >>= MHeapMap_Level1Bits;
	if(k != 0)
		throw("MHeapMap_Get");

	p2 = m->p[i1];
	if(p2 == nil)
		return nil;
	p3 = p2->p[i2];
	if(p3 == nil)
		return nil;
	return p3->s[i3];
}

void
MHeapMap_Set(MHeapMap *m, PageID k, MSpan *s)
{
	int32 i1, i2, i3;

	i3 = k & MHeapMap_Level3Mask;
	k >>= MHeapMap_Level3Bits;
	i2 = k & MHeapMap_Level2Mask;
	k >>= MHeapMap_Level2Bits;
	i1 = k & MHeapMap_Level1Mask;
	k >>= MHeapMap_Level1Bits;
	if(k != 0)
		throw("MHeapMap_Set");

	m->p[i1]->p[i2]->s[i3] = s;
}

// Allocate the storage required for entries [k, k+1, ..., k+len-1]
// so that Get and Set calls need not check for nil pointers.
bool
MHeapMap_Preallocate(MHeapMap *m, PageID k, uintptr len)
{
	uintptr end;
	int32 i1, i2;
	MHeapMapNode2 *p2;
	MHeapMapNode3 *p3;

	end = k+len;
	while(k < end) {
		if((k >> MHeapMap_TotalBits) != 0)
			return false;
		i2 = (k >> MHeapMap_Level3Bits) & MHeapMap_Level2Mask;
		i1 = (k >> (MHeapMap_Level3Bits + MHeapMap_Level2Bits)) & MHeapMap_Level1Mask;

		// first-level pointer
		if((p2 = m->p[i1]) == nil) {
			p2 = m->allocator(sizeof *p2);
			if(p2 == nil)
				return false;
			m->p[i1] = p2;
		}

		// second-level pointer
		if(p2->p[i2] == nil) {
			p3 = m->allocator(sizeof *p3);
			if(p3 == nil)
				return false;
			p2->p[i2] = p3;
		}

		// advance key past this leaf node
		k = ((k >> MHeapMap_Level3Bits) + 1) << MHeapMap_Level3Bits;
	}
	return true;
}

