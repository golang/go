// Copyright 2009 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// Free(v) must be able to determine the MSpan containing v.
// The MHeapMap is a 2-level radix tree mapping page numbers to MSpans.

typedef struct MHeapMapNode2 MHeapMapNode2;

enum
{
	// 32 bit address - 12 bit page size = 20 bits to map
	MHeapMap_Level1Bits = 10,
	MHeapMap_Level2Bits = 10,

	MHeapMap_TotalBits =
		MHeapMap_Level1Bits +
		MHeapMap_Level2Bits,

	MHeapMap_Level1Mask = (1<<MHeapMap_Level1Bits) - 1,
	MHeapMap_Level2Mask = (1<<MHeapMap_Level2Bits) - 1,
};

struct MHeapMap
{
	void *(*allocator)(uintptr);
	MHeapMapNode2 *p[1<<MHeapMap_Level1Bits];
};

struct MHeapMapNode2
{
	MSpan *s[1<<MHeapMap_Level2Bits];
};

void	MHeapMap_Init(MHeapMap *m, void *(*allocator)(uintptr));
bool	MHeapMap_Preallocate(MHeapMap *m, PageID k, uintptr npages);
MSpan*	MHeapMap_Get(MHeapMap *m, PageID k);
MSpan*	MHeapMap_GetMaybe(MHeapMap *m, PageID k);
void	MHeapMap_Set(MHeapMap *m, PageID k, MSpan *v);


