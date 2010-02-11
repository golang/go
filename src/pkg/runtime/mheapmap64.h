// Copyright 2009 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// Free(v) must be able to determine the MSpan containing v.
// The MHeapMap is a 3-level radix tree mapping page numbers to MSpans.
//
// NOTE(rsc): On a 32-bit platform (= 20-bit page numbers),
// we can swap in a 2-level radix tree.
//
// NOTE(rsc): We use a 3-level tree because tcmalloc does, but
// having only three levels requires approximately 1 MB per node
// in the tree, making the minimum map footprint 3 MB.
// Using a 4-level tree would cut the minimum footprint to 256 kB.
// On the other hand, it's just virtual address space: most of
// the memory is never going to be touched, thus never paged in.

typedef struct MHeapMapNode2 MHeapMapNode2;
typedef struct MHeapMapNode3 MHeapMapNode3;

enum
{
	// 64 bit address - 12 bit page size = 52 bits to map
	MHeapMap_Level1Bits = 18,
	MHeapMap_Level2Bits = 18,
	MHeapMap_Level3Bits = 16,

	MHeapMap_TotalBits =
		MHeapMap_Level1Bits +
		MHeapMap_Level2Bits +
		MHeapMap_Level3Bits,

	MHeapMap_Level1Mask = (1<<MHeapMap_Level1Bits) - 1,
	MHeapMap_Level2Mask = (1<<MHeapMap_Level2Bits) - 1,
	MHeapMap_Level3Mask = (1<<MHeapMap_Level3Bits) - 1,
};

struct MHeapMap
{
	void *(*allocator)(uintptr);
	MHeapMapNode2 *p[1<<MHeapMap_Level1Bits];
};

struct MHeapMapNode2
{
	MHeapMapNode3 *p[1<<MHeapMap_Level2Bits];
};

struct MHeapMapNode3
{
	MSpan *s[1<<MHeapMap_Level3Bits];
};

void	MHeapMap_Init(MHeapMap *m, void *(*allocator)(uintptr));
bool	MHeapMap_Preallocate(MHeapMap *m, PageID k, uintptr npages);
MSpan*	MHeapMap_Get(MHeapMap *m, PageID k);
MSpan*	MHeapMap_GetMaybe(MHeapMap *m, PageID k);
void	MHeapMap_Set(MHeapMap *m, PageID k, MSpan *v);


