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


// Much of the time, free(v) needs to know only the size class for v,
// not which span it came from.  The MHeapMap finds the size class
// by looking up the span.
//
// An MHeapMapCache is a simple direct-mapped cache translating
// page numbers to size classes.  It avoids the expensive MHeapMap
// lookup for hot pages.
//
// The cache entries are 64 bits, with the page number in the low part
// and the value at the top.
//
// NOTE(rsc): On a machine with 32-bit addresses (= 20-bit page numbers),
// we can use a 16-bit cache entry by not storing the redundant 12 bits
// of the key that are used as the entry index.  Here in 64-bit land,
// that trick won't work unless the hash table has 2^28 entries.
enum
{
	MHeapMapCache_HashBits = 12
};

struct MHeapMapCache
{
	uintptr array[1<<MHeapMapCache_HashBits];
};

// All macros for speed (sorry).
#define HMASK	((1<<MHeapMapCache_HashBits)-1)
#define KBITS	MHeapMap_TotalBits
#define KMASK	((1LL<<KBITS)-1)

#define MHeapMapCache_SET(cache, key, value) \
	((cache)->array[(key) & HMASK] = (key) | ((uintptr)(value) << KBITS))

#define MHeapMapCache_GET(cache, key, tmp) \
	(tmp = (cache)->array[(key) & HMASK], \
	 (tmp & KMASK) == (key) ? (tmp >> KBITS) : 0)
