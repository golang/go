// Copyright 2009 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// Memory allocator, based on tcmalloc.
// http://goog-perftools.sourceforge.net/doc/tcmalloc.html

// The main allocator works in runs of pages.
// Small allocation sizes (up to and including 32 kB) are
// rounded to one of about 100 size classes, each of which
// has its own free list of objects of exactly that size.
// Any free page of memory can be split into a set of objects
// of one size class, which are then managed using free list
// allocators.
//
// The allocator's data structures are:
//
//	FixAlloc: a free-list allocator for fixed-size objects,
//		used to manage storage used by the allocator.
//	MHeap: the malloc heap, managed at page (4096-byte) granularity.
//	MSpan: a run of pages managed by the MHeap.
//	MHeapMap: a mapping from page IDs to MSpans.
//	MHeapMapCache: a small cache of MHeapMap mapping page IDs
//		to size classes for pages used for small objects.
//	MCentral: a shared free list for a given size class.
//	MCache: a per-thread (in Go, per-M) cache for small objects.
//	MStats: allocation statistics.
//
// Allocating a small object proceeds up a hierarchy of caches:
//
//	1. Round the size up to one of the small size classes
//	   and look in the corresponding MCache free list.
//	   If the list is not empty, allocate an object from it.
//	   This can all be done without acquiring a lock.
//
//	2. If the MCache free list is empty, replenish it by
//	   taking a bunch of objects from the MCentral free list.
//	   Moving a bunch amortizes the cost of acquiring the MCentral lock.
//
//	3. If the MCentral free list is empty, replenish it by
//	   allocating a run of pages from the MHeap and then
//	   chopping that memory into a objects of the given size.
//	   Allocating many objects amortizes the cost of locking
//	   the heap.
//
//	4. If the MHeap is empty or has no page runs large enough,
//	   allocate a new group of pages (at least 1MB) from the
//	   operating system.  Allocating a large run of pages
//	   amortizes the cost of talking to the operating system.
//
// Freeing a small object proceeds up the same hierarchy:
//
//	1. Look up the size class for the object and add it to
//	   the MCache free list.
//
//	2. If the MCache free list is too long or the MCache has
//	   too much memory, return some to the MCentral free lists.
//
//	3. If all the objects in a given span have returned to
//	   the MCentral list, return that span to the page heap.
//
//	4. If the heap has too much memory, return some to the
//	   operating system.
//
//	TODO(rsc): Step 4 is not implemented.
//
// Allocating and freeing a large object uses the page heap
// directly, bypassing the MCache and MCentral free lists.
//
// This C code was written with an eye toward translating to Go
// in the future.  Methods have the form Type_Method(Type *t, ...).


typedef struct FixAlloc	FixAlloc;
typedef struct MCentral	MCentral;
typedef struct MHeap	MHeap;
typedef struct MHeapMap	MHeapMap;
typedef struct MHeapMapCache	MHeapMapCache;
typedef struct MSpan	MSpan;
typedef struct MStats	MStats;
typedef struct MLink	MLink;

enum
{
	PageShift	= 12,
	PageSize	= 1<<PageShift,
	PageMask	= PageSize - 1,
};
typedef	uintptr	PageID;		// address >> PageShift

enum
{
	// Tunable constants.
	NumSizeClasses = 67,		// Number of size classes (must match msize.c)
	MaxSmallSize = 32<<10,

	FixAllocChunk = 128<<10,	// Chunk size for FixAlloc
	MaxMCacheListLen = 256,		// Maximum objects on MCacheList
	MaxMCacheSize = 2<<20,		// Maximum bytes in one MCache
	MaxMHeapList = 1<<(20 - PageShift),	// Maximum page length for fixed-size list in MHeap.
	HeapAllocChunk = 1<<20,		// Chunk size for heap growth
};


// A generic linked list of blocks.  (Typically the block is bigger than sizeof(MLink).)
struct MLink
{
	MLink *next;
};

// SysAlloc obtains a large chunk of memory from the operating system,
// typically on the order of a hundred kilobytes or a megabyte.
//
// SysUnused notifies the operating system that the contents
// of the memory region are no longer needed and can be reused
// for other purposes.  The program reserves the right to start
// accessing those pages in the future.
//
// SysFree returns it unconditionally; this is only used if
// an out-of-memory error has been detected midway through
// an allocation.  It is okay if SysFree is a no-op.

void*	SysAlloc(uintptr nbytes);
void	SysFree(void *v, uintptr nbytes);
void	SysUnused(void *v, uintptr nbytes);


// FixAlloc is a simple free-list allocator for fixed size objects.
// Malloc uses a FixAlloc wrapped around SysAlloc to manages its
// MCache and MSpan objects.
//
// Memory returned by FixAlloc_Alloc is not zeroed.
// The caller is responsible for locking around FixAlloc calls.
// Callers can keep state in the object but the first word is
// smashed by freeing and reallocating.
struct FixAlloc
{
	uintptr size;
	void *(*alloc)(uintptr);
	void (*first)(void *arg, byte *p);	// called first time p is returned
	void *arg;
	MLink *list;
	byte *chunk;
	uint32 nchunk;
};

void	FixAlloc_Init(FixAlloc *f, uintptr size, void *(*alloc)(uintptr), void (*first)(void*, byte*), void *arg);
void*	FixAlloc_Alloc(FixAlloc *f);
void	FixAlloc_Free(FixAlloc *f, void *p);


// Statistics.
// Shared with Go: if you edit this structure, also edit ../lib/malloc.go.
struct MStats
{
	uint64	alloc;
	uint64	sys;
	uint64	stacks;
	uint64	inuse_pages;	// protected by mheap.Lock
	uint64	next_gc;	// protected by mheap.Lock
	bool	enablegc;
};
extern MStats mstats;


// Size classes.  Computed and initialized by InitSizes.
//
// SizeToClass(0 <= n <= MaxSmallSize) returns the size class,
//	1 <= sizeclass < NumSizeClasses, for n.
//	Size class 0 is reserved to mean "not small".
//
// class_to_size[i] = largest size in class i
// class_to_allocnpages[i] = number of pages to allocate when
// 	making new objects in class i
// class_to_transfercount[i] = number of objects to move when
//	taking a bunch of objects out of the central lists
//	and putting them in the thread free list.

int32	SizeToClass(int32);
extern	int32	class_to_size[NumSizeClasses];
extern	int32	class_to_allocnpages[NumSizeClasses];
extern	int32	class_to_transfercount[NumSizeClasses];
extern	void	InitSizes(void);


// Per-thread (in Go, per-M) cache for small objects.
// No locking needed because it is per-thread (per-M).
typedef struct MCacheList MCacheList;
struct MCacheList
{
	MLink *list;
	uint32 nlist;
	uint32 nlistmin;
};

struct MCache
{
	MCacheList list[NumSizeClasses];
	uint64 size;
};

void*	MCache_Alloc(MCache *c, int32 sizeclass, uintptr size);
void	MCache_Free(MCache *c, void *p, int32 sizeclass, uintptr size);


// An MSpan is a run of pages.
enum
{
	MSpanInUse = 0,
	MSpanFree,
	MSpanListHead,
	MSpanDead,
};
struct MSpan
{
	MSpan	*next;		// in a span linked list
	MSpan	*prev;		// in a span linked list
	MSpan	*allnext;		// in the list of all spans
	PageID	start;		// starting page number
	uintptr	npages;		// number of pages in span
	MLink	*freelist;	// list of free objects
	uint32	ref;		// number of allocated objects in this span
	uint32	sizeclass;	// size class
	uint32	state;		// MSpanInUse etc
	union {
		uint32	*gcref;	// sizeclass > 0
		uint32	gcref0;	// sizeclass == 0
	};
};

void	MSpan_Init(MSpan *span, PageID start, uintptr npages);

// Every MSpan is in one doubly-linked list,
// either one of the MHeap's free lists or one of the
// MCentral's span lists.  We use empty MSpan structures as list heads.
void	MSpanList_Init(MSpan *list);
bool	MSpanList_IsEmpty(MSpan *list);
void	MSpanList_Insert(MSpan *list, MSpan *span);
void	MSpanList_Remove(MSpan *span);	// from whatever list it is in


// Central list of free objects of a given size.
struct MCentral
{
	Lock;
	int32 sizeclass;
	MSpan nonempty;
	MSpan empty;
	int32 nfree;
};

void	MCentral_Init(MCentral *c, int32 sizeclass);
int32	MCentral_AllocList(MCentral *c, int32 n, MLink **first);
void	MCentral_FreeList(MCentral *c, int32 n, MLink *first);


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


// Main malloc heap.
// The heap itself is the "free[]" and "large" arrays,
// but all the other global data is here too.
struct MHeap
{
	Lock;
	MSpan free[MaxMHeapList];	// free lists of given length
	MSpan large;			// free lists length >= MaxMHeapList
	MSpan *allspans;

	// span lookup
	MHeapMap map;
	MHeapMapCache mapcache;

	// central free lists for small size classes.
	// the union makes sure that the MCentrals are
	// spaced 64 bytes apart, so that each MCentral.Lock
	// gets its own cache line.
	union {
		MCentral;
		byte pad[64];
	} central[NumSizeClasses];

	FixAlloc spanalloc;	// allocator for Span*
	FixAlloc cachealloc;	// allocator for MCache*
};
extern MHeap mheap;

void	MHeap_Init(MHeap *h, void *(*allocator)(uintptr));
MSpan*	MHeap_Alloc(MHeap *h, uintptr npage, int32 sizeclass);
void	MHeap_Free(MHeap *h, MSpan *s);
MSpan*	MHeap_Lookup(MHeap *h, PageID p);
MSpan*	MHeap_LookupMaybe(MHeap *h, PageID p);

int32	mlookup(void *v, byte **base, uintptr *size, uint32 **ref);
void	gc(int32 force);

enum
{
	RefcountOverhead = 4,	// one uint32 per object

	RefFree = 0,	// must be zero
	RefManual,	// manual allocation - don't free
	RefStack,		// stack segment - don't free and don't scan for pointers
	RefNone,		// no references
	RefSome,		// some references
};

