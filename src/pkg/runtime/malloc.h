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
// The small objects on the MCache and MCentral free lists
// may or may not be zeroed.  They are zeroed if and only if
// the second word of the object is zero.  The spans in the
// page heap are always zeroed.  When a span full of objects
// is returned to the page heap, the objects that need to be
// are zeroed first.  There are two main benefits to delaying the
// zeroing this way:
//
//	1. stack frames allocated from the small object lists
//	   can avoid zeroing altogether.
//	2. the cost of zeroing when reusing a small object is
//	   charged to the mutator, not the garbage collector.
//
// This C code was written with an eye toward translating to Go
// in the future.  Methods have the form Type_Method(Type *t, ...).

typedef struct MCentral	MCentral;
typedef struct MHeap	MHeap;
typedef struct MSpan	MSpan;
typedef struct MStats	MStats;
typedef struct MLink	MLink;
typedef struct MTypes	MTypes;
typedef struct GCStats	GCStats;

enum
{
	PageShift	= 12,
	PageSize	= 1<<PageShift,
	PageMask	= PageSize - 1,
};
typedef	uintptr	PageID;		// address >> PageShift

enum
{
	// Computed constant.  The definition of MaxSmallSize and the
	// algorithm in msize.c produce some number of different allocation
	// size classes.  NumSizeClasses is that number.  It's needed here
	// because there are static arrays of this length; when msize runs its
	// size choosing algorithm it double-checks that NumSizeClasses agrees.
	NumSizeClasses = 61,

	// Tunable constants.
	MaxSmallSize = 32<<10,

	FixAllocChunk = 128<<10,	// Chunk size for FixAlloc
	MaxMCacheListLen = 256,		// Maximum objects on MCacheList
	MaxMCacheSize = 2<<20,		// Maximum bytes in one MCache
	MaxMHeapList = 1<<(20 - PageShift),	// Maximum page length for fixed-size list in MHeap.
	HeapAllocChunk = 1<<20,		// Chunk size for heap growth

	// Number of bits in page to span calculations (4k pages).
	// On Windows 64-bit we limit the arena to 32GB or 35 bits (see below for reason).
	// On other 64-bit platforms, we limit the arena to 128GB, or 37 bits.
	// On 32-bit, we don't bother limiting anything, so we use the full 32-bit address.
#ifdef _64BIT
#ifdef GOOS_windows
	// Windows counts memory used by page table into committed memory
	// of the process, so we can't reserve too much memory.
	// See http://golang.org/issue/5402 and http://golang.org/issue/5236.
	MHeapMap_Bits = 35 - PageShift,
#else
	MHeapMap_Bits = 37 - PageShift,
#endif
#else
	MHeapMap_Bits = 32 - PageShift,
#endif

	// Max number of threads to run garbage collection.
	// 2, 3, and 4 are all plausible maximums depending
	// on the hardware details of the machine.  The garbage
	// collector scales well to 8 cpus.
	MaxGcproc = 8,
};

// Maximum memory allocation size, a hint for callers.
// This must be a #define instead of an enum because it
// is so large.
#ifdef _64BIT
#define	MaxMem	(1ULL<<(MHeapMap_Bits+PageShift))	/* 128 GB or 32 GB */
#else
#define	MaxMem	((uintptr)-1)
#endif

// A generic linked list of blocks.  (Typically the block is bigger than sizeof(MLink).)
struct MLink
{
	MLink *next;
};

// SysAlloc obtains a large chunk of zeroed memory from the
// operating system, typically on the order of a hundred kilobytes
// or a megabyte.  If the pointer argument is non-nil, the caller
// wants a mapping there or nowhere.
//
// SysUnused notifies the operating system that the contents
// of the memory region are no longer needed and can be reused
// for other purposes.  The program reserves the right to start
// accessing those pages in the future.
//
// SysFree returns it unconditionally; this is only used if
// an out-of-memory error has been detected midway through
// an allocation.  It is okay if SysFree is a no-op.
//
// SysReserve reserves address space without allocating memory.
// If the pointer passed to it is non-nil, the caller wants the
// reservation there, but SysReserve can still choose another
// location if that one is unavailable.
//
// SysMap maps previously reserved address space for use.

void*	runtime·SysAlloc(uintptr nbytes);
void	runtime·SysFree(void *v, uintptr nbytes);
void	runtime·SysUnused(void *v, uintptr nbytes);
void	runtime·SysMap(void *v, uintptr nbytes);
void*	runtime·SysReserve(void *v, uintptr nbytes);

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
	uintptr inuse;	// in-use bytes now
	uintptr sys;	// bytes obtained from system
};

void	runtime·FixAlloc_Init(FixAlloc *f, uintptr size, void *(*alloc)(uintptr), void (*first)(void*, byte*), void *arg);
void*	runtime·FixAlloc_Alloc(FixAlloc *f);
void	runtime·FixAlloc_Free(FixAlloc *f, void *p);


// Statistics.
// Shared with Go: if you edit this structure, also edit type MemStats in mem.go.
struct MStats
{
	// General statistics.
	uint64	alloc;		// bytes allocated and still in use
	uint64	total_alloc;	// bytes allocated (even if freed)
	uint64	sys;		// bytes obtained from system (should be sum of xxx_sys below, no locking, approximate)
	uint64	nlookup;	// number of pointer lookups
	uint64	nmalloc;	// number of mallocs
	uint64	nfree;  // number of frees

	// Statistics about malloc heap.
	// protected by mheap.Lock
	uint64	heap_alloc;	// bytes allocated and still in use
	uint64	heap_sys;	// bytes obtained from system
	uint64	heap_idle;	// bytes in idle spans
	uint64	heap_inuse;	// bytes in non-idle spans
	uint64	heap_released;	// bytes released to the OS
	uint64	heap_objects;	// total number of allocated objects

	// Statistics about allocation of low-level fixed-size structures.
	// Protected by FixAlloc locks.
	uint64	stacks_inuse;	// bootstrap stacks
	uint64	stacks_sys;
	uint64	mspan_inuse;	// MSpan structures
	uint64	mspan_sys;
	uint64	mcache_inuse;	// MCache structures
	uint64	mcache_sys;
	uint64	buckhash_sys;	// profiling bucket hash table

	// Statistics about garbage collector.
	// Protected by mheap or stopping the world during GC.
	uint64	next_gc;	// next GC (in heap_alloc time)
	uint64  last_gc;	// last GC (in absolute time)
	uint64	pause_total_ns;
	uint64	pause_ns[256];
	uint32	numgc;
	bool	enablegc;
	bool	debuggc;

	// Statistics about allocation size classes.
	struct {
		uint32 size;
		uint64 nmalloc;
		uint64 nfree;
	} by_size[NumSizeClasses];
};

#define mstats runtime·memStats	/* name shared with Go */
extern MStats mstats;

// Size classes.  Computed and initialized by InitSizes.
//
// SizeToClass(0 <= n <= MaxSmallSize) returns the size class,
//	1 <= sizeclass < NumSizeClasses, for n.
//	Size class 0 is reserved to mean "not small".
//
// class_to_size[i] = largest size in class i
// class_to_allocnpages[i] = number of pages to allocate when
//	making new objects in class i
// class_to_transfercount[i] = number of objects to move when
//	taking a bunch of objects out of the central lists
//	and putting them in the thread free list.

int32	runtime·SizeToClass(int32);
extern	int32	runtime·class_to_size[NumSizeClasses];
extern	int32	runtime·class_to_allocnpages[NumSizeClasses];
extern	int32	runtime·class_to_transfercount[NumSizeClasses];
extern	void	runtime·InitSizes(void);


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
	uintptr size;
	intptr local_cachealloc;	// bytes allocated (or freed) from cache since last lock of heap
	intptr local_objects;	// objects allocated (or freed) from cache since last lock of heap
	intptr local_alloc;	// bytes allocated (or freed) since last lock of heap
	uintptr local_total_alloc;	// bytes allocated (even if freed) since last lock of heap
	uintptr local_nmalloc;	// number of mallocs since last lock of heap
	uintptr local_nfree;	// number of frees since last lock of heap
	uintptr local_nlookup;	// number of pointer lookups since last lock of heap
	int32 next_sample;	// trigger heap sample after allocating this many bytes
	// Statistics about allocation size classes since last lock of heap
	struct {
		uintptr nmalloc;
		uintptr nfree;
	} local_by_size[NumSizeClasses];

};

void*	runtime·MCache_Alloc(MCache *c, int32 sizeclass, uintptr size, int32 zeroed);
void	runtime·MCache_Free(MCache *c, void *p, int32 sizeclass, uintptr size);
void	runtime·MCache_ReleaseAll(MCache *c);

// MTypes describes the types of blocks allocated within a span.
// The compression field describes the layout of the data.
//
// MTypes_Empty:
//     All blocks are free, or no type information is available for
//     allocated blocks.
//     The data field has no meaning.
// MTypes_Single:
//     The span contains just one block.
//     The data field holds the type information.
//     The sysalloc field has no meaning.
// MTypes_Words:
//     The span contains multiple blocks.
//     The data field points to an array of type [NumBlocks]uintptr,
//     and each element of the array holds the type of the corresponding
//     block.
// MTypes_Bytes:
//     The span contains at most seven different types of blocks.
//     The data field points to the following structure:
//         struct {
//             type  [8]uintptr       // type[0] is always 0
//             index [NumBlocks]byte
//         }
//     The type of the i-th block is: data.type[data.index[i]]
enum
{
	MTypes_Empty = 0,
	MTypes_Single = 1,
	MTypes_Words = 2,
	MTypes_Bytes = 3,
};
struct MTypes
{
	byte	compression;	// one of MTypes_*
	bool	sysalloc;	// whether (void*)data is from runtime·SysAlloc
	uintptr	data;
};

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
	PageID	start;		// starting page number
	uintptr	npages;		// number of pages in span
	MLink	*freelist;	// list of free objects
	uint32	ref;		// number of allocated objects in this span
	int32	sizeclass;	// size class
	uintptr	elemsize;	// computed from sizeclass or from npages
	uint32	state;		// MSpanInUse etc
	int64   unusedsince;	// First time spotted by GC in MSpanFree state
	uintptr npreleased;	// number of pages released to the OS
	byte	*limit;		// end of data in span
	MTypes	types;		// types of allocated objects in this span
};

void	runtime·MSpan_Init(MSpan *span, PageID start, uintptr npages);

// Every MSpan is in one doubly-linked list,
// either one of the MHeap's free lists or one of the
// MCentral's span lists.  We use empty MSpan structures as list heads.
void	runtime·MSpanList_Init(MSpan *list);
bool	runtime·MSpanList_IsEmpty(MSpan *list);
void	runtime·MSpanList_Insert(MSpan *list, MSpan *span);
void	runtime·MSpanList_Remove(MSpan *span);	// from whatever list it is in


// Central list of free objects of a given size.
struct MCentral
{
	Lock;
	int32 sizeclass;
	MSpan nonempty;
	MSpan empty;
	int32 nfree;
};

void	runtime·MCentral_Init(MCentral *c, int32 sizeclass);
int32	runtime·MCentral_AllocList(MCentral *c, int32 n, MLink **first);
void	runtime·MCentral_FreeList(MCentral *c, int32 n, MLink *first);
void	runtime·MCentral_FreeSpan(MCentral *c, MSpan *s, int32 n, MLink *start, MLink *end);

// Main malloc heap.
// The heap itself is the "free[]" and "large" arrays,
// but all the other global data is here too.
struct MHeap
{
	Lock;
	MSpan free[MaxMHeapList];	// free lists of given length
	MSpan large;			// free lists length >= MaxMHeapList
	MSpan **allspans;
	uint32	nspan;
	uint32	nspancap;

	// span lookup
	MSpan *map[1<<MHeapMap_Bits];

	// range of addresses we might see in the heap
	byte *bitmap;
	uintptr bitmap_mapped;
	byte *arena_start;
	byte *arena_used;
	byte *arena_end;

	// central free lists for small size classes.
	// the padding makes sure that the MCentrals are
	// spaced CacheLineSize bytes apart, so that each MCentral.Lock
	// gets its own cache line.
	struct {
		MCentral;
		byte pad[CacheLineSize];
	} central[NumSizeClasses];

	FixAlloc spanalloc;	// allocator for Span*
	FixAlloc cachealloc;	// allocator for MCache*
};
extern MHeap *runtime·mheap;

void	runtime·MHeap_Init(MHeap *h, void *(*allocator)(uintptr));
MSpan*	runtime·MHeap_Alloc(MHeap *h, uintptr npage, int32 sizeclass, int32 acct, int32 zeroed);
void	runtime·MHeap_Free(MHeap *h, MSpan *s, int32 acct);
MSpan*	runtime·MHeap_Lookup(MHeap *h, void *v);
MSpan*	runtime·MHeap_LookupMaybe(MHeap *h, void *v);
void	runtime·MGetSizeClassInfo(int32 sizeclass, uintptr *size, int32 *npages, int32 *nobj);
void*	runtime·MHeap_SysAlloc(MHeap *h, uintptr n);
void	runtime·MHeap_MapBits(MHeap *h);
void	runtime·MHeap_Scavenger(void);

void*	runtime·mallocgc(uintptr size, uint32 flag, int32 dogc, int32 zeroed);
int32	runtime·mlookup(void *v, byte **base, uintptr *size, MSpan **s);
void	runtime·gc(int32 force);
void	runtime·markallocated(void *v, uintptr n, bool noptr);
void	runtime·checkallocated(void *v, uintptr n);
void	runtime·markfreed(void *v, uintptr n);
void	runtime·checkfreed(void *v, uintptr n);
extern	int32	runtime·checking;
void	runtime·markspan(void *v, uintptr size, uintptr n, bool leftover);
void	runtime·unmarkspan(void *v, uintptr size);
bool	runtime·blockspecial(void*);
void	runtime·setblockspecial(void*, bool);
void	runtime·purgecachedstats(MCache*);
void*	runtime·cnew(Type*);

void	runtime·settype(void*, uintptr);
void	runtime·settype_flush(M*, bool);
void	runtime·settype_sysfree(MSpan*);
uintptr	runtime·gettype(void*);

enum
{
	// flags to malloc
	FlagNoPointers = 1<<0,	// no pointers here
	FlagNoProfiling = 1<<1,	// must not profile
	FlagNoGC = 1<<2,	// must not free or scan for pointers
};

void	runtime·MProf_Malloc(void*, uintptr);
void	runtime·MProf_Free(void*, uintptr);
void	runtime·MProf_GC(void);
int32	runtime·gcprocs(void);
void	runtime·helpgc(int32 nproc);
void	runtime·gchelper(void);

bool	runtime·getfinalizer(void *p, bool del, FuncVal **fn, uintptr *nret);
void	runtime·walkfintab(void (*fn)(void*));

enum
{
	TypeInfo_SingleObject = 0,
	TypeInfo_Array = 1,
	TypeInfo_Map = 2,
	TypeInfo_Chan = 3,

	// Enables type information at the end of blocks allocated from heap	
	DebugTypeAtBlockEnd = 0,
};

// defined in mgc0.go
void	runtime·gc_m_ptr(Eface*);
void	runtime·gc_itab_ptr(Eface*);

void	runtime·memorydump(void);
