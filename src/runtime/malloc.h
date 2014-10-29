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
//	MCache: a per-thread (in Go, per-P) cache for small objects.
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
// the second word of the object is zero.  A span in the
// page heap is zeroed unless s->needzero is set. When a span
// is allocated to break into small objects, it is zeroed if needed
// and s->needzero is set. There are two main benefits to delaying the
// zeroing this way:
//
//	1. stack frames allocated from the small object lists
//	   or the page heap can avoid zeroing altogether.
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
typedef struct GCStats	GCStats;

enum
{
	PageShift	= 13,
	PageSize	= 1<<PageShift,
	PageMask	= PageSize - 1,
};
typedef	uintptr	pageID;		// address >> PageShift

enum
{
	// Computed constant.  The definition of MaxSmallSize and the
	// algorithm in msize.c produce some number of different allocation
	// size classes.  NumSizeClasses is that number.  It's needed here
	// because there are static arrays of this length; when msize runs its
	// size choosing algorithm it double-checks that NumSizeClasses agrees.
	NumSizeClasses = 67,

	// Tunable constants.
	MaxSmallSize = 32<<10,

	// Tiny allocator parameters, see "Tiny allocator" comment in malloc.goc.
	TinySize = 16,
	TinySizeClass = 2,

	FixAllocChunk = 16<<10,		// Chunk size for FixAlloc
	MaxMHeapList = 1<<(20 - PageShift),	// Maximum page length for fixed-size list in MHeap.
	HeapAllocChunk = 1<<20,		// Chunk size for heap growth

	// Per-P, per order stack segment cache size.
	StackCacheSize = 32*1024,
	// Number of orders that get caching.  Order 0 is FixedStack
	// and each successive order is twice as large.
	NumStackOrders = 3,

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
	// collector scales well to 32 cpus.
	MaxGcproc = 32,
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

// sysAlloc obtains a large chunk of zeroed memory from the
// operating system, typically on the order of a hundred kilobytes
// or a megabyte.
// NOTE: sysAlloc returns OS-aligned memory, but the heap allocator
// may use larger alignment, so the caller must be careful to realign the
// memory obtained by sysAlloc.
//
// SysUnused notifies the operating system that the contents
// of the memory region are no longer needed and can be reused
// for other purposes.
// SysUsed notifies the operating system that the contents
// of the memory region are needed again.
//
// SysFree returns it unconditionally; this is only used if
// an out-of-memory error has been detected midway through
// an allocation.  It is okay if SysFree is a no-op.
//
// SysReserve reserves address space without allocating memory.
// If the pointer passed to it is non-nil, the caller wants the
// reservation there, but SysReserve can still choose another
// location if that one is unavailable.  On some systems and in some
// cases SysReserve will simply check that the address space is
// available and not actually reserve it.  If SysReserve returns
// non-nil, it sets *reserved to true if the address space is
// reserved, false if it has merely been checked.
// NOTE: SysReserve returns OS-aligned memory, but the heap allocator
// may use larger alignment, so the caller must be careful to realign the
// memory obtained by sysAlloc.
//
// SysMap maps previously reserved address space for use.
// The reserved argument is true if the address space was really
// reserved, not merely checked.
//
// SysFault marks a (already sysAlloc'd) region to fault
// if accessed.  Used only for debugging the runtime.

void*	runtime·sysAlloc(uintptr nbytes, uint64 *stat);
void	runtime·SysFree(void *v, uintptr nbytes, uint64 *stat);
void	runtime·SysUnused(void *v, uintptr nbytes);
void	runtime·SysUsed(void *v, uintptr nbytes);
void	runtime·SysMap(void *v, uintptr nbytes, bool reserved, uint64 *stat);
void*	runtime·SysReserve(void *v, uintptr nbytes, bool *reserved);
void	runtime·SysFault(void *v, uintptr nbytes);

// FixAlloc is a simple free-list allocator for fixed size objects.
// Malloc uses a FixAlloc wrapped around sysAlloc to manages its
// MCache and MSpan objects.
//
// Memory returned by FixAlloc_Alloc is not zeroed.
// The caller is responsible for locking around FixAlloc calls.
// Callers can keep state in the object but the first word is
// smashed by freeing and reallocating.
struct FixAlloc
{
	uintptr	size;
	void	(*first)(void *arg, byte *p);	// called first time p is returned
	void*	arg;
	MLink*	list;
	byte*	chunk;
	uint32	nchunk;
	uintptr	inuse;	// in-use bytes now
	uint64*	stat;
};

void	runtime·FixAlloc_Init(FixAlloc *f, uintptr size, void (*first)(void*, byte*), void *arg, uint64 *stat);
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
	// protected by mheap.lock
	uint64	heap_alloc;	// bytes allocated and still in use
	uint64	heap_sys;	// bytes obtained from system
	uint64	heap_idle;	// bytes in idle spans
	uint64	heap_inuse;	// bytes in non-idle spans
	uint64	heap_released;	// bytes released to the OS
	uint64	heap_objects;	// total number of allocated objects

	// Statistics about allocation of low-level fixed-size structures.
	// Protected by FixAlloc locks.
	uint64	stacks_inuse;	// this number is included in heap_inuse above
	uint64	stacks_sys;	// always 0 in mstats
	uint64	mspan_inuse;	// MSpan structures
	uint64	mspan_sys;
	uint64	mcache_inuse;	// MCache structures
	uint64	mcache_sys;
	uint64	buckhash_sys;	// profiling bucket hash table
	uint64	gc_sys;
	uint64	other_sys;

	// Statistics about garbage collector.
	// Protected by mheap or stopping the world during GC.
	uint64	next_gc;	// next GC (in heap_alloc time)
	uint64  last_gc;	// last GC (in absolute time)
	uint64	pause_total_ns;
	uint64	pause_ns[256];  // circular buffer of recent GC pause lengths
	uint64	pause_end[256]; // circular buffer of recent GC end times (nanoseconds since 1970)
	uint32	numgc;
	bool	enablegc;
	bool	debuggc;

	// Statistics about allocation size classes.
	
	struct MStatsBySize {
		uint32 size;
		uint64 nmalloc;
		uint64 nfree;
	} by_size[NumSizeClasses];
	
	uint64	tinyallocs;	// number of tiny allocations that didn't cause actual allocation; not exported to Go directly
};


#define mstats runtime·memstats
extern MStats mstats;
void	runtime·updatememstats(GCStats *stats);
void	runtime·ReadMemStats(MStats *stats);

// Size classes.  Computed and initialized by InitSizes.
//
// SizeToClass(0 <= n <= MaxSmallSize) returns the size class,
//	1 <= sizeclass < NumSizeClasses, for n.
//	Size class 0 is reserved to mean "not small".
//
// class_to_size[i] = largest size in class i
// class_to_allocnpages[i] = number of pages to allocate when
//	making new objects in class i

int32	runtime·SizeToClass(int32);
uintptr	runtime·roundupsize(uintptr);
extern	int32	runtime·class_to_size[NumSizeClasses];
extern	int32	runtime·class_to_allocnpages[NumSizeClasses];
extern	int8	runtime·size_to_class8[1024/8 + 1];
extern	int8	runtime·size_to_class128[(MaxSmallSize-1024)/128 + 1];
extern	void	runtime·InitSizes(void);

typedef struct MCacheList MCacheList;
struct MCacheList
{
	MLink *list;
	uint32 nlist;
};

typedef struct StackFreeList StackFreeList;
struct StackFreeList
{
	MLink *list;  // linked list of free stacks
	uintptr size; // total size of stacks in list
};

typedef struct SudoG SudoG;

// Per-thread (in Go, per-P) cache for small objects.
// No locking needed because it is per-thread (per-P).
struct MCache
{
	// The following members are accessed on every malloc,
	// so they are grouped here for better caching.
	int32 next_sample;		// trigger heap sample after allocating this many bytes
	intptr local_cachealloc;	// bytes allocated (or freed) from cache since last lock of heap
	// Allocator cache for tiny objects w/o pointers.
	// See "Tiny allocator" comment in malloc.goc.
	byte*	tiny;
	uintptr	tinysize;
	uintptr	local_tinyallocs;	// number of tiny allocs not counted in other stats
	// The rest is not accessed on every malloc.
	MSpan*	alloc[NumSizeClasses];	// spans to allocate from

	StackFreeList stackcache[NumStackOrders];

	SudoG*	sudogcache;

	void*	gcworkbuf;

	// Local allocator stats, flushed during GC.
	uintptr local_nlookup;		// number of pointer lookups
	uintptr local_largefree;	// bytes freed for large objects (>MaxSmallSize)
	uintptr local_nlargefree;	// number of frees for large objects (>MaxSmallSize)
	uintptr local_nsmallfree[NumSizeClasses];	// number of frees for small objects (<=MaxSmallSize)
};

MSpan*	runtime·MCache_Refill(MCache *c, int32 sizeclass);
void	runtime·MCache_ReleaseAll(MCache *c);
void	runtime·stackcache_clear(MCache *c);
void	runtime·gcworkbuffree(void *b);

enum
{
	KindSpecialFinalizer = 1,
	KindSpecialProfile = 2,
	// Note: The finalizer special must be first because if we're freeing
	// an object, a finalizer special will cause the freeing operation
	// to abort, and we want to keep the other special records around
	// if that happens.
};

typedef struct Special Special;
struct Special
{
	Special*	next;	// linked list in span
	uint16		offset;	// span offset of object
	byte		kind;	// kind of Special
};

// The described object has a finalizer set for it.
typedef struct SpecialFinalizer SpecialFinalizer;
struct SpecialFinalizer
{
	Special		special;
	FuncVal*	fn;
	uintptr		nret;
	Type*		fint;
	PtrType*	ot;
};

// The described object is being heap profiled.
typedef struct Bucket Bucket; // from mprof.h
typedef struct SpecialProfile SpecialProfile;
struct SpecialProfile
{
	Special	special;
	Bucket*	b;
};

// An MSpan is a run of pages.
enum
{
	MSpanInUse = 0, // allocated for garbage collected heap
	MSpanStack,     // allocated for use by stack allocator
	MSpanFree,
	MSpanListHead,
	MSpanDead,
};
struct MSpan
{
	MSpan	*next;		// in a span linked list
	MSpan	*prev;		// in a span linked list
	pageID	start;		// starting page number
	uintptr	npages;		// number of pages in span
	MLink	*freelist;	// list of free objects
	// sweep generation:
	// if sweepgen == h->sweepgen - 2, the span needs sweeping
	// if sweepgen == h->sweepgen - 1, the span is currently being swept
	// if sweepgen == h->sweepgen, the span is swept and ready to use
	// h->sweepgen is incremented by 2 after every GC
	uint32	sweepgen;
	uint16	ref;		// capacity - number of objects in freelist
	uint8	sizeclass;	// size class
	bool	incache;	// being used by an MCache
	uint8	state;		// MSpanInUse etc
	uint8	needzero;	// needs to be zeroed before allocation
	uintptr	elemsize;	// computed from sizeclass or from npages
	int64   unusedsince;	// First time spotted by GC in MSpanFree state
	uintptr npreleased;	// number of pages released to the OS
	byte	*limit;		// end of data in span
	Mutex	specialLock;	// guards specials list
	Special	*specials;	// linked list of special records sorted by offset.
};

void	runtime·MSpan_Init(MSpan *span, pageID start, uintptr npages);
void	runtime·MSpan_EnsureSwept(MSpan *span);
bool	runtime·MSpan_Sweep(MSpan *span, bool preserve);

// Every MSpan is in one doubly-linked list,
// either one of the MHeap's free lists or one of the
// MCentral's span lists.  We use empty MSpan structures as list heads.
void	runtime·MSpanList_Init(MSpan *list);
bool	runtime·MSpanList_IsEmpty(MSpan *list);
void	runtime·MSpanList_Insert(MSpan *list, MSpan *span);
void	runtime·MSpanList_InsertBack(MSpan *list, MSpan *span);
void	runtime·MSpanList_Remove(MSpan *span);	// from whatever list it is in


// Central list of free objects of a given size.
struct MCentral
{
	Mutex  lock;
	int32 sizeclass;
	MSpan nonempty;	// list of spans with a free object
	MSpan empty;	// list of spans with no free objects (or cached in an MCache)
};

void	runtime·MCentral_Init(MCentral *c, int32 sizeclass);
MSpan*	runtime·MCentral_CacheSpan(MCentral *c);
void	runtime·MCentral_UncacheSpan(MCentral *c, MSpan *s);
bool	runtime·MCentral_FreeSpan(MCentral *c, MSpan *s, int32 n, MLink *start, MLink *end, bool preserve);

// Main malloc heap.
// The heap itself is the "free[]" and "large" arrays,
// but all the other global data is here too.
struct MHeap
{
	Mutex  lock;
	MSpan free[MaxMHeapList];	// free lists of given length
	MSpan freelarge;		// free lists length >= MaxMHeapList
	MSpan busy[MaxMHeapList];	// busy lists of large objects of given length
	MSpan busylarge;		// busy lists of large objects length >= MaxMHeapList
	MSpan **allspans;		// all spans out there
	MSpan **gcspans;		// copy of allspans referenced by GC marker or sweeper
	uint32	nspan;
	uint32	nspancap;
	uint32	sweepgen;		// sweep generation, see comment in MSpan
	uint32	sweepdone;		// all spans are swept

	// span lookup
	MSpan**	spans;
	uintptr	spans_mapped;

	// range of addresses we might see in the heap
	byte *bitmap;
	uintptr bitmap_mapped;
	byte *arena_start;
	byte *arena_used;
	byte *arena_end;
	bool arena_reserved;

	// central free lists for small size classes.
	// the padding makes sure that the MCentrals are
	// spaced CacheLineSize bytes apart, so that each MCentral.lock
	// gets its own cache line.
	struct MHeapCentral {
		MCentral mcentral;
		byte pad[CacheLineSize];
	} central[NumSizeClasses];

	FixAlloc spanalloc;	// allocator for Span*
	FixAlloc cachealloc;	// allocator for MCache*
	FixAlloc specialfinalizeralloc;	// allocator for SpecialFinalizer*
	FixAlloc specialprofilealloc;	// allocator for SpecialProfile*
	Mutex speciallock; // lock for sepcial record allocators.

	// Malloc stats.
	uint64 largefree;	// bytes freed for large objects (>MaxSmallSize)
	uint64 nlargefree;	// number of frees for large objects (>MaxSmallSize)
	uint64 nsmallfree[NumSizeClasses];	// number of frees for small objects (<=MaxSmallSize)
};
#define runtime·mheap runtime·mheap_
extern MHeap runtime·mheap;

void	runtime·MHeap_Init(MHeap *h);
MSpan*	runtime·MHeap_Alloc(MHeap *h, uintptr npage, int32 sizeclass, bool large, bool needzero);
MSpan*	runtime·MHeap_AllocStack(MHeap *h, uintptr npage);
void	runtime·MHeap_Free(MHeap *h, MSpan *s, int32 acct);
void	runtime·MHeap_FreeStack(MHeap *h, MSpan *s);
MSpan*	runtime·MHeap_Lookup(MHeap *h, void *v);
MSpan*	runtime·MHeap_LookupMaybe(MHeap *h, void *v);
void*	runtime·MHeap_SysAlloc(MHeap *h, uintptr n);
void	runtime·MHeap_MapBits(MHeap *h);
void	runtime·MHeap_MapSpans(MHeap *h);
void	runtime·MHeap_Scavenge(int32 k, uint64 now, uint64 limit);

void*	runtime·persistentalloc(uintptr size, uintptr align, uint64 *stat);
int32	runtime·mlookup(void *v, byte **base, uintptr *size, MSpan **s);
uintptr	runtime·sweepone(void);
void	runtime·markspan(void *v, uintptr size, uintptr n, bool leftover);
void	runtime·unmarkspan(void *v, uintptr size);
void	runtime·purgecachedstats(MCache*);
void	runtime·tracealloc(void*, uintptr, Type*);
void	runtime·tracefree(void*, uintptr);
void	runtime·tracegc(void);

int32	runtime·gcpercent;
int32	runtime·readgogc(void);
void	runtime·clearpools(void);

enum
{
	// flags to malloc
	FlagNoScan	= 1<<0,	// GC doesn't have to scan object
	FlagNoZero	= 1<<1, // don't zero memory
};

void	runtime·mProf_Malloc(void*, uintptr);
void	runtime·mProf_Free(Bucket*, uintptr, bool);
void	runtime·mProf_GC(void);
void	runtime·iterate_memprof(void (**callback)(Bucket*, uintptr, uintptr*, uintptr, uintptr, uintptr));
int32	runtime·gcprocs(void);
void	runtime·helpgc(int32 nproc);
void	runtime·gchelper(void);
void	runtime·createfing(void);
G*	runtime·wakefing(void);
void	runtime·getgcmask(byte*, Type*, byte**, uintptr*);

// NOTE: Layout known to queuefinalizer.
typedef struct Finalizer Finalizer;
struct Finalizer
{
	FuncVal *fn;	// function to call
	void *arg;	// ptr to object
	uintptr nret;	// bytes of return values from fn
	Type *fint;	// type of first argument of fn
	PtrType *ot;	// type of ptr to object
};

typedef struct FinBlock FinBlock;
struct FinBlock
{
	FinBlock *alllink;
	FinBlock *next;
	int32 cnt;
	int32 cap;
	Finalizer fin[1];
};
extern Mutex	runtime·finlock;	// protects the following variables
extern G*	runtime·fing;
extern bool	runtime·fingwait;
extern bool	runtime·fingwake;
extern FinBlock	*runtime·finq;		// list of finalizers that are to be executed
extern FinBlock	*runtime·finc;		// cache of free blocks

void	runtime·setprofilebucket_m(void);

bool	runtime·addfinalizer(void*, FuncVal *fn, uintptr, Type*, PtrType*);
void	runtime·removefinalizer(void*);
void	runtime·queuefinalizer(byte *p, FuncVal *fn, uintptr nret, Type *fint, PtrType *ot);
bool	runtime·freespecial(Special *s, void *p, uintptr size, bool freed);

// Information from the compiler about the layout of stack frames.
struct BitVector
{
	int32 n; // # of bits
	uint8 *bytedata;
};
typedef struct StackMap StackMap;
struct StackMap
{
	int32 n; // number of bitmaps
	int32 nbit; // number of bits in each bitmap
	uint8 bytedata[]; // bitmaps, each starting on a 32-bit boundary
};
// Returns pointer map data for the given stackmap index
// (the index is encoded in PCDATA_StackMapIndex).
BitVector	runtime·stackmapdata(StackMap *stackmap, int32 n);

extern	BitVector	runtime·gcdatamask;
extern	BitVector	runtime·gcbssmask;

// defined in mgc0.go
void	runtime·gc_m_ptr(Eface*);
void	runtime·gc_g_ptr(Eface*);
void	runtime·gc_itab_ptr(Eface*);

void  runtime·setgcpercent_m(void);

// Value we use to mark dead pointers when GODEBUG=gcdead=1.
#define PoisonGC ((uintptr)0xf969696969696969ULL)
#define PoisonStack ((uintptr)0x6868686868686868ULL)
