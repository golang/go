// Copyright 2009 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package runtime

import "unsafe"

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

const (
	_PageShift = 13
	_PageSize  = 1 << _PageShift
	_PageMask  = _PageSize - 1
)

const (
	// _64bit = 1 on 64-bit systems, 0 on 32-bit systems
	_64bit = 1 << (^uintptr(0) >> 63) / 2

	// Computed constant.  The definition of MaxSmallSize and the
	// algorithm in msize.c produce some number of different allocation
	// size classes.  NumSizeClasses is that number.  It's needed here
	// because there are static arrays of this length; when msize runs its
	// size choosing algorithm it double-checks that NumSizeClasses agrees.
	_NumSizeClasses = 67

	// Tunable constants.
	_MaxSmallSize = 32 << 10

	// Tiny allocator parameters, see "Tiny allocator" comment in malloc.goc.
	_TinySize      = 16
	_TinySizeClass = 2

	_FixAllocChunk  = 16 << 10               // Chunk size for FixAlloc
	_MaxMHeapList   = 1 << (20 - _PageShift) // Maximum page length for fixed-size list in MHeap.
	_HeapAllocChunk = 1 << 20                // Chunk size for heap growth

	// Per-P, per order stack segment cache size.
	_StackCacheSize = 32 * 1024

	// Number of orders that get caching.  Order 0 is FixedStack
	// and each successive order is twice as large.
	_NumStackOrders = 3

	// Number of bits in page to span calculations (4k pages).
	// On Windows 64-bit we limit the arena to 32GB or 35 bits.
	// Windows counts memory used by page table into committed memory
	// of the process, so we can't reserve too much memory.
	// See http://golang.org/issue/5402 and http://golang.org/issue/5236.
	// On other 64-bit platforms, we limit the arena to 128GB, or 37 bits.
	// On 32-bit, we don't bother limiting anything, so we use the full 32-bit address.
	_MHeapMap_TotalBits = (_64bit*goos_windows)*35 + (_64bit*(1-goos_windows))*37 + (1-_64bit)*32
	_MHeapMap_Bits      = _MHeapMap_TotalBits - _PageShift

	_MaxMem = uintptr(1<<_MHeapMap_TotalBits - 1)

	// Max number of threads to run garbage collection.
	// 2, 3, and 4 are all plausible maximums depending
	// on the hardware details of the machine.  The garbage
	// collector scales well to 32 cpus.
	_MaxGcproc = 32
)

// A generic linked list of blocks.  (Typically the block is bigger than sizeof(MLink).)
// Since assignments to mlink.next will result in a write barrier being preformed
// this can not be used by some of the internal GC structures. For example when
// the sweeper is placing an unmarked object on the free list it does not want the
// write barrier to be called since that could result in the object being reachable.
type mlink struct {
	next *mlink
}

// A gclink is a node in a linked list of blocks, like mlink,
// but it is opaque to the garbage collector.
// The GC does not trace the pointers during collection,
// and the compiler does not emit write barriers for assignments
// of gclinkptr values. Code should store references to gclinks
// as gclinkptr, not as *gclink.
type gclink struct {
	next gclinkptr
}

// A gclinkptr is a pointer to a gclink, but it is opaque
// to the garbage collector.
type gclinkptr uintptr

// ptr returns the *gclink form of p.
// The result should be used for accessing fields, not stored
// in other data structures.
func (p gclinkptr) ptr() *gclink {
	return (*gclink)(unsafe.Pointer(p))
}

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

// FixAlloc is a simple free-list allocator for fixed size objects.
// Malloc uses a FixAlloc wrapped around sysAlloc to manages its
// MCache and MSpan objects.
//
// Memory returned by FixAlloc_Alloc is not zeroed.
// The caller is responsible for locking around FixAlloc calls.
// Callers can keep state in the object but the first word is
// smashed by freeing and reallocating.
type fixalloc struct {
	size   uintptr
	first  unsafe.Pointer // go func(unsafe.pointer, unsafe.pointer); f(arg, p) called first time p is returned
	arg    unsafe.Pointer
	list   *mlink
	chunk  *byte
	nchunk uint32
	inuse  uintptr // in-use bytes now
	stat   *uint64
}

// Statistics.
// Shared with Go: if you edit this structure, also edit type MemStats in mem.go.
type mstats struct {
	// General statistics.
	alloc       uint64 // bytes allocated and still in use
	total_alloc uint64 // bytes allocated (even if freed)
	sys         uint64 // bytes obtained from system (should be sum of xxx_sys below, no locking, approximate)
	nlookup     uint64 // number of pointer lookups
	nmalloc     uint64 // number of mallocs
	nfree       uint64 // number of frees

	// Statistics about malloc heap.
	// protected by mheap.lock
	heap_alloc    uint64 // bytes allocated and still in use
	heap_sys      uint64 // bytes obtained from system
	heap_idle     uint64 // bytes in idle spans
	heap_inuse    uint64 // bytes in non-idle spans
	heap_released uint64 // bytes released to the os
	heap_objects  uint64 // total number of allocated objects

	// Statistics about allocation of low-level fixed-size structures.
	// Protected by FixAlloc locks.
	stacks_inuse uint64 // this number is included in heap_inuse above
	stacks_sys   uint64 // always 0 in mstats
	mspan_inuse  uint64 // mspan structures
	mspan_sys    uint64
	mcache_inuse uint64 // mcache structures
	mcache_sys   uint64
	buckhash_sys uint64 // profiling bucket hash table
	gc_sys       uint64
	other_sys    uint64

	// Statistics about garbage collector.
	// Protected by mheap or stopping the world during GC.
	next_gc        uint64 // next gc (in heap_alloc time)
	last_gc        uint64 // last gc (in absolute time)
	pause_total_ns uint64
	pause_ns       [256]uint64 // circular buffer of recent gc pause lengths
	pause_end      [256]uint64 // circular buffer of recent gc end times (nanoseconds since 1970)
	numgc          uint32
	enablegc       bool
	debuggc        bool

	// Statistics about allocation size classes.

	by_size [_NumSizeClasses]struct {
		size    uint32
		nmalloc uint64
		nfree   uint64
	}

	tinyallocs uint64 // number of tiny allocations that didn't cause actual allocation; not exported to go directly
}

var memstats mstats

// Size classes.  Computed and initialized by InitSizes.
//
// SizeToClass(0 <= n <= MaxSmallSize) returns the size class,
//	1 <= sizeclass < NumSizeClasses, for n.
//	Size class 0 is reserved to mean "not small".
//
// class_to_size[i] = largest size in class i
// class_to_allocnpages[i] = number of pages to allocate when
//	making new objects in class i

var class_to_size [_NumSizeClasses]int32
var class_to_allocnpages [_NumSizeClasses]int32
var size_to_class8 [1024/8 + 1]int8
var size_to_class128 [(_MaxSmallSize-1024)/128 + 1]int8

type mcachelist struct {
	list  *mlink
	nlist uint32
}

type stackfreelist struct {
	list gclinkptr // linked list of free stacks
	size uintptr   // total size of stacks in list
}

// Per-thread (in Go, per-P) cache for small objects.
// No locking needed because it is per-thread (per-P).
type mcache struct {
	// The following members are accessed on every malloc,
	// so they are grouped here for better caching.
	next_sample      int32  // trigger heap sample after allocating this many bytes
	local_cachealloc intptr // bytes allocated (or freed) from cache since last lock of heap
	// Allocator cache for tiny objects w/o pointers.
	// See "Tiny allocator" comment in malloc.goc.
	tiny             *byte
	tinysize         uintptr
	local_tinyallocs uintptr // number of tiny allocs not counted in other stats

	// The rest is not accessed on every malloc.
	alloc [_NumSizeClasses]*mspan // spans to allocate from

	stackcache [_NumStackOrders]stackfreelist

	sudogcache *sudog

	// Local allocator stats, flushed during GC.
	local_nlookup    uintptr                  // number of pointer lookups
	local_largefree  uintptr                  // bytes freed for large objects (>maxsmallsize)
	local_nlargefree uintptr                  // number of frees for large objects (>maxsmallsize)
	local_nsmallfree [_NumSizeClasses]uintptr // number of frees for small objects (<=maxsmallsize)
}

const (
	_KindSpecialFinalizer = 1
	_KindSpecialProfile   = 2
	// Note: The finalizer special must be first because if we're freeing
	// an object, a finalizer special will cause the freeing operation
	// to abort, and we want to keep the other special records around
	// if that happens.
)

type special struct {
	next   *special // linked list in span
	offset uint16   // span offset of object
	kind   byte     // kind of special
}

// The described object has a finalizer set for it.
type specialfinalizer struct {
	special special
	fn      *funcval
	nret    uintptr
	fint    *_type
	ot      *ptrtype
}

// The described object is being heap profiled.
type specialprofile struct {
	special special
	b       *bucket
}

// An MSpan is a run of pages.
const (
	_MSpanInUse = iota // allocated for garbage collected heap
	_MSpanStack        // allocated for use by stack allocator
	_MSpanFree
	_MSpanListHead
	_MSpanDead
)

type mspan struct {
	next     *mspan    // in a span linked list
	prev     *mspan    // in a span linked list
	start    pageID    // starting page number
	npages   uintptr   // number of pages in span
	freelist gclinkptr // list of free objects
	// sweep generation:
	// if sweepgen == h->sweepgen - 2, the span needs sweeping
	// if sweepgen == h->sweepgen - 1, the span is currently being swept
	// if sweepgen == h->sweepgen, the span is swept and ready to use
	// h->sweepgen is incremented by 2 after every GC
	sweepgen    uint32
	ref         uint16   // capacity - number of objects in freelist
	sizeclass   uint8    // size class
	incache     bool     // being used by an mcache
	state       uint8    // mspaninuse etc
	needzero    uint8    // needs to be zeroed before allocation
	elemsize    uintptr  // computed from sizeclass or from npages
	unusedsince int64    // first time spotted by gc in mspanfree state
	npreleased  uintptr  // number of pages released to the os
	limit       uintptr  // end of data in span
	speciallock mutex    // guards specials list
	specials    *special // linked list of special records sorted by offset.
}

// Every MSpan is in one doubly-linked list,
// either one of the MHeap's free lists or one of the
// MCentral's span lists.  We use empty MSpan structures as list heads.

// Central list of free objects of a given size.
type mcentral struct {
	lock      mutex
	sizeclass int32
	nonempty  mspan // list of spans with a free object
	empty     mspan // list of spans with no free objects (or cached in an mcache)
}

// Main malloc heap.
// The heap itself is the "free[]" and "large" arrays,
// but all the other global data is here too.
type mheap struct {
	lock      mutex
	free      [_MaxMHeapList]mspan // free lists of given length
	freelarge mspan                // free lists length >= _MaxMHeapList
	busy      [_MaxMHeapList]mspan // busy lists of large objects of given length
	busylarge mspan                // busy lists of large objects length >= _MaxMHeapList
	allspans  **mspan              // all spans out there
	gcspans   **mspan              // copy of allspans referenced by gc marker or sweeper
	nspan     uint32
	sweepgen  uint32 // sweep generation, see comment in mspan
	sweepdone uint32 // all spans are swept

	// span lookup
	spans        **mspan
	spans_mapped uintptr

	// range of addresses we might see in the heap
	bitmap         uintptr
	bitmap_mapped  uintptr
	arena_start    uintptr
	arena_used     uintptr
	arena_end      uintptr
	arena_reserved bool

	// central free lists for small size classes.
	// the padding makes sure that the MCentrals are
	// spaced CacheLineSize bytes apart, so that each MCentral.lock
	// gets its own cache line.
	central [_NumSizeClasses]struct {
		mcentral mcentral
		pad      [_CacheLineSize]byte
	}

	spanalloc             fixalloc // allocator for span*
	cachealloc            fixalloc // allocator for mcache*
	specialfinalizeralloc fixalloc // allocator for specialfinalizer*
	specialprofilealloc   fixalloc // allocator for specialprofile*
	speciallock           mutex    // lock for sepcial record allocators.

	// Malloc stats.
	largefree  uint64                  // bytes freed for large objects (>maxsmallsize)
	nlargefree uint64                  // number of frees for large objects (>maxsmallsize)
	nsmallfree [_NumSizeClasses]uint64 // number of frees for small objects (<=maxsmallsize)
}

var mheap_ mheap

const (
	// flags to malloc
	_FlagNoScan = 1 << 0 // GC doesn't have to scan object
	_FlagNoZero = 1 << 1 // don't zero memory
)

// NOTE: Layout known to queuefinalizer.
type finalizer struct {
	fn   *funcval       // function to call
	arg  unsafe.Pointer // ptr to object
	nret uintptr        // bytes of return values from fn
	fint *_type         // type of first argument of fn
	ot   *ptrtype       // type of ptr to object
}

type finblock struct {
	alllink *finblock
	next    *finblock
	cnt     int32
	cap     int32
	fin     [1]finalizer
}

// Information from the compiler about the layout of stack frames.
type bitvector struct {
	n        int32 // # of bits
	bytedata *uint8
}

type stackmap struct {
	n        int32   // number of bitmaps
	nbit     int32   // number of bits in each bitmap
	bytedata [0]byte // bitmaps, each starting on a 32-bit boundary
}

// Returns pointer map data for the given stackmap index
// (the index is encoded in PCDATA_StackMapIndex).

// defined in mgc0.go
