// Copyright 2009 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// Memory statistics

package runtime

import (
	"internal/goarch"
	"runtime/internal/atomic"
	"unsafe"
)

// Statistics.
//
// For detailed descriptions see the documentation for MemStats.
// Fields that differ from MemStats are further documented here.
//
// Many of these fields are updated on the fly, while others are only
// updated when updatememstats is called.
type mstats struct {
	// General statistics.
	alloc       uint64 // bytes allocated and not yet freed
	total_alloc uint64 // bytes allocated (even if freed)
	sys         uint64 // bytes obtained from system (should be sum of xxx_sys below, no locking, approximate)
	nlookup     uint64 // number of pointer lookups (unused)
	nmalloc     uint64 // number of mallocs
	nfree       uint64 // number of frees

	// Statistics about malloc heap.
	// Updated atomically, or with the world stopped.
	//
	// Like MemStats, heap_sys and heap_inuse do not count memory
	// in manually-managed spans.
	heap_sys      sysMemStat // virtual address space obtained from system for GC'd heap
	heap_inuse    uint64     // bytes in mSpanInUse spans
	heap_released uint64     // bytes released to the os

	// heap_objects is not used by the runtime directly and instead
	// computed on the fly by updatememstats.
	heap_objects uint64 // total number of allocated objects

	// Statistics about stacks.
	stacks_inuse uint64     // bytes in manually-managed stack spans; computed by updatememstats
	stacks_sys   sysMemStat // only counts newosproc0 stack in mstats; differs from MemStats.StackSys

	// Statistics about allocation of low-level fixed-size structures.
	// Protected by FixAlloc locks.
	mspan_inuse  uint64 // mspan structures
	mspan_sys    sysMemStat
	mcache_inuse uint64 // mcache structures
	mcache_sys   sysMemStat
	buckhash_sys sysMemStat // profiling bucket hash table

	// Statistics about GC overhead.
	gcWorkBufInUse           uint64     // computed by updatememstats
	gcProgPtrScalarBitsInUse uint64     // computed by updatememstats
	gcMiscSys                sysMemStat // updated atomically or during STW

	// Miscellaneous statistics.
	other_sys sysMemStat // updated atomically or during STW

	// Statistics about the garbage collector.

	// Protected by mheap or stopping the world during GC.
	last_gc_unix    uint64 // last gc (in unix time)
	pause_total_ns  uint64
	pause_ns        [256]uint64 // circular buffer of recent gc pause lengths
	pause_end       [256]uint64 // circular buffer of recent gc end times (nanoseconds since 1970)
	numgc           uint32
	numforcedgc     uint32  // number of user-forced GCs
	gc_cpu_fraction float64 // fraction of CPU time used by GC
	enablegc        bool
	debuggc         bool

	// Statistics about allocation size classes.

	by_size [_NumSizeClasses]struct {
		size    uint32
		nmalloc uint64
		nfree   uint64
	}

	// Add an uint32 for even number of size classes to align below fields
	// to 64 bits for atomic operations on 32 bit platforms.
	_ [1 - _NumSizeClasses%2]uint32

	last_gc_nanotime uint64 // last gc (monotonic time)
	last_heap_inuse  uint64 // heap_inuse at mark termination of the previous GC

	// heapStats is a set of statistics
	heapStats consistentHeapStats

	// _ uint32 // ensure gcPauseDist is aligned

	// gcPauseDist represents the distribution of all GC-related
	// application pauses in the runtime.
	//
	// Each individual pause is counted separately, unlike pause_ns.
	gcPauseDist timeHistogram
}

var memstats mstats

// A MemStats records statistics about the memory allocator.
type MemStats struct {
	// General statistics.

	// Alloc is bytes of allocated heap objects.
	//
	// This is the same as HeapAlloc (see below).
	Alloc uint64

	// TotalAlloc is cumulative bytes allocated for heap objects.
	//
	// TotalAlloc increases as heap objects are allocated, but
	// unlike Alloc and HeapAlloc, it does not decrease when
	// objects are freed.
	TotalAlloc uint64

	// Sys is the total bytes of memory obtained from the OS.
	//
	// Sys is the sum of the XSys fields below. Sys measures the
	// virtual address space reserved by the Go runtime for the
	// heap, stacks, and other internal data structures. It's
	// likely that not all of the virtual address space is backed
	// by physical memory at any given moment, though in general
	// it all was at some point.
	Sys uint64

	// Lookups is the number of pointer lookups performed by the
	// runtime.
	//
	// This is primarily useful for debugging runtime internals.
	Lookups uint64

	// Mallocs is the cumulative count of heap objects allocated.
	// The number of live objects is Mallocs - Frees.
	Mallocs uint64

	// Frees is the cumulative count of heap objects freed.
	Frees uint64

	// Heap memory statistics.
	//
	// Interpreting the heap statistics requires some knowledge of
	// how Go organizes memory. Go divides the virtual address
	// space of the heap into "spans", which are contiguous
	// regions of memory 8K or larger. A span may be in one of
	// three states:
	//
	// An "idle" span contains no objects or other data. The
	// physical memory backing an idle span can be released back
	// to the OS (but the virtual address space never is), or it
	// can be converted into an "in use" or "stack" span.
	//
	// An "in use" span contains at least one heap object and may
	// have free space available to allocate more heap objects.
	//
	// A "stack" span is used for goroutine stacks. Stack spans
	// are not considered part of the heap. A span can change
	// between heap and stack memory; it is never used for both
	// simultaneously.

	// HeapAlloc is bytes of allocated heap objects.
	//
	// "Allocated" heap objects include all reachable objects, as
	// well as unreachable objects that the garbage collector has
	// not yet freed. Specifically, HeapAlloc increases as heap
	// objects are allocated and decreases as the heap is swept
	// and unreachable objects are freed. Sweeping occurs
	// incrementally between GC cycles, so these two processes
	// occur simultaneously, and as a result HeapAlloc tends to
	// change smoothly (in contrast with the sawtooth that is
	// typical of stop-the-world garbage collectors).
	HeapAlloc uint64

	// HeapSys is bytes of heap memory obtained from the OS.
	//
	// HeapSys measures the amount of virtual address space
	// reserved for the heap. This includes virtual address space
	// that has been reserved but not yet used, which consumes no
	// physical memory, but tends to be small, as well as virtual
	// address space for which the physical memory has been
	// returned to the OS after it became unused (see HeapReleased
	// for a measure of the latter).
	//
	// HeapSys estimates the largest size the heap has had.
	HeapSys uint64

	// HeapIdle is bytes in idle (unused) spans.
	//
	// Idle spans have no objects in them. These spans could be
	// (and may already have been) returned to the OS, or they can
	// be reused for heap allocations, or they can be reused as
	// stack memory.
	//
	// HeapIdle minus HeapReleased estimates the amount of memory
	// that could be returned to the OS, but is being retained by
	// the runtime so it can grow the heap without requesting more
	// memory from the OS. If this difference is significantly
	// larger than the heap size, it indicates there was a recent
	// transient spike in live heap size.
	HeapIdle uint64

	// HeapInuse is bytes in in-use spans.
	//
	// In-use spans have at least one object in them. These spans
	// can only be used for other objects of roughly the same
	// size.
	//
	// HeapInuse minus HeapAlloc estimates the amount of memory
	// that has been dedicated to particular size classes, but is
	// not currently being used. This is an upper bound on
	// fragmentation, but in general this memory can be reused
	// efficiently.
	HeapInuse uint64

	// HeapReleased is bytes of physical memory returned to the OS.
	//
	// This counts heap memory from idle spans that was returned
	// to the OS and has not yet been reacquired for the heap.
	HeapReleased uint64

	// HeapObjects is the number of allocated heap objects.
	//
	// Like HeapAlloc, this increases as objects are allocated and
	// decreases as the heap is swept and unreachable objects are
	// freed.
	HeapObjects uint64

	// Stack memory statistics.
	//
	// Stacks are not considered part of the heap, but the runtime
	// can reuse a span of heap memory for stack memory, and
	// vice-versa.

	// StackInuse is bytes in stack spans.
	//
	// In-use stack spans have at least one stack in them. These
	// spans can only be used for other stacks of the same size.
	//
	// There is no StackIdle because unused stack spans are
	// returned to the heap (and hence counted toward HeapIdle).
	StackInuse uint64

	// StackSys is bytes of stack memory obtained from the OS.
	//
	// StackSys is StackInuse, plus any memory obtained directly
	// from the OS for OS thread stacks (which should be minimal).
	StackSys uint64

	// Off-heap memory statistics.
	//
	// The following statistics measure runtime-internal
	// structures that are not allocated from heap memory (usually
	// because they are part of implementing the heap). Unlike
	// heap or stack memory, any memory allocated to these
	// structures is dedicated to these structures.
	//
	// These are primarily useful for debugging runtime memory
	// overheads.

	// MSpanInuse is bytes of allocated mspan structures.
	MSpanInuse uint64

	// MSpanSys is bytes of memory obtained from the OS for mspan
	// structures.
	MSpanSys uint64

	// MCacheInuse is bytes of allocated mcache structures.
	MCacheInuse uint64

	// MCacheSys is bytes of memory obtained from the OS for
	// mcache structures.
	MCacheSys uint64

	// BuckHashSys is bytes of memory in profiling bucket hash tables.
	BuckHashSys uint64

	// GCSys is bytes of memory in garbage collection metadata.
	GCSys uint64

	// OtherSys is bytes of memory in miscellaneous off-heap
	// runtime allocations.
	OtherSys uint64

	// Garbage collector statistics.

	// NextGC is the target heap size of the next GC cycle.
	//
	// The garbage collector's goal is to keep HeapAlloc ≤ NextGC.
	// At the end of each GC cycle, the target for the next cycle
	// is computed based on the amount of reachable data and the
	// value of GOGC.
	NextGC uint64

	// LastGC is the time the last garbage collection finished, as
	// nanoseconds since 1970 (the UNIX epoch).
	LastGC uint64

	// PauseTotalNs is the cumulative nanoseconds in GC
	// stop-the-world pauses since the program started.
	//
	// During a stop-the-world pause, all goroutines are paused
	// and only the garbage collector can run.
	PauseTotalNs uint64

	// PauseNs is a circular buffer of recent GC stop-the-world
	// pause times in nanoseconds.
	//
	// The most recent pause is at PauseNs[(NumGC+255)%256]. In
	// general, PauseNs[N%256] records the time paused in the most
	// recent N%256th GC cycle. There may be multiple pauses per
	// GC cycle; this is the sum of all pauses during a cycle.
	PauseNs [256]uint64

	// PauseEnd is a circular buffer of recent GC pause end times,
	// as nanoseconds since 1970 (the UNIX epoch).
	//
	// This buffer is filled the same way as PauseNs. There may be
	// multiple pauses per GC cycle; this records the end of the
	// last pause in a cycle.
	PauseEnd [256]uint64

	// NumGC is the number of completed GC cycles.
	NumGC uint32

	// NumForcedGC is the number of GC cycles that were forced by
	// the application calling the GC function.
	NumForcedGC uint32

	// GCCPUFraction is the fraction of this program's available
	// CPU time used by the GC since the program started.
	//
	// GCCPUFraction is expressed as a number between 0 and 1,
	// where 0 means GC has consumed none of this program's CPU. A
	// program's available CPU time is defined as the integral of
	// GOMAXPROCS since the program started. That is, if
	// GOMAXPROCS is 2 and a program has been running for 10
	// seconds, its "available CPU" is 20 seconds. GCCPUFraction
	// does not include CPU time used for write barrier activity.
	//
	// This is the same as the fraction of CPU reported by
	// GODEBUG=gctrace=1.
	GCCPUFraction float64

	// EnableGC indicates that GC is enabled. It is always true,
	// even if GOGC=off.
	EnableGC bool

	// DebugGC is currently unused.
	DebugGC bool

	// BySize reports per-size class allocation statistics.
	//
	// BySize[N] gives statistics for allocations of size S where
	// BySize[N-1].Size < S ≤ BySize[N].Size.
	//
	// This does not report allocations larger than BySize[60].Size.
	BySize [61]struct {
		// Size is the maximum byte size of an object in this
		// size class.
		Size uint32

		// Mallocs is the cumulative count of heap objects
		// allocated in this size class. The cumulative bytes
		// of allocation is Size*Mallocs. The number of live
		// objects in this size class is Mallocs - Frees.
		Mallocs uint64

		// Frees is the cumulative count of heap objects freed
		// in this size class.
		Frees uint64
	}
}

func init() {
	if offset := unsafe.Offsetof(memstats.heapStats); offset%8 != 0 {
		println(offset)
		throw("memstats.heapStats not aligned to 8 bytes")
	}
	if offset := unsafe.Offsetof(memstats.gcPauseDist); offset%8 != 0 {
		println(offset)
		throw("memstats.gcPauseDist not aligned to 8 bytes")
	}
	// Ensure the size of heapStatsDelta causes adjacent fields/slots (e.g.
	// [3]heapStatsDelta) to be 8-byte aligned.
	if size := unsafe.Sizeof(heapStatsDelta{}); size%8 != 0 {
		println(size)
		throw("heapStatsDelta not a multiple of 8 bytes in size")
	}
}

// ReadMemStats populates m with memory allocator statistics.
//
// The returned memory allocator statistics are up to date as of the
// call to ReadMemStats. This is in contrast with a heap profile,
// which is a snapshot as of the most recently completed garbage
// collection cycle.
func ReadMemStats(m *MemStats) {
	stopTheWorld("read mem stats")

	systemstack(func() {
		readmemstats_m(m)
	})

	startTheWorld()
}

func readmemstats_m(stats *MemStats) {
	updatememstats()

	stats.Alloc = memstats.alloc
	stats.TotalAlloc = memstats.total_alloc
	stats.Sys = memstats.sys
	stats.Mallocs = memstats.nmalloc
	stats.Frees = memstats.nfree
	stats.HeapAlloc = memstats.alloc
	stats.HeapSys = memstats.heap_sys.load()
	// By definition, HeapIdle is memory that was mapped
	// for the heap but is not currently used to hold heap
	// objects. It also specifically is memory that can be
	// used for other purposes, like stacks, but this memory
	// is subtracted out of HeapSys before it makes that
	// transition. Put another way:
	//
	// heap_sys = bytes allocated from the OS for the heap - bytes ultimately used for non-heap purposes
	// heap_idle = bytes allocated from the OS for the heap - bytes ultimately used for any purpose
	//
	// or
	//
	// heap_sys = sys - stacks_inuse - gcWorkBufInUse - gcProgPtrScalarBitsInUse
	// heap_idle = sys - stacks_inuse - gcWorkBufInUse - gcProgPtrScalarBitsInUse - heap_inuse
	//
	// => heap_idle = heap_sys - heap_inuse
	stats.HeapIdle = memstats.heap_sys.load() - memstats.heap_inuse
	stats.HeapInuse = memstats.heap_inuse
	stats.HeapReleased = memstats.heap_released
	stats.HeapObjects = memstats.heap_objects
	stats.StackInuse = memstats.stacks_inuse
	// memstats.stacks_sys is only memory mapped directly for OS stacks.
	// Add in heap-allocated stack memory for user consumption.
	stats.StackSys = memstats.stacks_inuse + memstats.stacks_sys.load()
	stats.MSpanInuse = memstats.mspan_inuse
	stats.MSpanSys = memstats.mspan_sys.load()
	stats.MCacheInuse = memstats.mcache_inuse
	stats.MCacheSys = memstats.mcache_sys.load()
	stats.BuckHashSys = memstats.buckhash_sys.load()
	// MemStats defines GCSys as an aggregate of all memory related
	// to the memory management system, but we track this memory
	// at a more granular level in the runtime.
	stats.GCSys = memstats.gcMiscSys.load() + memstats.gcWorkBufInUse + memstats.gcProgPtrScalarBitsInUse
	stats.OtherSys = memstats.other_sys.load()
	stats.NextGC = gcController.heapGoal
	stats.LastGC = memstats.last_gc_unix
	stats.PauseTotalNs = memstats.pause_total_ns
	stats.PauseNs = memstats.pause_ns
	stats.PauseEnd = memstats.pause_end
	stats.NumGC = memstats.numgc
	stats.NumForcedGC = memstats.numforcedgc
	stats.GCCPUFraction = memstats.gc_cpu_fraction
	stats.EnableGC = true

	// Handle BySize. Copy N values, where N is
	// the minimum of the lengths of the two arrays.
	// Unfortunately copy() won't work here because
	// the arrays have different structs.
	//
	// TODO(mknyszek): Consider renaming the fields
	// of by_size's elements to align so we can use
	// the copy built-in.
	bySizeLen := len(stats.BySize)
	if l := len(memstats.by_size); l < bySizeLen {
		bySizeLen = l
	}
	for i := 0; i < bySizeLen; i++ {
		stats.BySize[i].Size = memstats.by_size[i].size
		stats.BySize[i].Mallocs = memstats.by_size[i].nmalloc
		stats.BySize[i].Frees = memstats.by_size[i].nfree
	}
}

//go:linkname readGCStats runtime/debug.readGCStats
func readGCStats(pauses *[]uint64) {
	systemstack(func() {
		readGCStats_m(pauses)
	})
}

// readGCStats_m must be called on the system stack because it acquires the heap
// lock. See mheap for details.
//go:systemstack
func readGCStats_m(pauses *[]uint64) {
	p := *pauses
	// Calling code in runtime/debug should make the slice large enough.
	if cap(p) < len(memstats.pause_ns)+3 {
		throw("short slice passed to readGCStats")
	}

	// Pass back: pauses, pause ends, last gc (absolute time), number of gc, total pause ns.
	lock(&mheap_.lock)

	n := memstats.numgc
	if n > uint32(len(memstats.pause_ns)) {
		n = uint32(len(memstats.pause_ns))
	}

	// The pause buffer is circular. The most recent pause is at
	// pause_ns[(numgc-1)%len(pause_ns)], and then backward
	// from there to go back farther in time. We deliver the times
	// most recent first (in p[0]).
	p = p[:cap(p)]
	for i := uint32(0); i < n; i++ {
		j := (memstats.numgc - 1 - i) % uint32(len(memstats.pause_ns))
		p[i] = memstats.pause_ns[j]
		p[n+i] = memstats.pause_end[j]
	}

	p[n+n] = memstats.last_gc_unix
	p[n+n+1] = uint64(memstats.numgc)
	p[n+n+2] = memstats.pause_total_ns
	unlock(&mheap_.lock)
	*pauses = p[:n+n+3]
}

// Updates the memstats structure.
//
// The world must be stopped.
//
//go:nowritebarrier
func updatememstats() {
	assertWorldStopped()

	// Flush mcaches to mcentral before doing anything else.
	//
	// Flushing to the mcentral may in general cause stats to
	// change as mcentral data structures are manipulated.
	systemstack(flushallmcaches)

	memstats.mcache_inuse = uint64(mheap_.cachealloc.inuse)
	memstats.mspan_inuse = uint64(mheap_.spanalloc.inuse)
	memstats.sys = memstats.heap_sys.load() + memstats.stacks_sys.load() + memstats.mspan_sys.load() +
		memstats.mcache_sys.load() + memstats.buckhash_sys.load() + memstats.gcMiscSys.load() +
		memstats.other_sys.load()

	// Calculate memory allocator stats.
	// During program execution we only count number of frees and amount of freed memory.
	// Current number of alive objects in the heap and amount of alive heap memory
	// are calculated by scanning all spans.
	// Total number of mallocs is calculated as number of frees plus number of alive objects.
	// Similarly, total amount of allocated memory is calculated as amount of freed memory
	// plus amount of alive heap memory.
	memstats.alloc = 0
	memstats.total_alloc = 0
	memstats.nmalloc = 0
	memstats.nfree = 0
	for i := 0; i < len(memstats.by_size); i++ {
		memstats.by_size[i].nmalloc = 0
		memstats.by_size[i].nfree = 0
	}
	// Collect consistent stats, which are the source-of-truth in the some cases.
	var consStats heapStatsDelta
	memstats.heapStats.unsafeRead(&consStats)

	// Collect large allocation stats.
	totalAlloc := uint64(consStats.largeAlloc)
	memstats.nmalloc += uint64(consStats.largeAllocCount)
	totalFree := uint64(consStats.largeFree)
	memstats.nfree += uint64(consStats.largeFreeCount)

	// Collect per-sizeclass stats.
	for i := 0; i < _NumSizeClasses; i++ {
		// Malloc stats.
		a := uint64(consStats.smallAllocCount[i])
		totalAlloc += a * uint64(class_to_size[i])
		memstats.nmalloc += a
		memstats.by_size[i].nmalloc = a

		// Free stats.
		f := uint64(consStats.smallFreeCount[i])
		totalFree += f * uint64(class_to_size[i])
		memstats.nfree += f
		memstats.by_size[i].nfree = f
	}

	// Account for tiny allocations.
	memstats.nfree += uint64(consStats.tinyAllocCount)
	memstats.nmalloc += uint64(consStats.tinyAllocCount)

	// Calculate derived stats.
	memstats.total_alloc = totalAlloc
	memstats.alloc = totalAlloc - totalFree
	memstats.heap_objects = memstats.nmalloc - memstats.nfree

	memstats.stacks_inuse = uint64(consStats.inStacks)
	memstats.gcWorkBufInUse = uint64(consStats.inWorkBufs)
	memstats.gcProgPtrScalarBitsInUse = uint64(consStats.inPtrScalarBits)

	// We also count stacks_inuse, gcWorkBufInUse, and gcProgPtrScalarBitsInUse as sys memory.
	memstats.sys += memstats.stacks_inuse + memstats.gcWorkBufInUse + memstats.gcProgPtrScalarBitsInUse

	// The world is stopped, so the consistent stats (after aggregation)
	// should be identical to some combination of memstats. In particular:
	//
	// * heap_inuse == inHeap
	// * heap_released == released
	// * heap_sys - heap_released == committed - inStacks - inWorkBufs - inPtrScalarBits
	//
	// Check if that's actually true.
	//
	// TODO(mknyszek): Maybe don't throw here. It would be bad if a
	// bug in otherwise benign accounting caused the whole application
	// to crash.
	if memstats.heap_inuse != uint64(consStats.inHeap) {
		print("runtime: heap_inuse=", memstats.heap_inuse, "\n")
		print("runtime: consistent value=", consStats.inHeap, "\n")
		throw("heap_inuse and consistent stats are not equal")
	}
	if memstats.heap_released != uint64(consStats.released) {
		print("runtime: heap_released=", memstats.heap_released, "\n")
		print("runtime: consistent value=", consStats.released, "\n")
		throw("heap_released and consistent stats are not equal")
	}
	globalRetained := memstats.heap_sys.load() - memstats.heap_released
	consRetained := uint64(consStats.committed - consStats.inStacks - consStats.inWorkBufs - consStats.inPtrScalarBits)
	if globalRetained != consRetained {
		print("runtime: global value=", globalRetained, "\n")
		print("runtime: consistent value=", consRetained, "\n")
		throw("measures of the retained heap are not equal")
	}
}

// flushmcache flushes the mcache of allp[i].
//
// The world must be stopped.
//
//go:nowritebarrier
func flushmcache(i int) {
	assertWorldStopped()

	p := allp[i]
	c := p.mcache
	if c == nil {
		return
	}
	c.releaseAll()
	stackcache_clear(c)
}

// flushallmcaches flushes the mcaches of all Ps.
//
// The world must be stopped.
//
//go:nowritebarrier
func flushallmcaches() {
	assertWorldStopped()

	for i := 0; i < int(gomaxprocs); i++ {
		flushmcache(i)
	}
}

// sysMemStat represents a global system statistic that is managed atomically.
//
// This type must structurally be a uint64 so that mstats aligns with MemStats.
type sysMemStat uint64

// load atomically reads the value of the stat.
//
// Must be nosplit as it is called in runtime initialization, e.g. newosproc0.
//go:nosplit
func (s *sysMemStat) load() uint64 {
	return atomic.Load64((*uint64)(s))
}

// add atomically adds the sysMemStat by n.
//
// Must be nosplit as it is called in runtime initialization, e.g. newosproc0.
//go:nosplit
func (s *sysMemStat) add(n int64) {
	if s == nil {
		return
	}
	val := atomic.Xadd64((*uint64)(s), n)
	if (n > 0 && int64(val) < n) || (n < 0 && int64(val)+n < n) {
		print("runtime: val=", val, " n=", n, "\n")
		throw("sysMemStat overflow")
	}
}

// heapStatsDelta contains deltas of various runtime memory statistics
// that need to be updated together in order for them to be kept
// consistent with one another.
type heapStatsDelta struct {
	// Memory stats.
	committed       int64 // byte delta of memory committed
	released        int64 // byte delta of released memory generated
	inHeap          int64 // byte delta of memory placed in the heap
	inStacks        int64 // byte delta of memory reserved for stacks
	inWorkBufs      int64 // byte delta of memory reserved for work bufs
	inPtrScalarBits int64 // byte delta of memory reserved for unrolled GC prog bits

	// Allocator stats.
	tinyAllocCount  uintptr                  // number of tiny allocations
	largeAlloc      uintptr                  // bytes allocated for large objects
	largeAllocCount uintptr                  // number of large object allocations
	smallAllocCount [_NumSizeClasses]uintptr // number of allocs for small objects
	largeFree       uintptr                  // bytes freed for large objects (>maxSmallSize)
	largeFreeCount  uintptr                  // number of frees for large objects (>maxSmallSize)
	smallFreeCount  [_NumSizeClasses]uintptr // number of frees for small objects (<=maxSmallSize)

	// Add a uint32 to ensure this struct is a multiple of 8 bytes in size.
	// Only necessary on 32-bit platforms.
	_ [(goarch.PtrSize / 4) % 2]uint32
}

// merge adds in the deltas from b into a.
func (a *heapStatsDelta) merge(b *heapStatsDelta) {
	a.committed += b.committed
	a.released += b.released
	a.inHeap += b.inHeap
	a.inStacks += b.inStacks
	a.inWorkBufs += b.inWorkBufs
	a.inPtrScalarBits += b.inPtrScalarBits

	a.tinyAllocCount += b.tinyAllocCount
	a.largeAlloc += b.largeAlloc
	a.largeAllocCount += b.largeAllocCount
	for i := range b.smallAllocCount {
		a.smallAllocCount[i] += b.smallAllocCount[i]
	}
	a.largeFree += b.largeFree
	a.largeFreeCount += b.largeFreeCount
	for i := range b.smallFreeCount {
		a.smallFreeCount[i] += b.smallFreeCount[i]
	}
}

// consistentHeapStats represents a set of various memory statistics
// whose updates must be viewed completely to get a consistent
// state of the world.
//
// To write updates to memory stats use the acquire and release
// methods. To obtain a consistent global snapshot of these statistics,
// use read.
type consistentHeapStats struct {
	// stats is a ring buffer of heapStatsDelta values.
	// Writers always atomically update the delta at index gen.
	//
	// Readers operate by rotating gen (0 -> 1 -> 2 -> 0 -> ...)
	// and synchronizing with writers by observing each P's
	// statsSeq field. If the reader observes a P not writing,
	// it can be sure that it will pick up the new gen value the
	// next time it writes.
	//
	// The reader then takes responsibility by clearing space
	// in the ring buffer for the next reader to rotate gen to
	// that space (i.e. it merges in values from index (gen-2) mod 3
	// to index (gen-1) mod 3, then clears the former).
	//
	// Note that this means only one reader can be reading at a time.
	// There is no way for readers to synchronize.
	//
	// This process is why we need a ring buffer of size 3 instead
	// of 2: one is for the writers, one contains the most recent
	// data, and the last one is clear so writers can begin writing
	// to it the moment gen is updated.
	stats [3]heapStatsDelta

	// gen represents the current index into which writers
	// are writing, and can take on the value of 0, 1, or 2.
	// This value is updated atomically.
	gen uint32

	// noPLock is intended to provide mutual exclusion for updating
	// stats when no P is available. It does not block other writers
	// with a P, only other writers without a P and the reader. Because
	// stats are usually updated when a P is available, contention on
	// this lock should be minimal.
	noPLock mutex
}

// acquire returns a heapStatsDelta to be updated. In effect,
// it acquires the shard for writing. release must be called
// as soon as the relevant deltas are updated.
//
// The returned heapStatsDelta must be updated atomically.
//
// The caller's P must not change between acquire and
// release. This also means that the caller should not
// acquire a P or release its P in between. A P also must
// not acquire a given consistentHeapStats if it hasn't
// yet released it.
//
// nosplit because a stack growth in this function could
// lead to a stack allocation that could reenter the
// function.
//
//go:nosplit
func (m *consistentHeapStats) acquire() *heapStatsDelta {
	if pp := getg().m.p.ptr(); pp != nil {
		seq := atomic.Xadd(&pp.statsSeq, 1)
		if seq%2 == 0 {
			// Should have been incremented to odd.
			print("runtime: seq=", seq, "\n")
			throw("bad sequence number")
		}
	} else {
		lock(&m.noPLock)
	}
	gen := atomic.Load(&m.gen) % 3
	return &m.stats[gen]
}

// release indicates that the writer is done modifying
// the delta. The value returned by the corresponding
// acquire must no longer be accessed or modified after
// release is called.
//
// The caller's P must not change between acquire and
// release. This also means that the caller should not
// acquire a P or release its P in between.
//
// nosplit because a stack growth in this function could
// lead to a stack allocation that causes another acquire
// before this operation has completed.
//
//go:nosplit
func (m *consistentHeapStats) release() {
	if pp := getg().m.p.ptr(); pp != nil {
		seq := atomic.Xadd(&pp.statsSeq, 1)
		if seq%2 != 0 {
			// Should have been incremented to even.
			print("runtime: seq=", seq, "\n")
			throw("bad sequence number")
		}
	} else {
		unlock(&m.noPLock)
	}
}

// unsafeRead aggregates the delta for this shard into out.
//
// Unsafe because it does so without any synchronization. The
// world must be stopped.
func (m *consistentHeapStats) unsafeRead(out *heapStatsDelta) {
	assertWorldStopped()

	for i := range m.stats {
		out.merge(&m.stats[i])
	}
}

// unsafeClear clears the shard.
//
// Unsafe because the world must be stopped and values should
// be donated elsewhere before clearing.
func (m *consistentHeapStats) unsafeClear() {
	assertWorldStopped()

	for i := range m.stats {
		m.stats[i] = heapStatsDelta{}
	}
}

// read takes a globally consistent snapshot of m
// and puts the aggregated value in out. Even though out is a
// heapStatsDelta, the resulting values should be complete and
// valid statistic values.
//
// Not safe to call concurrently. The world must be stopped
// or metricsSema must be held.
func (m *consistentHeapStats) read(out *heapStatsDelta) {
	// Getting preempted after this point is not safe because
	// we read allp. We need to make sure a STW can't happen
	// so it doesn't change out from under us.
	mp := acquirem()

	// Get the current generation. We can be confident that this
	// will not change since read is serialized and is the only
	// one that modifies currGen.
	currGen := atomic.Load(&m.gen)
	prevGen := currGen - 1
	if currGen == 0 {
		prevGen = 2
	}

	// Prevent writers without a P from writing while we update gen.
	lock(&m.noPLock)

	// Rotate gen, effectively taking a snapshot of the state of
	// these statistics at the point of the exchange by moving
	// writers to the next set of deltas.
	//
	// This exchange is safe to do because we won't race
	// with anyone else trying to update this value.
	atomic.Xchg(&m.gen, (currGen+1)%3)

	// Allow P-less writers to continue. They'll be writing to the
	// next generation now.
	unlock(&m.noPLock)

	for _, p := range allp {
		// Spin until there are no more writers.
		for atomic.Load(&p.statsSeq)%2 != 0 {
		}
	}

	// At this point we've observed that each sequence
	// number is even, so any future writers will observe
	// the new gen value. That means it's safe to read from
	// the other deltas in the stats buffer.

	// Perform our responsibilities and free up
	// stats[prevGen] for the next time we want to take
	// a snapshot.
	m.stats[currGen].merge(&m.stats[prevGen])
	m.stats[prevGen] = heapStatsDelta{}

	// Finally, copy out the complete delta.
	*out = m.stats[currGen]

	releasem(mp)
}
