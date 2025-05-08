// Copyright 2009 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// Memory statistics

package runtime

import (
	"internal/runtime/atomic"
	"internal/runtime/gc"
	"unsafe"
)

type mstats struct {
	// Statistics about malloc heap.
	heapStats consistentHeapStats

	// Statistics about stacks.
	stacks_sys sysMemStat // only counts newosproc0 stack in mstats; differs from MemStats.StackSys

	// Statistics about allocation of low-level fixed-size structures.
	mspan_sys    sysMemStat
	mcache_sys   sysMemStat
	buckhash_sys sysMemStat // profiling bucket hash table

	// Statistics about GC overhead.
	gcMiscSys sysMemStat // updated atomically or during STW

	// Miscellaneous statistics.
	other_sys sysMemStat // updated atomically or during STW

	// Statistics about the garbage collector.

	// Protected by mheap or worldsema during GC.
	last_gc_unix    uint64 // last gc (in unix time)
	pause_total_ns  uint64
	pause_ns        [256]uint64 // circular buffer of recent gc pause lengths
	pause_end       [256]uint64 // circular buffer of recent gc end times (nanoseconds since 1970)
	numgc           uint32
	numforcedgc     uint32  // number of user-forced GCs
	gc_cpu_fraction float64 // fraction of CPU time used by GC

	last_gc_nanotime uint64 // last gc (monotonic time)
	lastHeapInUse    uint64 // heapInUse at mark termination of the previous GC

	lastScanStats [gc.NumSizeClasses]sizeClassScanStats

	enablegc bool
}

type sizeClassScanStats struct {
	spansDenseScanned     uint64
	spanObjsDenseScanned  uint64
	spansSparseScanned    uint64
	spanObjsSparseScanned uint64
	sparseObjsScanned     uint64
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
	// from the OS for OS thread stacks.
	//
	// In non-cgo programs this metric is currently equal to StackInuse
	// (but this should not be relied upon, and the value may change in
	// the future).
	//
	// In cgo programs this metric includes OS thread stacks allocated
	// directly from the OS. Currently, this only accounts for one stack in
	// c-shared and c-archive build modes and other sources of stacks from
	// the OS (notably, any allocated by C code) are not currently measured.
	// Note this too may change in the future.
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
	_ = m.Alloc // nil check test before we switch stacks, see issue 61158
	stw := stopTheWorld(stwReadMemStats)

	systemstack(func() {
		readmemstats_m(m)
	})

	startTheWorld(stw)
}

// doubleCheckReadMemStats controls a double-check mode for ReadMemStats that
// ensures consistency between the values that ReadMemStats is using and the
// runtime-internal stats.
var doubleCheckReadMemStats = false

// readmemstats_m populates stats for internal runtime values.
//
// The world must be stopped.
func readmemstats_m(stats *MemStats) {
	assertWorldStopped()

	// Flush mcaches to mcentral before doing anything else.
	//
	// Flushing to the mcentral may in general cause stats to
	// change as mcentral data structures are manipulated.
	systemstack(flushallmcaches)

	// Calculate memory allocator stats.
	// During program execution we only count number of frees and amount of freed memory.
	// Current number of alive objects in the heap and amount of alive heap memory
	// are calculated by scanning all spans.
	// Total number of mallocs is calculated as number of frees plus number of alive objects.
	// Similarly, total amount of allocated memory is calculated as amount of freed memory
	// plus amount of alive heap memory.

	// Collect consistent stats, which are the source-of-truth in some cases.
	var consStats heapStatsDelta
	memstats.heapStats.unsafeRead(&consStats)

	// Collect large allocation stats.
	totalAlloc := consStats.largeAlloc
	nMalloc := consStats.largeAllocCount
	totalFree := consStats.largeFree
	nFree := consStats.largeFreeCount

	// Collect per-sizeclass stats.
	var bySize [gc.NumSizeClasses]struct {
		Size    uint32
		Mallocs uint64
		Frees   uint64
	}
	for i := range bySize {
		bySize[i].Size = uint32(gc.SizeClassToSize[i])

		// Malloc stats.
		a := consStats.smallAllocCount[i]
		totalAlloc += a * uint64(gc.SizeClassToSize[i])
		nMalloc += a
		bySize[i].Mallocs = a

		// Free stats.
		f := consStats.smallFreeCount[i]
		totalFree += f * uint64(gc.SizeClassToSize[i])
		nFree += f
		bySize[i].Frees = f
	}

	// Account for tiny allocations.
	// For historical reasons, MemStats includes tiny allocations
	// in both the total free and total alloc count. This double-counts
	// memory in some sense because their tiny allocation block is also
	// counted. Tracking the lifetime of individual tiny allocations is
	// currently not done because it would be too expensive.
	nFree += consStats.tinyAllocCount
	nMalloc += consStats.tinyAllocCount

	// Calculate derived stats.

	stackInUse := uint64(consStats.inStacks)
	gcWorkBufInUse := uint64(consStats.inWorkBufs)

	totalMapped := gcController.heapInUse.load() + gcController.heapFree.load() + gcController.heapReleased.load() +
		memstats.stacks_sys.load() + memstats.mspan_sys.load() + memstats.mcache_sys.load() +
		memstats.buckhash_sys.load() + memstats.gcMiscSys.load() + memstats.other_sys.load() +
		stackInUse + gcWorkBufInUse

	heapGoal := gcController.heapGoal()

	if doubleCheckReadMemStats {
		// Only check this if we're debugging. It would be bad to crash an application
		// just because the debugging stats are wrong. We mostly rely on tests to catch
		// these issues, and we enable the double check mode for tests.
		//
		// The world is stopped, so the consistent stats (after aggregation)
		// should be identical to some combination of memstats. In particular:
		//
		// * memstats.heapInUse == inHeap
		// * memstats.heapReleased == released
		// * memstats.heapInUse + memstats.heapFree == committed - inStacks - inWorkBufs
		// * memstats.totalAlloc == totalAlloc
		// * memstats.totalFree == totalFree
		//
		// Check if that's actually true.
		//
		// Prevent sysmon and the tracer from skewing the stats since they can
		// act without synchronizing with a STW. See #64401.
		lock(&sched.sysmonlock)
		lock(&trace.lock)
		if gcController.heapInUse.load() != uint64(consStats.inHeap) {
			print("runtime: heapInUse=", gcController.heapInUse.load(), "\n")
			print("runtime: consistent value=", consStats.inHeap, "\n")
			throw("heapInUse and consistent stats are not equal")
		}
		if gcController.heapReleased.load() != uint64(consStats.released) {
			print("runtime: heapReleased=", gcController.heapReleased.load(), "\n")
			print("runtime: consistent value=", consStats.released, "\n")
			throw("heapReleased and consistent stats are not equal")
		}
		heapRetained := gcController.heapInUse.load() + gcController.heapFree.load()
		consRetained := uint64(consStats.committed - consStats.inStacks - consStats.inWorkBufs)
		if heapRetained != consRetained {
			print("runtime: global value=", heapRetained, "\n")
			print("runtime: consistent value=", consRetained, "\n")
			throw("measures of the retained heap are not equal")
		}
		if gcController.totalAlloc.Load() != totalAlloc {
			print("runtime: totalAlloc=", gcController.totalAlloc.Load(), "\n")
			print("runtime: consistent value=", totalAlloc, "\n")
			throw("totalAlloc and consistent stats are not equal")
		}
		if gcController.totalFree.Load() != totalFree {
			print("runtime: totalFree=", gcController.totalFree.Load(), "\n")
			print("runtime: consistent value=", totalFree, "\n")
			throw("totalFree and consistent stats are not equal")
		}
		// Also check that mappedReady lines up with totalMapped - released.
		// This isn't really the same type of "make sure consistent stats line up" situation,
		// but this is an opportune time to check.
		if gcController.mappedReady.Load() != totalMapped-uint64(consStats.released) {
			print("runtime: mappedReady=", gcController.mappedReady.Load(), "\n")
			print("runtime: totalMapped=", totalMapped, "\n")
			print("runtime: released=", uint64(consStats.released), "\n")
			print("runtime: totalMapped-released=", totalMapped-uint64(consStats.released), "\n")
			throw("mappedReady and other memstats are not equal")
		}
		unlock(&trace.lock)
		unlock(&sched.sysmonlock)
	}

	// We've calculated all the values we need. Now, populate stats.

	stats.Alloc = totalAlloc - totalFree
	stats.TotalAlloc = totalAlloc
	stats.Sys = totalMapped
	stats.Mallocs = nMalloc
	stats.Frees = nFree
	stats.HeapAlloc = totalAlloc - totalFree
	stats.HeapSys = gcController.heapInUse.load() + gcController.heapFree.load() + gcController.heapReleased.load()
	// By definition, HeapIdle is memory that was mapped
	// for the heap but is not currently used to hold heap
	// objects. It also specifically is memory that can be
	// used for other purposes, like stacks, but this memory
	// is subtracted out of HeapSys before it makes that
	// transition. Put another way:
	//
	// HeapSys = bytes allocated from the OS for the heap - bytes ultimately used for non-heap purposes
	// HeapIdle = bytes allocated from the OS for the heap - bytes ultimately used for any purpose
	//
	// or
	//
	// HeapSys = sys - stacks_inuse - gcWorkBufInUse
	// HeapIdle = sys - stacks_inuse - gcWorkBufInUse - heapInUse
	//
	// => HeapIdle = HeapSys - heapInUse = heapFree + heapReleased
	stats.HeapIdle = gcController.heapFree.load() + gcController.heapReleased.load()
	stats.HeapInuse = gcController.heapInUse.load()
	stats.HeapReleased = gcController.heapReleased.load()
	stats.HeapObjects = nMalloc - nFree
	stats.StackInuse = stackInUse
	// memstats.stacks_sys is only memory mapped directly for OS stacks.
	// Add in heap-allocated stack memory for user consumption.
	stats.StackSys = stackInUse + memstats.stacks_sys.load()
	stats.MSpanInuse = uint64(mheap_.spanalloc.inuse)
	stats.MSpanSys = memstats.mspan_sys.load()
	stats.MCacheInuse = uint64(mheap_.cachealloc.inuse)
	stats.MCacheSys = memstats.mcache_sys.load()
	stats.BuckHashSys = memstats.buckhash_sys.load()
	// MemStats defines GCSys as an aggregate of all memory related
	// to the memory management system, but we track this memory
	// at a more granular level in the runtime.
	stats.GCSys = memstats.gcMiscSys.load() + gcWorkBufInUse
	stats.OtherSys = memstats.other_sys.load()
	stats.NextGC = heapGoal
	stats.LastGC = memstats.last_gc_unix
	stats.PauseTotalNs = memstats.pause_total_ns
	stats.PauseNs = memstats.pause_ns
	stats.PauseEnd = memstats.pause_end
	stats.NumGC = memstats.numgc
	stats.NumForcedGC = memstats.numforcedgc
	stats.GCCPUFraction = memstats.gc_cpu_fraction
	stats.EnableGC = true

	// stats.BySize and bySize might not match in length.
	// That's OK, stats.BySize cannot change due to backwards
	// compatibility issues. copy will copy the minimum amount
	// of values between the two of them.
	copy(stats.BySize[:], bySize[:])
}

//go:linkname readGCStats runtime/debug.readGCStats
func readGCStats(pauses *[]uint64) {
	systemstack(func() {
		readGCStats_m(pauses)
	})
}

// readGCStats_m must be called on the system stack because it acquires the heap
// lock. See mheap for details.
//
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
//
//go:nosplit
func (s *sysMemStat) load() uint64 {
	return atomic.Load64((*uint64)(s))
}

// add atomically adds the sysMemStat by n.
//
// Must be nosplit as it is called in runtime initialization, e.g. newosproc0.
//
//go:nosplit
func (s *sysMemStat) add(n int64) {
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
	committed  int64 // byte delta of memory committed
	released   int64 // byte delta of released memory generated
	inHeap     int64 // byte delta of memory placed in the heap
	inStacks   int64 // byte delta of memory reserved for stacks
	inWorkBufs int64 // byte delta of memory reserved for work bufs

	// Allocator stats.
	//
	// These are all uint64 because they're cumulative, and could quickly wrap
	// around otherwise.
	tinyAllocCount  uint64                    // number of tiny allocations
	largeAlloc      uint64                    // bytes allocated for large objects
	largeAllocCount uint64                    // number of large object allocations
	smallAllocCount [gc.NumSizeClasses]uint64 // number of allocs for small objects
	largeFree       uint64                    // bytes freed for large objects (>maxSmallSize)
	largeFreeCount  uint64                    // number of frees for large objects (>maxSmallSize)
	smallFreeCount  [gc.NumSizeClasses]uint64 // number of frees for small objects (<=maxSmallSize)

	// NOTE: This struct must be a multiple of 8 bytes in size because it
	// is stored in an array. If it's not, atomic accesses to the above
	// fields may be unaligned and fail on 32-bit platforms.
}

// merge adds in the deltas from b into a.
func (a *heapStatsDelta) merge(b *heapStatsDelta) {
	a.committed += b.committed
	a.released += b.released
	a.inHeap += b.inHeap
	a.inStacks += b.inStacks
	a.inWorkBufs += b.inWorkBufs

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
	gen atomic.Uint32

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
		seq := pp.statsSeq.Add(1)
		if seq%2 == 0 {
			// Should have been incremented to odd.
			print("runtime: seq=", seq, "\n")
			throw("bad sequence number")
		}
	} else {
		lock(&m.noPLock)
	}
	gen := m.gen.Load() % 3
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
		seq := pp.statsSeq.Add(1)
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

	clear(m.stats[:])
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
	currGen := m.gen.Load()
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
	m.gen.Swap((currGen + 1) % 3)

	// Allow P-less writers to continue. They'll be writing to the
	// next generation now.
	unlock(&m.noPLock)

	for _, p := range allp {
		// Spin until there are no more writers.
		for p.statsSeq.Load()%2 != 0 {
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

type cpuStats struct {
	// All fields are CPU time in nanoseconds computed by comparing
	// calls of nanotime. This means they're all overestimates, because
	// they don't accurately compute on-CPU time (so some of the time
	// could be spent scheduled away by the OS).

	GCAssistTime    int64 // GC assists
	GCDedicatedTime int64 // GC dedicated mark workers + pauses
	GCIdleTime      int64 // GC idle mark workers
	GCPauseTime     int64 // GC pauses (all GOMAXPROCS, even if just 1 is running)
	GCTotalTime     int64

	ScavengeAssistTime int64 // background scavenger
	ScavengeBgTime     int64 // scavenge assists
	ScavengeTotalTime  int64

	IdleTime int64 // Time Ps spent in _Pidle.
	UserTime int64 // Time Ps spent in _Prunning or _Psyscall that's not any of the above.

	TotalTime int64 // GOMAXPROCS * (monotonic wall clock time elapsed)
}

// accumulateGCPauseTime add dt*stwProcs to the GC CPU pause time stats. dt should be
// the actual time spent paused, for orthogonality. maxProcs should be GOMAXPROCS,
// not work.stwprocs, since this number must be comparable to a total time computed
// from GOMAXPROCS.
func (s *cpuStats) accumulateGCPauseTime(dt int64, maxProcs int32) {
	cpu := dt * int64(maxProcs)
	s.GCPauseTime += cpu
	s.GCTotalTime += cpu
}

// accumulate takes a cpuStats and adds in the current state of all GC CPU
// counters.
//
// gcMarkPhase indicates that we're in the mark phase and that certain counter
// values should be used.
func (s *cpuStats) accumulate(now int64, gcMarkPhase bool) {
	// N.B. Mark termination and sweep termination pauses are
	// accumulated in work.cpuStats at the end of their respective pauses.
	var (
		markAssistCpu     int64
		markDedicatedCpu  int64
		markFractionalCpu int64
		markIdleCpu       int64
	)
	if gcMarkPhase {
		// N.B. These stats may have stale values if the GC is not
		// currently in the mark phase.
		markAssistCpu = gcController.assistTime.Load()
		markDedicatedCpu = gcController.dedicatedMarkTime.Load()
		markFractionalCpu = gcController.fractionalMarkTime.Load()
		markIdleCpu = gcController.idleMarkTime.Load()
	}

	// The rest of the stats below are either derived from the above or
	// are reset on each mark termination.

	scavAssistCpu := scavenge.assistTime.Load()
	scavBgCpu := scavenge.backgroundTime.Load()

	// Update cumulative GC CPU stats.
	s.GCAssistTime += markAssistCpu
	s.GCDedicatedTime += markDedicatedCpu + markFractionalCpu
	s.GCIdleTime += markIdleCpu
	s.GCTotalTime += markAssistCpu + markDedicatedCpu + markFractionalCpu + markIdleCpu

	// Update cumulative scavenge CPU stats.
	s.ScavengeAssistTime += scavAssistCpu
	s.ScavengeBgTime += scavBgCpu
	s.ScavengeTotalTime += scavAssistCpu + scavBgCpu

	// Update total CPU.
	s.TotalTime = sched.totaltime + (now-sched.procresizetime)*int64(gomaxprocs)
	s.IdleTime += sched.idleTime.Load()

	// Compute userTime. We compute this indirectly as everything that's not the above.
	//
	// Since time spent in _Pgcstop is covered by gcPauseTime, and time spent in _Pidle
	// is covered by idleTime, what we're left with is time spent in _Prunning and _Psyscall,
	// the latter of which is fine because the P will either go idle or get used for something
	// else via sysmon. Meanwhile if we subtract GC time from whatever's left, we get non-GC
	// _Prunning time. Note that this still leaves time spent in sweeping and in the scheduler,
	// but that's fine. The overwhelming majority of this time will be actual user time.
	s.UserTime = s.TotalTime - (s.GCTotalTime + s.ScavengeTotalTime + s.IdleTime)
}
