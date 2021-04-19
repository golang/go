// Copyright 2009 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// Memory statistics

package runtime

import (
	"runtime/internal/atomic"
	"runtime/internal/sys"
	"unsafe"
)

// Statistics.
// If you edit this structure, also edit type MemStats below.
// Their layouts must match exactly.
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
	nlookup     uint64 // number of pointer lookups
	nmalloc     uint64 // number of mallocs
	nfree       uint64 // number of frees

	// Statistics about malloc heap.
	// Protected by mheap.lock
	//
	// In mstats, heap_sys and heap_inuse includes stack memory,
	// while in MemStats stack memory is separated out from the
	// heap stats.
	heap_alloc    uint64 // bytes allocated and not yet freed (same as alloc above)
	heap_sys      uint64 // virtual address space obtained from system
	heap_idle     uint64 // bytes in idle spans
	heap_inuse    uint64 // bytes in non-idle spans
	heap_released uint64 // bytes released to the os
	heap_objects  uint64 // total number of allocated objects

	// TODO(austin): heap_released is both useless and inaccurate
	// in its current form. It's useless because, from the user's
	// and OS's perspectives, there's no difference between a page
	// that has not yet been faulted in and a page that has been
	// released back to the OS. We could fix this by considering
	// newly mapped spans to be "released". It's inaccurate
	// because when we split a large span for allocation, we
	// "unrelease" all pages in the large span and not just the
	// ones we split off for use. This is trickier to fix because
	// we currently don't know which pages of a span we've
	// released. We could fix it by separating "free" and
	// "released" spans, but then we have to allocate from runs of
	// free and released spans.

	// Statistics about allocation of low-level fixed-size structures.
	// Protected by FixAlloc locks.
	stacks_inuse uint64 // this number is included in heap_inuse above; differs from MemStats.StackInuse
	stacks_sys   uint64 // only counts newosproc0 stack in mstats; differs from MemStats.StackSys
	mspan_inuse  uint64 // mspan structures
	mspan_sys    uint64
	mcache_inuse uint64 // mcache structures
	mcache_sys   uint64
	buckhash_sys uint64 // profiling bucket hash table
	gc_sys       uint64
	other_sys    uint64

	// Statistics about garbage collector.
	// Protected by mheap or stopping the world during GC.
	next_gc         uint64 // goal heap_live for when next GC ends; ^0 if disabled
	last_gc         uint64 // last gc (in absolute time)
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

	// Statistics below here are not exported to MemStats directly.

	tinyallocs uint64 // number of tiny allocations that didn't cause actual allocation; not exported to go directly

	// gc_trigger is the heap size that triggers marking.
	//
	// When heap_live ≥ gc_trigger, the mark phase will start.
	// This is also the heap size by which proportional sweeping
	// must be complete.
	gc_trigger uint64

	// heap_live is the number of bytes considered live by the GC.
	// That is: retained by the most recent GC plus allocated
	// since then. heap_live <= heap_alloc, since heap_alloc
	// includes unmarked objects that have not yet been swept (and
	// hence goes up as we allocate and down as we sweep) while
	// heap_live excludes these objects (and hence only goes up
	// between GCs).
	//
	// This is updated atomically without locking. To reduce
	// contention, this is updated only when obtaining a span from
	// an mcentral and at this point it counts all of the
	// unallocated slots in that span (which will be allocated
	// before that mcache obtains another span from that
	// mcentral). Hence, it slightly overestimates the "true" live
	// heap size. It's better to overestimate than to
	// underestimate because 1) this triggers the GC earlier than
	// necessary rather than potentially too late and 2) this
	// leads to a conservative GC rate rather than a GC rate that
	// is potentially too low.
	//
	// Whenever this is updated, call traceHeapAlloc() and
	// gcController.revise().
	heap_live uint64

	// heap_scan is the number of bytes of "scannable" heap. This
	// is the live heap (as counted by heap_live), but omitting
	// no-scan objects and no-scan tails of objects.
	//
	// Whenever this is updated, call gcController.revise().
	heap_scan uint64

	// heap_marked is the number of bytes marked by the previous
	// GC. After mark termination, heap_live == heap_marked, but
	// unlike heap_live, heap_marked does not change until the
	// next mark termination.
	heap_marked uint64
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
	// HeapInuse minus HeapAlloc esimates the amount of memory
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

// Size of the trailing by_size array differs between mstats and MemStats,
// and all data after by_size is local to runtime, not exported.
// NumSizeClasses was changed, but we cannot change MemStats because of backward compatibility.
// sizeof_C_MStats is the size of the prefix of mstats that
// corresponds to MemStats. It should match Sizeof(MemStats{}).
var sizeof_C_MStats = unsafe.Offsetof(memstats.by_size) + 61*unsafe.Sizeof(memstats.by_size[0])

func init() {
	var memStats MemStats
	if sizeof_C_MStats != unsafe.Sizeof(memStats) {
		println(sizeof_C_MStats, unsafe.Sizeof(memStats))
		throw("MStats vs MemStatsType size mismatch")
	}

	if unsafe.Offsetof(memstats.heap_live)%8 != 0 {
		println(unsafe.Offsetof(memstats.heap_live))
		throw("memstats.heap_live not aligned to 8 bytes")
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
	updatememstats(nil)

	// The size of the trailing by_size array differs between
	// mstats and MemStats. NumSizeClasses was changed, but we
	// cannot change MemStats because of backward compatibility.
	memmove(unsafe.Pointer(stats), unsafe.Pointer(&memstats), sizeof_C_MStats)

	// Stack numbers are part of the heap numbers, separate those out for user consumption
	stats.StackSys += stats.StackInuse
	stats.HeapInuse -= stats.StackInuse
	stats.HeapSys -= stats.StackInuse
}

//go:linkname readGCStats runtime/debug.readGCStats
func readGCStats(pauses *[]uint64) {
	systemstack(func() {
		readGCStats_m(pauses)
	})
}

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

	p[n+n] = memstats.last_gc
	p[n+n+1] = uint64(memstats.numgc)
	p[n+n+2] = memstats.pause_total_ns
	unlock(&mheap_.lock)
	*pauses = p[:n+n+3]
}

//go:nowritebarrier
func updatememstats(stats *gcstats) {
	if stats != nil {
		*stats = gcstats{}
	}
	for mp := allm; mp != nil; mp = mp.alllink {
		if stats != nil {
			src := (*[unsafe.Sizeof(gcstats{}) / 8]uint64)(unsafe.Pointer(&mp.gcstats))
			dst := (*[unsafe.Sizeof(gcstats{}) / 8]uint64)(unsafe.Pointer(stats))
			for i, v := range src {
				dst[i] += v
			}
			mp.gcstats = gcstats{}
		}
	}

	memstats.mcache_inuse = uint64(mheap_.cachealloc.inuse)
	memstats.mspan_inuse = uint64(mheap_.spanalloc.inuse)
	memstats.sys = memstats.heap_sys + memstats.stacks_sys + memstats.mspan_sys +
		memstats.mcache_sys + memstats.buckhash_sys + memstats.gc_sys + memstats.other_sys

	// Calculate memory allocator stats.
	// During program execution we only count number of frees and amount of freed memory.
	// Current number of alive object in the heap and amount of alive heap memory
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

	// Flush MCache's to MCentral.
	systemstack(flushallmcaches)

	// Aggregate local stats.
	cachestats()

	// Scan all spans and count number of alive objects.
	lock(&mheap_.lock)
	for _, s := range mheap_.allspans {
		if s.state != mSpanInUse {
			continue
		}
		if s.sizeclass == 0 {
			memstats.nmalloc++
			memstats.alloc += uint64(s.elemsize)
		} else {
			memstats.nmalloc += uint64(s.allocCount)
			memstats.by_size[s.sizeclass].nmalloc += uint64(s.allocCount)
			memstats.alloc += uint64(s.allocCount) * uint64(s.elemsize)
		}
	}
	unlock(&mheap_.lock)

	// Aggregate by size class.
	smallfree := uint64(0)
	memstats.nfree = mheap_.nlargefree
	for i := 0; i < len(memstats.by_size); i++ {
		memstats.nfree += mheap_.nsmallfree[i]
		memstats.by_size[i].nfree = mheap_.nsmallfree[i]
		memstats.by_size[i].nmalloc += mheap_.nsmallfree[i]
		smallfree += mheap_.nsmallfree[i] * uint64(class_to_size[i])
	}
	memstats.nfree += memstats.tinyallocs
	memstats.nmalloc += memstats.nfree

	// Calculate derived stats.
	memstats.total_alloc = memstats.alloc + mheap_.largefree + smallfree
	memstats.heap_alloc = memstats.alloc
	memstats.heap_objects = memstats.nmalloc - memstats.nfree
}

//go:nowritebarrier
func cachestats() {
	for i := 0; ; i++ {
		p := allp[i]
		if p == nil {
			break
		}
		c := p.mcache
		if c == nil {
			continue
		}
		purgecachedstats(c)
	}
}

// flushmcache flushes the mcache of allp[i].
//
// The world must be stopped.
//
//go:nowritebarrier
func flushmcache(i int) {
	p := allp[i]
	if p == nil {
		return
	}
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
	for i := 0; i < int(gomaxprocs); i++ {
		flushmcache(i)
	}
}

//go:nosplit
func purgecachedstats(c *mcache) {
	// Protected by either heap or GC lock.
	h := &mheap_
	memstats.heap_scan += uint64(c.local_scan)
	c.local_scan = 0
	memstats.tinyallocs += uint64(c.local_tinyallocs)
	c.local_tinyallocs = 0
	memstats.nlookup += uint64(c.local_nlookup)
	c.local_nlookup = 0
	h.largefree += uint64(c.local_largefree)
	c.local_largefree = 0
	h.nlargefree += uint64(c.local_nlargefree)
	c.local_nlargefree = 0
	for i := 0; i < len(c.local_nsmallfree); i++ {
		h.nsmallfree[i] += uint64(c.local_nsmallfree[i])
		c.local_nsmallfree[i] = 0
	}
}

// Atomically increases a given *system* memory stat. We are counting on this
// stat never overflowing a uintptr, so this function must only be used for
// system memory stats.
//
// The current implementation for little endian architectures is based on
// xadduintptr(), which is less than ideal: xadd64() should really be used.
// Using xadduintptr() is a stop-gap solution until arm supports xadd64() that
// doesn't use locks.  (Locks are a problem as they require a valid G, which
// restricts their useability.)
//
// A side-effect of using xadduintptr() is that we need to check for
// overflow errors.
//go:nosplit
func mSysStatInc(sysStat *uint64, n uintptr) {
	if sys.BigEndian != 0 {
		atomic.Xadd64(sysStat, int64(n))
		return
	}
	if val := atomic.Xadduintptr((*uintptr)(unsafe.Pointer(sysStat)), n); val < n {
		print("runtime: stat overflow: val ", val, ", n ", n, "\n")
		exit(2)
	}
}

// Atomically decreases a given *system* memory stat. Same comments as
// mSysStatInc apply.
//go:nosplit
func mSysStatDec(sysStat *uint64, n uintptr) {
	if sys.BigEndian != 0 {
		atomic.Xadd64(sysStat, -int64(n))
		return
	}
	if val := atomic.Xadduintptr((*uintptr)(unsafe.Pointer(sysStat)), uintptr(-int64(n))); val+n < n {
		print("runtime: stat underflow: val ", val, ", n ", n, "\n")
		exit(2)
	}
}
