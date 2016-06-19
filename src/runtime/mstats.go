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
type mstats struct {
	// General statistics.
	alloc       uint64 // bytes allocated and not yet freed
	total_alloc uint64 // bytes allocated (even if freed)
	sys         uint64 // bytes obtained from system (should be sum of xxx_sys below, no locking, approximate)
	nlookup     uint64 // number of pointer lookups
	nmalloc     uint64 // number of mallocs
	nfree       uint64 // number of frees

	// Statistics about malloc heap.
	// protected by mheap.lock
	heap_alloc    uint64 // bytes allocated and not yet freed (same as alloc above)
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
	next_gc         uint64 // next gc (in heap_live time)
	last_gc         uint64 // last gc (in absolute time)
	pause_total_ns  uint64
	pause_ns        [256]uint64 // circular buffer of recent gc pause lengths
	pause_end       [256]uint64 // circular buffer of recent gc end times (nanoseconds since 1970)
	numgc           uint32
	gc_cpu_fraction float64 // fraction of CPU time used by GC
	enablegc        bool
	debuggc         bool

	// Statistics about allocation size classes.

	by_size [_NumSizeClasses]struct {
		size    uint32
		nmalloc uint64
		nfree   uint64
	}

	// Statistics below here are not exported to Go directly.

	tinyallocs uint64 // number of tiny allocations that didn't cause actual allocation; not exported to go directly

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

	// heap_reachable is an estimate of the reachable heap bytes
	// at the end of the previous GC.
	heap_reachable uint64
}

var memstats mstats

// A MemStats records statistics about the memory allocator.
type MemStats struct {
	// General statistics.
	Alloc      uint64 // bytes allocated and not yet freed
	TotalAlloc uint64 // bytes allocated (even if freed)
	Sys        uint64 // bytes obtained from system (sum of XxxSys below)
	Lookups    uint64 // number of pointer lookups
	Mallocs    uint64 // number of mallocs
	Frees      uint64 // number of frees

	// Main allocation heap statistics.
	HeapAlloc    uint64 // bytes allocated and not yet freed (same as Alloc above)
	HeapSys      uint64 // bytes obtained from system
	HeapIdle     uint64 // bytes in idle spans
	HeapInuse    uint64 // bytes in non-idle span
	HeapReleased uint64 // bytes released to the OS
	HeapObjects  uint64 // total number of allocated objects

	// Low-level fixed-size structure allocator statistics.
	//	Inuse is bytes used now.
	//	Sys is bytes obtained from system.
	StackInuse  uint64 // bytes used by stack allocator
	StackSys    uint64
	MSpanInuse  uint64 // mspan structures
	MSpanSys    uint64
	MCacheInuse uint64 // mcache structures
	MCacheSys   uint64
	BuckHashSys uint64 // profiling bucket hash table
	GCSys       uint64 // GC metadata
	OtherSys    uint64 // other system allocations

	// Garbage collector statistics.
	NextGC        uint64 // next collection will happen when HeapAlloc ≥ this amount
	LastGC        uint64 // end time of last collection (nanoseconds since 1970)
	PauseTotalNs  uint64
	PauseNs       [256]uint64 // circular buffer of recent GC pause durations, most recent at [(NumGC+255)%256]
	PauseEnd      [256]uint64 // circular buffer of recent GC pause end times
	NumGC         uint32
	GCCPUFraction float64 // fraction of CPU time used by GC
	EnableGC      bool
	DebugGC       bool

	// Per-size allocation statistics.
	// 61 is NumSizeClasses in the C code.
	BySize [61]struct {
		Size    uint32
		Mallocs uint64
		Frees   uint64
	}
}

// Size of the trailing by_size array differs between Go and C,
// and all data after by_size is local to runtime, not exported.
// NumSizeClasses was changed, but we cannot change Go struct because of backward compatibility.
// sizeof_C_MStats is what C thinks about size of Go struct.
var sizeof_C_MStats = unsafe.Offsetof(memstats.by_size) + 61*unsafe.Sizeof(memstats.by_size[0])

func init() {
	var memStats MemStats
	if sizeof_C_MStats != unsafe.Sizeof(memStats) {
		println(sizeof_C_MStats, unsafe.Sizeof(memStats))
		throw("MStats vs MemStatsType size mismatch")
	}
}

// ReadMemStats populates m with memory allocator statistics.
func ReadMemStats(m *MemStats) {
	stopTheWorld("read mem stats")

	systemstack(func() {
		readmemstats_m(m)
	})

	startTheWorld()
}

func readmemstats_m(stats *MemStats) {
	updatememstats(nil)

	// Size of the trailing by_size array differs between Go and C,
	// NumSizeClasses was changed, but we cannot change Go struct because of backward compatibility.
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
	for i := uint32(0); i < mheap_.nspan; i++ {
		s := h_allspans[i]
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

//go:nowritebarrier
func flushallmcaches() {
	for i := 0; ; i++ {
		p := allp[i]
		if p == nil {
			break
		}
		c := p.mcache
		if c == nil {
			continue
		}
		c.releaseAll()
		stackcache_clear(c)
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
