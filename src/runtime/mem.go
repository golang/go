// Copyright 2009 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package runtime

import "unsafe"

// Note: the MemStats struct should be kept in sync with
// struct MStats in malloc.h

// A MemStats records statistics about the memory allocator.
type MemStats struct {
	// General statistics.
	Alloc      uint64 // bytes allocated and still in use
	TotalAlloc uint64 // bytes allocated (even if freed)
	Sys        uint64 // bytes obtained from system (sum of XxxSys below)
	Lookups    uint64 // number of pointer lookups
	Mallocs    uint64 // number of mallocs
	Frees      uint64 // number of frees

	// Main allocation heap statistics.
	HeapAlloc    uint64 // bytes allocated and still in use
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
	NextGC       uint64 // next collection will happen when HeapAlloc ≥ this amount
	LastGC       uint64 // end time of last collection (nanoseconds since 1970)
	PauseTotalNs uint64
	PauseNs      [256]uint64 // circular buffer of recent GC pause durations, most recent at [(NumGC+255)%256]
	PauseEnd     [256]uint64 // circular buffer of recent GC pause end times
	NumGC        uint32
	EnableGC     bool
	DebugGC      bool

	// Per-size allocation statistics.
	// 61 is NumSizeClasses in the C code.
	BySize [61]struct {
		Size    uint32
		Mallocs uint64
		Frees   uint64
	}
}

var sizeof_C_MStats uintptr // filled in by malloc.goc

func init() {
	var memStats MemStats
	if sizeof_C_MStats != unsafe.Sizeof(memStats) {
		println(sizeof_C_MStats, unsafe.Sizeof(memStats))
		gothrow("MStats vs MemStatsType size mismatch")
	}
}

// ReadMemStats populates m with memory allocator statistics.
func ReadMemStats(m *MemStats) {
	// Have to acquire worldsema to stop the world,
	// because stoptheworld can only be used by
	// one goroutine at a time, and there might be
	// a pending garbage collection already calling it.
	semacquire(&worldsema, false)
	gp := getg()
	gp.m.gcing = 1
	onM(stoptheworld)

	gp.m.ptrarg[0] = noescape(unsafe.Pointer(m))
	onM(readmemstats_m)

	gp.m.gcing = 0
	gp.m.locks++
	semrelease(&worldsema)
	onM(starttheworld)
	gp.m.locks--
}

// Implementation of runtime/debug.WriteHeapDump
func writeHeapDump(fd uintptr) {
	semacquire(&worldsema, false)
	gp := getg()
	gp.m.gcing = 1
	onM(stoptheworld)

	gp.m.scalararg[0] = fd
	onM(writeheapdump_m)

	gp.m.gcing = 0
	gp.m.locks++
	semrelease(&worldsema)
	onM(starttheworld)
	gp.m.locks--
}
