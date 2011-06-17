// Copyright 2009 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package runtime

import "unsafe"

type MemStatsType struct {
	// General statistics.
	// Not locked during update; approximate.
	Alloc      uint64 // bytes allocated and still in use
	TotalAlloc uint64 // bytes allocated (even if freed)
	Sys        uint64 // bytes obtained from system (should be sum of XxxSys below)
	Lookups    uint64 // number of pointer lookups
	Mallocs    uint64 // number of mallocs
	Frees      uint64 // number of frees

	// Main allocation heap statistics.
	HeapAlloc   uint64 // bytes allocated and still in use
	HeapSys     uint64 // bytes obtained from system
	HeapIdle    uint64 // bytes in idle spans
	HeapInuse   uint64 // bytes in non-idle span
	HeapObjects uint64 // total number of allocated objects

	// Low-level fixed-size structure allocator statistics.
	//	Inuse is bytes used now.
	//	Sys is bytes obtained from system.
	StackInuse  uint64 // bootstrap stacks
	StackSys    uint64
	MSpanInuse  uint64 // mspan structures
	MSpanSys    uint64
	MCacheInuse uint64 // mcache structures
	MCacheSys   uint64
	BuckHashSys uint64 // profiling bucket hash table

	// Garbage collector statistics.
	NextGC       uint64
	PauseTotalNs uint64
	PauseNs      [256]uint64 // most recent GC pause times
	NumGC        uint32
	EnableGC     bool
	DebugGC      bool

	// Per-size allocation statistics.
	// Not locked during update; approximate.
	// 61 is NumSizeClasses in the C code.
	BySize [61]struct {
		Size    uint32
		Mallocs uint64
		Frees   uint64
	}
}

var sizeof_C_MStats uintptr // filled in by malloc.goc

func init() {
	if sizeof_C_MStats != unsafe.Sizeof(MemStats) {
		println(sizeof_C_MStats, unsafe.Sizeof(MemStats))
		panic("MStats vs MemStatsType size mismatch")
	}
}

// MemStats holds statistics about the memory system.
// The statistics are only approximate, as they are not interlocked on update.
var MemStats MemStatsType

// GC runs a garbage collection.
func GC()
