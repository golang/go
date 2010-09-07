// Copyright 2009 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package runtime

// Breakpoint() executes a breakpoint trap.
func Breakpoint()

// LockOSThread wires the calling goroutine to its current operating system thread.
// Until the calling goroutine exits or calls UnlockOSThread, it will always
// execute in that thread, and no other goroutine can.
// LockOSThread cannot be used during init functions.
func LockOSThread()

// UnlockOSThread unwires the calling goroutine from its fixed operating system thread.
// If the calling goroutine has not called LockOSThread, UnlockOSThread is a no-op.
func UnlockOSThread()

// GOMAXPROCS sets the maximum number of CPUs that can be executing
// simultaneously and returns the previous setting.  If n < 1, it does not
// change the current setting.
// This call will go away when the scheduler improves.
func GOMAXPROCS(n int) int

// Cgocalls returns the number of cgo calls made by the current process.
func Cgocalls() int64

type MemStatsType struct {
	// General statistics.
	// Not locked during update; approximate.
	Alloc      uint64 // bytes allocated and still in use
	TotalAlloc uint64 // bytes allocated (even if freed)
	Sys        uint64 // bytes obtained from system (should be sum of XxxSys below)
	Lookups    uint64 // number of pointer lookups
	Mallocs    uint64 // number of mallocs

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
	MHeapMapSys uint64 // heap map
	BuckHashSys uint64 // profiling bucket hash table

	// Garbage collector statistics.
	NextGC   uint64
	PauseNs  uint64
	NumGC    uint32
	EnableGC bool
	DebugGC  bool

	// Per-size allocation statistics.
	// Not locked during update; approximate.
	BySize [67]struct {
		Size    uint32
		Mallocs uint64
		Frees   uint64
	}
}

// MemStats holds statistics about the memory system.
// The statistics are only approximate, as they are not interlocked on update.
var MemStats MemStatsType

// Alloc allocates a block of the given size.
// FOR TESTING AND DEBUGGING ONLY.
func Alloc(uintptr) *byte

// Free frees the block starting at the given pointer.
// FOR TESTING AND DEBUGGING ONLY.
func Free(*byte)

// Lookup returns the base and size of the block containing the given pointer.
// FOR TESTING AND DEBUGGING ONLY.
func Lookup(*byte) (*byte, uintptr)

// GC runs a garbage collection.
func GC()

// MemProfileRate controls the fraction of memory allocations
// that are recorded and reported in the memory profile.
// The profiler aims to sample an average of
// one allocation per MemProfileRate bytes allocated.
//
// To include every allocated block in the profile, set MemProfileRate to 1.
// To turn off profiling entirely, set MemProfileRate to 0.
//
// The tools that process the memory profiles assume that the
// profile rate is constant across the lifetime of the program
// and equal to the current value.  Programs that change the
// memory profiling rate should do so just once, as early as
// possible in the execution of the program (for example,
// at the beginning of main).
var MemProfileRate int = 512 * 1024

// A MemProfileRecord describes the live objects allocated
// by a particular call sequence (stack trace).
type MemProfileRecord struct {
	AllocBytes, FreeBytes     int64       // number of bytes allocated, freed
	AllocObjects, FreeObjects int64       // number of objects allocated, freed
	Stack0                    [32]uintptr // stack trace for this record; ends at first 0 entry
}

// InUseBytes returns the number of bytes in use (AllocBytes - FreeBytes).
func (r *MemProfileRecord) InUseBytes() int64 { return r.AllocBytes - r.FreeBytes }

// InUseObjects returns the number of objects in use (AllocObjects - FreeObjects).
func (r *MemProfileRecord) InUseObjects() int64 {
	return r.AllocObjects - r.FreeObjects
}

// Stack returns the stack trace associated with the record,
// a prefix of r.Stack0.
func (r *MemProfileRecord) Stack() []uintptr {
	for i, v := range r.Stack0 {
		if v == 0 {
			return r.Stack0[0:i]
		}
	}
	return r.Stack0[0:]
}

// MemProfile returns n, the number of records in the current memory profile.
// If len(p) >= n, MemProfile copies the profile into p and returns n, true.
// If len(p) < n, MemProfile does not change p and returns n, false.
//
// If inuseZero is true, the profile includes allocation records
// where r.AllocBytes > 0 but r.AllocBytes == r.FreeBytes.
// These are sites where memory was allocated, but it has all
// been released back to the runtime.
func MemProfile(p []MemProfileRecord, inuseZero bool) (n int, ok bool)
