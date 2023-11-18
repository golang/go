// Copyright 2013 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package debug

import (
	"runtime"
	"sort"
	"time"
)

// GCStats collect information about recent garbage collections.
type GCStats struct {
	LastGC         time.Time       // time of last collection
	NumGC          int64           // number of garbage collections
	PauseTotal     time.Duration   // total pause for all collections
	Pause          []time.Duration // pause history, most recent first
	PauseEnd       []time.Time     // pause end times history, most recent first
	PauseQuantiles []time.Duration
}

// ReadGCStats reads statistics about garbage collection into stats.
// The number of entries in the pause history is system-dependent;
// stats.Pause slice will be reused if large enough, reallocated otherwise.
// ReadGCStats may use the full capacity of the stats.Pause slice.
// If stats.PauseQuantiles is non-empty, ReadGCStats fills it with quantiles
// summarizing the distribution of pause time. For example, if
// len(stats.PauseQuantiles) is 5, it will be filled with the minimum,
// 25%, 50%, 75%, and maximum pause times.
func ReadGCStats(stats *GCStats) {
	// Create a buffer with space for at least two copies of the
	// pause history tracked by the runtime. One will be returned
	// to the caller and the other will be used as transfer buffer
	// for end times history and as a temporary buffer for
	// computing quantiles.
	const maxPause = len(((*runtime.MemStats)(nil)).PauseNs)
	if cap(stats.Pause) < 2*maxPause+3 {
		stats.Pause = make([]time.Duration, 2*maxPause+3)
	}

	// readGCStats fills in the pause and end times histories (up to
	// maxPause entries) and then three more: Unix ns time of last GC,
	// number of GC, and total pause time in nanoseconds. Here we
	// depend on the fact that time.Duration's native unit is
	// nanoseconds, so the pauses and the total pause time do not need
	// any conversion.
	readGCStats(&stats.Pause)
	n := len(stats.Pause) - 3
	stats.LastGC = time.Unix(0, int64(stats.Pause[n]))
	stats.NumGC = int64(stats.Pause[n+1])
	stats.PauseTotal = stats.Pause[n+2]
	n /= 2 // buffer holds pauses and end times
	stats.Pause = stats.Pause[:n]

	if cap(stats.PauseEnd) < maxPause {
		stats.PauseEnd = make([]time.Time, 0, maxPause)
	}
	stats.PauseEnd = stats.PauseEnd[:0]
	for _, ns := range stats.Pause[n : n+n] {
		stats.PauseEnd = append(stats.PauseEnd, time.Unix(0, int64(ns)))
	}

	if len(stats.PauseQuantiles) > 0 {
		if n == 0 {
			for i := range stats.PauseQuantiles {
				stats.PauseQuantiles[i] = 0
			}
		} else {
			// There's room for a second copy of the data in stats.Pause.
			// See the allocation at the top of the function.
			sorted := stats.Pause[n : n+n]
			copy(sorted, stats.Pause)
			sort.Slice(sorted, func(i, j int) bool { return sorted[i] < sorted[j] })
			nq := len(stats.PauseQuantiles) - 1
			for i := 0; i < nq; i++ {
				stats.PauseQuantiles[i] = sorted[len(sorted)*i/nq]
			}
			stats.PauseQuantiles[nq] = sorted[len(sorted)-1]
		}
	}
}

// SetGCPercent sets the garbage collection target percentage:
// a collection is triggered when the ratio of freshly allocated data
// to live data remaining after the previous collection reaches this percentage.
// SetGCPercent returns the previous setting.
// The initial setting is the value of the GOGC environment variable
// at startup, or 100 if the variable is not set.
// This setting may be effectively reduced in order to maintain a memory
// limit.
// A negative percentage effectively disables garbage collection, unless
// the memory limit is reached.
// See SetMemoryLimit for more details.
func SetGCPercent(percent int) int {
	return int(setGCPercent(int32(percent)))
}

// FreeOSMemory forces a garbage collection followed by an
// attempt to return as much memory to the operating system
// as possible. (Even if this is not called, the runtime gradually
// returns memory to the operating system in a background task.)
func FreeOSMemory() {
	freeOSMemory()
}

// SetMaxStack sets the maximum amount of memory that
// can be used by a single goroutine stack.
// If any goroutine exceeds this limit while growing its stack,
// the program crashes.
// SetMaxStack returns the previous setting.
// The initial setting is 1 GB on 64-bit systems, 250 MB on 32-bit systems.
// There may be a system-imposed maximum stack limit regardless
// of the value provided to SetMaxStack.
//
// SetMaxStack is useful mainly for limiting the damage done by
// goroutines that enter an infinite recursion. It only limits future
// stack growth.
func SetMaxStack(bytes int) int {
	return setMaxStack(bytes)
}

// SetMaxThreads sets the maximum number of operating system
// threads that the Go program can use. If it attempts to use more than
// this many, the program crashes.
// SetMaxThreads returns the previous setting.
// The initial setting is 10,000 threads.
//
// The limit controls the number of operating system threads, not the number
// of goroutines. A Go program creates a new thread only when a goroutine
// is ready to run but all the existing threads are blocked in system calls, cgo calls,
// or are locked to other goroutines due to use of runtime.LockOSThread.
//
// SetMaxThreads is useful mainly for limiting the damage done by
// programs that create an unbounded number of threads. The idea is
// to take down the program before it takes down the operating system.
func SetMaxThreads(threads int) int {
	return setMaxThreads(threads)
}

// SetPanicOnFault controls the runtime's behavior when a program faults
// at an unexpected (non-nil) address. Such faults are typically caused by
// bugs such as runtime memory corruption, so the default response is to crash
// the program. Programs working with memory-mapped files or unsafe
// manipulation of memory may cause faults at non-nil addresses in less
// dramatic situations; SetPanicOnFault allows such programs to request
// that the runtime trigger only a panic, not a crash.
// The runtime.Error that the runtime panics with may have an additional method:
//
//	Addr() uintptr
//
// If that method exists, it returns the memory address which triggered the fault.
// The results of Addr are best-effort and the veracity of the result
// may depend on the platform.
// SetPanicOnFault applies only to the current goroutine.
// It returns the previous setting.
func SetPanicOnFault(enabled bool) bool {
	return setPanicOnFault(enabled)
}

// WriteHeapDump writes a description of the heap and the objects in
// it to the given file descriptor.
//
// WriteHeapDump suspends the execution of all goroutines until the heap
// dump is completely written.  Thus, the file descriptor must not be
// connected to a pipe or socket whose other end is in the same Go
// process; instead, use a temporary file or network socket.
//
// The heap dump format is defined at https://golang.org/s/go15heapdump.
func WriteHeapDump(fd uintptr)

// SetTraceback sets the amount of detail printed by the runtime in
// the traceback it prints before exiting due to an unrecovered panic
// or an internal runtime error.
// The level argument takes the same values as the GOTRACEBACK
// environment variable. For example, SetTraceback("all") ensure
// that the program prints all goroutines when it crashes.
// See the package runtime documentation for details.
// If SetTraceback is called with a level lower than that of the
// environment variable, the call is ignored.
func SetTraceback(level string)

// SetMemoryLimit provides the runtime with a soft memory limit.
//
// The runtime undertakes several processes to try to respect this
// memory limit, including adjustments to the frequency of garbage
// collections and returning memory to the underlying system more
// aggressively. This limit will be respected even if GOGC=off (or,
// if SetGCPercent(-1) is executed).
//
// The input limit is provided as bytes, and includes all memory
// mapped, managed, and not released by the Go runtime. Notably, it
// does not account for space used by the Go binary and memory
// external to Go, such as memory managed by the underlying system
// on behalf of the process, or memory managed by non-Go code inside
// the same process. Examples of excluded memory sources include: OS
// kernel memory held on behalf of the process, memory allocated by
// C code, and memory mapped by syscall.Mmap (because it is not
// managed by the Go runtime).
//
// More specifically, the following expression accurately reflects
// the value the runtime attempts to maintain as the limit:
//
//	runtime.MemStats.Sys - runtime.MemStats.HeapReleased
//
// or in terms of the runtime/metrics package:
//
//	/memory/classes/total:bytes - /memory/classes/heap/released:bytes
//
// A zero limit or a limit that's lower than the amount of memory
// used by the Go runtime may cause the garbage collector to run
// nearly continuously. However, the application may still make
// progress.
//
// The memory limit is always respected by the Go runtime, so to
// effectively disable this behavior, set the limit very high.
// [math.MaxInt64] is the canonical value for disabling the limit,
// but values much greater than the available memory on the underlying
// system work just as well.
//
// See https://go.dev/doc/gc-guide for a detailed guide explaining
// the soft memory limit in more detail, as well as a variety of common
// use-cases and scenarios.
//
// The initial setting is math.MaxInt64 unless the GOMEMLIMIT
// environment variable is set, in which case it provides the initial
// setting. GOMEMLIMIT is a numeric value in bytes with an optional
// unit suffix. The supported suffixes include B, KiB, MiB, GiB, and
// TiB. These suffixes represent quantities of bytes as defined by
// the IEC 80000-13 standard. That is, they are based on powers of
// two: KiB means 2^10 bytes, MiB means 2^20 bytes, and so on.
//
// SetMemoryLimit returns the previously set memory limit.
// A negative input does not adjust the limit, and allows for
// retrieval of the currently set memory limit.
func SetMemoryLimit(limit int64) int64 {
	return setMemoryLimit(limit)
}
