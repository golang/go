// Copyright 2009 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package runtime

import (
	"internal/runtime/atomic"
	"unsafe"
)

// GOMAXPROCS sets the maximum number of CPUs that can be executing
// simultaneously and returns the previous setting. If n < 1, it does not change
// the current setting.
//
// If the GOMAXPROCS environment variable is set to a positive whole number,
// GOMAXPROCS defaults to that value.
//
// Otherwise, the Go runtime selects an appropriate default value based on the
// number of logical CPUs on the machine, the process’s CPU affinity mask, and,
// on Linux, the process’s average CPU throughput limit based on cgroup CPU
// quota, if any.
//
// The Go runtime periodically updates the default value based on changes to
// the total logical CPU count, the CPU affinity mask, or cgroup quota. Setting
// a custom value with the GOMAXPROCS environment variable or by calling
// GOMAXPROCS disables automatic updates. The default value and automatic
// updates can be restored by calling [SetDefaultGOMAXPROCS].
//
// If GODEBUG=containermaxprocs=0 is set, GOMAXPROCS defaults to the value of
// [runtime.NumCPU]. If GODEBUG=updatemaxprocs=0 is set, the Go runtime does
// not perform automatic GOMAXPROCS updating.
//
// The default GOMAXPROCS behavior may change as the scheduler improves.
func GOMAXPROCS(n int) int {
	if GOARCH == "wasm" && n > 1 {
		n = 1 // WebAssembly has no threads yet, so only one CPU is possible.
	}

	lock(&sched.lock)
	ret := int(gomaxprocs)
	unlock(&sched.lock)
	if n <= 0 || n == ret {
		return ret
	}

	stw := stopTheWorldGC(stwGOMAXPROCS)

	// newprocs will be processed by startTheWorld
	//
	// TODO(prattmic): this could use a nicer API. Perhaps add it to the
	// stw parameter?
	newprocs = int32(n)
	newprocsCustom = true

	startTheWorldGC(stw)
	return ret
}

// SetDefaultGOMAXPROCS updates the GOMAXPROCS setting to the runtime
// default, as described by [GOMAXPROCS], ignoring the GOMAXPROCS
// environment variable.
//
// SetDefaultGOMAXPROCS can be used to enable the default automatic updating
// GOMAXPROCS behavior if it has been disabled by the GOMAXPROCS
// environment variable or a prior call to [GOMAXPROCS], or to force an immediate
// update if the caller is aware of a change to the total logical CPU count, CPU
// affinity mask or cgroup quota.
func SetDefaultGOMAXPROCS() {
	// SetDefaultGOMAXPROCS conceptually means "[re]do what the runtime
	// would do at startup if the GOMAXPROCS environment variable were
	// unset." It still respects GODEBUG.

	procs := defaultGOMAXPROCS(0)

	lock(&sched.lock)
	curr := gomaxprocs
	custom := sched.customGOMAXPROCS
	unlock(&sched.lock)

	if !custom && procs == curr {
		// Nothing to do if we're already using automatic GOMAXPROCS
		// and the limit is unchanged.
		return
	}

	stw := stopTheWorldGC(stwGOMAXPROCS)

	// newprocs will be processed by startTheWorld
	//
	// TODO(prattmic): this could use a nicer API. Perhaps add it to the
	// stw parameter?
	newprocs = procs
	newprocsCustom = false

	startTheWorldGC(stw)
}

// NumCPU returns the number of logical CPUs usable by the current process.
//
// The set of available CPUs is checked by querying the operating system
// at process startup. Changes to operating system CPU allocation after
// process startup are not reflected.
func NumCPU() int {
	return int(numCPUStartup)
}

// NumCgoCall returns the number of cgo calls made by the current process.
func NumCgoCall() int64 {
	var n = int64(atomic.Load64(&ncgocall))
	for mp := (*m)(atomic.Loadp(unsafe.Pointer(&allm))); mp != nil; mp = mp.alllink {
		n += int64(mp.ncgocall)
	}
	return n
}

func totalMutexWaitTimeNanos() int64 {
	total := sched.totalMutexWaitTime.Load()

	total += sched.totalRuntimeLockWaitTime.Load()
	for mp := (*m)(atomic.Loadp(unsafe.Pointer(&allm))); mp != nil; mp = mp.alllink {
		total += mp.mLockProfile.waitTime.Load()
	}

	return total
}

// NumGoroutine returns the number of goroutines that currently exist.
func NumGoroutine() int {
	return int(gcount())
}

//go:linkname debug_modinfo runtime/debug.modinfo
func debug_modinfo() string {
	return modinfo
}

// mayMoreStackPreempt is a maymorestack hook that forces a preemption
// at every possible cooperative preemption point.
//
// This is valuable to apply to the runtime, which can be sensitive to
// preemption points. To apply this to all preemption points in the
// runtime and runtime-like code, use the following in bash or zsh:
//
//	X=(-{gc,asm}flags={runtime/...,reflect,sync}=-d=maymorestack=runtime.mayMoreStackPreempt) GOFLAGS=${X[@]}
//
// This must be deeply nosplit because it is called from a function
// prologue before the stack is set up and because the compiler will
// call it from any splittable prologue (leading to infinite
// recursion).
//
// Ideally it should also use very little stack because the linker
// doesn't currently account for this in nosplit stack depth checking.
//
// Ensure mayMoreStackPreempt can be called for all ABIs.
//
//go:nosplit
//go:linkname mayMoreStackPreempt
func mayMoreStackPreempt() {
	// Don't do anything on the g0 or gsignal stack.
	gp := getg()
	if gp == gp.m.g0 || gp == gp.m.gsignal {
		return
	}
	// Force a preemption, unless the stack is already poisoned.
	if gp.stackguard0 < stackPoisonMin {
		gp.stackguard0 = stackPreempt
	}
}

// mayMoreStackMove is a maymorestack hook that forces stack movement
// at every possible point.
//
// See mayMoreStackPreempt.
//
//go:nosplit
//go:linkname mayMoreStackMove
func mayMoreStackMove() {
	// Don't do anything on the g0 or gsignal stack.
	gp := getg()
	if gp == gp.m.g0 || gp == gp.m.gsignal {
		return
	}
	// Force stack movement, unless the stack is already poisoned.
	if gp.stackguard0 < stackPoisonMin {
		gp.stackguard0 = stackForceMove
	}
}

// debugPinnerKeepUnpin is used to make runtime.(*Pinner).Unpin reachable.
var debugPinnerKeepUnpin bool = false

// debugPinnerV1 returns a new Pinner that pins itself. This function can be
// used by debuggers to easily obtain a Pinner that will not be garbage
// collected (or moved in memory) even if no references to it exist in the
// target program. This pinner in turn can be used to extend this property
// to other objects, which debuggers can use to simplify the evaluation of
// expressions involving multiple call injections.
func debugPinnerV1() *Pinner {
	p := new(Pinner)
	p.Pin(unsafe.Pointer(p))
	if debugPinnerKeepUnpin {
		// Make Unpin reachable.
		p.Unpin()
	}
	return p
}
