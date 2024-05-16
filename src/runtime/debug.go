// Copyright 2009 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package runtime

import (
	"internal/runtime/atomic"
	"unsafe"
)

// GOMAXPROCS sets the maximum number of CPUs that can be executing
// simultaneously and returns the previous setting. It defaults to
// the value of [runtime.NumCPU]. If n < 1, it does not change the current setting.
// This call will go away when the scheduler improves.
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
	newprocs = int32(n)

	startTheWorldGC(stw)
	return ret
}

// NumCPU returns the number of logical CPUs usable by the current process.
//
// The set of available CPUs is checked by querying the operating system
// at process startup. Changes to operating system CPU allocation after
// process startup are not reflected.
func NumCPU() int {
	return int(ncpu)
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
