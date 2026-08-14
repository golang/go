// Copyright 2025 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

//go:build goexperiment.jit

// Package jit provides support for registering user code frames with
// the Go runtime, enabling JIT compilers, WASM engines, and other code
// generators to interoperate safely with Go's stack unwinder, panic/recover
// mechanism, and garbage collector.
//
// A user frame region is a range of executable memory containing code that
// was not produced by the Go compiler. Without registration, any Go operation
// that inspects the call stack (panic, recover, runtime.Callers, GC stack
// scanning) will crash if it encounters a PC from such a region.
//
// Registering a region makes the runtime aware of the user code and tells
// it how to handle frames in that code during stack unwinding and GC.
//
// # Unwind modes
//
// When the runtime encounters a frame whose PC falls in a registered
// region, it consults the region's unwind mode:
//
//   - [UnwindStop]: The traceback ends at the user frame boundary.
//   - [UnwindSkip]: The user frame is skipped and the traceback continues
//     past it using the provided Next callback.
//   - [UnwindDeclare]: Like UnwindSkip, but the user frame is also
//     described in tracebacks using the provided Describe callback.
//
// # GC integration
//
// If user code holds Go pointers (e.g., on a shadow stack), the
// ScanStack callback must be provided so the GC can find and mark those
// pointers. Failure to do so may cause the GC to collect live objects.
package jit

import (
	"sync"
	_ "unsafe" // for go:linkname
)

// UnwindMode controls how the Go runtime handles user frames during
// stack unwinding (tracebacks, panics, GC).
type UnwindMode uint8

const (
	// UnwindStop ends the traceback at the user frame boundary.
	UnwindStop UnwindMode = iota

	// UnwindSkip skips user frames and continues unwinding using
	// the Next callback. The user frames do not appear in tracebacks.
	UnwindSkip

	// UnwindDeclare is like UnwindSkip but also describes user frames
	// in tracebacks using the Describe callback.
	UnwindDeclare
)

// Region describes a range of executable memory containing user code.
type Region struct {
	// Start is the first byte of the code region (inclusive).
	Start uintptr

	// End is one past the last byte of the code region (exclusive).
	End uintptr

	// Unwind controls the behavior when the runtime encounters a
	// frame in this region during stack unwinding.
	Unwind UnwindMode

	// Describe returns human-readable information about a PC in this
	// region for use in tracebacks. Only called when Unwind is
	// UnwindDeclare. Must not allocate Go memory. May be nil.
	Describe func(pc uintptr) (name, file string, line int, ok bool)

	// Next returns the caller's PC, SP, and BP for a frame at the
	// given PC and SP. This allows the runtime to unwind past user
	// frames. Must not allocate Go memory.
	// Required for UnwindSkip and UnwindDeclare modes.
	// May be nil for UnwindStop (traceback simply ends).
	//
	// The callback runs on the system stack and must not grow the stack.
	// It receives:
	//   pc: the return address pointing into user code
	//   sp: the frame pointer of the Go callee (= user frame's SP before call)
	// It must return:
	//   callerPC: the return address of the Go caller (above the user frame)
	//   callerSP: the SP of the Go caller
	//   callerBP: unused, reserved for future use
	Next func(pc, sp uintptr) (callerPC, callerSP, callerBP uintptr, ok bool)

	// ScanStack is called during garbage collection to report Go
	// pointers held in user stack frames or shadow stacks.
	// Must not allocate Go memory. May be nil.
	ScanStack func(report func(ptr uintptr))
}

// Handle represents a registered user frame region.
type Handle struct {
	handle uintptr
}

// liveRegions keeps function pointers (closures) alive so the GC does
// not collect them. The runtime stores these pointers in persistentalloc
// memory that is invisible to the GC, so we must retain a reference here.
var (
	liveRegionsMu sync.Mutex
	liveRegions   map[uintptr]Region
)

// Register makes the runtime aware of a user code region.
func Register(r Region) Handle {
	h := registerUserFrameRegion(
		r.Start, r.End,
		uint8(r.Unwind),
		r.Describe,
		r.Next,
		r.ScanStack,
	)

	// Keep closures alive for the GC.
	liveRegionsMu.Lock()
	if liveRegions == nil {
		liveRegions = make(map[uintptr]Region)
	}
	liveRegions[h] = r
	liveRegionsMu.Unlock()

	return Handle{handle: h}
}

// Unregister removes the user frame region from the runtime.
func (h Handle) Unregister() {
	unregisterUserFrameRegion(h.handle)

	liveRegionsMu.Lock()
	delete(liveRegions, h.handle)
	liveRegionsMu.Unlock()
}

// Preempt reports whether the Go runtime is requesting the current
// goroutine to yield (e.g., for garbage collection or scheduling).
//
// JIT-compiled code should call Preempt at regular safepoints such as
// loop back-edges and function entries. When Preempt returns true, the
// JIT code should return to its Go caller as soon as possible so the
// runtime can perform pending operations (GC stop-the-world, goroutine
// preemption, etc.).
//
// JIT code that frequently calls Go functions may not need explicit
// Preempt checks, because Go function prologues already check for
// preemption requests. Preempt is primarily useful for long-running
// JIT loops that do not call back into Go.
//
// Preempt is very lightweight: it reads a single flag from the current
// goroutine with no synchronization overhead.
func Preempt() bool {
	return userFramePreempt()
}

//go:linkname userFramePreempt runtime/jit.userFramePreempt
func userFramePreempt() bool

//go:linkname registerUserFrameRegion runtime/jit.registerUserFrameRegion
func registerUserFrameRegion(start, end uintptr, unwindMode uint8,
	describe func(pc uintptr) (string, string, int, bool),
	next func(pc, sp uintptr) (uintptr, uintptr, uintptr, bool),
	scanStack func(report func(ptr uintptr)),
) uintptr

//go:linkname unregisterUserFrameRegion runtime/jit.unregisterUserFrameRegion
func unregisterUserFrameRegion(handle uintptr)
