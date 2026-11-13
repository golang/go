// Copyright 2026 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

//go:build goexperiment.cgobindings && (darwin || (linux && !android && (amd64 || arm64)))

package cgo

import (
	"internal/abi"
	"unsafe"
)

//go:linkname _cgo_tsan_acquire_fn _cgo_tsan_acquire_fn
var _cgo_tsan_acquire_fn uintptr

//go:linkname _cgo_tsan_release_fn _cgo_tsan_release_fn
var _cgo_tsan_release_fn uintptr

//go:linkname _cgo_msan_unpoison_fn _cgo_msan_unpoison_fn
var _cgo_msan_unpoison_fn uintptr

//go:linknamestd runtime_asmcgocall runtime.asmcgocall
//go:noescape
func runtime_asmcgocall(fn, arg unsafe.Pointer) int32

//go:linkname runtime_cgocall runtime.cgocall
//go:noescape
func runtime_cgocall(fn, arg unsafe.Pointer) int32

// cgoCallFrame holds the target function, up to 6 integer/pointer arguments,
// and the return value for cgoCallTrampoline.
type cgoCallFrame struct {
	fn                     uintptr
	a0, a1, a2, a3, a4, a5 uintptr
	r1                     uintptr
}

func cgoCallTrampoline()

// asmcgocall calls a C function pointer fn with up to 6 integer/pointer
// arguments on the system stack via runtime.asmcgocall without entering
// the scheduler.
//
//go:nosplit
//go:norace
//go:nocheckptr
func asmcgocall(fn, a0, a1, a2, a3, a4, a5 uintptr) uintptr {
	f := cgoCallFrame{
		fn: fn,
		a0: a0,
		a1: a1,
		a2: a2,
		a3: a3,
		a4: a4,
		a5: a5,
	}
	runtime_asmcgocall(unsafe.Pointer(abi.FuncPCABI0(cgoCallTrampoline)), unsafe.Pointer(&f))
	return f.r1
}

// cgocall calls a C function pointer fn with up to 6 integer/pointer
// arguments via runtime.cgocall, entering and exiting the scheduler.
//
//go:norace
//go:nocheckptr
func cgocall(fn, a0, a1, a2, a3, a4, a5 uintptr) uintptr {
	f := cgoCallFrame{
		fn: fn,
		a0: a0,
		a1: a1,
		a2: a2,
		a3: a3,
		a4: a4,
		a5: a5,
	}
	runtime_cgocall(unsafe.Pointer(abi.FuncPCABI0(cgoCallTrampoline)), unsafe.Pointer(&f))
	return f.r1
}

//go:nosplit
//go:norace
func tsanAcquire() {
	if _cgo_tsan_acquire_fn != 0 {
		asmcgocall(_cgo_tsan_acquire_fn, 0, 0, 0, 0, 0, 0)
	}
}

//go:nosplit
//go:norace
func tsanRelease() {
	if _cgo_tsan_release_fn != 0 {
		asmcgocall(_cgo_tsan_release_fn, 0, 0, 0, 0, 0, 0)
	}
}

//go:nosplit
//go:norace
func msanUnpoison(addr unsafe.Pointer, size uintptr) {
	if _cgo_msan_unpoison_fn != 0 {
		asmcgocall(_cgo_msan_unpoison_fn, uintptr(addr), size, 0, 0, 0, 0)
	}
}
