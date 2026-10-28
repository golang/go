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

//go:linknamestd runtime_asmcgocall_no_g runtime.asmcgocall_no_g
//go:noescape
func runtime_asmcgocall_no_g(fn, arg unsafe.Pointer)

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
	var f cgoCallFrame
	f.fn = fn
	f.a0 = a0
	f.a1 = a1
	f.a2 = a2
	f.a3 = a3
	f.a4 = a4
	f.a5 = a5
	f.r1 = 0
	runtime_asmcgocall(unsafe.Pointer(abi.FuncPCABI0(cgoCallTrampoline)), unsafe.Pointer(&f))
	return f.r1
}

// asmcgocallNoG calls a C function pointer fn with up to 6 integer/pointer
// arguments on the current stack via runtime.asmcgocall_no_g without touching
// g or TLS. It is used before runtime initialization (for example, libpreinit,
// libInit, or x_cgo_init) or on a C/signal stack where switching to g0 is
// unsafe.
//
//go:nosplit
//go:norace
//go:nocheckptr
func asmcgocallNoG(fn, a0, a1, a2, a3, a4, a5 uintptr) uintptr {
	var f cgoCallFrame
	f.fn = fn
	f.a0 = a0
	f.a1 = a1
	f.a2 = a2
	f.a3 = a3
	f.a4 = a4
	f.a5 = a5
	f.r1 = 0
	runtime_asmcgocall_no_g(unsafe.Pointer(abi.FuncPCABI0(cgoCallTrampoline)), unsafe.Pointer(&f))
	return f.r1
}

// cgocall calls a C function pointer fn with up to 6 integer/pointer
// arguments via runtime.cgocall, entering and exiting the scheduler.
//
//go:norace
//go:nocheckptr
func cgocall(fn, a0, a1, a2, a3, a4, a5 uintptr) uintptr {
	var f cgoCallFrame
	f.fn = fn
	f.a0 = a0
	f.a1 = a1
	f.a2 = a2
	f.a3 = a3
	f.a4 = a4
	f.a5 = a5
	f.r1 = 0
	runtime_cgocall(unsafe.Pointer(abi.FuncPCABI0(cgoCallTrampoline)), unsafe.Pointer(&f))
	return f.r1
}

//go:nosplit
//go:norace
func tsanAcquire() {
	if fn := _cgo_tsan_acquire_fn; fn != 0 {
		runtime_asmcgocall(*(*unsafe.Pointer)(unsafe.Pointer(&fn)), nil)
	}
}

//go:nosplit
//go:norace
func tsanAcquireNoG() {
	if fn := _cgo_tsan_acquire_fn; fn != 0 {
		runtime_asmcgocall_no_g(*(*unsafe.Pointer)(unsafe.Pointer(&fn)), nil)
	}
}

//go:nosplit
//go:norace
func tsanRelease() {
	if fn := _cgo_tsan_release_fn; fn != 0 {
		runtime_asmcgocall(*(*unsafe.Pointer)(unsafe.Pointer(&fn)), nil)
	}
}

//go:nosplit
//go:norace
func tsanReleaseNoG() {
	if fn := _cgo_tsan_release_fn; fn != 0 {
		runtime_asmcgocall_no_g(*(*unsafe.Pointer)(unsafe.Pointer(&fn)), nil)
	}
}

//go:nosplit
//go:norace
func msanUnpoison(addr unsafe.Pointer, size uintptr) {
	if _cgo_msan_unpoison_fn != 0 {
		asmcgocall(_cgo_msan_unpoison_fn, uintptr(addr), size, 0, 0, 0, 0)
	}
}

//go:nosplit
//go:norace
func msanUnpoisonNoG(addr unsafe.Pointer, size uintptr) {
	if _cgo_msan_unpoison_fn != 0 {
		asmcgocallNoG(_cgo_msan_unpoison_fn, uintptr(addr), size, 0, 0, 0, 0)
	}
}

//go:nosplit
//go:norace
func msanUnpoisonMode(noG bool, addr unsafe.Pointer, size uintptr) {
	if _cgo_msan_unpoison_fn != 0 {
		if noG {
			asmcgocallNoG(_cgo_msan_unpoison_fn, uintptr(addr), size, 0, 0, 0, 0)
		} else {
			asmcgocall(_cgo_msan_unpoison_fn, uintptr(addr), size, 0, 0, 0, 0)
		}
	}
}
