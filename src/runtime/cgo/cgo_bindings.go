// Copyright 2026 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

//go:build goexperiment.runtimecgopurego && !msan && !tsan

package cgo

import (
	"internal/abi"
	"internal/cpu"
	"internal/runtime/atomic"
	"unsafe"
)

const offsetX86HasAVX = unsafe.Offsetof(cpu.X86.HasAVX)

//go:cgo_export_static _cgo_wait_runtime_init_done
//go:cgo_export_dynamic _cgo_wait_runtime_init_done
//go:cgo_export_static _cgo_release_context
//go:cgo_export_dynamic _cgo_release_context

var pthread_key_created uintptr

//go:linkname _cgo_pthread_key_created _cgo_pthread_key_created
var _cgo_pthread_key_created = &pthread_key_created

// x_crosscall2_ptr and _crosscall2_ptr are vestigial on Go-backed runtime/cgo
// targets (nothing reads x_crosscall2_ptr), but runtime.main still calls
// runtime.set_crosscall2, which writes through _crosscall2_ptr and is kept for
// external linkname compatibility (for example, github.com/ebitengine/purego).
var x_crosscall2_ptr uintptr

//go:linkname _crosscall2_ptr _crosscall2_ptr
var _crosscall2_ptr = &x_crosscall2_ptr

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
	f := cgoCallFrame{fn: fn, a0: a0, a1: a1, a2: a2, a3: a3, a4: a4, a5: a5}
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
	f := cgoCallFrame{fn: fn, a0: a0, a1: a1, a2: a2, a3: a3, a4: a4, a5: a5}
	runtime_asmcgocall_no_g(unsafe.Pointer(abi.FuncPCABI0(cgoCallTrampoline)), unsafe.Pointer(&f))
	return f.r1
}

// cgocall calls a C function pointer fn with up to 6 integer/pointer
// arguments via runtime.cgocall, entering and exiting the scheduler.
//
//go:norace
//go:nocheckptr
func cgocall(fn, a0, a1, a2, a3, a4, a5 uintptr) uintptr {
	f := cgoCallFrame{fn: fn, a0: a0, a1: a1, a2: a2, a3: a3, a4: a4, a5: a5}
	runtime_cgocall(unsafe.Pointer(abi.FuncPCABI0(cgoCallTrampoline)), unsafe.Pointer(&f))
	return f.r1
}

//go:nosplit
//go:norace
func tsanAcquire() {}

//go:nosplit
//go:norace
func tsanAcquireNoG() {}

//go:nosplit
//go:norace
func tsanRelease() {}

//go:nosplit
//go:norace
func tsanReleaseNoG() {}

//go:nosplit
//go:norace
func msanUnpoison(addr unsafe.Pointer, size uintptr) {}

//go:nosplit
//go:norace
func msanUnpoisonNoG(addr unsafe.Pointer, size uintptr) {}

//go:nosplit
//go:norace
func msanUnpoisonMode(noG bool, addr unsafe.Pointer, size uintptr) {}

type cgoContextArg struct {
	context uintptr
}

var (
	runtime_init_done    uint32
	cgo_context_function uintptr
)

//go:nosplit
//go:norace
//go:nocheckptr
func cgoWaitRuntimeInitDone() uintptr {
	if atomic.Load(&runtime_init_done) != 2 {
		for atomic.Load(&runtime_init_done) == 0 {
			cgoYieldWaitRuntimeInit()
		}
		atomic.Store(&runtime_init_done, 2)
	}
	// TODO(iant): For the case of a new C thread calling into Go, such
	// as when using -buildmode=c-archive, we know that Go runtime
	// initialization is complete but we do not know that all Go init
	// functions have been run. We should not fetch cgo_context_function
	// until they have been, because that is where a call to
	// SetCgoTraceback is likely to occur. We are going to wait for Go
	// initialization to be complete anyhow, later, by waiting for
	// main_init_done to be closed in cgocallbackg1. We should wait here
	// instead. See also issue #15943.
	if pfn := atomic.Loaduintptr(&cgo_context_function); pfn != 0 {
		var arg cgoContextArg
		msanUnpoisonNoG(unsafe.Pointer(&arg), unsafe.Sizeof(arg))
		asmcgocallNoG(pfn, uintptr(unsafe.Pointer(&arg)), 0, 0, 0, 0, 0)
		return arg.context
	}
	return 0
}

//go:linknamestd _cgoNotifyRuntimeInitDone runtime.cgoNotifyRuntimeInitDone
var _cgoNotifyRuntimeInitDone = cgoNotifyRuntimeInitDone

//go:norace
//go:nocheckptr
func cgoNotifyRuntimeInitDone() {
	cgoNotifyRuntimeInitDoneOS()
	atomic.Store(&runtime_init_done, 1)
}

//go:nosplit
//go:norace
//go:nocheckptr
func cgoReleaseContext(ctxt uintptr) {
	if ctxt == 0 {
		return
	}
	if pfn := atomic.Loaduintptr(&cgo_context_function); pfn != 0 {
		arg := cgoContextArg{context: ctxt}
		msanUnpoisonNoG(unsafe.Pointer(&arg), unsafe.Sizeof(arg))
		asmcgocallNoG(pfn, uintptr(unsafe.Pointer(&arg)), 0, 0, 0, 0, 0)
	}
}

//go:linknamestd _cgoSetContextFunction runtime.cgoSetContextFunction
var _cgoSetContextFunction = cgoSetContextFunction

//go:norace
func cgoSetContextFunction(context uintptr) {
	atomic.Storeuintptr(&cgo_context_function, context)
}
