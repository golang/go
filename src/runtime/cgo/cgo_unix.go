// Copyright 2026 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

//go:build goexperiment.cgobindings && (darwin || (linux && !android && (amd64 || arm64)))

package cgo

import (
	"internal/abi"
	"internal/runtime/sys"
	"unsafe"
)

//go:cgo_import_static x_cgo_bindm
//go:linkname x_cgo_bindm x_cgo_bindm
//go:linkname _cgo_bindm _cgo_bindm
var x_cgo_bindm byte
var _cgo_bindm = &x_cgo_bindm

//go:linkname _cgo_tsan_acquire_fn _cgo_tsan_acquire_fn
var _cgo_tsan_acquire_fn uintptr

//go:linkname _cgo_tsan_release_fn _cgo_tsan_release_fn
var _cgo_tsan_release_fn uintptr

//go:linkname _cgo_msan_unpoison_fn _cgo_msan_unpoison_fn
var _cgo_msan_unpoison_fn uintptr

//go:linkname _libc_setenv libc_setenv
var _libc_setenv byte
var libc_setenv = unsafe.Pointer(&_libc_setenv)

//go:linkname _libc_unsetenv libc_unsetenv
var _libc_unsetenv byte
var libc_unsetenv = unsafe.Pointer(&_libc_unsetenv)

//go:linkname _libc_pthread_create libc_pthread_create
var _libc_pthread_create byte
var libc_pthread_create = unsafe.Pointer(&_libc_pthread_create)

//go:linkname _libc_pthread_attr_init libc_pthread_attr_init
var _libc_pthread_attr_init byte
var libc_pthread_attr_init = unsafe.Pointer(&_libc_pthread_attr_init)

//go:linkname _libc_pthread_attr_setdetachstate libc_pthread_attr_setdetachstate
var _libc_pthread_attr_setdetachstate byte
var libc_pthread_attr_setdetachstate = unsafe.Pointer(&_libc_pthread_attr_setdetachstate)

//go:linkname _libc_pthread_attr_destroy libc_pthread_attr_destroy
var _libc_pthread_attr_destroy byte
var libc_pthread_attr_destroy = unsafe.Pointer(&_libc_pthread_attr_destroy)

//go:linkname _libc_pthread_self libc_pthread_self
var _libc_pthread_self byte
var libc_pthread_self = unsafe.Pointer(&_libc_pthread_self)

//go:linkname _libc_pthread_sigmask libc_pthread_sigmask
var _libc_pthread_sigmask byte
var libc_pthread_sigmask = unsafe.Pointer(&_libc_pthread_sigmask)

//go:linkname _libc_sigfillset libc_sigfillset
var _libc_sigfillset byte
var libc_sigfillset = unsafe.Pointer(&_libc_sigfillset)

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

//go:linknamestd runtime_setg runtime.setg
func runtime_setg(gg uintptr)

//go:linknamestd runtime_mstart runtime.mstart
func runtime_mstart()

//go:linknamestd runtime_cgoUsleep runtime.cgoUsleep
func runtime_cgoUsleep(usec uint32)

//go:linknamestd runtime_writeErrStr runtime.writeErrStr
//go:noescape
func runtime_writeErrStr(s string)

//go:linkname runtime_abort runtime.abort
func runtime_abort()

// x_cgo_threadentry is a C-callable assembly entrypoint.
func x_cgo_threadentry()

//go:linknamestd _cgoSetenv runtime.cgoSetenv
var _cgoSetenv = cgoSetenv

//go:norace
func cgoSetenv(k, v unsafe.Pointer) {
	tsanAcquire()
	asmcgocall(uintptr(libc_setenv), uintptr(k), uintptr(v), 1, 0, 0, 0)
	tsanRelease()
}

//go:linknamestd _cgoUnsetenv runtime.cgoUnsetenv
var _cgoUnsetenv = cgoUnsetenv

//go:norace
func cgoUnsetenv(k unsafe.Pointer) {
	tsanAcquire()
	asmcgocall(uintptr(libc_unsetenv), uintptr(k), 0, 0, 0, 0, 0)
	tsanRelease()
}

//go:linknamestd _cgoGetStackBound runtime.cgoGetStackBound
var _cgoGetStackBound = cgoGetStackBound

// g matches the first few fields that we need from the runtime's g struct.
type g struct {
	_       sys.NotInHeap
	stacklo uintptr
	stackhi uintptr
}

// pthreadAttr corresponds to pthread_attr_t.
type pthreadAttr struct {
	data [8]uintptr
}

type sigset struct {
	mask [16]uintptr
}

var fullSigset sigset

//go:nosplit
//go:norace
//go:nocheckptr
func cgoInit(g *g, _ uintptr) {
	asmcgocallNoG(uintptr(libc_sigfillset), uintptr(unsafe.Pointer(&fullSigset)), 0, 0, 0, 0, 0)
	lo, _ := cgoGetStackBound()
	g.stacklo = lo
	if g.stacklo >= g.stackhi {
		runtime_writeErrStr("runtime/cgo: bad stack bounds\n")
		runtime_abort()
	}
}

//go:linknamestd _cgoThreadStart runtime.cgoThreadStart
var _cgoThreadStart = cgoThreadStart

//go:nosplit
//go:norace
//go:nocheckptr
func cgoBlockSigs(oset *sigset) {
	asmcgocall(uintptr(libc_pthread_sigmask), sigSetMask, uintptr(unsafe.Pointer(&fullSigset)), uintptr(unsafe.Pointer(oset)), 0, 0, 0)
}

//go:nosplit
//go:norace
//go:nocheckptr
func cgoRestoreSigs(oset *sigset) {
	asmcgocall(uintptr(libc_pthread_sigmask), sigSetMask, uintptr(unsafe.Pointer(oset)), 0, 0, 0, 0)
}

//go:nosplit
//go:norace
//go:nocheckptr
func cgoInitThreadAttr(attr *pthreadAttr) int32 {
	asmcgocall(uintptr(libc_pthread_attr_init), uintptr(unsafe.Pointer(attr)), 0, 0, 0, 0, 0)
	return int32(asmcgocall(uintptr(libc_pthread_attr_setdetachstate), uintptr(unsafe.Pointer(attr)), pthreadCreateDetached, 0, 0, 0, 0))
}

//go:nosplit
//go:norace
//go:nocheckptr
func cgoDestroyThreadAttr(attr *pthreadAttr) {
	asmcgocall(uintptr(libc_pthread_attr_destroy), uintptr(unsafe.Pointer(attr)), 0, 0, 0, 0, 0)
}

//go:nosplit
//go:norace
//go:nocheckptr
func cgoPthreadCreate(noG bool, thread *uintptr, attr *pthreadAttr, pfn, arg uintptr) int32 {
	if noG {
		return int32(asmcgocallNoG(uintptr(libc_pthread_create),
			uintptr(unsafe.Pointer(thread)),
			uintptr(unsafe.Pointer(attr)),
			pfn, arg, 0, 0))
	}
	return int32(asmcgocall(uintptr(libc_pthread_create),
		uintptr(unsafe.Pointer(thread)),
		uintptr(unsafe.Pointer(attr)),
		pfn, arg, 0, 0))
}

// cgoThreadStart is called by runtime.newm1 on the system stack.
//
//go:norace
//go:nocheckptr
func cgoThreadStart(gp *g) (int32, string) {
	var attr pthreadAttr
	var oset sigset
	var thread uintptr

	cgoBlockSigs(&oset)
	if err := cgoInitThreadAttr(&attr); err != 0 {
		cgoDestroyThreadAttr(&attr)
		cgoRestoreSigs(&oset)
		return err, "pthread_attr_setdetachstate failed"
	}
	gp.stackhi = cgoThreadStackSize(&attr)

	err := cgoTryPthreadCreate(false, &thread, &attr, abi.FuncPCABI0(x_cgo_threadentry), uintptr(unsafe.Pointer(gp)))
	cgoDestroyThreadAttr(&attr)
	cgoRestoreSigs(&oset)

	if err != 0 {
		return err, "pthread_create failed"
	}
	return 0, ""
}

//go:nosplit
//go:norace
//go:nocheckptr
func threadentry(v unsafe.Pointer) {
	runtime_setg(uintptr(v))
	runtime_mstart()
}

//go:linknamestd _cgoSysThreadCreate runtime.cgoSysThreadCreate
var _cgoSysThreadCreate = cgoSysThreadCreate

//go:nosplit
//go:norace
//go:nocheckptr
func cgoSysThreadCreate(fn unsafe.Pointer) int32 {
	var attr pthreadAttr
	var p uintptr

	asmcgocallNoG(uintptr(libc_pthread_attr_init), uintptr(unsafe.Pointer(&attr)), 0, 0, 0, 0, 0)
	if err := int32(asmcgocallNoG(uintptr(libc_pthread_attr_setdetachstate), uintptr(unsafe.Pointer(&attr)), pthreadCreateDetached, 0, 0, 0, 0)); err != 0 {
		asmcgocallNoG(uintptr(libc_pthread_attr_destroy), uintptr(unsafe.Pointer(&attr)), 0, 0, 0, 0, 0)
		return err
	}
	err := cgoTryPthreadCreate(true, &p, &attr, uintptr(fn), 0)
	asmcgocallNoG(uintptr(libc_pthread_attr_destroy), uintptr(unsafe.Pointer(&attr)), 0, 0, 0, 0, 0)
	return err
}

//go:nosplit
//go:norace
//go:nocheckptr
func cgoTryPthreadCreate(noG bool, thread *uintptr, attr *pthreadAttr, pfn, arg uintptr) int32 {
	for tries := uint32(0); tries < 20; tries++ {
		err := cgoPthreadCreate(noG, thread, attr, pfn, arg)
		if err == 0 {
			return 0
		}
		if err != _EAGAIN {
			return err
		}
		runtime_cgoUsleep((tries + 1) * 1000) // Milliseconds.
	}
	return _EAGAIN
}
