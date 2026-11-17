// Copyright 2026 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

//go:build goexperiment.cgobindings && (darwin || (linux && !android && (amd64 || arm64)))

package cgo

import (
	"internal/abi"
	"internal/cpu"
	"internal/runtime/atomic"
	"internal/runtime/sys"
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

//go:linkname _libc_pthread_key_create libc_pthread_key_create
var _libc_pthread_key_create byte
var libc_pthread_key_create = unsafe.Pointer(&_libc_pthread_key_create)

//go:linkname _libc_pthread_setspecific libc_pthread_setspecific
var _libc_pthread_setspecific byte
var libc_pthread_setspecific = unsafe.Pointer(&_libc_pthread_setspecific)

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

// pthreadKeyDestructor and x_cgo_threadentry are C-callable assembly entrypoints.
func pthreadKeyDestructor()
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

type cgoContextArg struct {
	context uintptr
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

var (
	runtime_init_done    uint32
	pthread_g            uintptr
	fullSigset           sigset
	cgo_context_function uintptr
)

//go:nosplit
//go:norace
//go:nocheckptr
func cgoWaitRuntimeInitDone() uintptr {
	if atomic.Load(&runtime_init_done) != 2 {
		for atomic.Load(&runtime_init_done) == 0 {
			runtime_cgoUsleep(100) // 100us
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
	tsanAcquire()
	r := asmcgocall(uintptr(libc_pthread_key_create),
		uintptr(unsafe.Pointer(&pthread_g)),
		abi.FuncPCABI0(pthreadKeyDestructor), 0, 0, 0, 0)
	tsanRelease()
	if int32(r) == 0 {
		atomic.Storeuintptr(&pthread_key_created, 1)
	}
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

//go:linknamestd _cgoBindM runtime.cgoBindM
var _cgoBindM = cgoBindM

//go:nosplit
//go:norace
func cgoBindM(g uintptr) {
	asmcgocall(uintptr(libc_pthread_setspecific), atomic.Loaduintptr(&pthread_g), g, 0, 0, 0, 0)
}

//go:linknamestd _cgoSetContextFunction runtime.cgoSetContextFunction
var _cgoSetContextFunction = cgoSetContextFunction

//go:norace
func cgoSetContextFunction(context uintptr) {
	atomic.Storeuintptr(&cgo_context_function, context)
}

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
	asmcgocall(uintptr(libc_pthread_sigmask), _SIG_SETMASK, uintptr(unsafe.Pointer(&fullSigset)), uintptr(unsafe.Pointer(oset)), 0, 0, 0)
}

//go:nosplit
//go:norace
//go:nocheckptr
func cgoRestoreSigs(oset *sigset) {
	asmcgocall(uintptr(libc_pthread_sigmask), _SIG_SETMASK, uintptr(unsafe.Pointer(oset)), 0, 0, 0, 0)
}

//go:nosplit
//go:norace
//go:nocheckptr
func cgoInitThreadAttr(attr *pthreadAttr) int32 {
	asmcgocall(uintptr(libc_pthread_attr_init), uintptr(unsafe.Pointer(attr)), 0, 0, 0, 0, 0)
	return int32(asmcgocall(uintptr(libc_pthread_attr_setdetachstate), uintptr(unsafe.Pointer(attr)), _PTHREAD_CREATE_DETACHED, 0, 0, 0, 0))
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
	if err := int32(asmcgocallNoG(uintptr(libc_pthread_attr_setdetachstate), uintptr(unsafe.Pointer(&attr)), _PTHREAD_CREATE_DETACHED, 0, 0, 0, 0)); err != 0 {
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
