// Copyright 2026 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

//go:build goexperiment.runtimecgopurego && !msan && !tsan && (darwin || (linux && (amd64 || arm64)))

package cgo

import (
	"internal/abi"
	"internal/runtime/atomic"
	"internal/runtime/sys"
	"unsafe"
)

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

// libc_dlopen, libc_dlsym, and libc_dlerror are imported only when the
// cgobindings experiment is enabled (see library_linux.go and
// library_darwin.go), for Open.

//go:linkname _libc_dlopen libc_dlopen
var _libc_dlopen byte
var libc_dlopen = unsafe.Pointer(&_libc_dlopen)

//go:linkname _libc_dlsym libc_dlsym
var _libc_dlsym byte
var libc_dlsym = unsafe.Pointer(&_libc_dlsym)

//go:linkname _libc_dlerror libc_dlerror
var _libc_dlerror byte
var libc_dlerror = unsafe.Pointer(&_libc_dlerror)

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
	data [9]uintptr
}

type sigset struct {
	mask [128 / unsafe.Sizeof(uintptr(0))]uintptr
}

var (
	pthread_g  uintptr
	fullSigset sigset
)

//go:nosplit
//go:norace
func cgoYieldWaitRuntimeInit() {
	runtime_cgoUsleep(100) // 100us
}

//go:norace
//go:nocheckptr
func cgoNotifyRuntimeInitDoneOS() {
	var key pthreadKey
	tsanAcquire()
	r := asmcgocall(uintptr(libc_pthread_key_create),
		uintptr(unsafe.Pointer(&key)),
		abi.FuncPCABI0(pthreadKeyDestructor), 0, 0, 0, 0)
	tsanRelease()
	if int32(r) == 0 {
		atomic.Storeuintptr(&pthread_g, uintptr(key))
		atomic.Storeuintptr(&pthread_key_created, 1)
	}
}

//go:linknamestd _cgoBindM runtime.cgoBindM
var _cgoBindM = cgoBindM

//go:nosplit
//go:norace
func cgoBindM(g uintptr) {
	asmcgocall(uintptr(libc_pthread_setspecific), atomic.Loaduintptr(&pthread_g), g, 0, 0, 0, 0)
}

//go:nosplit
//go:norace
//go:nocheckptr
func cgoInit(g *g, _ uintptr, tlsg, tlsbase uintptr) {
	if tlsg != 0 {
		cgoInitTLS(tlsg, tlsbase)
	}
	asmcgocallNoG(uintptr(libc_sigfillset), uintptr(unsafe.Pointer(&fullSigset)), 0, 0, 0, 0, 0)
	lo, _ := cgoGetStackBound()
	g.stacklo = lo
	if g.stacklo >= g.stackhi {
		cgoFatal("runtime/cgo: bad stack bounds\n\x00")
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
