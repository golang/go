// Copyright 2026 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

//go:build goexperiment.runtimecgopurego && unix

package cgo

import (
	"internal/goos"
	"internal/runtime/atomic"
	"internal/runtime/sys"
	"runtime"
	"unsafe"
)

// Export TSAN hook pointers and _cgo_sync statically so cmd/cgo's constructor
// in _cgo_export.c can populate them when C code is compiled with
// -fsanitize=thread.
//
//go:cgo_export_static _cgo_tsan_acquire_fn
//go:cgo_export_static _cgo_tsan_release_fn
//go:cgo_export_static _cgo_sync
//go:cgo_export_static _cgo_yield

//go:linkname tsanAcquireFn _cgo_tsan_acquire_fn
//go:linkname tsanReleaseFn _cgo_tsan_release_fn
//go:linkname cgoSync _cgo_sync
//go:linkname cgoYield _cgo_yield
var (
	tsanAcquireFn uintptr
	tsanReleaseFn uintptr
	cgoSync       int64
	cgoYield      uintptr
)

// Access the statically exported TSAN variables through non-exported pointers
// so .text does not reference global data symbols directly (which fails under
// -buildmode=c-shared -Wl,-Bsymbolic-functions).
var (
	tsanAcquireFnPtr = &tsanAcquireFn
	tsanReleaseFnPtr = &tsanReleaseFn
	cgoSyncPtr       = &cgoSync
	cgoYieldPtr      = &cgoYield
)

// tsanCallSlow, tsanCallNoGSlow, and tsanCallModeSlow are out-of-line slow
// paths so the fast-path nil checks in tsanAcquire/tsanRelease/tsanCall inline.
//
//go:noinline
//go:nosplit
//go:norace
//go:nocheckptr
func tsanCallSlow(fn uintptr) {
	runtime_asmcgocall(*(*unsafe.Pointer)(unsafe.Pointer(&fn)), unsafe.Pointer(cgoSyncPtr))
}

//go:noinline
//go:nosplit
//go:norace
//go:nocheckptr
func tsanCallNoGSlow(fn uintptr) {
	runtime_asmcgocall_no_g(*(*unsafe.Pointer)(unsafe.Pointer(&fn)), unsafe.Pointer(cgoSyncPtr))
}

//go:nosplit
//go:norace
//go:nocheckptr
func tsanCallModeSlow(fn uintptr, noG bool) {
	if noG {
		runtime_asmcgocall_no_g(*(*unsafe.Pointer)(unsafe.Pointer(&fn)), unsafe.Pointer(cgoSyncPtr))
	} else {
		runtime_asmcgocall(*(*unsafe.Pointer)(unsafe.Pointer(&fn)), unsafe.Pointer(cgoSyncPtr))
	}
}

// tsanCall calls *fnp(&_cgo_sync), if set, on the system stack, or on the
// current stack if noG is set.
//
//go:nosplit
//go:norace
func tsanCall(fnp *uintptr, noG bool) {
	if fn := atomic.LoadAcquintptr(fnp); fn != 0 {
		tsanCallModeSlow(fn, noG)
	}
}

//go:nosplit
//go:norace
func tsanAcquire() {
	if fn := atomic.LoadAcquintptr(tsanAcquireFnPtr); fn != 0 {
		tsanCallSlow(fn)
	}
}

//go:nosplit
//go:norace
func tsanAcquireNoG() {
	if fn := atomic.LoadAcquintptr(tsanAcquireFnPtr); fn != 0 {
		tsanCallNoGSlow(fn)
	}
}

//go:nosplit
//go:norace
func tsanRelease() {
	if fn := atomic.LoadAcquintptr(tsanReleaseFnPtr); fn != 0 {
		tsanCallSlow(fn)
	}
}

//go:nosplit
//go:norace
func tsanReleaseNoG() {
	if fn := atomic.LoadAcquintptr(tsanReleaseFnPtr); fn != 0 {
		tsanCallNoGSlow(fn)
	}
}

//go:linkname _cgo_libc_setenv _cgo_libc_setenv
var _cgo_libc_setenv byte
var libc_setenv = unsafe.Pointer(&_cgo_libc_setenv)

//go:linkname _cgo_libc_unsetenv _cgo_libc_unsetenv
var _cgo_libc_unsetenv byte
var libc_unsetenv = unsafe.Pointer(&_cgo_libc_unsetenv)

//go:linkname _cgo_libc_pthread_create _cgo_libc_pthread_create
var _cgo_libc_pthread_create byte
var libc_pthread_create = unsafe.Pointer(&_cgo_libc_pthread_create)

//go:linkname _cgo_libc_pthread_attr_init _cgo_libc_pthread_attr_init
var _cgo_libc_pthread_attr_init byte
var libc_pthread_attr_init = unsafe.Pointer(&_cgo_libc_pthread_attr_init)

//go:linkname _cgo_libc_pthread_attr_setdetachstate _cgo_libc_pthread_attr_setdetachstate
var _cgo_libc_pthread_attr_setdetachstate byte
var libc_pthread_attr_setdetachstate = unsafe.Pointer(&_cgo_libc_pthread_attr_setdetachstate)

//go:linkname _cgo_libc_pthread_attr_getstacksize _cgo_libc_pthread_attr_getstacksize
var _cgo_libc_pthread_attr_getstacksize byte
var libc_pthread_attr_getstacksize = unsafe.Pointer(&_cgo_libc_pthread_attr_getstacksize)

//go:linkname _cgo_libc_pthread_attr_setstacksize _cgo_libc_pthread_attr_setstacksize
var _cgo_libc_pthread_attr_setstacksize byte
var libc_pthread_attr_setstacksize = unsafe.Pointer(&_cgo_libc_pthread_attr_setstacksize)

//go:linkname _cgo_libc_pthread_attr_getstack _cgo_libc_pthread_attr_getstack
var _cgo_libc_pthread_attr_getstack byte
var libc_pthread_attr_getstack = unsafe.Pointer(&_cgo_libc_pthread_attr_getstack)

//go:linkname _cgo_libc_pthread_attr_destroy _cgo_libc_pthread_attr_destroy
var _cgo_libc_pthread_attr_destroy byte
var libc_pthread_attr_destroy = unsafe.Pointer(&_cgo_libc_pthread_attr_destroy)

//go:linkname _cgo_libc_pthread_key_create _cgo_libc_pthread_key_create
var _cgo_libc_pthread_key_create byte
var libc_pthread_key_create = unsafe.Pointer(&_cgo_libc_pthread_key_create)

//go:linkname _cgo_libc_pthread_setspecific _cgo_libc_pthread_setspecific
var _cgo_libc_pthread_setspecific byte
var libc_pthread_setspecific = unsafe.Pointer(&_cgo_libc_pthread_setspecific)

//go:linkname _cgo_libc_pthread_self _cgo_libc_pthread_self
var _cgo_libc_pthread_self byte
var libc_pthread_self = unsafe.Pointer(&_cgo_libc_pthread_self)

//go:linkname _cgo_libc_pthread_get_stackaddr_np _cgo_libc_pthread_get_stackaddr_np
var _cgo_libc_pthread_get_stackaddr_np byte
var libc_pthread_get_stackaddr_np = unsafe.Pointer(&_cgo_libc_pthread_get_stackaddr_np)

//go:linkname _cgo_libc_pthread_get_stacksize_np _cgo_libc_pthread_get_stacksize_np
var _cgo_libc_pthread_get_stacksize_np byte
var libc_pthread_get_stacksize_np = unsafe.Pointer(&_cgo_libc_pthread_get_stacksize_np)

//go:linkname _cgo_libc_pthread_getattr_np _cgo_libc_pthread_getattr_np
var _cgo_libc_pthread_getattr_np byte
var libc_pthread_getattr_np = unsafe.Pointer(&_cgo_libc_pthread_getattr_np)

//go:linkname _cgo_libc_pthread_attr_get_np _cgo_libc_pthread_attr_get_np
var _cgo_libc_pthread_attr_get_np byte
var libc_pthread_attr_get_np = unsafe.Pointer(&_cgo_libc_pthread_attr_get_np)

//go:linkname _cgo_libc_pthread_sigmask _cgo_libc_pthread_sigmask
var _cgo_libc_pthread_sigmask byte
var libc_pthread_sigmask = unsafe.Pointer(&_cgo_libc_pthread_sigmask)

//go:linkname _cgo_libc_sigfillset _cgo_libc_sigfillset
var _cgo_libc_sigfillset byte
var libc_sigfillset = unsafe.Pointer(&_cgo_libc_sigfillset)

//go:linkname _cgo_libc_sigaltstack _cgo_libc_sigaltstack
var _cgo_libc_sigaltstack byte
var libc_sigaltstack = unsafe.Pointer(&_cgo_libc_sigaltstack)

// libc_dlopen, libc_dlsym, and libc_dlerror are imported for
// Open only when the cgobindings experiment is enabled (see library_*.go),
// except on Android (TLS initialization) and Darwin (TSAN libInit detection).

//go:linkname _cgo_libc_dlopen _cgo_libc_dlopen
var _cgo_libc_dlopen byte
var libc_dlopen = unsafe.Pointer(&_cgo_libc_dlopen)

//go:linkname _cgo_libc_dlsym _cgo_libc_dlsym
var _cgo_libc_dlsym byte
var libc_dlsym = unsafe.Pointer(&_cgo_libc_dlsym)

//go:linkname _cgo_libc_dlerror _cgo_libc_dlerror
var _cgo_libc_dlerror byte
var libc_dlerror = unsafe.Pointer(&_cgo_libc_dlerror)

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

//go:linknamestd _cgoSetenv runtime.cgoSetenv
var _cgoSetenv = cgoSetenv

//go:norace
func cgoSetenv(k, v unsafe.Pointer) {
	if acq := atomic.LoadAcquintptr(tsanAcquireFnPtr); acq != 0 {
		runtime.LockOSThread()
		tsanCallSlow(acq)
		asmcgocall(uintptr(libc_setenv), uintptr(k), uintptr(v), 1, 0, 0, 0)
		tsanRelease()
		runtime.UnlockOSThread()
		return
	}
	asmcgocall(uintptr(libc_setenv), uintptr(k), uintptr(v), 1, 0, 0, 0)
}

//go:linknamestd _cgoUnsetenv runtime.cgoUnsetenv
var _cgoUnsetenv = cgoUnsetenv

//go:norace
func cgoUnsetenv(k unsafe.Pointer) {
	if acq := atomic.LoadAcquintptr(tsanAcquireFnPtr); acq != 0 {
		runtime.LockOSThread()
		tsanCallSlow(acq)
		asmcgocall(uintptr(libc_unsetenv), uintptr(k), 0, 0, 0, 0, 0)
		tsanRelease()
		runtime.UnlockOSThread()
		return
	}
	asmcgocall(uintptr(libc_unsetenv), uintptr(k), 0, 0, 0, 0, 0)
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
//go:nocheckptr
func cgoThreadStackSize(attr *pthreadAttr) uintptr {
	if goos.IsDarwin|goos.IsIos != 0 {
		// Copy stack size from parent thread instead of using the
		// non-main thread default stack size.
		self := asmcgocall(uintptr(libc_pthread_self), 0, 0, 0, 0, 0, 0)
		size := asmcgocall(uintptr(libc_pthread_get_stacksize_np), self, 0, 0, 0, 0, 0)
		asmcgocall(uintptr(libc_pthread_attr_setstacksize), uintptr(unsafe.Pointer(attr)), size, 0, 0, 0, 0)
		return size
	}
	var size uintptr
	asmcgocall(uintptr(libc_pthread_attr_getstacksize), uintptr(unsafe.Pointer(attr)), uintptr(unsafe.Pointer(&size)), 0, 0, 0, 0)
	if goos.IsSolaris|goos.IsIllumos != 0 && size == 0 {
		// Solaris can report 0 stack size; fix it.
		size = 2 << 20
		if int32(asmcgocall(uintptr(libc_pthread_attr_setstacksize), uintptr(unsafe.Pointer(attr)), size, 0, 0, 0, 0)) != 0 {
			runtime_writeErrStr("runtime/cgo: pthread_attr_setstacksize failed\n")
			runtime_abort()
		}
	}
	return size
}

// cgoGetStackBound returns the current OS thread's stack bounds [lo, hi).
// It must be called while pinned to the M (or on a system/C stack) so
// pthread_self refers to a fixed OS thread.
//
//go:nosplit
//go:norace
//go:nocheckptr
func cgoGetStackBound() (uintptr, uintptr) {
	switch {
	case goos.IsDarwin|goos.IsIos != 0:
		tsanAcquireNoG()
		self := asmcgocallNoG(uintptr(libc_pthread_self), 0, 0, 0, 0, 0, 0)
		high := asmcgocallNoG(uintptr(libc_pthread_get_stackaddr_np), self, 0, 0, 0, 0, 0)
		size := asmcgocallNoG(uintptr(libc_pthread_get_stacksize_np), self, 0, 0, 0, 0, 0)
		tsanReleaseNoG()
		return high - size, high

	case goos.IsLinux|goos.IsAndroid|goos.IsSolaris|goos.IsIllumos != 0:
		var attr pthreadAttr
		var addr, size uintptr
		// Use if/else rather than overwriting a default so that
		// -gcflags=-N does not keep a reference to the function
		// missing on the other OS.
		var getattr unsafe.Pointer
		if goos.IsIllumos != 0 {
			getattr = libc_pthread_attr_get_np
		} else {
			getattr = libc_pthread_getattr_np
		}
		tsanAcquireNoG()
		// Needed before pthread_getattr_np, too, since before glibc 2.32
		// it did not call pthread_attr_init in all cases (see #65625).
		asmcgocallNoG(uintptr(libc_pthread_attr_init), uintptr(unsafe.Pointer(&attr)), 0, 0, 0, 0, 0)
		self := asmcgocallNoG(uintptr(libc_pthread_self), 0, 0, 0, 0, 0, 0)
		if int32(asmcgocallNoG(uintptr(getattr), self, uintptr(unsafe.Pointer(&attr)), 0, 0, 0, 0)) == 0 {
			asmcgocallNoG(uintptr(libc_pthread_attr_getstack), uintptr(unsafe.Pointer(&attr)), uintptr(unsafe.Pointer(&addr)), uintptr(unsafe.Pointer(&size)), 0, 0, 0)
		}
		asmcgocallNoG(uintptr(libc_pthread_attr_destroy), uintptr(unsafe.Pointer(&attr)), 0, 0, 0, 0, 0)
		tsanReleaseNoG()
		return addr, addr + size

	default:
		return 0, 0
	}
}

//go:nosplit
//go:norace
func cgoYieldWaitRuntimeInit() {
	runtime_cgoUsleep(100) // 100us
}

//go:norace
//go:nocheckptr
func cgoNotifyRuntimeInitDoneOS() {
	var r, keyVal uintptr
	if goos.IsDarwin|goos.IsIos != 0 {
		var key uintptr
		tsanAcquire()
		r = asmcgocall(uintptr(libc_pthread_key_create),
			uintptr(unsafe.Pointer(&key)),
			pthreadKeyDestructorPC(), 0, 0, 0, 0)
		tsanRelease()
		keyVal = key
	} else {
		var key uint32
		tsanAcquire()
		r = asmcgocall(uintptr(libc_pthread_key_create),
			uintptr(unsafe.Pointer(&key)),
			pthreadKeyDestructorPC(), 0, 0, 0, 0)
		tsanRelease()
		keyVal = uintptr(key)
	}
	if int32(r) == 0 {
		atomic.Storeuintptr(&pthread_g, keyVal)
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
func cgoPthreadCreate(thread *uintptr, attr *pthreadAttr, pfn, arg uintptr) int32 {
	return int32(asmcgocall(uintptr(libc_pthread_create),
		uintptr(unsafe.Pointer(thread)),
		uintptr(unsafe.Pointer(attr)),
		pfn, arg, 0, 0))
}

// cgoPthreadCreateNoG is like cgoPthreadCreate, but for callers without a g.
//
//go:nosplit
//go:norace
//go:nocheckptr
func cgoPthreadCreateNoG(thread *uintptr, attr *pthreadAttr, pfn, arg uintptr) int32 {
	return int32(asmcgocallNoG(uintptr(libc_pthread_create),
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

	err := cgoTryPthreadCreate(&thread, &attr, threadentryPC(), uintptr(unsafe.Pointer(gp)))
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
	if goos.IsNetbsd != 0 {
		// On NetBSD, a new thread inherits the signal stack of the
		// creating thread. Remove that signal stack before mstart so
		// minit can install the new thread's own signal stack.
		// Do this before setting g: the g0 stack bounds are not
		// set up yet, so we cannot switch stacks with asmcgocall.
		const _SS_DISABLE = 4
		ss := struct {
			sp    uintptr
			size  uintptr
			flags int32
		}{flags: _SS_DISABLE}
		asmcgocallNoG(uintptr(libc_sigaltstack), uintptr(unsafe.Pointer(&ss)), 0, 0, 0, 0, 0)
	}
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
	err := cgoTryPthreadCreateNoG(&p, &attr, uintptr(fn), 0)
	asmcgocallNoG(uintptr(libc_pthread_attr_destroy), uintptr(unsafe.Pointer(&attr)), 0, 0, 0, 0, 0)
	return err
}

//go:nosplit
//go:norace
//go:nocheckptr
func cgoTryPthreadCreate(thread *uintptr, attr *pthreadAttr, pfn, arg uintptr) int32 {
	for tries := uint32(0); tries < 20; tries++ {
		err := cgoPthreadCreate(thread, attr, pfn, arg)
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

// cgoTryPthreadCreateNoG is like cgoTryPthreadCreate, but for callers
// without a g. It is separate, rather than taking a flag, so that those
// callers do not statically reach asmcgocall: the linker's nosplit check
// would count it, as dead branches remain with -gcflags=-N.
//
//go:nosplit
//go:norace
//go:nocheckptr
func cgoTryPthreadCreateNoG(thread *uintptr, attr *pthreadAttr, pfn, arg uintptr) int32 {
	for tries := uint32(0); tries < 20; tries++ {
		err := cgoPthreadCreateNoG(thread, attr, pfn, arg)
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
