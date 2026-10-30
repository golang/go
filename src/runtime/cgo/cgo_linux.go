// Copyright 2026 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

//go:build goexperiment.cgobindings && linux && !android && (amd64 || arm64)

package cgo

import (
	"internal/abi"
	"unsafe"
)

const (
	sigSetMask            = 2
	_EAGAIN               = 11
	pthreadCreateDetached = 1
)

//go:linkname _libc_pthread_attr_getstacksize libc_pthread_attr_getstacksize
var _libc_pthread_attr_getstacksize byte
var libc_pthread_attr_getstacksize = unsafe.Pointer(&_libc_pthread_attr_getstacksize)

//go:linkname _libc_pthread_getattr_np libc_pthread_getattr_np
var _libc_pthread_getattr_np byte
var libc_pthread_getattr_np = unsafe.Pointer(&_libc_pthread_getattr_np)

//go:linkname _libc_pthread_attr_getstack libc_pthread_attr_getstack
var _libc_pthread_attr_getstack byte
var libc_pthread_attr_getstack = unsafe.Pointer(&_libc_pthread_attr_getstack)

//go:linkname _libc_mmap libc_mmap
var _libc_mmap byte
var libc_mmap = unsafe.Pointer(&_libc_mmap)

//go:linkname _libc_munmap libc_munmap
var _libc_munmap byte
var libc_munmap = unsafe.Pointer(&_libc_munmap)

//go:linkname _libc_sigaction libc_sigaction
var _libc_sigaction byte
var libc_sigaction = unsafe.Pointer(&_libc_sigaction)

//go:linkname _libc_errno_location libc_errno_location
var _libc_errno_location byte
var libc_errno_location = unsafe.Pointer(&_libc_errno_location)

//go:linkname _libc_clearenv libc_clearenv
var _libc_clearenv byte
var libc_clearenv = unsafe.Pointer(&_libc_clearenv)

//go:linkname _libc_setegid libc_setegid
//go:linkname cgo_libc_setegid syscall.cgo_libc_setegid
var _libc_setegid byte
var cgo_libc_setegid = unsafe.Pointer(&_libc_setegid)

//go:linkname _libc_seteuid libc_seteuid
//go:linkname cgo_libc_seteuid syscall.cgo_libc_seteuid
var _libc_seteuid byte
var cgo_libc_seteuid = unsafe.Pointer(&_libc_seteuid)

//go:linkname _libc_setgid libc_setgid
//go:linkname cgo_libc_setgid syscall.cgo_libc_setgid
var _libc_setgid byte
var cgo_libc_setgid = unsafe.Pointer(&_libc_setgid)

//go:linkname _libc_setgroups libc_setgroups
//go:linkname cgo_libc_setgroups syscall.cgo_libc_setgroups
var _libc_setgroups byte
var cgo_libc_setgroups = unsafe.Pointer(&_libc_setgroups)

//go:linkname _libc_setregid libc_setregid
//go:linkname cgo_libc_setregid syscall.cgo_libc_setregid
var _libc_setregid byte
var cgo_libc_setregid = unsafe.Pointer(&_libc_setregid)

//go:linkname _libc_setresgid libc_setresgid
//go:linkname cgo_libc_setresgid syscall.cgo_libc_setresgid
var _libc_setresgid byte
var cgo_libc_setresgid = unsafe.Pointer(&_libc_setresgid)

//go:linkname _libc_setresuid libc_setresuid
//go:linkname cgo_libc_setresuid syscall.cgo_libc_setresuid
var _libc_setresuid byte
var cgo_libc_setresuid = unsafe.Pointer(&_libc_setresuid)

//go:linkname _libc_setreuid libc_setreuid
//go:linkname cgo_libc_setreuid syscall.cgo_libc_setreuid
var _libc_setreuid byte
var cgo_libc_setreuid = unsafe.Pointer(&_libc_setreuid)

//go:linkname _libc_setuid libc_setuid
//go:linkname cgo_libc_setuid syscall.cgo_libc_setuid
var _libc_setuid byte
var cgo_libc_setuid = unsafe.Pointer(&_libc_setuid)

//go:linknamestd _cgoMmap runtime.cgoMmap
var _cgoMmap = cgoMmap

//go:nosplit
//go:norace
//go:nocheckptr
func cgoMmap(addr unsafe.Pointer, length uintptr, prot, flags, fd int32, offset uint32) uintptr {
	tsanAcquire()
	p := asmcgocall(uintptr(libc_mmap), uintptr(addr), length, uintptr(uint32(prot)), uintptr(uint32(flags)), uintptr(fd), uintptr(offset))
	if int(p) == -1 {
		errnoPtr := asmcgocall(uintptr(libc_errno_location), 0, 0, 0, 0, 0, 0)
		errno := uintptr(*(*int32)(unsafe.Pointer(errnoPtr)))
		tsanRelease()
		return errno
	}
	tsanRelease()
	return p
}

//go:linknamestd _cgoMunmap runtime.cgoMunmap
var _cgoMunmap = cgoMunmap

//go:nosplit
//go:norace
func cgoMunmap(addr unsafe.Pointer, length uintptr) int32 {
	tsanAcquire()
	r := int32(asmcgocall(uintptr(libc_munmap), uintptr(addr), length, 0, 0, 0, 0))
	tsanRelease()
	return r
}

// goSigaction matches runtime.sigactiont on linux/amd64 and linux/arm64.
type goSigaction struct {
	handler  uintptr
	flags    uint64
	restorer uintptr
	mask     uint64
}

// cSigaction matches libc's struct sigaction on 64-bit Linux (glibc and musl).
type cSigaction struct {
	handler  uintptr
	mask     [16]uintptr // sigset_t
	flags    int32
	restorer uintptr
}

//go:linknamestd _cgoSigaction runtime.cgoSigaction
var _cgoSigaction = cgoSigaction

//go:nosplit
//go:norace
func sigactionCall(noG bool, fn, a0, a1, a2 uintptr) uintptr {
	if noG {
		return asmcgocallNoG(fn, a0, a1, a2, 0, 0, 0)
	}
	return asmcgocall(fn, a0, a1, a2, 0, 0, 0)
}

//go:nosplit
//go:norace
//go:nocheckptr
func cgoSigaction(signum uintptr, goact, oldgoact *goSigaction, noG bool) int32 {
	// SA_RESTORER is part of the kernel interface.
	// This is Linux i386/amd64 specific.
	// See the comment on fixSigactionForCgo.
	const SA_RESTORER = 0x4000000

	if noG {
		tsanAcquireNoG()
	} else {
		tsanAcquire()
	}

	var oldact cSigaction
	var actPtr, oldactPtr uintptr

	if goact != nil {
		var act cSigaction
		act.handler = goact.handler
		sigactionCall(noG, uintptr(libc_sigfillset), uintptr(unsafe.Pointer(&act.mask)), 0, 0)
		act.mask[0] &= uintptr(goact.mask)
		act.flags = int32(goact.flags &^ SA_RESTORER)
		msanUnpoisonMode(noG, unsafe.Pointer(&act), unsafe.Sizeof(act))
		actPtr = uintptr(unsafe.Pointer(&act))
	}
	if oldgoact != nil {
		oldactPtr = uintptr(unsafe.Pointer(&oldact))
	}

	ret := int32(sigactionCall(noG, uintptr(libc_sigaction), uintptr(int32(signum)), actPtr, oldactPtr))
	if ret == -1 {
		// runtime.rt_sigaction expects _cgo_sigaction to return errno on error.
		errnoPtr := sigactionCall(noG, uintptr(libc_errno_location), 0, 0, 0)
		errno := *(*int32)(unsafe.Pointer(errnoPtr))
		if noG {
			tsanReleaseNoG()
		} else {
			tsanRelease()
		}
		return errno
	}

	if oldgoact != nil {
		oldgoact.handler = oldact.handler
		oldgoact.mask = uint64(oldact.mask[0])
		oldgoact.flags = uint64(int64(oldact.flags))
		msanUnpoisonMode(noG, unsafe.Pointer(oldgoact), unsafe.Sizeof(*oldgoact))
	}

	if noG {
		tsanReleaseNoG()
	} else {
		tsanRelease()
	}
	return ret
}

//go:linknamestd _cgoClearenv runtime.cgoClearenv
var _cgoClearenv = cgoClearenv

//go:norace
func cgoClearenv() {
	tsanAcquire()
	asmcgocall(uintptr(libc_clearenv), 0, 0, 0, 0, 0, 0)
	tsanRelease()
}

// cgoSyscallFrame holds the target libc syscall function, the errno location
// function, up to 3 arguments, and the result (0 on success, or errno on
// failure) for cgoSyscallTrampoline.
type cgoSyscallFrame struct {
	fn         uintptr
	a0, a1, a2 uintptr
	r1         uintptr
}

func cgoSyscallTrampoline()

//go:linknamestd _cgoSyscall runtime.cgoSyscall
var _cgoSyscall = cgoSyscall

//go:nosplit
//go:norace
//go:nocheckptr
func cgoSyscall(fn unsafe.Pointer, a0, a1, a2 uintptr) uintptr {
	var f cgoSyscallFrame
	f.fn = uintptr(fn)
	f.a0 = a0
	f.a1 = a1
	f.a2 = a2
	runtime_cgocall(unsafe.Pointer(abi.FuncPCABI0(cgoSyscallTrampoline)), unsafe.Pointer(&f))
	return f.r1
}

//go:nosplit
//go:norace
//go:nocheckptr
func cgoThreadStackSize(attr *pthreadAttr) uintptr {
	var size uintptr
	asmcgocall(uintptr(libc_pthread_attr_getstacksize), uintptr(unsafe.Pointer(attr)), uintptr(unsafe.Pointer(&size)), 0, 0, 0, 0)
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
	var attr pthreadAttr
	var addr, size uintptr

	// Needed before pthread_getattr_np, too, since before glibc 2.32
	// it did not call pthread_attr_init in all cases (see #65625).
	asmcgocallNoG(uintptr(libc_pthread_attr_init), uintptr(unsafe.Pointer(&attr)), 0, 0, 0, 0, 0)
	self := asmcgocallNoG(uintptr(libc_pthread_self), 0, 0, 0, 0, 0, 0)
	if int32(asmcgocallNoG(uintptr(libc_pthread_getattr_np), self, uintptr(unsafe.Pointer(&attr)), 0, 0, 0, 0)) == 0 {
		asmcgocallNoG(uintptr(libc_pthread_attr_getstack), uintptr(unsafe.Pointer(&attr)), uintptr(unsafe.Pointer(&addr)), uintptr(unsafe.Pointer(&size)), 0, 0, 0)
	}
	asmcgocallNoG(uintptr(libc_pthread_attr_destroy), uintptr(unsafe.Pointer(&attr)), 0, 0, 0, 0, 0)

	return addr, addr + size
}

//go:cgo_ldflag "-pthread"

// Specify "libc.so.6" so internal linking without C host objects emits a
// DT_NEEDED entry for glibc's dynamic loader; external linking ignores the
// library field, !android excludes bionic, and musl treats any "libc.so.*"
// SONAME as libc itself.
//go:cgo_import_dynamic _ _ "libpthread.so.0"
//go:cgo_import_dynamic libc_mmap mmap "libc.so.6"
//go:cgo_import_dynamic libc_munmap munmap "libc.so.6"
//go:cgo_import_dynamic libc_sigaction sigaction "libc.so.6"
//go:cgo_import_dynamic libc_sigfillset sigfillset "libc.so.6"
//go:cgo_import_dynamic libc_errno_location __errno_location "libc.so.6"
//go:cgo_import_dynamic libc_setenv setenv "libc.so.6"
//go:cgo_import_dynamic libc_unsetenv unsetenv "libc.so.6"
//go:cgo_import_dynamic libc_clearenv clearenv "libc.so.6"
//go:cgo_import_dynamic libc_pthread_create pthread_create "libpthread.so.0"
//go:cgo_import_dynamic libc_pthread_attr_init pthread_attr_init "libpthread.so.0"
//go:cgo_import_dynamic libc_pthread_attr_setdetachstate pthread_attr_setdetachstate "libpthread.so.0"
//go:cgo_import_dynamic libc_pthread_attr_getstacksize pthread_attr_getstacksize "libpthread.so.0"
//go:cgo_import_dynamic libc_pthread_attr_destroy pthread_attr_destroy "libpthread.so.0"
//go:cgo_import_dynamic libc_pthread_self pthread_self "libpthread.so.0"
//go:cgo_import_dynamic libc_pthread_getattr_np pthread_getattr_np "libpthread.so.0"
//go:cgo_import_dynamic libc_pthread_attr_getstack pthread_attr_getstack "libpthread.so.0"
//go:cgo_import_dynamic libc_pthread_sigmask pthread_sigmask "libpthread.so.0"
//go:cgo_import_dynamic libc_setegid setegid "libc.so.6"
//go:cgo_import_dynamic libc_seteuid seteuid "libc.so.6"
//go:cgo_import_dynamic libc_setgid setgid "libc.so.6"
//go:cgo_import_dynamic libc_setgroups setgroups "libc.so.6"
//go:cgo_import_dynamic libc_setregid setregid "libc.so.6"
//go:cgo_import_dynamic libc_setresgid setresgid "libc.so.6"
//go:cgo_import_dynamic libc_setresuid setresuid "libc.so.6"
//go:cgo_import_dynamic libc_setreuid setreuid "libc.so.6"
//go:cgo_import_dynamic libc_setuid setuid "libc.so.6"
