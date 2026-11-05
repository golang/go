// Copyright 2026 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

//go:build goexperiment.runtimecgopurego && !msan && !tsan && linux && (amd64 || arm64)

package cgo

import (
	"internal/goarch"
	"unsafe"
)

const (
	// SIG_SETMASK is 3 on MIPS Linux and 2 on all other Linux architectures.
	_SIG_SETMASK             = 2 + (goarch.IsMips | goarch.IsMipsle | goarch.IsMips64 | goarch.IsMips64le)
	_EAGAIN                  = 11
	_PTHREAD_CREATE_DETACHED = 1
)

type pthreadKey uint32

//go:linkname _libc_pthread_attr_getstacksize libc_pthread_attr_getstacksize
var _libc_pthread_attr_getstacksize byte
var libc_pthread_attr_getstacksize = unsafe.Pointer(&_libc_pthread_attr_getstacksize)

//go:linkname _libc_pthread_getattr_np libc_pthread_getattr_np
var _libc_pthread_getattr_np byte
var libc_pthread_getattr_np = unsafe.Pointer(&_libc_pthread_getattr_np)

//go:linkname _libc_pthread_attr_getstack libc_pthread_attr_getstack
var _libc_pthread_attr_getstack byte
var libc_pthread_attr_getstack = unsafe.Pointer(&_libc_pthread_attr_getstack)

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

//go:linknamestd _cgoClearenv runtime.cgoClearenv
var _cgoClearenv = cgoClearenv

//go:norace
func cgoClearenv() {
	tsanAcquire()
	asmcgocall(uintptr(libc_clearenv), 0, 0, 0, 0, 0, 0)
	tsanRelease()
}

//go:linknamestd _cgoSyscall runtime.cgoSyscall
var _cgoSyscall = cgoSyscall

//go:nosplit
//go:norace
//go:nocheckptr
func cgoSyscall(fn unsafe.Pointer, a0, a1, a2 uintptr) uintptr {
	if fn != cgo_libc_setgroups {
		// The other functions take only uid_t or gid_t arguments.
		a0, a1, a2 = cUint32Arg(a0), cUint32Arg(a1), cUint32Arg(a2)
	}
	r := int32(cgocall(uintptr(fn), a0, a1, a2, 0, 0, 0))
	if r == -1 {
		errnoPtr := asmcgocall(uintptr(libc_errno_location), 0, 0, 0, 0, 0, 0)
		return uintptr(*(*int32)(unsafe.Pointer(errnoPtr)))
	}
	return uintptr(int(r))
}

// cUint32Arg converts x, holding a 32-bit unsigned C argument such as a
// uid_t, to the register representation required by the C ABI. Callers
// pass values like uintptr(-1), which must not leak into the upper bits:
// 64-bit RISC-V, LoongArch, and MIPS require 32-bit values to be
// sign-extended, while the other 64-bit ABIs zero-extend them (or ignore
// the upper bits).
//
//go:nosplit
func cUint32Arg(x uintptr) uintptr {
	if goarch.IsRiscv64|goarch.IsLoong64|goarch.IsMips64|goarch.IsMips64le != 0 {
		return uintptr(int32(x))
	}
	return uintptr(uint32(x))
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
