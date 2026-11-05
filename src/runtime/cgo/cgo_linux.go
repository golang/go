// Copyright 2026 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

//go:build goexperiment.runtimecgopurego && !msan && !tsan && linux && (386 || amd64 || arm || arm64 || loong64 || riscv64)

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

//go:linkname _cgo_libc_errno_location _cgo_libc_errno_location
var _cgo_libc_errno_location byte
var libc_errno_location = unsafe.Pointer(&_cgo_libc_errno_location)

//go:linkname _cgo_libc_clearenv _cgo_libc_clearenv
var _cgo_libc_clearenv byte
var libc_clearenv = unsafe.Pointer(&_cgo_libc_clearenv)

//go:linkname _cgo_libc_setegid _cgo_libc_setegid
//go:linkname cgo_libc_setegid syscall.cgo_libc_setegid
var _cgo_libc_setegid byte
var cgo_libc_setegid = unsafe.Pointer(&_cgo_libc_setegid)

//go:linkname _cgo_libc_seteuid _cgo_libc_seteuid
//go:linkname cgo_libc_seteuid syscall.cgo_libc_seteuid
var _cgo_libc_seteuid byte
var cgo_libc_seteuid = unsafe.Pointer(&_cgo_libc_seteuid)

//go:linkname _cgo_libc_setgid _cgo_libc_setgid
//go:linkname cgo_libc_setgid syscall.cgo_libc_setgid
var _cgo_libc_setgid byte
var cgo_libc_setgid = unsafe.Pointer(&_cgo_libc_setgid)

//go:linkname _cgo_libc_setgroups _cgo_libc_setgroups
//go:linkname cgo_libc_setgroups syscall.cgo_libc_setgroups
var _cgo_libc_setgroups byte
var cgo_libc_setgroups = unsafe.Pointer(&_cgo_libc_setgroups)

//go:linkname _cgo_libc_setregid _cgo_libc_setregid
//go:linkname cgo_libc_setregid syscall.cgo_libc_setregid
var _cgo_libc_setregid byte
var cgo_libc_setregid = unsafe.Pointer(&_cgo_libc_setregid)

//go:linkname _cgo_libc_setresgid _cgo_libc_setresgid
//go:linkname cgo_libc_setresgid syscall.cgo_libc_setresgid
var _cgo_libc_setresgid byte
var cgo_libc_setresgid = unsafe.Pointer(&_cgo_libc_setresgid)

//go:linkname _cgo_libc_setresuid _cgo_libc_setresuid
//go:linkname cgo_libc_setresuid syscall.cgo_libc_setresuid
var _cgo_libc_setresuid byte
var cgo_libc_setresuid = unsafe.Pointer(&_cgo_libc_setresuid)

//go:linkname _cgo_libc_setreuid _cgo_libc_setreuid
//go:linkname cgo_libc_setreuid syscall.cgo_libc_setreuid
var _cgo_libc_setreuid byte
var cgo_libc_setreuid = unsafe.Pointer(&_cgo_libc_setreuid)

//go:linkname _cgo_libc_setuid _cgo_libc_setuid
//go:linkname cgo_libc_setuid syscall.cgo_libc_setuid
var _cgo_libc_setuid byte
var cgo_libc_setuid = unsafe.Pointer(&_cgo_libc_setuid)

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
