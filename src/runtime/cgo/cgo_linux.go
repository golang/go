// Copyright 2026 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

//go:build goexperiment.cgobindings && linux && !android && (amd64 || arm64)

package cgo

import "unsafe"

//go:linkname _libc_mmap libc_mmap
var _libc_mmap byte
var libc_mmap = unsafe.Pointer(&_libc_mmap)

//go:linkname _libc_munmap libc_munmap
var _libc_munmap byte
var libc_munmap = unsafe.Pointer(&_libc_munmap)

//go:linkname _libc_sigaction libc_sigaction
var _libc_sigaction byte
var libc_sigaction = unsafe.Pointer(&_libc_sigaction)

//go:linkname _libc_sigfillset libc_sigfillset
var _libc_sigfillset byte
var libc_sigfillset = unsafe.Pointer(&_libc_sigfillset)

//go:linkname _libc_errno_location libc_errno_location
var _libc_errno_location byte
var libc_errno_location = unsafe.Pointer(&_libc_errno_location)

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

// Specify "libc.so.6" so internal linking without C host objects emits a
// DT_NEEDED entry for glibc's dynamic loader; external linking ignores the
// library field, !android excludes bionic, and musl treats any "libc.so.*"
// SONAME as libc itself.
//go:cgo_import_dynamic libc_mmap mmap "libc.so.6"
//go:cgo_import_dynamic libc_munmap munmap "libc.so.6"
//go:cgo_import_dynamic libc_sigaction sigaction "libc.so.6"
//go:cgo_import_dynamic libc_sigfillset sigfillset "libc.so.6"
//go:cgo_import_dynamic libc_errno_location __errno_location "libc.so.6"
