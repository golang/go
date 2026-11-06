// Copyright 2026 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

//go:build goexperiment.runtimecgopurego && !msan && !tsan && freebsd && amd64

package cgo

import "unsafe"

//go:linkname _cgo_libc_sigaction _cgo_libc_sigaction
var _cgo_libc_sigaction byte
var libc_sigaction = unsafe.Pointer(&_cgo_libc_sigaction)

//go:linkname _cgo_libc_errno_location _cgo_libc_errno_location
var _cgo_libc_errno_location byte
var libc_errno_location = unsafe.Pointer(&_cgo_libc_errno_location)

//go:cgo_import_dynamic _cgo_libc_mmap mmap "libc.so.7"
//go:cgo_import_dynamic _cgo_libc_munmap munmap "libc.so.7"
//go:cgo_import_dynamic _cgo_libc_sigaction sigaction "libc.so.7"
//go:cgo_import_dynamic _cgo_libc_errno_location __error "libc.so.7"

// freebsdSigaction matches both runtime.sigactiont (os_freebsd.go) and
// FreeBSD's C struct sigaction on freebsd/amd64.
type freebsdSigaction struct {
	handler uintptr
	flags   int32
	mask    [4]uint32
}

//go:linknamestd _cgoSigaction runtime.cgoSigaction
var _cgoSigaction = cgoSigaction

//go:nosplit
//go:norace
//go:nocheckptr
func cgoSigaction(signum uintptr, goact, oldgoact *freebsdSigaction, noG bool) int32 {
	tsanAcquireMode(noG)

	var act, oldact freebsdSigaction
	var actPtr, oldactPtr uintptr

	if goact != nil {
		act = *goact
		msanUnpoisonMode(noG, unsafe.Pointer(&act), unsafe.Sizeof(act))
		actPtr = uintptr(unsafe.Pointer(&act))
	}
	if oldgoact != nil {
		oldactPtr = uintptr(unsafe.Pointer(&oldact))
	}

	ret := int32(asmcgocallMode(noG, uintptr(libc_sigaction), uintptr(int32(signum)), actPtr, oldactPtr, 0, 0, 0))
	if ret == -1 {
		errnoPtr := asmcgocallMode(noG, uintptr(libc_errno_location), 0, 0, 0, 0, 0, 0)
		errno := *(*int32)(unsafe.Pointer(errnoPtr))
		tsanReleaseMode(noG)
		return errno
	}

	if oldgoact != nil {
		*oldgoact = oldact
		msanUnpoisonMode(noG, unsafe.Pointer(oldgoact), unsafe.Sizeof(*oldgoact))
	}

	tsanReleaseMode(noG)
	return ret
}
