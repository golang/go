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

// Specify "libc.so.6" so internal linking without C host objects emits a
// DT_NEEDED entry for glibc's dynamic loader; external linking ignores the
// library field, !android excludes bionic, and musl treats any "libc.so.*"
// SONAME as libc itself.
//go:cgo_import_dynamic libc_mmap mmap "libc.so.6"
//go:cgo_import_dynamic libc_munmap munmap "libc.so.6"
//go:cgo_import_dynamic libc_errno_location __errno_location "libc.so.6"
