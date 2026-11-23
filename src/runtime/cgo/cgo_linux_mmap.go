// Copyright 2026 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

//go:build goexperiment.runtimecgopurego && ((linux && (amd64 || arm64 || loong64 || riscv64)) || (freebsd && amd64))

package cgo

import "unsafe"

// Despite the file name, this file is also used on freebsd/amd64, where the
// runtime calls mmap and munmap through runtime/cgo as on Linux.

//go:linkname _cgo_libc_mmap _cgo_libc_mmap
var _cgo_libc_mmap byte
var libc_mmap = unsafe.Pointer(&_cgo_libc_mmap)

//go:linkname _cgo_libc_munmap _cgo_libc_munmap
var _cgo_libc_munmap byte
var libc_munmap = unsafe.Pointer(&_cgo_libc_munmap)

//go:linknamestd _cgoMmap runtime.cgoMmap
var _cgoMmap = cgoMmap

//go:nosplit
//go:norace
//go:nocheckptr
func cgoMmap(addr unsafe.Pointer, length uintptr, prot, flags, fd int32, offset uint32) uintptr {
	tsanAcquire()
	p := asmcgocall(uintptr(libc_mmap), uintptr(addr), length, uintptr(prot), uintptr(flags), uintptr(fd), uintptr(offset))
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
