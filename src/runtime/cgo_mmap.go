// Copyright 2015 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// Support for memory sanitizer. See runtime/cgo/mmap.go.

// +build linux,amd64 linux,arm64

package runtime

import "unsafe"

// _cgo_mmap is filled in by runtime/cgo when it is linked into the
// program, so it is only non-nil when using cgo.
//go:linkname _cgo_mmap _cgo_mmap
var _cgo_mmap unsafe.Pointer

// _cgo_munmap is filled in by runtime/cgo when it is linked into the
// program, so it is only non-nil when using cgo.
//go:linkname _cgo_munmap _cgo_munmap
var _cgo_munmap unsafe.Pointer

// mmap is used to route the mmap system call through C code when using cgo, to
// support sanitizer interceptors. Don't allow stack splits, since this function
// (used by sysAlloc) is called in a lot of low-level parts of the runtime and
// callers often assume it won't acquire any locks.
//go:nosplit
func mmap(addr unsafe.Pointer, n uintptr, prot, flags, fd int32, off uint32) (unsafe.Pointer, int) {
	if _cgo_mmap != nil {
		// Make ret a uintptr so that writing to it in the
		// function literal does not trigger a write barrier.
		// A write barrier here could break because of the way
		// that mmap uses the same value both as a pointer and
		// an errno value.
		var ret uintptr
		systemstack(func() {
			ret = callCgoMmap(addr, n, prot, flags, fd, off)
		})
		if ret < 4096 {
			return nil, int(ret)
		}
		return unsafe.Pointer(ret), 0
	}
	return sysMmap(addr, n, prot, flags, fd, off)
}

func munmap(addr unsafe.Pointer, n uintptr) {
	if _cgo_munmap != nil {
		systemstack(func() { callCgoMunmap(addr, n) })
		return
	}
	sysMunmap(addr, n)
}

// sysMmap calls the mmap system call. It is implemented in assembly.
func sysMmap(addr unsafe.Pointer, n uintptr, prot, flags, fd int32, off uint32) (p unsafe.Pointer, err int)

// callCgoMmap calls the mmap function in the runtime/cgo package
// using the GCC calling convention. It is implemented in assembly.
func callCgoMmap(addr unsafe.Pointer, n uintptr, prot, flags, fd int32, off uint32) uintptr

// sysMunmap calls the munmap system call. It is implemented in assembly.
func sysMunmap(addr unsafe.Pointer, n uintptr)

// callCgoMunmap calls the munmap function in the runtime/cgo package
// using the GCC calling convention. It is implemented in assembly.
func callCgoMunmap(addr unsafe.Pointer, n uintptr)
