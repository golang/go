// Copyright 2015 The Go Authors.  All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// Support for memory sanitizer.  See runtime/cgo/mmap.go.

// +build linux,amd64

package runtime

import "unsafe"

// _cgo_mmap is filled in by runtime/cgo when it is linked into the
// program, so it is only non-nil when using cgo.
//go:linkname _cgo_mmap _cgo_mmap
var _cgo_mmap unsafe.Pointer

func mmap(addr unsafe.Pointer, n uintptr, prot, flags, fd int32, off uint32) (ret unsafe.Pointer) {
	if _cgo_mmap != nil {
		systemstack(func() {
			ret = callCgoMmap(addr, n, prot, flags, fd, off)
		})
		return
	}
	return sysMmap(addr, n, prot, flags, fd, off)
}

// sysMmap calls the mmap system call.  It is implemented in assembly.
func sysMmap(addr unsafe.Pointer, n uintptr, prot, flags, fd int32, off uint32) unsafe.Pointer

// cgoMmap calls the mmap function in the runtime/cgo package on the
// callCgoMmap calls the mmap function in the runtime/cgo package
// using the GCC calling convention.  It is implemented in assembly.
func callCgoMmap(addr unsafe.Pointer, n uintptr, prot, flags, fd int32, off uint32) unsafe.Pointer
