// Copyright 2024 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package unix

import (
	"internal/abi"
	"syscall"
	"unsafe"
)

//go:linkname syscall_syscall syscall.syscall
func syscall_syscall(fn, a1, a2, a3 uintptr) (r1, r2 uintptr, err syscall.Errno)

//go:cgo_import_dynamic libc_arc4random_buf arc4random_buf "libc.so"

func libc_arc4random_buf_trampoline()

func ARC4Random(p []byte) {
	syscall_syscall(abi.FuncPCABI0(libc_arc4random_buf_trampoline),
		uintptr(unsafe.Pointer(unsafe.SliceData(p))), uintptr(len(p)), 0)
}
