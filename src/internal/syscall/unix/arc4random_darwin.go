// Copyright 2024 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package unix

import (
	"internal/abi"
	"unsafe"
)

//go:cgo_import_dynamic libc_arc4random_buf arc4random_buf "/usr/lib/libSystem.B.dylib"

func libc_arc4random_buf_trampoline()

// ARC4Random calls the macOS arc4random_buf(3) function.
func ARC4Random(p []byte) {
	// macOS 11 and 12 abort if length is 0.
	if len(p) == 0 {
		return
	}
	syscall_syscall(abi.FuncPCABI0(libc_arc4random_buf_trampoline),
		uintptr(unsafe.Pointer(unsafe.SliceData(p))), uintptr(len(p)), 0)
}
