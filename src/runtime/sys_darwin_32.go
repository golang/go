// Copyright 2018 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// +build darwin
// +build 386 arm

package runtime

import "unsafe"

//go:linkname syscall_syscall9 syscall.syscall9
//go:nosplit
//go:cgo_unsafe_args
func syscall_syscall9(fn, a1, a2, a3, a4, a5, a6, a7, a8, a9 uintptr) (r1, r2, err uintptr) {
	entersyscallblock()
	libcCall(unsafe.Pointer(funcPC(syscall9)), unsafe.Pointer(&fn))
	exitsyscall()
	return
}
func syscall9()
