// Copyright 2018 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// +build darwin
// +build amd64 arm64

package runtime

import "unsafe"

//go:linkname syscall_syscallX syscall.syscallX
//go:nosplit
//go:cgo_unsafe_args
func syscall_syscallX(fn, a1, a2, a3 uintptr) (r1, r2, err uintptr) {
	entersyscallblock()
	libcCall(unsafe.Pointer(funcPC(syscallX)), unsafe.Pointer(&fn))
	exitsyscall()
	return
}
func syscallX()
