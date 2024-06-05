// Copyright 2022 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

//go:build openbsd

package unix

import _ "unsafe"

// Implemented in the runtime package (runtime/sys_openbsd3.go)
func syscall_syscall(fn, a1, a2, a3 uintptr) (r1, r2 uintptr, err Errno)
func syscall_syscall6(fn, a1, a2, a3, a4, a5, a6 uintptr) (r1, r2 uintptr, err Errno)
func syscall_syscall10(fn, a1, a2, a3, a4, a5, a6, a7, a8, a9, a10 uintptr) (r1, r2 uintptr, err Errno)
func syscall_rawSyscall(fn, a1, a2, a3 uintptr) (r1, r2 uintptr, err Errno)
func syscall_rawSyscall6(fn, a1, a2, a3, a4, a5, a6 uintptr) (r1, r2 uintptr, err Errno)

//go:linkname syscall_syscall syscall.syscall
//go:linkname syscall_syscall6 syscall.syscall6
//go:linkname syscall_syscall10 syscall.syscall10
//go:linkname syscall_rawSyscall syscall.rawSyscall
//go:linkname syscall_rawSyscall6 syscall.rawSyscall6

func syscall_syscall9(fn, a1, a2, a3, a4, a5, a6, a7, a8, a9 uintptr) (r1, r2 uintptr, err Errno) {
	return syscall_syscall10(fn, a1, a2, a3, a4, a5, a6, a7, a8, a9, 0)
}
