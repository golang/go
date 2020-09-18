// Copyright 2015 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// +build gccgo
// +build !aix

package unix

import "syscall"

// We can't use the gc-syntax .s files for gccgo. On the plus side
// much of the functionality can be written directly in Go.

//extern gccgoRealSyscallNoError
func realSyscallNoError(trap, a1, a2, a3, a4, a5, a6, a7, a8, a9 uintptr) (r uintptr)

//extern gccgoRealSyscall
func realSyscall(trap, a1, a2, a3, a4, a5, a6, a7, a8, a9 uintptr) (r, errno uintptr)

func SyscallNoError(trap, a1, a2, a3 uintptr) (r1, r2 uintptr) {
	syscall.Entersyscall()
	r := realSyscallNoError(trap, a1, a2, a3, 0, 0, 0, 0, 0, 0)
	syscall.Exitsyscall()
	return r, 0
}

func Syscall(trap, a1, a2, a3 uintptr) (r1, r2 uintptr, err syscall.Errno) {
	syscall.Entersyscall()
	r, errno := realSyscall(trap, a1, a2, a3, 0, 0, 0, 0, 0, 0)
	syscall.Exitsyscall()
	return r, 0, syscall.Errno(errno)
}

func Syscall6(trap, a1, a2, a3, a4, a5, a6 uintptr) (r1, r2 uintptr, err syscall.Errno) {
	syscall.Entersyscall()
	r, errno := realSyscall(trap, a1, a2, a3, a4, a5, a6, 0, 0, 0)
	syscall.Exitsyscall()
	return r, 0, syscall.Errno(errno)
}

func Syscall9(trap, a1, a2, a3, a4, a5, a6, a7, a8, a9 uintptr) (r1, r2 uintptr, err syscall.Errno) {
	syscall.Entersyscall()
	r, errno := realSyscall(trap, a1, a2, a3, a4, a5, a6, a7, a8, a9)
	syscall.Exitsyscall()
	return r, 0, syscall.Errno(errno)
}

func RawSyscallNoError(trap, a1, a2, a3 uintptr) (r1, r2 uintptr) {
	r := realSyscallNoError(trap, a1, a2, a3, 0, 0, 0, 0, 0, 0)
	return r, 0
}

func RawSyscall(trap, a1, a2, a3 uintptr) (r1, r2 uintptr, err syscall.Errno) {
	r, errno := realSyscall(trap, a1, a2, a3, 0, 0, 0, 0, 0, 0)
	return r, 0, syscall.Errno(errno)
}

func RawSyscall6(trap, a1, a2, a3, a4, a5, a6 uintptr) (r1, r2 uintptr, err syscall.Errno) {
	r, errno := realSyscall(trap, a1, a2, a3, a4, a5, a6, 0, 0, 0)
	return r, 0, syscall.Errno(errno)
}
