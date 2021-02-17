// Copyright 2019 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// Minimal copy of x/sys/unix so the cpu package can make a
// system call on AIX without depending on x/sys/unix.
// (See golang.org/issue/32102)

// +build aix,ppc64
// +build gc

package cpu

import (
	"syscall"
	"unsafe"
)

//go:cgo_import_dynamic libc_getsystemcfg getsystemcfg "libc.a/shr_64.o"

//go:linkname libc_getsystemcfg libc_getsystemcfg

type syscallFunc uintptr

var libc_getsystemcfg syscallFunc

type errno = syscall.Errno

// Implemented in runtime/syscall_aix.go.
func rawSyscall6(trap, nargs, a1, a2, a3, a4, a5, a6 uintptr) (r1, r2 uintptr, err errno)
func syscall6(trap, nargs, a1, a2, a3, a4, a5, a6 uintptr) (r1, r2 uintptr, err errno)

func callgetsystemcfg(label int) (r1 uintptr, e1 errno) {
	r1, _, e1 = syscall6(uintptr(unsafe.Pointer(&libc_getsystemcfg)), 1, uintptr(label), 0, 0, 0, 0, 0)
	return
}
