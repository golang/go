// Copyright 2018 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package unix

import "syscall"

// Implemented as sysvicall6 in runtime/syscall_solaris.go.
func syscall6(trap, nargs, a1, a2, a3, a4, a5, a6 uintptr) (r1, r2 uintptr, err syscall.Errno)

// Implemented as rawsysvicall6 in runtime/syscall_solaris.go.
func rawSyscall6(trap, nargs, a1, a2, a3, a4, a5, a6 uintptr) (r1, r2 uintptr, err syscall.Errno)

//go:cgo_import_dynamic libc_fstatat fstatat "libc.so"
//go:cgo_import_dynamic libc_openat openat "libc.so"
//go:cgo_import_dynamic libc_unlinkat unlinkat "libc.so"
//go:cgo_import_dynamic libc_uname uname "libc.so"

const (
	AT_REMOVEDIR        = 0x1
	AT_SYMLINK_NOFOLLOW = 0x1000

	UTIME_OMIT = -0x2
)
