// Copyright 2022 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// Package syscall provides the syscall primitives required for the runtime.
package syscall

import (
	_ "unsafe" // for go:linkname
)

// TODO(https://go.dev/issue/51087): This package is incomplete and currently
// only contains very minimal support for Linux.

// Syscall6 calls system call number 'num' with arguments a1-6.
func Syscall6(num, a1, a2, a3, a4, a5, a6 uintptr) (r1, r2, errno uintptr)

// syscall_RawSyscall6 is a push linkname to export Syscall6 as
// syscall.RawSyscall6.
//
// //go:uintptrkeepalive because the uintptr argument may be converted pointers
// that need to be kept alive in the caller (this is implied for Syscall6 since
// it has no body).
//
// //go:nosplit because stack copying does not account for uintptrkeepalive, so
// the stack must not grow. Stack copying cannot blindly assume that all
// uintptr arguments are pointers, because some values may look like pointers,
// but not really be pointers, and adjusting their value would break the call.
//
// This is a separate wrapper because we can't export one function as two
// names. The assembly implementations name themselves Syscall6 would not be
// affected by a linkname.
//
//go:uintptrkeepalive
//go:nosplit
//go:linkname syscall_RawSyscall6 syscall.RawSyscall6
func syscall_RawSyscall6(num, a1, a2, a3, a4, a5, a6 uintptr) (r1, r2, errno uintptr) {
	return Syscall6(num, a1, a2, a3, a4, a5, a6)
}
