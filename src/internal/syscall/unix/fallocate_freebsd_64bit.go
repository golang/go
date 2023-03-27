// Copyright 2023 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

//go:build freebsd && (amd64 || arm64 || riscv64)

package unix

import "syscall"

// FreeBSD posix_fallocate system call number.
const posixFallocateTrap uintptr = 530

func PosixFallocate(fd int, off int64, size int64) error {
	_, _, errno := syscall.Syscall(posixFallocateTrap, uintptr(fd), uintptr(off), uintptr(size))
	if errno != 0 {
		return errno
	}
	return nil
}
