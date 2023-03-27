// Copyright 2023 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

//go:build freebsd && (386 || arm)

package unix

import "syscall"

// FreeBSD posix_fallocate system call number.
const posixFallocateTrap uintptr = 530

func PosixFallocate(fd int, off int64, size int64) error {
	_, _, errno := syscall.Syscall6(posixFallocateTrap, uintptr(fd), uintptr(off), uintptr(off>>32), uintptr(size), uintptr(size>>32), 0)
	if errno != 0 {
		return errno
	}
	return nil
}
