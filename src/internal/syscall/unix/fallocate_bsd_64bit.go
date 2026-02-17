// Copyright 2023 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

//go:build (freebsd || netbsd) && (amd64 || arm64 || riscv64)

package unix

import "syscall"

func PosixFallocate(fd int, off int64, size int64) error {
	// If successful, posix_fallocate() returns zero. It returns an error on failure, without
	// setting errno. See https://man.freebsd.org/cgi/man.cgi?query=posix_fallocate&sektion=2&n=1
	// and https://man.netbsd.org/posix_fallocate.2#RETURN%20VALUES
	r1, _, _ := syscall.Syscall(posixFallocateTrap, uintptr(fd), uintptr(off), uintptr(size))
	if r1 != 0 {
		return syscall.Errno(r1)
	}
	return nil
}
