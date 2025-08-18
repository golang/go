// Copyright 2023 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

//go:build (freebsd || netbsd) && arm

package unix

import "syscall"

func PosixFallocate(fd int, off int64, size int64) error {
	// If successful, posix_fallocate() returns zero. It returns an error on failure, without
	// setting errno. See https://man.freebsd.org/cgi/man.cgi?query=posix_fallocate&sektion=2&n=1
	// and https://man.netbsd.org/posix_fallocate.2#RETURN%20VALUES
	//
	// The padding 0 argument is needed because the ARM calling convention requires that if an
	// argument (off in this case) needs double-word alignment (8-byte), the NCRN (next core
	// register number) is rounded up to the next even register number.
	// See https://github.com/ARM-software/abi-aa/blob/2bcab1e3b22d55170c563c3c7940134089176746/aapcs32/aapcs32.rst#parameter-passing
	r1, _, _ := syscall.Syscall6(posixFallocateTrap, uintptr(fd), 0, uintptr(off), uintptr(off>>32), uintptr(size), uintptr(size>>32))
	if r1 != 0 {
		return syscall.Errno(r1)
	}
	return nil
}
