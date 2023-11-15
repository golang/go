// Copyright 2014 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

//go:build dragonfly || freebsd || linux || netbsd || openbsd

package unix

import "unsafe"

// fcntl64Syscall is usually SYS_FCNTL, but is overridden on 32-bit Linux
// systems by fcntl_linux_32bit.go to be SYS_FCNTL64.
var fcntl64Syscall uintptr = SYS_FCNTL

func fcntl(fd int, cmd, arg int) (int, error) {
	valptr, _, errno := Syscall(fcntl64Syscall, uintptr(fd), uintptr(cmd), uintptr(arg))
	var err error
	if errno != 0 {
		err = errno
	}
	return int(valptr), err
}

// FcntlInt performs a fcntl syscall on fd with the provided command and argument.
func FcntlInt(fd uintptr, cmd, arg int) (int, error) {
	return fcntl(int(fd), cmd, arg)
}

// FcntlFlock performs a fcntl syscall for the F_GETLK, F_SETLK or F_SETLKW command.
func FcntlFlock(fd uintptr, cmd int, lk *Flock_t) error {
	_, _, errno := Syscall(fcntl64Syscall, fd, uintptr(cmd), uintptr(unsafe.Pointer(lk)))
	if errno == 0 {
		return nil
	}
	return errno
}
