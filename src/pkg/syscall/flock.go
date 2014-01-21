// +build linux darwin freebsd openbsd netbsd dragonfly

// Copyright 2014 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package syscall

import "unsafe"

// fcntl64Syscall is usually SYS_FCNTL, but is overridden on 32-bit Linux
// systems by flock_linux_32bit.go to be SYS_FCNTL64.
var fcntl64Syscall uintptr = SYS_FCNTL

// Lock performs a fcntl syscall for F_GETLK, F_SETLK or F_SETLKW commands.
func (lk *Flock_t) Lock(fd uintptr, cmd int) error {
	_, _, errno := Syscall(fcntl64Syscall, fd, uintptr(cmd), uintptr(unsafe.Pointer(lk)))
	if errno == 0 {
		return nil
	}
	return errno
}
