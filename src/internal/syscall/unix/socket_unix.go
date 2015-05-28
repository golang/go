// Copyright 2015 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// +build darwin dragonfly freebsd linux,!386 netbsd openbsd

package unix

import (
	"syscall"
	"unsafe"
)

func getsockname(s int, addr []byte) error {
	l := uint32(len(addr))
	_, _, errno := syscall.RawSyscall(syscall.SYS_GETSOCKNAME, uintptr(s), uintptr(unsafe.Pointer(&addr[0])), uintptr(unsafe.Pointer(&l)))
	if errno != 0 {
		return error(errno)
	}
	return nil
}

func getpeername(s int, addr []byte) error {
	l := uint32(len(addr))
	_, _, errno := syscall.RawSyscall(syscall.SYS_GETPEERNAME, uintptr(s), uintptr(unsafe.Pointer(&addr[0])), uintptr(unsafe.Pointer(&l)))
	if errno != 0 {
		return error(errno)
	}
	return nil
}

func recvfrom(s int, b []byte, flags int, from []byte) (int, error) {
	var p unsafe.Pointer
	if len(b) > 0 {
		p = unsafe.Pointer(&b[0])
	} else {
		p = unsafe.Pointer(&emptyPayload)
	}
	l := uint32(len(from))
	n, _, errno := syscall.Syscall6(syscall.SYS_RECVFROM, uintptr(s), uintptr(p), uintptr(len(b)), uintptr(flags), uintptr(unsafe.Pointer(&from[0])), uintptr(unsafe.Pointer(&l)))
	if errno != 0 {
		return int(n), error(errno)
	}
	return int(n), nil
}

func sendto(s int, b []byte, flags int, to []byte) (int, error) {
	var p unsafe.Pointer
	if len(b) > 0 {
		p = unsafe.Pointer(&b[0])
	} else {
		p = unsafe.Pointer(&emptyPayload)
	}
	n, _, errno := syscall.Syscall6(syscall.SYS_SENDTO, uintptr(s), uintptr(p), uintptr(len(b)), uintptr(flags), uintptr(unsafe.Pointer(&to[0])), uintptr(len(to)))
	if errno != 0 {
		return int(n), error(errno)
	}
	return int(n), nil
}
