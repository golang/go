// Copyright 2023 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

//go:build wasip1

package unix

import (
	"syscall"
	_ "unsafe"
)

func RecvfromInet4(fd int, p []byte, flags int, from *syscall.SockaddrInet4) (int, error) {
	return 0, syscall.ENOSYS
}

func RecvfromInet6(fd int, p []byte, flags int, from *syscall.SockaddrInet6) (n int, err error) {
	return 0, syscall.ENOSYS
}

func SendtoInet4(fd int, p []byte, flags int, to *syscall.SockaddrInet4) (err error) {
	return syscall.ENOSYS
}

func SendtoInet6(fd int, p []byte, flags int, to *syscall.SockaddrInet6) (err error) {
	return syscall.ENOSYS
}

func SendmsgNInet4(fd int, p, oob []byte, to *syscall.SockaddrInet4, flags int) (n int, err error) {
	return 0, syscall.ENOSYS
}

func SendmsgNInet6(fd int, p, oob []byte, to *syscall.SockaddrInet6, flags int) (n int, err error) {
	return 0, syscall.ENOSYS
}

func RecvmsgInet4(fd int, p, oob []byte, flags int, from *syscall.SockaddrInet4) (n, oobn int, recvflags int, err error) {
	return 0, 0, 0, syscall.ENOSYS
}

func RecvmsgInet6(fd int, p, oob []byte, flags int, from *syscall.SockaddrInet6) (n, oobn int, recvflags int, err error) {
	return 0, 0, 0, syscall.ENOSYS
}
