// Copyright 2021 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

//go:build aix || darwin || dragonfly || freebsd || linux || netbsd || openbsd || solaris

package unix

import (
	"syscall"
	_ "unsafe"
)

//go:linkname RecvfromInet4 syscall.recvfromInet4
//go:noescape
func RecvfromInet4(fd int, p []byte, flags int, from *syscall.SockaddrInet4) (int, error)

//go:linkname RecvfromInet6 syscall.recvfromInet6
//go:noescape
func RecvfromInet6(fd int, p []byte, flags int, from *syscall.SockaddrInet6) (n int, err error)

//go:linkname SendtoInet4 syscall.sendtoInet4
//go:noescape
func SendtoInet4(fd int, p []byte, flags int, to *syscall.SockaddrInet4) (err error)

//go:linkname SendtoInet6 syscall.sendtoInet6
//go:noescape
func SendtoInet6(fd int, p []byte, flags int, to *syscall.SockaddrInet6) (err error)

//go:linkname SendmsgNInet4 syscall.sendmsgNInet4
//go:noescape
func SendmsgNInet4(fd int, p, oob []byte, to *syscall.SockaddrInet4, flags int) (n int, err error)

//go:linkname SendmsgNInet6 syscall.sendmsgNInet6
//go:noescape
func SendmsgNInet6(fd int, p, oob []byte, to *syscall.SockaddrInet6, flags int) (n int, err error)

//go:linkname RecvmsgInet4 syscall.recvmsgInet4
//go:noescape
func RecvmsgInet4(fd int, p, oob []byte, flags int, from *syscall.SockaddrInet4) (n, oobn int, recvflags int, err error)

//go:linkname RecvmsgInet6 syscall.recvmsgInet6
//go:noescape
func RecvmsgInet6(fd int, p, oob []byte, flags int, from *syscall.SockaddrInet6) (n, oobn int, recvflags int, err error)
