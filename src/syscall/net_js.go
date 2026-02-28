// Copyright 2018 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// js/wasm uses fake networking directly implemented in the net package.
// This file only exists to make the compiler happy.

//go:build js && wasm

package syscall

const (
	AF_UNSPEC = iota
	AF_UNIX
	AF_INET
	AF_INET6
)

const (
	SOCK_STREAM = 1 + iota
	SOCK_DGRAM
	SOCK_RAW
	SOCK_SEQPACKET
)

const (
	IPPROTO_IP   = 0
	IPPROTO_IPV4 = 4
	IPPROTO_IPV6 = 0x29
	IPPROTO_TCP  = 6
	IPPROTO_UDP  = 0x11
)

const (
	_ = iota
	IPV6_V6ONLY
	SOMAXCONN
	SO_ERROR
)

// Misc constants expected by package net but not supported.
const (
	_ = iota
	F_DUPFD_CLOEXEC
	SYS_FCNTL = 500 // unsupported
)

type Sockaddr any

type SockaddrInet4 struct {
	Port int
	Addr [4]byte
}

type SockaddrInet6 struct {
	Port   int
	ZoneId uint32
	Addr   [16]byte
}

type SockaddrUnix struct {
	Name string
}

func Socket(proto, sotype, unused int) (fd int, err error) {
	return 0, ENOSYS
}

func Bind(fd int, sa Sockaddr) error {
	return ENOSYS
}

func StopIO(fd int) error {
	return ENOSYS
}

func Listen(fd int, backlog int) error {
	return ENOSYS
}

func Accept(fd int) (newfd int, sa Sockaddr, err error) {
	return 0, nil, ENOSYS
}

func Connect(fd int, sa Sockaddr) error {
	return ENOSYS
}

func Recvfrom(fd int, p []byte, flags int) (n int, from Sockaddr, err error) {
	return 0, nil, ENOSYS
}

func Sendto(fd int, p []byte, flags int, to Sockaddr) error {
	return ENOSYS
}

func Recvmsg(fd int, p, oob []byte, flags int) (n, oobn, recvflags int, from Sockaddr, err error) {
	return 0, 0, 0, nil, ENOSYS
}

func SendmsgN(fd int, p, oob []byte, to Sockaddr, flags int) (n int, err error) {
	return 0, ENOSYS
}

func GetsockoptInt(fd, level, opt int) (value int, err error) {
	return 0, ENOSYS
}

func SetsockoptInt(fd, level, opt int, value int) error {
	return nil
}

func SetReadDeadline(fd int, t int64) error {
	return ENOSYS
}

func SetWriteDeadline(fd int, t int64) error {
	return ENOSYS
}

func Shutdown(fd int, how int) error {
	return ENOSYS
}

func SetNonblock(fd int, nonblocking bool) error {
	return nil
}
