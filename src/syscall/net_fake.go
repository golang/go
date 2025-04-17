// Copyright 2023 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// Fake networking for js/wasm and wasip1/wasm.

//go:build js || wasip1

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
	SOMAXCONN = 0x80
)

const (
	_ = iota
	IPV6_V6ONLY
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
