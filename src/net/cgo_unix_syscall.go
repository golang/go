// Copyright 2022 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

//go:build !netgo && darwin

package net

import (
	"internal/syscall/unix"
	"runtime"
	"syscall"
	"unsafe"
)

const (
	_C_AF_INET      = syscall.AF_INET
	_C_AF_INET6     = syscall.AF_INET6
	_C_AF_UNSPEC    = syscall.AF_UNSPEC
	_C_EAI_AGAIN    = unix.EAI_AGAIN
	_C_EAI_NONAME   = unix.EAI_NONAME
	_C_EAI_NODATA   = unix.EAI_NODATA
	_C_EAI_OVERFLOW = unix.EAI_OVERFLOW
	_C_EAI_SYSTEM   = unix.EAI_SYSTEM
	_C_IPPROTO_TCP  = syscall.IPPROTO_TCP
	_C_IPPROTO_UDP  = syscall.IPPROTO_UDP
	_C_SOCK_DGRAM   = syscall.SOCK_DGRAM
	_C_SOCK_STREAM  = syscall.SOCK_STREAM
)

type (
	_C_char               = byte
	_C_int                = int32
	_C_uchar              = byte
	_C_uint               = uint32
	_C_socklen_t          = int
	_C_struct___res_state = unix.ResState
	_C_struct_addrinfo    = unix.Addrinfo
	_C_struct_sockaddr    = syscall.RawSockaddr
)

func _C_GoString(p *_C_char) string {
	return unix.GoString(p)
}

func _C_free(p unsafe.Pointer) { runtime.KeepAlive(p) }

func _C_malloc(n uintptr) unsafe.Pointer {
	if n <= 0 {
		n = 1
	}
	return unsafe.Pointer(&make([]byte, n)[0])
}

func _C_ai_addr(ai *_C_struct_addrinfo) **_C_struct_sockaddr { return &ai.Addr }
func _C_ai_family(ai *_C_struct_addrinfo) *_C_int            { return &ai.Family }
func _C_ai_flags(ai *_C_struct_addrinfo) *_C_int             { return &ai.Flags }
func _C_ai_next(ai *_C_struct_addrinfo) **_C_struct_addrinfo { return &ai.Next }
func _C_ai_protocol(ai *_C_struct_addrinfo) *_C_int          { return &ai.Protocol }
func _C_ai_socktype(ai *_C_struct_addrinfo) *_C_int          { return &ai.Socktype }

func _C_freeaddrinfo(ai *_C_struct_addrinfo) {
	unix.Freeaddrinfo(ai)
}

func _C_gai_strerror(eai _C_int) string {
	return unix.GaiStrerror(int(eai))
}

func _C_getaddrinfo(hostname, servname *byte, hints *_C_struct_addrinfo, res **_C_struct_addrinfo) (int, error) {
	return unix.Getaddrinfo(hostname, servname, hints, res)
}

func _C_res_ninit(state *_C_struct___res_state) error {
	unix.ResNinit(state)
	return nil
}

func _C_res_nsearch(state *_C_struct___res_state, dname *_C_char, class, typ int, ans *_C_char, anslen int) (int, error) {
	return unix.ResNsearch(state, dname, class, typ, ans, anslen)
}

func _C_res_nclose(state *_C_struct___res_state) {
	unix.ResNclose(state)
}

func cgoNameinfoPTR(b []byte, sa *syscall.RawSockaddr, salen int) (int, error) {
	gerrno, err := unix.Getnameinfo(sa, salen, &b[0], len(b), nil, 0, unix.NI_NAMEREQD)
	return int(gerrno), err
}

func cgoSockaddrInet4(ip IP) *syscall.RawSockaddr {
	sa := syscall.RawSockaddrInet4{Len: syscall.SizeofSockaddrInet4, Family: syscall.AF_INET}
	copy(sa.Addr[:], ip)
	return (*syscall.RawSockaddr)(unsafe.Pointer(&sa))
}

func cgoSockaddrInet6(ip IP, zone int) *syscall.RawSockaddr {
	sa := syscall.RawSockaddrInet6{Len: syscall.SizeofSockaddrInet6, Family: syscall.AF_INET6, Scope_id: uint32(zone)}
	copy(sa.Addr[:], ip)
	return (*syscall.RawSockaddr)(unsafe.Pointer(&sa))
}
