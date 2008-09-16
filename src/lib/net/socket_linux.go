// Copyright 2009 The Go Authors.  All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// Low-level socket interface.
// Only for implementing net package.
// DO NOT USE DIRECTLY.

package socket

import (
	"os";
	"ip";
	"syscall"
)

export const (
	SOCKET = 41;
	CONNECT = 42;
	ACCEPT = 43;
	SETSOCKOPT = 54;
	GETSOCKOPT = 55;
	BIND = 49;
	LISTEN = 50;

	AF_UNIX = 1;
	AF_INET = 2;
	AF_INET6 = 10;

	SOCK_STREAM = 1;
	SOCK_DGRAM = 2;
	SOCK_RAW = 3;
	SOCK_RDM = 4;
	SOCK_SEQPACKET = 5;

	SOL_SOCKET = 1;

	SO_DEBUG = 1;
	SO_REUSEADDR = 2;
	SO_TYPE = 3;
	SO_ERROR = 4;
	SO_DONTROUTE = 5;
	SO_BROADCAST = 6;
	SO_SNDBUF = 7;
	SO_RCVBUF = 8;
	SO_SNDBUFFORCE = 32;
	SO_RCVBUFFORCE = 33;
	SO_KEEPALIVE = 9;
	SO_OOBINLINE = 10;
	SO_NO_CHECK = 11;
	SO_PRIORITY = 12;
	SO_LINGER = 13;
	SO_BSDCOMPAT = 14;
	SO_PASSCRED = 16;
	SO_PEERCRED = 17;
	SO_RCVLOWAT = 18;
	SO_SNDLOWAT = 19;
	SO_RCVTIMEO = 20;
	SO_SNDTIMEO = 21;
	SO_BINDTODEVICE = 25;

	IPPROTO_TCP = 6;
	IPPROTO_UDP = 17;

	TCP_NODELAY = 0x01;
)

export type SockaddrUnix struct {
	family	uint16;
	path	[108]byte
}
export const SizeofSockaddrUnix = 110

export type SockaddrInet4 struct {
	family	uint16;
	port	[2]byte;
	addr	[4]byte;
	zero	[8]byte
}
export const SizeofSockaddrInet4 = 16

export type SockaddrInet6 struct {
	family	uint16;
	port	[2]byte;
	flowinfo	[4]byte;
	addr	[16]byte;
	scopeid	[4]byte;
}
export const SizeofSockaddrInet6 = 28

export type Sockaddr struct {
	family	uint16;
	opaque	[126]byte
}
export const SizeofSockaddr = 128

export type Timeval struct {
	sec int32;
	usec int32;
}
export type Linger struct {
	yes int32;
	sec int32;
}

func (s *Sockaddr) Len() int64 {
	switch s.family {
	case AF_UNIX:
		return SizeofSockaddrUnix
	case AF_INET:
		return SizeofSockaddrInet4
	case AF_INET6:
		return SizeofSockaddrInet6
	}
	return 0
}

func SockaddrToSockaddrInet4(s *Sockaddr) *SockaddrInet4;
func SockaddrToSockaddrInet6(s *Sockaddr) *SockaddrInet6;
func SockaddrInet4ToSockaddr(s *SockaddrInet4) *Sockaddr;
func SockaddrInet6ToSockaddr(s *SockaddrInet6) *Sockaddr;
func SockaddrPtr(s *Sockaddr) int64;
func Int32Ptr(ip *int32) int64;
func TimevalPtr(tv *Timeval) int64;
func LingerPtr(l *Linger) int64;

export func socket(domain, proto, typ int64) (ret int64, err *os.Error) {
	r1, r2, e := syscall.Syscall(SOCKET, domain, proto, typ);
	return r1, os.ErrnoToError(e)
}

export func connect(fd int64, sa *Sockaddr) (ret int64, err *os.Error) {
	r1, r2, e := syscall.Syscall(CONNECT, fd, SockaddrPtr(sa), sa.Len());
	return r1, os.ErrnoToError(e)
}

export func bind(fd int64, sa *Sockaddr) (ret int64, err *os.Error) {
	r1, r2, e := syscall.Syscall(BIND, fd, SockaddrPtr(sa), sa.Len());
	return r1, os.ErrnoToError(e)
}

export func listen(fd, n int64) (ret int64, err *os.Error) {
	r1, r2, e := syscall.Syscall(LISTEN, fd, n, 0);
	return r1, os.ErrnoToError(e)
}

export func accept(fd int64, sa *Sockaddr) (ret int64, err *os.Error) {
	n := int32(sa.Len());
	r1, r2, e := syscall.Syscall(ACCEPT, fd, SockaddrPtr(sa), Int32Ptr(&n));
	return r1, os.ErrnoToError(e)
}

export func setsockopt(fd, level, opt, valueptr, length int64) (ret int64, err *os.Error) {
	if fd < 0 {
		return -1, os.EINVAL
	}
	r1, r2, e := syscall.Syscall6(SETSOCKOPT, fd, level, opt, valueptr, length, 0);
	return r1, os.ErrnoToError(e)
}

export func setsockopt_int(fd, level, opt int64, value int) *os.Error {
	n := int32(opt);
	r1, e := setsockopt(fd, level, opt, Int32Ptr(&n), 4)
	return e
}

export func setsockopt_tv(fd, level, opt, nsec int64) *os.Error {
	var tv Timeval;
	nsec += 999;
	tv.sec = int32(nsec/1000000000);
	tv.usec = int32(nsec%1000000000);
	r1, e := setsockopt(fd, level, opt, TimevalPtr(&tv), 4)
	return e
}

export func setsockopt_linger(fd, level, opt int64, sec int) *os.Error {
	var l Linger;
	if sec != 0 {
		l.yes = 1;
		l.sec = sec
	} else {
		l.yes = 0;
		l.sec = 0
	}
	r1, err := setsockopt(fd, level, opt, LingerPtr(&l), 8)
	return err
}

/*
export func getsockopt(fd, level, opt, valueptr, lenptr int64) (ret int64, errno int64) {
	r1, r2, err := syscall.Syscall6(GETSOCKOPT, fd, level, opt, valueptr, lenptr, 0);
	return r1, err;
}
*/

export func IPv4ToSockaddr(p *[]byte, port int) (sa1 *Sockaddr, err *os.Error) {
	p = ip.ToIPv4(p)
	if p == nil || port < 0 || port > 0xFFFF {
		return nil, os.EINVAL
	}
	sa := new(SockaddrInet4);
	sa.family = AF_INET;
	sa.port[0] = byte(port>>8);
	sa.port[1] = byte(port);
	for i := 0; i < ip.IPv4len; i++ {
		sa.addr[i] = p[i]
	}
	return SockaddrInet4ToSockaddr(sa), nil
}

export func IPv6ToSockaddr(p *[]byte, port int) (sa1 *Sockaddr, err *os.Error) {
	p = ip.ToIPv6(p)
	if p == nil || port < 0 || port > 0xFFFF {
		return nil, os.EINVAL
	}
	sa := new(SockaddrInet6);
	sa.family = AF_INET6;
	sa.port[0] = byte(port>>8);
	sa.port[1] = byte(port);
	for i := 0; i < ip.IPv6len; i++ {
		sa.addr[i] = p[i]
	}
	return SockaddrInet6ToSockaddr(sa), nil
}

export func SockaddrToIP(sa1 *Sockaddr) (p *[]byte, port int, err *os.Error) {
	switch sa1.family {
	case AF_INET:
		sa := SockaddrToSockaddrInet4(sa1);
		a := ip.ToIPv6(&sa.addr)
		if a == nil {
			return nil, 0, os.EINVAL
		}
		return a, int(sa.port[0])<<8 + int(sa.port[1]), nil
	case AF_INET6:
		sa := SockaddrToSockaddrInet6(sa1);
		a := ip.ToIPv6(&sa.addr)
		if a == nil {
			return nil, 0, os.EINVAL
		}
		return a, int(sa.port[0])<<8 + int(sa.port[1]), nil
	default:
		return nil, 0, os.EINVAL
	}
	return nil, 0, nil	// not reached
}

