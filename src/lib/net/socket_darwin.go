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
	ACCEPT = 30;
	SOCKET = 97;
	CONNECT = 98;
	GETSOCKOPT = 118;
	BIND = 104;
	SETSOCKOPT = 105;
	LISTEN = 106;

	AF_UNIX = 1;
	AF_INET = 2;
	AF_DATAKIT = 9;
	AF_INET6 = 30;

	SOCK_STREAM = 1;
	SOCK_DGRAM = 2;
	SOCK_RAW = 3;
	SOCK_RDM = 4;
	SOCK_SEQPACKET = 5;

	SOL_SOCKET = 0xffff;

	SO_REUSEADDR = 0x0004;
	SO_KEEPALIVE = 0x0008;
	SO_DONTROUTE = 0x0010;
	SO_BROADCAST = 0x0020;
	SO_USELOOPBACK = 0x0040;
	SO_LINGER = 0x1080;
	SO_REUSEPORT = 0x0200;
	SO_SNDBUF = 0x1001;
	SO_RCVBUF = 0x1002;
	SO_SNDTIMEO = 0x1005;
	SO_RCVTIMEO = 0x1006;
	SO_NOSIGPIPE = 0x1022;

	IPPROTO_TCP = 6;
	IPPROTO_UDP = 17;

	TCP_NODELAY = 0x01;
)

export type SockaddrUnix struct {
	len	byte;
	family	byte;
	path	[104]byte
}
export const SizeofSockaddrUnix = 106

export type SockaddrInet4 struct {
	len	byte;
	family	byte;
	port	[2]byte;
	addr	[4]byte;
	zero	[8]byte
}
export const SizeofSockaddrInet4 = 16

export type SockaddrInet6 struct {
	len	byte;
	family	byte;
	port	[2]byte;
	flowinfo	[4]byte;
	addr	[16]byte;
	scopeid	[4]byte;
}
export const SizeofSockaddrInet6 = 28

export type Sockaddr struct {
	len	byte;
	family	byte;
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
	r1, r2, e := syscall.Syscall(CONNECT, fd, SockaddrPtr(sa), int64(sa.len));
	return r1, os.ErrnoToError(e)
}

export func bind(fd int64, sa *Sockaddr) (ret int64, err *os.Error) {
	r1, r2, e := syscall.Syscall(BIND, fd, SockaddrPtr(sa), int64(sa.len));
	return r1, os.ErrnoToError(e)
}

export func listen(fd, n int64) (ret int64, err *os.Error) {
	r1, r2, e := syscall.Syscall(LISTEN, fd, n, 0);
	return r1, os.ErrnoToError(e)
}

export func accept(fd int64, sa *Sockaddr) (ret int64, err *os.Error) {
	n := int32(sa.len);
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
	sa.len = SizeofSockaddrInet4;
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
	sa.len = SizeofSockaddrInet6;
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

