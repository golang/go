// Copyright 2009 The Go Authors.  All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// Types and defined constants.
// Should be automatically generated, but is not.

package syscall


// Time

export type Timespec struct {
	sec	int64;
	nsec	uint64;
}
export func TimespecPtr(t *Timespec) int64;

export type Timeval struct {
	sec	int64;
	usec	uint64;
}
export func TimevalPtr(t *Timeval) int64;


// Files

export const (
	O_RDONLY = 0x0;
	O_WRONLY = 0x1;
	O_RDWR = 0x2;
	O_APPEND = 0x400;
	O_ASYNC = 0x2000;
	O_CREAT = 0x40;
	O_NOCTTY = 0x100;
	O_NONBLOCK = 0x800;
	O_NDELAY = O_NONBLOCK;
	O_SYNC = 0x1000;
	O_TRUNC = 0x200;

	F_GETFL = 3;
	F_SETFL = 4;
)

export type Stat struct {
	dev	uint64;
	ino	uint64;
	nlink	uint64;
	mode	uint32;
	uid	uint32;
	gid	uint32;
	_pad0	uint32;
	rdev	uint64;
	size	int64;
	blksize	int64;
	blocks	int64;
	atime	Timespec;
	mtime	Timespec;
	ctime	Timespec;
	_unused	[3]int64
}
export func StatPtr(s *Stat) int64;


// Sockets

export const (
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

	SOMAXCONN = 128;
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
export func SockaddrPtr(s *Sockaddr) int64;

export type Linger struct {
	yes int32;
	sec int32;
}
export func LingerPtr(l *Linger) int64;


// Events (epoll)

export const (
	// EpollEvent.events
	EPOLLIN = 0x1;
	EPOLLOUT = 0x4;
	EPOLLRDHUP = 0x2000;
	EPOLLPRI = 0x2;
	EPOLLERR = 0x8;
	EPOLLET = 0x80000000;
	EPOLLONESHOT = 0x40000000;

	// op
	EPOLL_CTL_ADD = 0x1;
	EPOLL_CTL_MOD = 0x3;
	EPOLL_CTL_DEL = 0x2;
)

export type EpollEvent struct {
	events uint32;
	fd int32;
	pad int32;
}
export func EpollEventPtr(ev *EpollEvent) int64;
