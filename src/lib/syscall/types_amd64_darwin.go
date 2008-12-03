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
	usec	uint32;
}
export func TimevalPtr(t *Timeval) int64;


// Processes

export type Rusage struct {
	utime	Timeval;
	stime	Timeval;
	maxrss	int64;
	ixrss	int64;
	idrss	int64;
	isrss	int64;
	minflt	int64;
	majflt	int64;
	nswap	int64;
	inblock	int64;
	oublock	int64;
	msgsnd	int64;
	msgrcv	int64;
	nsignals	int64;
	nvcsw	int64;
	nivcsw	int64;
}
export func RusagePtr(r *Rusage) int64;


// Files

export const (
	O_RDONLY = 0x0;
	O_WRONLY = 0x1;
	O_RDWR = 0x2;
	O_APPEND = 0x8;
	O_ASYNC = 0x40;
	O_CREAT = 0x200;
	O_NOCTTY = 0x20000;
	O_NONBLOCK = 0x4;
	O_NDELAY = O_NONBLOCK;
	O_SYNC = 0x80;
	O_TRUNC = 0x400;

	F_GETFD = 1;
	F_SETFD = 2;

	F_GETFL = 3;
	F_SETFL = 4;

	FD_CLOEXEC = 1;
)

export type Stat struct {
	dev	uint32;
	mode	uint16;
	nlink	uint16;
	ino	uint64;
	uid	uint32;
	gid	uint32;
	rdev	uint32;
	pad1	uint32;
	atime Timespec;
	mtime Timespec;
	ctime Timespec;
	birthtime Timespec;
	size uint64;
	blocks uint64;
	blksize uint32;
	flags uint32;
	gen uint32;
	lspare uint32;
	qspare [2]uint64;
}
export func StatPtr(s *Stat) int64;


// Sockets

export const (
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

	SOMAXCONN = 128;
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
export func SockaddrPtr(s *Sockaddr) int64;

export type Linger struct {
	yes int32;
	sec int32;
}
export func LingerPtr(l *Linger) int64;


// Events (kqueue, kevent)

export const (
	// filters
	EVFILT_READ = -1;
	EVFILT_WRITE = -2;
	EVFILT_AIO = -3;
	EVFILT_VNODE = -4;
	EVFILT_PROC = -5;
	EVFILT_SIGNAL = -6;
	EVFILT_TIMER = -7;
	EVFILT_MACHPORT = -8;
	EVFILT_FS = -9;

	EVFILT_SYSCOUNT = 9;

	// actions
	EV_ADD = 0x0001;
	EV_DELETE = 0x0002;
	EV_DISABLE = 0x0008;
	EV_RECEIPT = 0x0040;

	// flags
	EV_ONESHOT = 0x0010;
	EV_CLEAR = 0x0020;
	EV_RECEIPT = 0x40;
	EV_SYSFLAGS = 0xF000;
	EV_FLAG0 = 0x1000;
	EV_FLAG1 = 0x2000;

	// returned values
	EV_EOF = 0x8000;
	EV_ERROR = 0x4000
)

export type Kevent struct {
	ident int64;
	filter int16;
	flags uint16;
	fflags uint32;
	data int64;
	udata int64;
}
export func KeventPtr(e *Kevent) int64;

