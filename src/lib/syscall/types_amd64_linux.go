// Copyright 2009 The Go Authors.  All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// Types and defined constants.
// Should be automatically generated, but is not.

package syscall

import "syscall"

const OS = "linux"

// Time

type Timespec struct {
	Sec	int64;
	Nsec	uint64;
}

type Timeval struct {
	Sec	int64;
	Usec	uint64;
}


// Processes

type Rusage struct {
	Utime	Timeval;
	Stime	Timeval;
	Maxrss	int64;
	Ixrss	int64;
	Idrss	int64;
	Isrss	int64;
	Minflt	int64;
	Majflt	int64;
	Nswap	int64;
	Inblock	int64;
	Oublock	int64;
	Msgsnd	int64;
	Msgrcv	int64;
	Nsignals	int64;
	Nvcsw	int64;
	Nivcsw	int64;
}


// Files

const (
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
	O_CLOEXEC = 0x80000;

	F_GETFD = 1;
	F_SETFD = 2;

	F_GETFL = 3;
	F_SETFL = 4;

	FD_CLOEXEC = 1;

	NAME_MAX = 255;
)

// Dir.Mode bits
const (
	S_IFMT = 0170000;	      /* type of file */
		S_IFIFO  = 0010000;  /* named pipe (fifo) */
		S_IFCHR  = 0020000;  /* character special */
		S_IFDIR  = 0040000;  /* directory */
		S_IFBLK  = 0060000;  /* block special */
		S_IFREG  = 0100000;  /* regular */
		S_IFLNK  = 0120000;  /* symbolic link */
		S_IFSOCK = 0140000;  /* socket */
	S_ISUID = 0004000;  /* set user id on execution */
	S_ISGID = 0002000;  /* set group id on execution */
	S_ISVTX = 0001000;  /* save swapped text even after use */
	S_IRUSR = 0000400;  /* read permission, owner */
	S_IWUSR = 0000200;  /* write permission, owner */
	S_IXUSR = 0000100;  /* execute/search permission, owner */
)

type Stat_t struct {
	Dev	uint64;
	Ino	uint64;
	Nlink	uint64;
	Mode	uint32;
	Uid	uint32;
	Gid	uint32;
	_pad0	uint32;
	Rdev	uint64;
	Size	int64;
	Blksize	int64;
	Blocks	int64;
	Atime	Timespec;
	Mtime	Timespec;
	Ctime	Timespec;
	_unused	[3]int64
}

type Dirent struct {
	Ino	uint64;
	Off	uint64;
	Reclen	uint16;
	Type	uint8;
	Name	[NAME_MAX+1]byte;
}

// Sockets

const (
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

type SockaddrUnix struct {
	Family	uint16;
	Path	[108]byte;
	Length	int64;	// Not part of the kernel structure; used internally
}
const SizeofSockaddrUnix = 110

type SockaddrInet4 struct {
	Family	uint16;
	Port	[2]byte;
	Addr	[4]byte;
	Zero	[8]byte
}
const SizeofSockaddrInet4 = 16

type SockaddrInet6 struct {
	Family	uint16;
	Port	[2]byte;
	Flowinfo	[4]byte;
	Addr	[16]byte;
	Scopeid	[4]byte;
}
const SizeofSockaddrInet6 = 28

type Sockaddr struct {
	Family	uint16;
	Opaque	[126]byte
}
const SizeofSockaddr = 128

type Linger struct {
	Yes int32;
	Sec int32;
}


// Events (epoll)

const (
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

type EpollEvent struct {
	Events uint32;
	Fd int32;
	Pad int32;
}


// Wait status.
// See /usr/include/bits/waitstatus.h

const (
	WNOHANG = 1;
	WUNTRACED = 2;
	WSTOPPED = 2;	// same as WUNTRACED
	WEXITED = 4;
	WCONTINUED = 8;
	WNOWAIT = 0x01000000;
	WNOTHREAD = 0x20000000;
	WALL = 0x40000000;
	WCLONE = 0x80000000;
)

type WaitStatus uint32;

// TODO(rsc): should be method on WaitStatus,
// not *WaitStatus, but causes problems when
// embedding in a *Waitmsg in package os.
// Need to find the 6g bug.

// Wait status is 7 bits at bottom, either 0 (exited),
// 0x7F (stopped), or a signal number that caused an exit.
// The 0x80 bit is whether there was a core dump.
// An extra number (exit code, signal causing a stop)
// is in the high bits.  At least that's the idea.
// There are various irregularities.  For example, the
// "continued" status is 0xFFFF, distinguishing itself
// from stopped via the core dump bit.

const (
	mask = 0x7F;
	core = 0x80;
	exited = 0x00;
	stopped = 0x7F;
	shift = 8;

	// types_amd64_darwin.go refers to SIGSTOP.
	// do the same here so the dependencies are
	// the same on Linux as on Darwin.
	__unused = SIGSTOP;
)

func (wp *WaitStatus) Exited() bool {
	w := *wp;  // TODO(rsc): no pointer
	return w&mask == exited;
}

func (wp *WaitStatus) Signaled() bool {
	w := *wp;  // TODO(rsc): no pointer
	return w&mask != stopped && w&mask != exited;
}

func (wp *WaitStatus) Stopped() bool {
	w := *wp;  // TODO(rsc): no pointer
	return w&0xFF == stopped;
}

func (wp *WaitStatus) Continued() bool {
	w := *wp;  // TODO(rsc): no pointer
	return w == 0xFFFF;
}

func (wp *WaitStatus) CoreDump() bool {
	w := *wp;  // TODO(rsc): no pointer
	return w.Signaled() && w&core != 0;
}

func (wp *WaitStatus) ExitStatus() int {
	w := *wp;  // TODO(rsc): no pointer
	if !w.Exited() {
		return -1;
	}
	return int(w >> shift) & 0xFF;
}

func (wp *WaitStatus) Signal() int {
	w := *wp;  // TODO(rsc): no pointer
	if !w.Signaled() {
		return -1;
	}
	return int(w & mask);
}

func (wp *WaitStatus) StopSignal() int {
	w := *wp;  // TODO(rsc): no pointer
	if !w.Stopped() {
		return -1;
	}
	return int(w >> shift) & 0xFF;
}

