// Copyright 2009 The Go Authors.  All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// Types and defined constants.
// Should be automatically generated, but is not.

package syscall

import "syscall"

const OS = "darwin"

// Time

type Timespec struct {
	Sec	int64;
	Nsec	uint64;
}

type Timeval struct {
	Sec	int64;
	Usec	uint32;
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
	O_APPEND = 0x8;
	O_ASYNC = 0x40;
	O_CREAT = 0x200;
	O_NOCTTY = 0x20000;
	O_NONBLOCK = 0x4;
	O_NDELAY = O_NONBLOCK;
	O_SYNC = 0x80;
	O_TRUNC = 0x400;
	O_CLOEXEC = 0;  // not supported

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
		S_IFWHT  = 0160000;  /* whiteout */
	S_ISUID = 0004000;  /* set user id on execution */
	S_ISGID = 0002000;  /* set group id on execution */
	S_ISVTX = 0001000;  /* save swapped text even after use */
	S_IRUSR = 0000400;  /* read permission, owner */
	S_IWUSR = 0000200;  /* write permission, owner */
	S_IXUSR = 0000100;  /* execute/search permission, owner */
)

type Stat_t struct {
	Dev	uint32;
	Mode	uint16;
	Nlink	uint16;
	Ino	uint64;
	Uid	uint32;
	Gid	uint32;
	Rdev	uint32;
	Pad1	uint32;
	Atime	Timespec;
	Mtime	Timespec;
	Ctime	Timespec;
	Birthtime	Timespec;
	Size	uint64;
	Blocks	uint64;
	Blksize	uint32;
	Flags	uint32;
	Gen	uint32;
	Lspare	uint32;
	Qspare	[2]uint64;
}

type Dirent struct {
	Ino	uint64;
	Off	uint64;
	Reclen	uint16;
	Namlen	uint16;
	Type	uint8;
	Name	[NAME_MAX+1]byte;
}

// Sockets

const (
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

type SockaddrUnix struct {
	Len	byte;
	Family	byte;
	Path	[104]byte
}
const SizeofSockaddrUnix = 106

type SockaddrInet4 struct {
	Len	byte;
	Family	byte;
	Port	[2]byte;
	Addr	[4]byte;
	Zero	[8]byte
}
const SizeofSockaddrInet4 = 16

type SockaddrInet6 struct {
	Len	byte;
	Family	byte;
	Port	[2]byte;
	Flowinfo	[4]byte;
	Addr	[16]byte;
	Scopeid	[4]byte;
}
const SizeofSockaddrInet6 = 28

type Sockaddr struct {
	Len	byte;
	Family	byte;
	Opaque	[126]byte
}
const SizeofSockaddr = 128

type Linger struct {
	Yes int32;
	Sec int32;
}


// Events (kqueue, kevent)

const (
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
//	EV_RECEIPT = 0x40;
	EV_SYSFLAGS = 0xF000;
	EV_FLAG0 = 0x1000;
	EV_FLAG1 = 0x2000;

	// returned values
	EV_EOF = 0x8000;
	EV_ERROR = 0x4000
)

type Kevent_t struct {
	Ident int64;
	Filter int16;
	Flags uint16;
	Fflags uint32;
	Data int64;
	Udata int64;
}


// Wait status.
// See /usr/include/bits/waitstatus.h

const (
	WNOHANG = 1;
	WUNTRACED = 2;
	WEXITED = 4;
	WSTOPPED = 8;
	WCONTINUED = 0x10;
	WNOWAIT = 0x20;
)

type WaitStatus uint32;

// Wait status is 7 bits at bottom, either 0 (exited),
// 0x7F (stopped), or a signal number that caused an exit.
// The 0x80 bit is whether there was a core dump.
// An extra number (exit code, signal causing a stop)
// is in the high bits.

const (
	mask = 0x7F;
	core = 0x80;
	shift = 8;

	exited = 0;
	stopped = 0x7F;
)

func (w WaitStatus) Exited() bool {
	return w&mask == exited;
}

func (w WaitStatus) ExitStatus() int {
	if w&mask != exited {
		return -1;
	}
	return int(w >> shift);
}

func (w WaitStatus) Signaled() bool {
	return w&mask != stopped && w&mask != 0;
}

func (w WaitStatus) Signal() int {
	sig := int(w & mask);
	if sig == stopped || sig == 0 {
		return -1;
	}
	return sig;
}

func (w WaitStatus) CoreDump() bool {
	return w.Signaled() && w&core != 0;
}

func (w WaitStatus) Stopped() bool {
	return w&mask == stopped && w>>shift != SIGSTOP;
}

func (w WaitStatus) Continued() bool {
	return w&mask == stopped && w>>shift == SIGSTOP;
}

func (w WaitStatus) StopSignal() int {
	if !w.Stopped() {
		return -1;
	}
	return int(w >> shift) & 0xFF;
}

