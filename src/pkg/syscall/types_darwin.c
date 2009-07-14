// Copyright 2009 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

/*
Input to godefs.  See PORT.
 */

#define __DARWIN_UNIX03 0
#define KERNEL
#define _DARWIN_USE_64_BIT_INODE
#include <dirent.h>
#include <fcntl.h>
#include <mach/mach.h>
#include <mach/message.h>
#include <netinet/in.h>
#include <netinet/tcp.h>
#include <signal.h>
#include <stdio.h>
#include <sys/event.h>
#include <sys/mman.h>
#include <sys/mount.h>
#include <sys/param.h>
#include <sys/ptrace.h>
#include <sys/resource.h>
#include <sys/select.h>
#include <sys/signal.h>
#include <sys/stat.h>
#include <sys/time.h>
#include <sys/types.h>
#include <sys/types.h>
#include <sys/un.h>
#include <sys/wait.h>
#include <unistd.h>

// Machine characteristics; for internal use.

enum
{
	$sizeofPtr = sizeof(void*),
	$sizeofShort = sizeof(short),
	$sizeofInt = sizeof(int),
	$sizeofLong = sizeof(long),
	$sizeofLongLong = sizeof(long long),
};


// Basic types

typedef short $_C_short;
typedef int $_C_int;
typedef long $_C_long;
typedef long long $_C_long_long;

// Time

typedef struct timespec $Timespec;
typedef struct timeval $Timeval;

// Processes

typedef struct rusage $Rusage;
typedef struct rlimit $Rlimit;

typedef gid_t $_Gid_t;

// Files

enum
{
	$O_RDONLY = O_RDONLY,
	$O_WRONLY = O_WRONLY,
	$O_RDWR = O_RDWR,
	$O_APPEND = O_APPEND,
	$O_ASYNC = O_ASYNC,
	$O_CREAT = O_CREAT,
	$O_NOCTTY = O_NOCTTY,
	$O_NONBLOCK = O_NONBLOCK,
	$O_SYNC = O_SYNC,
	$O_TRUNC = O_TRUNC,
	$O_CLOEXEC = 0,	// not supported

	$F_GETFD = F_GETFD,
	$F_SETFD = F_SETFD,

	$F_GETFL = F_GETFL,
	$F_SETFL = F_SETFL,

	$FD_CLOEXEC = FD_CLOEXEC,

	$NAME_MAX = NAME_MAX
};

enum
{	// Directory mode bits
	$S_IFMT = S_IFMT,
	$S_IFIFO = S_IFIFO,
	$S_IFCHR = S_IFCHR,
	$S_IFDIR = S_IFDIR,
	$S_IFBLK = S_IFBLK,
	$S_IFREG = S_IFREG,
	$S_IFLNK = S_IFLNK,
	$S_IFSOCK = S_IFSOCK,
	$S_IFWHT = S_IFWHT,
	$S_ISUID = S_ISUID,
	$S_ISGID = S_ISGID,
	$S_ISVTX = S_ISVTX,
	$S_IRUSR = S_IRUSR,
	$S_IWUSR = S_IWUSR,
	$S_IXUSR = S_IXUSR,
};

typedef struct stat64 $Stat_t;
typedef struct statfs64 $Statfs_t;

typedef struct dirent $Dirent;

// Wait status.

enum
{
	$WNOHANG = WNOHANG,
	$WUNTRACED = WUNTRACED,
	$WEXITED = WEXITED,
	$WSTOPPED = WSTOPPED,
	$WCONTINUED = WCONTINUED,
	$WNOWAIT = WNOWAIT,
};

// Sockets

enum
{
	$AF_UNIX = AF_UNIX,
	$AF_INET = AF_INET,
	$AF_DATAKIT = AF_DATAKIT,
	$AF_INET6 = AF_INET6,

	$SOCK_STREAM = SOCK_STREAM,
	$SOCK_DGRAM = SOCK_DGRAM,
	$SOCK_RAW = SOCK_RAW,
	$SOCK_SEQPACKET = SOCK_SEQPACKET,

	$SOL_SOCKET = SOL_SOCKET,

	$SO_REUSEADDR = SO_REUSEADDR,
	$SO_KEEPALIVE = SO_KEEPALIVE,
	$SO_DONTROUTE = SO_DONTROUTE,
	$SO_BROADCAST = SO_BROADCAST,
	$SO_USELOOPBACK = SO_USELOOPBACK,
	$SO_LINGER = SO_LINGER,
	$SO_REUSEPORT = SO_REUSEPORT,
	$SO_SNDBUF = SO_SNDBUF,
	$SO_RCVBUF = SO_RCVBUF,
	$SO_SNDTIMEO = SO_SNDTIMEO,
	$SO_RCVTIMEO = SO_RCVTIMEO,
	$SO_NOSIGPIPE = SO_NOSIGPIPE,

	$IPPROTO_TCP = IPPROTO_TCP,
	$IPPROTO_UDP = IPPROTO_UDP,

	$TCP_NODELAY = TCP_NODELAY,

	$SOMAXCONN = SOMAXCONN
};

typedef struct sockaddr_in $RawSockaddrInet4;
typedef struct sockaddr_in6 $RawSockaddrInet6;
typedef struct sockaddr_un $RawSockaddrUnix;
typedef struct sockaddr $RawSockaddr;

union sockaddr_all {
	struct sockaddr s1;	// this one gets used for fields
	struct sockaddr_in s2;	// these pad it out
	struct sockaddr_in6 s3;
};

struct sockaddr_any {
	struct sockaddr addr;
	char pad[sizeof(union sockaddr_all) - sizeof(struct sockaddr)];
};

enum {
	$SizeofSockaddrInet4 = sizeof(struct sockaddr_in),
	$SizeofSockaddrInet6 = sizeof(struct sockaddr_in6),
	$SizeofSockaddrAny = sizeof(struct sockaddr_any),
	$SizeofSockaddrUnix = sizeof(struct sockaddr_un),
};

typedef struct sockaddr_any $RawSockaddrAny;
typedef socklen_t $_Socklen;
typedef struct linger $Linger;

// Ptrace requests
enum {
	$_PTRACE_TRACEME = PT_TRACE_ME,
	$_PTRACE_CONT = PT_CONTINUE,
	$_PTRACE_KILL = PT_KILL,
};


// Events (kqueue, kevent)

enum {
	// filters
	$EVFILT_READ = EVFILT_READ,
	$EVFILT_WRITE = EVFILT_WRITE,
	$EVFILT_AIO = EVFILT_AIO,
	$EVFILT_VNODE = EVFILT_VNODE,
	$EVFILT_PROC = EVFILT_PROC,
	$EVFILT_SIGNAL = EVFILT_SIGNAL,
	$EVFILT_TIMER = EVFILT_TIMER,
	$EVFILT_MACHPORT = EVFILT_MACHPORT,
	$EVFILT_FS = EVFILT_FS,

	$EVFILT_SYSCOUNT = EVFILT_SYSCOUNT,

	// actions
	$EV_ADD = EV_ADD,
	$EV_DELETE = EV_DELETE,
	$EV_DISABLE = EV_DISABLE,
	$EV_RECEIPT = EV_RECEIPT,

	// flags
	$EV_ONESHOT = EV_ONESHOT,
	$EV_CLEAR = EV_CLEAR,
	$EV_SYSFLAGS = EV_SYSFLAGS,
	$EV_FLAG0 = EV_FLAG0,
	$EV_FLAG1 = EV_FLAG1,

	// returned values
	$EV_EOF = EV_EOF,
	$EV_ERROR = EV_ERROR,
};

typedef struct kevent $Kevent_t;

// Select

typedef fd_set $FdSet;
