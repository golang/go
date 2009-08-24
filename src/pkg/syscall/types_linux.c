// Copyright 2009 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

/*
Input to godefs.  See PORT.sh
 */

#define _LARGEFILE_SOURCE
#define _LARGEFILE64_SOURCE
#define _FILE_OFFSET_BITS 64
#define _GNU_SOURCE

#include <dirent.h>
#include <fcntl.h>
#include <linux/user.h>
#include <netinet/in.h>
#include <netinet/tcp.h>
#include <signal.h>
#include <stdio.h>
#include <sys/epoll.h>
#include <sys/mman.h>
#include <sys/mount.h>
#include <sys/param.h>
#include <sys/ptrace.h>
#include <sys/resource.h>
#include <sys/select.h>
#include <sys/signal.h>
#include <sys/stat.h>
#include <sys/statfs.h>
#include <sys/sysinfo.h>
#include <sys/time.h>
#include <sys/times.h>
#include <sys/timex.h>
#include <sys/types.h>
#include <sys/un.h>
#include <sys/utsname.h>
#include <sys/wait.h>
#include <time.h>
#include <unistd.h>
#include <ustat.h>
#include <utime.h>

// Machine characteristics; for internal use.

enum
{
	$sizeofPtr = sizeof(void*),
	$sizeofShort = sizeof(short),
	$sizeofInt = sizeof(int),
	$sizeofLong = sizeof(long),
	$sizeofLongLong = sizeof(long long),
	$PathMax = PATH_MAX,
};

// Basic types

typedef short $_C_short;
typedef int $_C_int;
typedef long $_C_long;
typedef long long $_C_long_long;

// Time

typedef struct timespec $Timespec;
typedef struct timeval $Timeval;
typedef struct timex $Timex;
typedef time_t $Time_t;
typedef struct tms $Tms;
typedef struct utimbuf $Utimbuf;

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
	$S_ISUID = S_ISUID,
	$S_ISGID = S_ISGID,
	$S_ISVTX = S_ISVTX,
	$S_IRUSR = S_IRUSR,
	$S_IWUSR = S_IWUSR,
	$S_IXUSR = S_IXUSR,
};

typedef struct stat $Stat_t;
typedef struct statfs $Statfs_t;

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

        // Linux-specific
        $WCLONE = __WCLONE,
        $WALL = __WALL,
        $WNOTHREAD = __WNOTHREAD,
};

// Sockets

enum
{
	$AF_UNIX = AF_UNIX,
	$AF_INET = AF_INET,
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
	$SO_LINGER = SO_LINGER,
	$SO_SNDBUF = SO_SNDBUF,
	$SO_RCVBUF = SO_RCVBUF,
	$SO_SNDTIMEO = SO_SNDTIMEO,
	$SO_RCVTIMEO = SO_RCVTIMEO,

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

// Ptrace

// Ptrace requests
enum {
	$_PTRACE_TRACEME = PTRACE_TRACEME,
	$_PTRACE_PEEKTEXT = PTRACE_PEEKTEXT,
	$_PTRACE_PEEKDATA = PTRACE_PEEKDATA,
	$_PTRACE_PEEKUSER = PTRACE_PEEKUSER,
	$_PTRACE_POKETEXT = PTRACE_POKETEXT,
	$_PTRACE_POKEDATA = PTRACE_POKEDATA,
	$_PTRACE_POKEUSER = PTRACE_POKEUSER,
	$_PTRACE_CONT = PTRACE_CONT,
	$_PTRACE_KILL = PTRACE_KILL,
	$_PTRACE_SINGLESTEP = PTRACE_SINGLESTEP,
	$_PTRACE_GETREGS = PTRACE_GETREGS,
	$_PTRACE_SETREGS = PTRACE_SETREGS,
	$_PTRACE_GETFPREGS = PTRACE_GETFPREGS,
	$_PTRACE_SETFPREGS = PTRACE_SETFPREGS,
	$_PTRACE_ATTACH = PTRACE_ATTACH,
	$_PTRACE_DETACH = PTRACE_DETACH,
	$_PTRACE_GETFPXREGS = PTRACE_GETFPXREGS,
	$_PTRACE_SETFPXREGS = PTRACE_SETFPXREGS,
	$_PTRACE_SYSCALL = PTRACE_SYSCALL,
	$_PTRACE_SETOPTIONS = PTRACE_SETOPTIONS,
	$_PTRACE_GETEVENTMSG = PTRACE_GETEVENTMSG,
	$_PTRACE_GETSIGINFO = PTRACE_GETSIGINFO,
	$_PTRACE_SETSIGINFO = PTRACE_SETSIGINFO,
};

// PTRACE_SETOPTIONS options
enum {
	$PTRACE_O_TRACESYSGOOD = PTRACE_O_TRACESYSGOOD,
	$PTRACE_O_TRACEFORK = PTRACE_O_TRACEFORK,
	$PTRACE_O_TRACEVFORK = PTRACE_O_TRACEVFORK,
	$PTRACE_O_TRACECLONE = PTRACE_O_TRACECLONE,
	$PTRACE_O_TRACEEXEC = PTRACE_O_TRACEEXEC,
	$PTRACE_O_TRACEVFORKDONE = PTRACE_O_TRACEVFORKDONE,
	$PTRACE_O_TRACEEXIT = PTRACE_O_TRACEEXIT,
	$PTRACE_O_MASK = PTRACE_O_MASK,
};

// Extended result codes
enum {
	$PTRACE_EVENT_FORK = PTRACE_EVENT_FORK,
	$PTRACE_EVENT_VFORK = PTRACE_EVENT_VFORK,
	$PTRACE_EVENT_CLONE = PTRACE_EVENT_CLONE,
	$PTRACE_EVENT_EXEC = PTRACE_EVENT_EXEC,
	$PTRACE_EVENT_VFORK_DONE = PTRACE_EVENT_VFORK_DONE,
	$PTRACE_EVENT_EXIT = PTRACE_EVENT_EXIT,
};

// Register structures
typedef struct user_regs_struct $PtraceRegs;

// Misc

enum {
	$EPOLLIN = EPOLLIN,
	$EPOLLRDHUP = EPOLLRDHUP,
	$EPOLLOUT = EPOLLOUT,
	$EPOLLONESHOT = EPOLLONESHOT,
	$EPOLL_CTL_MOD = EPOLL_CTL_MOD,
	$EPOLL_CTL_ADD = EPOLL_CTL_ADD,
	$EPOLL_CTL_DEL = EPOLL_CTL_DEL,
};

typedef fd_set $FdSet;
typedef struct sysinfo $Sysinfo_t;
typedef struct utsname $Utsname;
typedef struct ustat $Ustat_t;

// The real epoll_event is a union, and godefs doesn't handle it well.
struct my_epoll_event {
	uint32_t events;
	int32_t fd;
	int32_t pad;
};

typedef struct my_epoll_event $EpollEvent;
