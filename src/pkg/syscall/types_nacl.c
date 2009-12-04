// Copyright 2009 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

/*
Input to godefs.  See also mkerrors.sh and mkall.sh
 */

#define _LARGEFILE_SOURCE
#define _LARGEFILE64_SOURCE
#define _FILE_OFFSET_BITS 64
#define _GNU_SOURCE

#define __native_client__ 1

#define suseconds_t nacl_suseconds_t_1
#include <sys/types.h>
#undef suseconds_t

#include <sys/dirent.h>
#include <sys/mman.h>
#include <sys/fcntl.h>
#include <sys/stat.h>
#include <sys/time.h>
#include <sys/unistd.h>
#include <sys/mman.h>

// Machine characteristics; for internal use.

enum
{
	$sizeofPtr = sizeof(void*),
	$sizeofShort = sizeof(short),
	$sizeofInt = sizeof(int),
	$sizeofLong = sizeof(long),
	$sizeofLongLong = sizeof(long long),
};

// Mmap constants
enum {
	$PROT_READ = PROT_READ,
	$PROT_WRITE = PROT_WRITE,
	$MAP_SHARED = MAP_SHARED,
};

// Unimplemented system calls
enum {
	$SYS_FORK = 0,
	$SYS_PTRACE = 0,
	$SYS_CHDIR = 0,
	$SYS_DUP2 = 0,
	$SYS_FCNTL = 0,
	$SYS_EXECVE = 0,
};

// Basic types

typedef short $_C_short;
typedef int $_C_int;
typedef long $_C_long;
typedef long long $_C_long_long;
typedef off_t $_C_off_t;

// Time

typedef struct timespec $Timespec;
typedef struct timeval $Timeval;
typedef time_t $Time_t;

// Processes

//typedef struct rusage $Rusage;
//typedef struct rlimit $Rlimit;

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
	$O_NOCTTY = 0,	// not supported
	$O_NONBLOCK = O_NONBLOCK,
	$O_SYNC = O_SYNC,
	$O_TRUNC = O_TRUNC,
	$O_EXCL = O_EXCL,
	$O_CLOEXEC = 0,	// not supported

	$F_GETFD = F_GETFD,
	$F_SETFD = F_SETFD,

	$F_GETFL = F_GETFL,
	$F_SETFL = F_SETFL,

	$FD_CLOEXEC = 0,	// not supported
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

typedef struct dirent $Dirent;
