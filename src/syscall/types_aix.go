// Copyright 2018 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

//go:build ignore

/*
Input to cgo -godefs.  See also mkerrors.sh and mkall.sh
*/

// +godefs map struct_in_addr [4]byte /* in_addr */
// +godefs map struct_in6_addr [16]byte /* in6_addr */

package syscall

/*
#include <sys/types.h>
#include <sys/time.h>
#include <sys/limits.h>
#include <sys/un.h>
#include <sys/utsname.h>
#include <sys/ptrace.h>
#include <sys/statfs.h>

#include <net/if.h>
#include <net/if_dl.h>
#include <netinet/in.h>
#include <netinet/icmp6.h>

#include <termios.h>

#include <dirent.h>
#include <fcntl.h>
#include <gcrypt.h>

enum {
	sizeofPtr = sizeof(void*),
};

union sockaddr_all {
	struct sockaddr s1;	// this one gets used for fields
	struct sockaddr_in s2;	// these pad it out
	struct sockaddr_in6 s3;
	struct sockaddr_un s4;
};

struct sockaddr_any {
	struct sockaddr addr;
	char pad[sizeof(union sockaddr_all) - sizeof(struct sockaddr)];
};

*/
import "C"

// Machine characteristics; for internal use.

const (
	sizeofPtr      = C.sizeofPtr
	sizeofShort    = C.sizeof_short
	sizeofInt      = C.sizeof_int
	sizeofLong     = C.sizeof_long
	sizeofLongLong = C.sizeof_longlong
	PathMax        = C.PATH_MAX
)

// Basic types

type (
	_C_short     C.short
	_C_int       C.int
	_C_long      C.long
	_C_long_long C.longlong
)

// Time

type Timespec C.struct_timespec

type Timeval C.struct_timeval

type Timeval32 C.struct_timeval32

type Timezone C.struct_timezone

// Processes

type Rusage C.struct_rusage

type Rlimit C.struct_rlimit

type _Pid_t C.pid_t

type _Gid_t C.gid_t

// Files

type Flock_t C.struct_flock

type Stat_t C.struct_stat

type Statfs_t C.struct_statfs

type Fsid64_t C.fsid64_t

type StTimespec_t C.st_timespec_t

type Dirent C.struct_dirent

// Sockets

type RawSockaddrInet4 C.struct_sockaddr_in

type RawSockaddrInet6 C.struct_sockaddr_in6

type RawSockaddrUnix C.struct_sockaddr_un

type RawSockaddrDatalink C.struct_sockaddr_dl

type RawSockaddr C.struct_sockaddr

type RawSockaddrAny C.struct_sockaddr_any

type _Socklen C.socklen_t

type Cmsghdr C.struct_cmsghdr

type ICMPv6Filter C.struct_icmp6_filter

type Iovec C.struct_iovec

type IPMreq C.struct_ip_mreq

type IPv6Mreq C.struct_ipv6_mreq

type Linger C.struct_linger

type Msghdr C.struct_msghdr

const (
	SizeofSockaddrInet4    = C.sizeof_struct_sockaddr_in
	SizeofSockaddrInet6    = C.sizeof_struct_sockaddr_in6
	SizeofSockaddrAny      = C.sizeof_struct_sockaddr_any
	SizeofSockaddrUnix     = C.sizeof_struct_sockaddr_un
	SizeofSockaddrDatalink = C.sizeof_struct_sockaddr_dl
	SizeofLinger           = C.sizeof_struct_linger
	SizeofIPMreq           = C.sizeof_struct_ip_mreq
	SizeofIPv6Mreq         = C.sizeof_struct_ipv6_mreq
	SizeofMsghdr           = C.sizeof_struct_msghdr
	SizeofCmsghdr          = C.sizeof_struct_cmsghdr
	SizeofICMPv6Filter     = C.sizeof_struct_icmp6_filter
)

// Ptrace requests

const (
	PTRACE_TRACEME = C.PT_TRACE_ME
	PTRACE_CONT    = C.PT_CONTINUE
	PTRACE_KILL    = C.PT_KILL
)

// Routing and interface messages

const (
	SizeofIfMsghdr = C.sizeof_struct_if_msghdr
)

type IfMsgHdr C.struct_if_msghdr

// Misc

type Utsname C.struct_utsname

const (
	_AT_FDCWD            = C.AT_FDCWD
	_AT_REMOVEDIR        = C.AT_REMOVEDIR
	_AT_SYMLINK_NOFOLLOW = C.AT_SYMLINK_NOFOLLOW
)

// Terminal handling

type Termios C.struct_termios
