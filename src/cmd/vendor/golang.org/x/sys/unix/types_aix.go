// Copyright 2018 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// +build ignore
// +build aix

/*
Input to cgo -godefs.  See also mkerrors.sh and mkall.sh
*/

// +godefs map struct_in_addr [4]byte /* in_addr */
// +godefs map struct_in6_addr [16]byte /* in6_addr */

package unix

/*
#include <sys/types.h>
#include <sys/time.h>
#include <sys/limits.h>
#include <sys/un.h>
#include <utime.h>
#include <sys/utsname.h>
#include <sys/poll.h>
#include <sys/resource.h>
#include <sys/stat.h>
#include <sys/statfs.h>
#include <sys/termio.h>
#include <sys/ioctl.h>

#include <termios.h>

#include <net/if.h>
#include <net/if_dl.h>
#include <netinet/in.h>
#include <netinet/icmp6.h>


#include <dirent.h>
#include <fcntl.h>

enum {
	sizeofPtr = sizeof(void*),
};

union sockaddr_all {
	struct sockaddr s1;     // this one gets used for fields
	struct sockaddr_in s2;  // these pad it out
	struct sockaddr_in6 s3;
	struct sockaddr_un s4;
	struct sockaddr_dl s5;
};

struct sockaddr_any {
	struct sockaddr addr;
	char pad[sizeof(union sockaddr_all) - sizeof(struct sockaddr)];
};

*/
import "C"

// Machine characteristics

const (
	SizeofPtr      = C.sizeofPtr
	SizeofShort    = C.sizeof_short
	SizeofInt      = C.sizeof_int
	SizeofLong     = C.sizeof_long
	SizeofLongLong = C.sizeof_longlong
	PathMax        = C.PATH_MAX
)

// Basic types

type (
	_C_short     C.short
	_C_int       C.int
	_C_long      C.long
	_C_long_long C.longlong
)

type off64 C.off64_t
type off C.off_t
type Mode_t C.mode_t

// Time

type Timespec C.struct_timespec

type StTimespec C.struct_st_timespec

type Timeval C.struct_timeval

type Timeval32 C.struct_timeval32

type Timex C.struct_timex

type Time_t C.time_t

type Tms C.struct_tms

type Utimbuf C.struct_utimbuf

type Timezone C.struct_timezone

// Processes

type Rusage C.struct_rusage

type Rlimit C.struct_rlimit64

type Pid_t C.pid_t

type _Gid_t C.gid_t

type dev_t C.dev_t

// Files

type Stat_t C.struct_stat

type StatxTimestamp C.struct_statx_timestamp

type Statx_t C.struct_statx

type Dirent C.struct_dirent

// Sockets

type RawSockaddrInet4 C.struct_sockaddr_in

type RawSockaddrInet6 C.struct_sockaddr_in6

type RawSockaddrUnix C.struct_sockaddr_un

type RawSockaddr C.struct_sockaddr

type RawSockaddrAny C.struct_sockaddr_any

type _Socklen C.socklen_t

type Cmsghdr C.struct_cmsghdr

type ICMPv6Filter C.struct_icmp6_filter

type Iovec C.struct_iovec

type IPMreq C.struct_ip_mreq

type IPv6Mreq C.struct_ipv6_mreq

type IPv6MTUInfo C.struct_ip6_mtuinfo

type Linger C.struct_linger

type Msghdr C.struct_msghdr

const (
	SizeofSockaddrInet4 = C.sizeof_struct_sockaddr_in
	SizeofSockaddrInet6 = C.sizeof_struct_sockaddr_in6
	SizeofSockaddrAny   = C.sizeof_struct_sockaddr_any
	SizeofSockaddrUnix  = C.sizeof_struct_sockaddr_un
	SizeofLinger        = C.sizeof_struct_linger
	SizeofIPMreq        = C.sizeof_struct_ip_mreq
	SizeofIPv6Mreq      = C.sizeof_struct_ipv6_mreq
	SizeofIPv6MTUInfo   = C.sizeof_struct_ip6_mtuinfo
	SizeofMsghdr        = C.sizeof_struct_msghdr
	SizeofCmsghdr       = C.sizeof_struct_cmsghdr
	SizeofICMPv6Filter  = C.sizeof_struct_icmp6_filter
)

// Routing and interface messages

const (
	SizeofIfMsghdr = C.sizeof_struct_if_msghdr
)

type IfMsgHdr C.struct_if_msghdr

// Misc

type FdSet C.fd_set

type Utsname C.struct_utsname

type Ustat_t C.struct_ustat

type Sigset_t C.sigset_t

const (
	AT_FDCWD            = C.AT_FDCWD
	AT_REMOVEDIR        = C.AT_REMOVEDIR
	AT_SYMLINK_NOFOLLOW = C.AT_SYMLINK_NOFOLLOW
)

// Terminal handling

type Termios C.struct_termios

type Termio C.struct_termio

type Winsize C.struct_winsize

//poll

type PollFd struct {
	Fd      int32
	Events  uint16
	Revents uint16
}

const (
	POLLERR    = C.POLLERR
	POLLHUP    = C.POLLHUP
	POLLIN     = C.POLLIN
	POLLNVAL   = C.POLLNVAL
	POLLOUT    = C.POLLOUT
	POLLPRI    = C.POLLPRI
	POLLRDBAND = C.POLLRDBAND
	POLLRDNORM = C.POLLRDNORM
	POLLWRBAND = C.POLLWRBAND
	POLLWRNORM = C.POLLWRNORM
)

//flock_t

type Flock_t C.struct_flock64

// Statfs

type Fsid_t C.struct_fsid_t
type Fsid64_t C.struct_fsid64_t

type Statfs_t C.struct_statfs

const RNDGETENTCNT = 0x80045200
