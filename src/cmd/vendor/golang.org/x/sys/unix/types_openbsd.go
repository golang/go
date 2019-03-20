// Copyright 2009 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// +build ignore

/*
Input to cgo -godefs.  See README.md
*/

// +godefs map struct_in_addr [4]byte /* in_addr */
// +godefs map struct_in6_addr [16]byte /* in6_addr */

package unix

/*
#define KERNEL
#include <dirent.h>
#include <fcntl.h>
#include <poll.h>
#include <signal.h>
#include <termios.h>
#include <stdio.h>
#include <unistd.h>
#include <sys/param.h>
#include <sys/types.h>
#include <sys/event.h>
#include <sys/mman.h>
#include <sys/mount.h>
#include <sys/ptrace.h>
#include <sys/resource.h>
#include <sys/select.h>
#include <sys/signal.h>
#include <sys/socket.h>
#include <sys/stat.h>
#include <sys/time.h>
#include <sys/uio.h>
#include <sys/un.h>
#include <sys/utsname.h>
#include <sys/wait.h>
#include <uvm/uvmexp.h>
#include <net/bpf.h>
#include <net/if.h>
#include <net/if_dl.h>
#include <net/route.h>
#include <netinet/in.h>
#include <netinet/icmp6.h>
#include <netinet/tcp.h>

enum {
	sizeofPtr = sizeof(void*),
};

union sockaddr_all {
	struct sockaddr s1;	// this one gets used for fields
	struct sockaddr_in s2;	// these pad it out
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

// Processes

type Rusage C.struct_rusage

type Rlimit C.struct_rlimit

type _Gid_t C.gid_t

// Files

type Stat_t C.struct_stat

type Statfs_t C.struct_statfs

type Flock_t C.struct_flock

type Dirent C.struct_dirent

type Fsid C.fsid_t

// File system limits

const (
	PathMax = C.PATH_MAX
)

// Sockets

type RawSockaddrInet4 C.struct_sockaddr_in

type RawSockaddrInet6 C.struct_sockaddr_in6

type RawSockaddrUnix C.struct_sockaddr_un

type RawSockaddrDatalink C.struct_sockaddr_dl

type RawSockaddr C.struct_sockaddr

type RawSockaddrAny C.struct_sockaddr_any

type _Socklen C.socklen_t

type Linger C.struct_linger

type Iovec C.struct_iovec

type IPMreq C.struct_ip_mreq

type IPv6Mreq C.struct_ipv6_mreq

type Msghdr C.struct_msghdr

type Cmsghdr C.struct_cmsghdr

type Inet6Pktinfo C.struct_in6_pktinfo

type IPv6MTUInfo C.struct_ip6_mtuinfo

type ICMPv6Filter C.struct_icmp6_filter

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
	SizeofInet6Pktinfo     = C.sizeof_struct_in6_pktinfo
	SizeofIPv6MTUInfo      = C.sizeof_struct_ip6_mtuinfo
	SizeofICMPv6Filter     = C.sizeof_struct_icmp6_filter
)

// Ptrace requests

const (
	PTRACE_TRACEME = C.PT_TRACE_ME
	PTRACE_CONT    = C.PT_CONTINUE
	PTRACE_KILL    = C.PT_KILL
)

// Events (kqueue, kevent)

type Kevent_t C.struct_kevent

// Select

type FdSet C.fd_set

// Routing and interface messages

const (
	SizeofIfMsghdr         = C.sizeof_struct_if_msghdr
	SizeofIfData           = C.sizeof_struct_if_data
	SizeofIfaMsghdr        = C.sizeof_struct_ifa_msghdr
	SizeofIfAnnounceMsghdr = C.sizeof_struct_if_announcemsghdr
	SizeofRtMsghdr         = C.sizeof_struct_rt_msghdr
	SizeofRtMetrics        = C.sizeof_struct_rt_metrics
)

type IfMsghdr C.struct_if_msghdr

type IfData C.struct_if_data

type IfaMsghdr C.struct_ifa_msghdr

type IfAnnounceMsghdr C.struct_if_announcemsghdr

type RtMsghdr C.struct_rt_msghdr

type RtMetrics C.struct_rt_metrics

type Mclpool C.struct_mclpool

// Berkeley packet filter

const (
	SizeofBpfVersion = C.sizeof_struct_bpf_version
	SizeofBpfStat    = C.sizeof_struct_bpf_stat
	SizeofBpfProgram = C.sizeof_struct_bpf_program
	SizeofBpfInsn    = C.sizeof_struct_bpf_insn
	SizeofBpfHdr     = C.sizeof_struct_bpf_hdr
)

type BpfVersion C.struct_bpf_version

type BpfStat C.struct_bpf_stat

type BpfProgram C.struct_bpf_program

type BpfInsn C.struct_bpf_insn

type BpfHdr C.struct_bpf_hdr

type BpfTimeval C.struct_bpf_timeval

// Terminal handling

type Termios C.struct_termios

type Winsize C.struct_winsize

// fchmodat-like syscalls.

const (
	AT_FDCWD            = C.AT_FDCWD
	AT_SYMLINK_NOFOLLOW = C.AT_SYMLINK_NOFOLLOW
)

// poll

type PollFd C.struct_pollfd

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

// Signal Sets

type Sigset_t C.sigset_t

// Uname

type Utsname C.struct_utsname

// Uvmexp

const SizeofUvmexp = C.sizeof_struct_uvmexp

type Uvmexp C.struct_uvmexp

// Clockinfo

const SizeofClockinfo = C.sizeof_struct_clockinfo

type Clockinfo C.struct_clockinfo
