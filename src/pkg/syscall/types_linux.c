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

#include <dirent.h>
#include <fcntl.h>
#include <netinet/in.h>
#include <netinet/tcp.h>
#include <netpacket/packet.h>
#include <signal.h>
#include <stdio.h>
#include <sys/epoll.h>
#include <sys/inotify.h>
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
#include <sys/user.h>
#include <sys/utsname.h>
#include <sys/wait.h>
#include <linux/filter.h>
#include <linux/netlink.h>
#include <linux/rtnetlink.h>
#include <termios.h>
#include <time.h>
#include <unistd.h>
#include <ustat.h>
#include <utime.h>

// Machine characteristics; for internal use.

enum {
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

typedef struct stat $Stat_t;
typedef struct statfs $Statfs_t;

typedef struct dirent $Dirent;

// Sockets

union sockaddr_all {
	struct sockaddr s1;	// this one gets used for fields
	struct sockaddr_in s2;	// these pad it out
	struct sockaddr_in6 s3;
	struct sockaddr_un s4;
	struct sockaddr_ll s5;
	struct sockaddr_nl s6;
};

struct sockaddr_any {
	struct sockaddr addr;
	char pad[sizeof(union sockaddr_all) - sizeof(struct sockaddr)];
};

// copied from /usr/include/linux/un.h 
struct my_sockaddr_un {
	sa_family_t sun_family;
#ifdef __ARM_EABI__
	// on ARM char is by default unsigned
	signed char sun_path[108];
#else
	char sun_path[108];
#endif
};

typedef struct sockaddr_in $RawSockaddrInet4;
typedef struct sockaddr_in6 $RawSockaddrInet6;
typedef struct my_sockaddr_un $RawSockaddrUnix;
typedef struct sockaddr_ll $RawSockaddrLinklayer;
typedef struct sockaddr_nl $RawSockaddrNetlink;
typedef struct sockaddr $RawSockaddr;
typedef struct sockaddr_any $RawSockaddrAny;
typedef socklen_t $_Socklen;
typedef struct linger $Linger;
typedef struct iovec $Iovec;
typedef struct ip_mreq $IPMreq;
typedef struct ip_mreqn $IPMreqn;
typedef struct ipv6_mreq $IPv6Mreq;
typedef struct msghdr $Msghdr;
typedef struct cmsghdr $Cmsghdr;
typedef struct in_pktinfo $Inet4Pktinfo;
typedef struct in6_pktinfo $Inet6Pktinfo;
typedef struct ucred $Ucred;

enum {
	$SizeofSockaddrInet4 = sizeof(struct sockaddr_in),
	$SizeofSockaddrInet6 = sizeof(struct sockaddr_in6),
	$SizeofSockaddrAny = sizeof(struct sockaddr_any),
	$SizeofSockaddrUnix = sizeof(struct sockaddr_un),
	$SizeofSockaddrLinklayer = sizeof(struct sockaddr_ll),
	$SizeofSockaddrNetlink = sizeof(struct sockaddr_nl),
	$SizeofLinger = sizeof(struct linger),
	$SizeofIPMreq = sizeof(struct ip_mreq),
	$SizeofIPMreqn = sizeof(struct ip_mreqn),
	$SizeofIPv6Mreq = sizeof(struct ipv6_mreq),
	$SizeofMsghdr = sizeof(struct msghdr),
	$SizeofCmsghdr = sizeof(struct cmsghdr),
	$SizeofInet4Pktinfo = sizeof(struct in_pktinfo),
	$SizeofInet6Pktinfo = sizeof(struct in6_pktinfo),
	$SizeofUcred = sizeof(struct ucred),
};

// Netlink routing and interface messages

enum {
	$IFA_UNSPEC = IFA_UNSPEC,
	$IFA_ADDRESS = IFA_ADDRESS,
	$IFA_LOCAL = IFA_LOCAL,
	$IFA_LABEL = IFA_LABEL,
	$IFA_BROADCAST = IFA_BROADCAST,
	$IFA_ANYCAST = IFA_ANYCAST,
	$IFA_CACHEINFO = IFA_CACHEINFO,
	$IFA_MULTICAST = IFA_MULTICAST,
	$IFLA_UNSPEC = IFLA_UNSPEC,
	$IFLA_ADDRESS = IFLA_ADDRESS,
	$IFLA_BROADCAST = IFLA_BROADCAST,
	$IFLA_IFNAME = IFLA_IFNAME,
	$IFLA_MTU = IFLA_MTU,
	$IFLA_LINK = IFLA_LINK,
	$IFLA_QDISC = IFLA_QDISC,
	$IFLA_STATS = IFLA_STATS,
	$IFLA_COST = IFLA_COST,
	$IFLA_PRIORITY = IFLA_PRIORITY,
	$IFLA_MASTER = IFLA_MASTER,
	$IFLA_WIRELESS = IFLA_WIRELESS,
	$IFLA_PROTINFO = IFLA_PROTINFO,
	$IFLA_TXQLEN = IFLA_TXQLEN,
	$IFLA_MAP = IFLA_MAP,
	$IFLA_WEIGHT = IFLA_WEIGHT,
	$IFLA_OPERSTATE = IFLA_OPERSTATE,
	$IFLA_LINKMODE = IFLA_LINKMODE,
	$IFLA_LINKINFO = IFLA_LINKINFO,
	$IFLA_NET_NS_PID = IFLA_NET_NS_PID,
	$IFLA_IFALIAS = IFLA_IFALIAS,
	$IFLA_MAX = IFLA_MAX,
	$RT_SCOPE_UNIVERSE = RT_SCOPE_UNIVERSE,
	$RT_SCOPE_SITE = RT_SCOPE_SITE,
	$RT_SCOPE_LINK = RT_SCOPE_LINK,
	$RT_SCOPE_HOST = RT_SCOPE_HOST,
	$RT_SCOPE_NOWHERE = RT_SCOPE_NOWHERE,
	$RT_TABLE_UNSPEC = RT_TABLE_UNSPEC,
	$RT_TABLE_COMPAT = RT_TABLE_COMPAT,
	$RT_TABLE_DEFAULT = RT_TABLE_DEFAULT,
	$RT_TABLE_MAIN = RT_TABLE_MAIN,
	$RT_TABLE_LOCAL = RT_TABLE_LOCAL,
	$RT_TABLE_MAX = RT_TABLE_MAX,
	$RTA_UNSPEC = RTA_UNSPEC,
	$RTA_DST = RTA_DST,
	$RTA_SRC = RTA_SRC,
	$RTA_IIF = RTA_IIF,
	$RTA_OIF = RTA_OIF,
	$RTA_GATEWAY = RTA_GATEWAY,
	$RTA_PRIORITY = RTA_PRIORITY,
	$RTA_PREFSRC = RTA_PREFSRC,
	$RTA_METRICS = RTA_METRICS,
	$RTA_MULTIPATH = RTA_MULTIPATH,
	$RTA_FLOW = RTA_FLOW,
	$RTA_CACHEINFO = RTA_CACHEINFO,
	$RTA_TABLE = RTA_TABLE,
	$RTN_UNSPEC = RTN_UNSPEC,
	$RTN_UNICAST = RTN_UNICAST,
	$RTN_LOCAL = RTN_LOCAL,
	$RTN_BROADCAST = RTN_BROADCAST,
	$RTN_ANYCAST = RTN_ANYCAST,
	$RTN_MULTICAST = RTN_MULTICAST,
	$RTN_BLACKHOLE = RTN_BLACKHOLE,
	$RTN_UNREACHABLE = RTN_UNREACHABLE,
	$RTN_PROHIBIT = RTN_PROHIBIT,
	$RTN_THROW = RTN_THROW,
	$RTN_NAT = RTN_NAT,
	$RTN_XRESOLVE = RTN_XRESOLVE,
	$SizeofNlMsghdr = sizeof(struct nlmsghdr),
	$SizeofNlMsgerr = sizeof(struct nlmsgerr),
	$SizeofRtGenmsg = sizeof(struct rtgenmsg),
	$SizeofNlAttr = sizeof(struct nlattr),
	$SizeofRtAttr = sizeof(struct rtattr),
	$SizeofIfInfomsg = sizeof(struct ifinfomsg),
	$SizeofIfAddrmsg = sizeof(struct ifaddrmsg),
	$SizeofRtMsg = sizeof(struct rtmsg),
	$SizeofRtNexthop = sizeof(struct rtnexthop),
};

typedef struct nlmsghdr $NlMsghdr;
typedef struct nlmsgerr $NlMsgerr;
typedef struct rtgenmsg $RtGenmsg;
typedef struct nlattr $NlAttr;
typedef struct rtattr $RtAttr;
typedef struct ifinfomsg $IfInfomsg;
typedef struct ifaddrmsg $IfAddrmsg;
typedef struct rtmsg $RtMsg;
typedef struct rtnexthop $RtNexthop;

// Linux socket filter

enum {
	$SizeofSockFilter = sizeof(struct sock_filter),
	$SizeofSockFprog = sizeof(struct sock_fprog),
};

typedef struct sock_filter $SockFilter;
typedef struct sock_fprog $SockFprog;

// Inotify

typedef struct inotify_event $InotifyEvent;

enum {
	$SizeofInotifyEvent = sizeof(struct inotify_event)
};

// Ptrace

// Register structures
#ifdef __ARM_EABI__
	typedef struct user_regs $PtraceRegs;
#else
	typedef struct user_regs_struct $PtraceRegs;
#endif

// Misc

typedef fd_set $FdSet;
typedef struct sysinfo $Sysinfo_t;
typedef struct utsname $Utsname;
typedef struct ustat $Ustat_t;

// The real epoll_event is a union, and godefs doesn't handle it well.
struct my_epoll_event {
	uint32_t events;
#ifdef __ARM_EABI__
	// padding is not specified in linux/eventpoll.h but added to conform to the
	// alignment requirements of EABI
	int32_t padFd;
#endif
	int32_t fd;
	int32_t pad;
};

typedef struct my_epoll_event $EpollEvent;

// Terminal handling

typedef struct termios $Termios;

enum {
	$VINTR = VINTR,
	$VQUIT = VQUIT,
	$VERASE = VERASE,
	$VKILL = VKILL,
	$VEOF = VEOF,
	$VTIME = VTIME,
	$VMIN = VMIN,
	$VSWTC = VSWTC,
	$VSTART = VSTART,
	$VSTOP = VSTOP,
	$VSUSP = VSUSP,
	$VEOL = VEOL,
	$VREPRINT = VREPRINT,
	$VDISCARD = VDISCARD,
	$VWERASE = VWERASE,
	$VLNEXT = VLNEXT,
	$VEOL2 = VEOL2,
	$IGNBRK = IGNBRK,
	$BRKINT = BRKINT,
	$IGNPAR = IGNPAR,
	$PARMRK = PARMRK,
	$INPCK = INPCK,
	$ISTRIP = ISTRIP,
	$INLCR = INLCR,
	$IGNCR = IGNCR,
	$ICRNL = ICRNL,
	$IUCLC = IUCLC,
	$IXON = IXON,
	$IXANY = IXANY,
	$IXOFF = IXOFF,
	$IMAXBEL = IMAXBEL,
	$IUTF8 = IUTF8,
	$OPOST = OPOST,
	$OLCUC = OLCUC,
	$ONLCR = ONLCR,
	$OCRNL = OCRNL,
	$ONOCR = ONOCR,
	$ONLRET = ONLRET,
	$OFILL = OFILL,
	$OFDEL = OFDEL,
	$B0 = B0,
	$B50 = B50,
	$B75 = B75,
	$B110 = B110,
	$B134 = B134,
	$B150 = B150,
	$B200 = B200,
	$B300 = B300,
	$B600 = B600,
	$B1200 = B1200,
	$B1800 = B1800,
	$B2400 = B2400,
	$B4800 = B4800,
	$B9600 = B9600,
	$B19200 = B19200,
	$B38400 = B38400,
	$CSIZE = CSIZE,
	$CS5 = CS5,
	$CS6 = CS6,
	$CS7 = CS7,
	$CS8 = CS8,
	$CSTOPB = CSTOPB,
	$CREAD = CREAD,
	$PARENB = PARENB,
	$PARODD = PARODD,
	$HUPCL = HUPCL,
	$CLOCAL = CLOCAL,
	$B57600 = B57600,
	$B115200 = B115200,
	$B230400 = B230400,
	$B460800 = B460800,
	$B500000 = B500000,
	$B576000 = B576000,
	$B921600 = B921600,
	$B1000000 = B1000000,
	$B1152000 = B1152000,
	$B1500000 = B1500000,
	$B2000000 = B2000000,
	$B2500000 = B2500000,
	$B3000000 = B3000000,
	$B3500000 = B3500000,
	$B4000000 = B4000000,
	$ISIG = ISIG,
	$ICANON = ICANON,
	$XCASE = XCASE,
	$ECHO = ECHO,
	$ECHOE = ECHOE,
	$ECHOK = ECHOK,
	$ECHONL = ECHONL,
	$NOFLSH = NOFLSH,
	$TOSTOP = TOSTOP,
	$ECHOCTL = ECHOCTL,
	$ECHOPRT = ECHOPRT,
	$ECHOKE = ECHOKE,
	$FLUSHO = FLUSHO,
	$PENDIN = PENDIN,
	$IEXTEN = IEXTEN,
	$TCGETS = TCGETS,
	$TCSETS = TCSETS,
};
