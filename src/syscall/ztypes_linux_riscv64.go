// Created by cgo -godefs - DO NOT EDIT
// cgo -godefs types_linux.go

package syscall

const (
	sizeofPtr      = 0x8
	sizeofShort    = 0x2
	sizeofInt      = 0x4
	sizeofLong     = 0x8
	sizeofLongLong = 0x8
	PathMax        = 0x1000
)

type (
	_C_short     int16
	_C_int       int32
	_C_long      int64
	_C_long_long int64
)

type Timespec struct {
	Sec  int64
	Nsec int64
}

type Timeval struct {
	Sec  int64
	Usec int64
}

type Timex struct {
	Modes     uint32
	Offset    int64
	Freq      int64
	Maxerror  int64
	Esterror  int64
	Status    int32
	Constant  int64
	Precision int64
	Tolerance int64
	Time      Timeval
	Tick      int64
	Ppsfreq   int64
	Jitter    int64
	Shift     int32
	Stabil    int64
	Jitcnt    int64
	Calcnt    int64
	Errcnt    int64
	Stbcnt    int64
	Tai       int32
	_         [44]byte
}

type Time_t int64

type Tms struct {
	Utime  int64
	Stime  int64
	Cutime int64
	Cstime int64
}

type Utimbuf struct {
	Actime  int64
	Modtime int64
}

type Rusage struct {
	Utime    Timeval
	Stime    Timeval
	Maxrss   int64
	Ixrss    int64
	Idrss    int64
	Isrss    int64
	Minflt   int64
	Majflt   int64
	Nswap    int64
	Inblock  int64
	Oublock  int64
	Msgsnd   int64
	Msgrcv   int64
	Nsignals int64
	Nvcsw    int64
	Nivcsw   int64
}

type Rlimit struct {
	Cur uint64
	Max uint64
}

type _Gid_t uint32

type Stat_t struct {
	Dev               uint64
	Ino               uint64
	Mode              uint32
	Nlink             uint32
	Uid               uint32
	Gid               uint32
	Rdev              uint64
	X__pad1           uint64
	Size              int64
	Blksize           int32
	X__pad2           int32
	Blocks            int64
	Atim              Timespec
	Mtim              Timespec
	Ctim              Timespec
	X__glibc_reserved [2]int32
}

type Statfs_t struct {
	Type    int64
	Bsize   int64
	Blocks  uint64
	Bfree   uint64
	Bavail  uint64
	Files   uint64
	Ffree   uint64
	Fsid    Fsid
	Namelen int64
	Frsize  int64
	Flags   int64
	Spare   [4]int64
}

type Dirent struct {
	Ino    uint64
	Off    int64
	Reclen uint16
	Type   uint8
	Name   [256]uint8
	_      [5]byte
}

type Fsid struct {
	X__val [2]int32
}

type Flock_t struct {
	Type   int16
	Whence int16
	Start  int64
	Len    int64
	Pid    int32
	_      [4]byte
}

type RawSockaddrInet4 struct {
	Family uint16
	Port   uint16
	Addr   [4]byte /* in_addr */
	Zero   [8]uint8
}

type RawSockaddrInet6 struct {
	Family   uint16
	Port     uint16
	Flowinfo uint32
	Addr     [16]byte /* in6_addr */
	Scope_id uint32
}

type RawSockaddrUnix struct {
	Family uint16
	Path   [108]int8
}

type RawSockaddrLinklayer struct {
	Family   uint16
	Protocol uint16
	Ifindex  int32
	Hatype   uint16
	Pkttype  uint8
	Halen    uint8
	Addr     [8]uint8
}

type RawSockaddrNetlink struct {
	Family uint16
	Pad    uint16
	Pid    uint32
	Groups uint32
}

type RawSockaddr struct {
	Family uint16
	Data   [14]uint8
}

type RawSockaddrAny struct {
	Addr RawSockaddr
	Pad  [96]uint8
}

type _Socklen uint32

type Linger struct {
	Onoff  int32
	Linger int32
}

type Iovec struct {
	Base *byte
	Len  uint64
}

type IPMreq struct {
	Multiaddr [4]byte /* in_addr */
	Interface [4]byte /* in_addr */
}

type IPMreqn struct {
	Multiaddr [4]byte /* in_addr */
	Address   [4]byte /* in_addr */
	Ifindex   int32
}

type IPv6Mreq struct {
	Multiaddr [16]byte /* in6_addr */
	Interface uint32
}

type Msghdr struct {
	Name       *byte
	Namelen    uint32
	Iov        *Iovec
	Iovlen     uint64
	Control    *byte
	Controllen uint64
	Flags      int32
	_          [4]byte
}

type Cmsghdr struct {
	Len   uint64
	Level int32
	Type  int32
}

type Inet4Pktinfo struct {
	Ifindex  int32
	Spec_dst [4]byte /* in_addr */
	Addr     [4]byte /* in_addr */
}

type Inet6Pktinfo struct {
	Addr    [16]byte /* in6_addr */
	Ifindex uint32
}

type IPv6MTUInfo struct {
	Addr RawSockaddrInet6
	Mtu  uint32
}

type ICMPv6Filter struct {
	Data [8]uint32
}

type Ucred struct {
	Pid int32
	Uid uint32
	Gid uint32
}

type TCPInfo struct {
	State          uint8
	Ca_state       uint8
	Retransmits    uint8
	Probes         uint8
	Backoff        uint8
	Options        uint8
	Rto            uint32
	Ato            uint32
	Snd_mss        uint32
	Rcv_mss        uint32
	Unacked        uint32
	Sacked         uint32
	Lost           uint32
	Retrans        uint32
	Fackets        uint32
	Last_data_sent uint32
	Last_ack_sent  uint32
	Last_data_recv uint32
	Last_ack_recv  uint32
	Pmtu           uint32
	Rcv_ssthresh   uint32
	Rtt            uint32
	Rttvar         uint32
	Snd_ssthresh   uint32
	Snd_cwnd       uint32
	Advmss         uint32
	Reordering     uint32
	Rcv_rtt        uint32
	Rcv_space      uint32
	Total_retrans  uint32
}

const (
	SizeofSockaddrInet4     = 0x10
	SizeofSockaddrInet6     = 0x1c
	SizeofSockaddrAny       = 0x70
	SizeofSockaddrUnix      = 0x6e
	SizeofSockaddrLinklayer = 0x14
	SizeofSockaddrNetlink   = 0xc
	SizeofLinger            = 0x8
	SizeofIPMreq            = 0x8
	SizeofIPMreqn           = 0xc
	SizeofIPv6Mreq          = 0x14
	SizeofMsghdr            = 0x38
	SizeofCmsghdr           = 0x10
	SizeofInet4Pktinfo      = 0xc
	SizeofInet6Pktinfo      = 0x14
	SizeofIPv6MTUInfo       = 0x20
	SizeofICMPv6Filter      = 0x20
	SizeofUcred             = 0xc
	SizeofTCPInfo           = 0x68
)

const (
	IFA_UNSPEC          = 0x0
	IFA_ADDRESS         = 0x1
	IFA_LOCAL           = 0x2
	IFA_LABEL           = 0x3
	IFA_BROADCAST       = 0x4
	IFA_ANYCAST         = 0x5
	IFA_CACHEINFO       = 0x6
	IFA_MULTICAST       = 0x7
	IFLA_UNSPEC         = 0x0
	IFLA_ADDRESS        = 0x1
	IFLA_BROADCAST      = 0x2
	IFLA_IFNAME         = 0x3
	IFLA_MTU            = 0x4
	IFLA_LINK           = 0x5
	IFLA_QDISC          = 0x6
	IFLA_STATS          = 0x7
	IFLA_COST           = 0x8
	IFLA_PRIORITY       = 0x9
	IFLA_MASTER         = 0xa
	IFLA_WIRELESS       = 0xb
	IFLA_PROTINFO       = 0xc
	IFLA_TXQLEN         = 0xd
	IFLA_MAP            = 0xe
	IFLA_WEIGHT         = 0xf
	IFLA_OPERSTATE      = 0x10
	IFLA_LINKMODE       = 0x11
	IFLA_LINKINFO       = 0x12
	IFLA_NET_NS_PID     = 0x13
	IFLA_IFALIAS        = 0x14
	IFLA_MAX            = 0x26
	RT_SCOPE_UNIVERSE   = 0x0
	RT_SCOPE_SITE       = 0xc8
	RT_SCOPE_LINK       = 0xfd
	RT_SCOPE_HOST       = 0xfe
	RT_SCOPE_NOWHERE    = 0xff
	RT_TABLE_UNSPEC     = 0x0
	RT_TABLE_COMPAT     = 0xfc
	RT_TABLE_DEFAULT    = 0xfd
	RT_TABLE_MAIN       = 0xfe
	RT_TABLE_LOCAL      = 0xff
	RT_TABLE_MAX        = 0xffffffff
	RTA_UNSPEC          = 0x0
	RTA_DST             = 0x1
	RTA_SRC             = 0x2
	RTA_IIF             = 0x3
	RTA_OIF             = 0x4
	RTA_GATEWAY         = 0x5
	RTA_PRIORITY        = 0x6
	RTA_PREFSRC         = 0x7
	RTA_METRICS         = 0x8
	RTA_MULTIPATH       = 0x9
	RTA_FLOW            = 0xb
	RTA_CACHEINFO       = 0xc
	RTA_TABLE           = 0xf
	RTN_UNSPEC          = 0x0
	RTN_UNICAST         = 0x1
	RTN_LOCAL           = 0x2
	RTN_BROADCAST       = 0x3
	RTN_ANYCAST         = 0x4
	RTN_MULTICAST       = 0x5
	RTN_BLACKHOLE       = 0x6
	RTN_UNREACHABLE     = 0x7
	RTN_PROHIBIT        = 0x8
	RTN_THROW           = 0x9
	RTN_NAT             = 0xa
	RTN_XRESOLVE        = 0xb
	RTNLGRP_NONE        = 0x0
	RTNLGRP_LINK        = 0x1
	RTNLGRP_NOTIFY      = 0x2
	RTNLGRP_NEIGH       = 0x3
	RTNLGRP_TC          = 0x4
	RTNLGRP_IPV4_IFADDR = 0x5
	RTNLGRP_IPV4_MROUTE = 0x6
	RTNLGRP_IPV4_ROUTE  = 0x7
	RTNLGRP_IPV4_RULE   = 0x8
	RTNLGRP_IPV6_IFADDR = 0x9
	RTNLGRP_IPV6_MROUTE = 0xa
	RTNLGRP_IPV6_ROUTE  = 0xb
	RTNLGRP_IPV6_IFINFO = 0xc
	RTNLGRP_IPV6_PREFIX = 0x12
	RTNLGRP_IPV6_RULE   = 0x13
	RTNLGRP_ND_USEROPT  = 0x14
	SizeofNlMsghdr      = 0x10
	SizeofNlMsgerr      = 0x14
	SizeofRtGenmsg      = 0x1
	SizeofNlAttr        = 0x4
	SizeofRtAttr        = 0x4
	SizeofIfInfomsg     = 0x10
	SizeofIfAddrmsg     = 0x8
	SizeofRtMsg         = 0xc
	SizeofRtNexthop     = 0x8
)

type NlMsghdr struct {
	Len   uint32
	Type  uint16
	Flags uint16
	Seq   uint32
	Pid   uint32
}

type NlMsgerr struct {
	Error int32
	Msg   NlMsghdr
}

type RtGenmsg struct {
	Family uint8
}

type NlAttr struct {
	Len  uint16
	Type uint16
}

type RtAttr struct {
	Len  uint16
	Type uint16
}

type IfInfomsg struct {
	Family     uint8
	X__ifi_pad uint8
	Type       uint16
	Index      int32
	Flags      uint32
	Change     uint32
}

type IfAddrmsg struct {
	Family    uint8
	Prefixlen uint8
	Flags     uint8
	Scope     uint8
	Index     uint32
}

type RtMsg struct {
	Family   uint8
	Dst_len  uint8
	Src_len  uint8
	Tos      uint8
	Table    uint8
	Protocol uint8
	Scope    uint8
	Type     uint8
	Flags    uint32
}

type RtNexthop struct {
	Len     uint16
	Flags   uint8
	Hops    uint8
	Ifindex int32
}

const (
	SizeofSockFilter = 0x8
	SizeofSockFprog  = 0x10
)

type SockFilter struct {
	Code uint16
	Jt   uint8
	Jf   uint8
	K    uint32
}

type SockFprog struct {
	Len    uint16
	Filter *SockFilter
}

type InotifyEvent struct {
	Wd     int32
	Mask   uint32
	Cookie uint32
	Len    uint32
}

const SizeofInotifyEvent = 0x10

type PtraceRegs struct {
	Pc  uint64
	Ra  uint64
	Sp  uint64
	Gp  uint64
	Tp  uint64
	T0  uint64
	T1  uint64
	T2  uint64
	S0  uint64
	S1  uint64
	A0  uint64
	A1  uint64
	A2  uint64
	A3  uint64
	A4  uint64
	A5  uint64
	A6  uint64
	A7  uint64
	S2  uint64
	S3  uint64
	S4  uint64
	S5  uint64
	S6  uint64
	S7  uint64
	S8  uint64
	S9  uint64
	S10 uint64
	S11 uint64
	T3  uint64
	T4  uint64
	T5  uint64
	T6  uint64
}

type ptracePsw struct {
}

type ptraceFpregs struct {
}

type ptracePer struct {
}

type FdSet struct {
	Bits [16]int64
}

type Sysinfo_t struct {
	Uptime    int64
	Loads     [3]uint64
	Totalram  uint64
	Freeram   uint64
	Sharedram uint64
	Bufferram uint64
	Totalswap uint64
	Freeswap  uint64
	Procs     uint16
	Pad       uint16
	Totalhigh uint64
	Freehigh  uint64
	Unit      uint32
	_         [0]uint8
	_         [4]byte
}

type Utsname struct {
	Sysname    [65]uint8
	Nodename   [65]uint8
	Release    [65]uint8
	Version    [65]uint8
	Machine    [65]uint8
	Domainname [65]uint8
}

type Ustat_t struct {
	Tfree  int32
	Tinode uint64
	Fname  [6]uint8
	Fpack  [6]uint8
	_      [4]byte
}

type EpollEvent struct {
	Events uint32
	_      int32
	Fd     int32
	Pad    int32
}

const (
	_AT_FDCWD            = -0x64
	_AT_REMOVEDIR        = 0x200
	_AT_SYMLINK_NOFOLLOW = 0x100
	_AT_EACCESS          = 0x200
)

type pollFd struct {
	Fd      int32
	Events  int16
	Revents int16
}

type Termios struct {
	Iflag  uint32
	Oflag  uint32
	Cflag  uint32
	Lflag  uint32
	Line   uint8
	Cc     [19]uint8
	Ispeed uint32
	Ospeed uint32
}

const (
	IUCLC  = 0x200
	OLCUC  = 0x2
	TCGETS = 0x5401
	TCSETS = 0x5402
	XCASE  = 0x4
)
