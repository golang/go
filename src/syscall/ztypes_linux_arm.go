// Created by cgo -godefs - DO NOT EDIT
// cgo -godefs types_linux.go

//go:build arm && linux

package syscall

const (
	sizeofPtr      = 0x4
	sizeofShort    = 0x2
	sizeofInt      = 0x4
	sizeofLong     = 0x4
	sizeofLongLong = 0x8
	PathMax        = 0x1000
)

type (
	_C_short     int16
	_C_int       int32
	_C_long      int32
	_C_long_long int64
)

type Timespec struct {
	Sec  int32
	Nsec int32
}

type Timeval struct {
	Sec  int32
	Usec int32
}

type Timex struct {
	Modes     uint32
	Offset    int32
	Freq      int32
	Maxerror  int32
	Esterror  int32
	Status    int32
	Constant  int32
	Precision int32
	Tolerance int32
	Time      Timeval
	Tick      int32
	Ppsfreq   int32
	Jitter    int32
	Shift     int32
	Stabil    int32
	Jitcnt    int32
	Calcnt    int32
	Errcnt    int32
	Stbcnt    int32
	Tai       int32
	Pad_cgo_0 [44]byte
}

type Time_t int32

type Tms struct {
	Utime  int32
	Stime  int32
	Cutime int32
	Cstime int32
}

type Utimbuf struct {
	Actime  int32
	Modtime int32
}

type Rusage struct {
	Utime    Timeval
	Stime    Timeval
	Maxrss   int32
	Ixrss    int32
	Idrss    int32
	Isrss    int32
	Minflt   int32
	Majflt   int32
	Nswap    int32
	Inblock  int32
	Oublock  int32
	Msgsnd   int32
	Msgrcv   int32
	Nsignals int32
	Nvcsw    int32
	Nivcsw   int32
}

type Rlimit struct {
	Cur uint64
	Max uint64
}

type _Gid_t uint32

type Stat_t struct {
	Dev       uint64
	X__pad1   uint16
	Pad_cgo_0 [2]byte
	X__st_ino uint32
	Mode      uint32
	Nlink     uint32
	Uid       uint32
	Gid       uint32
	Rdev      uint64
	X__pad2   uint16
	Pad_cgo_1 [6]byte
	Size      int64
	Blksize   int32
	Pad_cgo_2 [4]byte
	Blocks    int64
	Atim      Timespec
	Mtim      Timespec
	Ctim      Timespec
	Ino       uint64
}

type Statfs_t struct {
	Type      int32
	Bsize     int32
	Blocks    uint64
	Bfree     uint64
	Bavail    uint64
	Files     uint64
	Ffree     uint64
	Fsid      Fsid
	Namelen   int32
	Frsize    int32
	Flags     int32
	Spare     [4]int32
	Pad_cgo_0 [4]byte
}

type Dirent struct {
	Ino       uint64
	Off       int64
	Reclen    uint16
	Type      uint8
	Name      [256]uint8
	Pad_cgo_0 [5]byte
}

type Fsid struct {
	X__val [2]int32
}

type Flock_t struct {
	Type      int16
	Whence    int16
	Pad_cgo_0 [4]byte
	Start     int64
	Len       int64
	Pid       int32
	Pad_cgo_1 [4]byte
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
	Len  uint32
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
	Iovlen     uint32
	Control    *byte
	Controllen uint32
	Flags      int32
}

type Cmsghdr struct {
	Len   uint32
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
	Pad_cgo_0      [2]byte
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
	SizeofMsghdr            = 0x1c
	SizeofCmsghdr           = 0xc
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
	IFLA_MAX            = 0x1d
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
	SizeofSockFprog  = 0x8
)

type SockFilter struct {
	Code uint16
	Jt   uint8
	Jf   uint8
	K    uint32
}

type SockFprog struct {
	Len       uint16
	Pad_cgo_0 [2]byte
	Filter    *SockFilter
}

type InotifyEvent struct {
	Wd     int32
	Mask   uint32
	Cookie uint32
	Len    uint32
	Name   [0]uint8
}

const SizeofInotifyEvent = 0x10

type PtraceRegs struct {
	Uregs [18]uint32
}

type FdSet struct {
	Bits [32]int32
}

type Sysinfo_t struct {
	Uptime    int32
	Loads     [3]uint32
	Totalram  uint32
	Freeram   uint32
	Sharedram uint32
	Bufferram uint32
	Totalswap uint32
	Freeswap  uint32
	Procs     uint16
	Pad       uint16
	Totalhigh uint32
	Freehigh  uint32
	Unit      uint32
	X_f       [8]uint8
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
	Tinode uint32
	Fname  [6]uint8
	Fpack  [6]uint8
}

type EpollEvent struct {
	Events uint32
	PadFd  int32
	Fd     int32
	Pad    int32
}

const (
	_AT_FDCWD            = -0x64
	_AT_REMOVEDIR        = 0x200
	_AT_SYMLINK_NOFOLLOW = 0x100
	_AT_EACCESS          = 0x200
	_AT_EMPTY_PATH       = 0x1000
)

type pollFd struct {
	Fd      int32
	Events  int16
	Revents int16
}

type Termios struct {
	Iflag     uint32
	Oflag     uint32
	Cflag     uint32
	Lflag     uint32
	Line      uint8
	Cc        [32]uint8
	Pad_cgo_0 [3]byte
	Ispeed    uint32
	Ospeed    uint32
}

const (
	VINTR    = 0x0
	VQUIT    = 0x1
	VERASE   = 0x2
	VKILL    = 0x3
	VEOF     = 0x4
	VTIME    = 0x5
	VMIN     = 0x6
	VSWTC    = 0x7
	VSTART   = 0x8
	VSTOP    = 0x9
	VSUSP    = 0xa
	VEOL     = 0xb
	VREPRINT = 0xc
	VDISCARD = 0xd
	VWERASE  = 0xe
	VLNEXT   = 0xf
	VEOL2    = 0x10
	IGNBRK   = 0x1
	BRKINT   = 0x2
	IGNPAR   = 0x4
	PARMRK   = 0x8
	INPCK    = 0x10
	ISTRIP   = 0x20
	INLCR    = 0x40
	IGNCR    = 0x80
	ICRNL    = 0x100
	IUCLC    = 0x200
	IXON     = 0x400
	IXANY    = 0x800
	IXOFF    = 0x1000
	IMAXBEL  = 0x2000
	IUTF8    = 0x4000
	OPOST    = 0x1
	OLCUC    = 0x2
	ONLCR    = 0x4
	OCRNL    = 0x8
	ONOCR    = 0x10
	ONLRET   = 0x20
	OFILL    = 0x40
	OFDEL    = 0x80
	B0       = 0x0
	B50      = 0x1
	B75      = 0x2
	B110     = 0x3
	B134     = 0x4
	B150     = 0x5
	B200     = 0x6
	B300     = 0x7
	B600     = 0x8
	B1200    = 0x9
	B1800    = 0xa
	B2400    = 0xb
	B4800    = 0xc
	B9600    = 0xd
	B19200   = 0xe
	B38400   = 0xf
	CSIZE    = 0x30
	CS5      = 0x0
	CS6      = 0x10
	CS7      = 0x20
	CS8      = 0x30
	CSTOPB   = 0x40
	CREAD    = 0x80
	PARENB   = 0x100
	PARODD   = 0x200
	HUPCL    = 0x400
	CLOCAL   = 0x800
	B57600   = 0x1001
	B115200  = 0x1002
	B230400  = 0x1003
	B460800  = 0x1004
	B500000  = 0x1005
	B576000  = 0x1006
	B921600  = 0x1007
	B1000000 = 0x1008
	B1152000 = 0x1009
	B1500000 = 0x100a
	B2000000 = 0x100b
	B2500000 = 0x100c
	B3000000 = 0x100d
	B3500000 = 0x100e
	B4000000 = 0x100f
	ISIG     = 0x1
	ICANON   = 0x2
	XCASE    = 0x4
	ECHO     = 0x8
	ECHOE    = 0x10
	ECHOK    = 0x20
	ECHONL   = 0x40
	NOFLSH   = 0x80
	TOSTOP   = 0x100
	ECHOCTL  = 0x200
	ECHOPRT  = 0x400
	ECHOKE   = 0x800
	FLUSHO   = 0x1000
	PENDIN   = 0x4000
	IEXTEN   = 0x8000
	TCGETS   = 0x5401
	TCSETS   = 0x5402
)
