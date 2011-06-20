// godefs -gsyscall -f-m32 types_linux.c

// MACHINE GENERATED - DO NOT EDIT.

package syscall

// Constants
const (
	sizeofPtr               = 0x4
	sizeofShort             = 0x2
	sizeofInt               = 0x4
	sizeofLong              = 0x4
	sizeofLongLong          = 0x8
	PathMax                 = 0x1000
	SizeofSockaddrInet4     = 0x10
	SizeofSockaddrInet6     = 0x1c
	SizeofSockaddrAny       = 0x70
	SizeofSockaddrUnix      = 0x6e
	SizeofSockaddrLinklayer = 0x14
	SizeofSockaddrNetlink   = 0xc
	SizeofLinger            = 0x8
	SizeofIPMreq            = 0x8
	SizeofIPv6Mreq          = 0x14
	SizeofMsghdr            = 0x1c
	SizeofCmsghdr           = 0xc
	SizeofInet6Pktinfo      = 0x14
	SizeofUcred             = 0xc
	IFA_UNSPEC              = 0
	IFA_ADDRESS             = 0x1
	IFA_LOCAL               = 0x2
	IFA_LABEL               = 0x3
	IFA_BROADCAST           = 0x4
	IFA_ANYCAST             = 0x5
	IFA_CACHEINFO           = 0x6
	IFA_MULTICAST           = 0x7
	IFLA_UNSPEC             = 0
	IFLA_ADDRESS            = 0x1
	IFLA_BROADCAST          = 0x2
	IFLA_IFNAME             = 0x3
	IFLA_MTU                = 0x4
	IFLA_LINK               = 0x5
	IFLA_QDISC              = 0x6
	IFLA_STATS              = 0x7
	IFLA_COST               = 0x8
	IFLA_PRIORITY           = 0x9
	IFLA_MASTER             = 0xa
	IFLA_WIRELESS           = 0xb
	IFLA_PROTINFO           = 0xc
	IFLA_TXQLEN             = 0xd
	IFLA_MAP                = 0xe
	IFLA_WEIGHT             = 0xf
	IFLA_OPERSTATE          = 0x10
	IFLA_LINKMODE           = 0x11
	IFLA_LINKINFO           = 0x12
	IFLA_NET_NS_PID         = 0x13
	IFLA_IFALIAS            = 0x14
	IFLA_MAX                = 0x14
	RT_SCOPE_UNIVERSE       = 0
	RT_SCOPE_SITE           = 0xc8
	RT_SCOPE_LINK           = 0xfd
	RT_SCOPE_HOST           = 0xfe
	RT_SCOPE_NOWHERE        = 0xff
	RT_TABLE_UNSPEC         = 0
	RT_TABLE_COMPAT         = 0xfc
	RT_TABLE_DEFAULT        = 0xfd
	RT_TABLE_MAIN           = 0xfe
	RT_TABLE_LOCAL          = 0xff
	RT_TABLE_MAX            = 0xffffffff
	RTA_UNSPEC              = 0
	RTA_DST                 = 0x1
	RTA_SRC                 = 0x2
	RTA_IIF                 = 0x3
	RTA_OIF                 = 0x4
	RTA_GATEWAY             = 0x5
	RTA_PRIORITY            = 0x6
	RTA_PREFSRC             = 0x7
	RTA_METRICS             = 0x8
	RTA_MULTIPATH           = 0x9
	RTA_FLOW                = 0xb
	RTA_CACHEINFO           = 0xc
	RTA_TABLE               = 0xf
	RTN_UNSPEC              = 0
	RTN_UNICAST             = 0x1
	RTN_LOCAL               = 0x2
	RTN_BROADCAST           = 0x3
	RTN_ANYCAST             = 0x4
	RTN_MULTICAST           = 0x5
	RTN_BLACKHOLE           = 0x6
	RTN_UNREACHABLE         = 0x7
	RTN_PROHIBIT            = 0x8
	RTN_THROW               = 0x9
	RTN_NAT                 = 0xa
	RTN_XRESOLVE            = 0xb
	SizeofNlMsghdr          = 0x10
	SizeofNlMsgerr          = 0x14
	SizeofRtGenmsg          = 0x1
	SizeofNlAttr            = 0x4
	SizeofRtAttr            = 0x4
	SizeofIfInfomsg         = 0x10
	SizeofIfAddrmsg         = 0x8
	SizeofRtmsg             = 0xc
	SizeofRtNexthop         = 0x8
	SizeofSockFilter        = 0x8
	SizeofSockFprog         = 0x8
	SizeofInotifyEvent      = 0x10
)

// Types

type _C_short int16

type _C_int int32

type _C_long int32

type _C_long_long int64

type Timespec struct {
	Sec  int32
	Nsec int32
}

type Timeval struct {
	Sec  int32
	Usec int32
}

type Timex struct {
	Modes         uint32
	Offset        int32
	Freq          int32
	Maxerror      int32
	Esterror      int32
	Status        int32
	Constant      int32
	Precision     int32
	Tolerance     int32
	Time          Timeval
	Tick          int32
	Ppsfreq       int32
	Jitter        int32
	Shift         int32
	Stabil        int32
	Jitcnt        int32
	Calcnt        int32
	Errcnt        int32
	Stbcnt        int32
	Tai           int32
	Pad_godefs_0  int32
	Pad_godefs_1  int32
	Pad_godefs_2  int32
	Pad_godefs_3  int32
	Pad_godefs_4  int32
	Pad_godefs_5  int32
	Pad_godefs_6  int32
	Pad_godefs_7  int32
	Pad_godefs_8  int32
	Pad_godefs_9  int32
	Pad_godefs_10 int32
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
	Dev          uint64
	X__pad1      uint16
	Pad_godefs_0 [2]byte
	X__st_ino    uint32
	Mode         uint32
	Nlink        uint32
	Uid          uint32
	Gid          uint32
	Rdev         uint64
	X__pad2      uint16
	Pad_godefs_1 [2]byte
	Size         int64
	Blksize      int32
	Blocks       int64
	Atim         Timespec
	Mtim         Timespec
	Ctim         Timespec
	Ino          uint64
}

type Statfs_t struct {
	Type    int32
	Bsize   int32
	Blocks  uint64
	Bfree   uint64
	Bavail  uint64
	Files   uint64
	Ffree   uint64
	Fsid    [8]byte /* __fsid_t */
	Namelen int32
	Frsize  int32
	Spare   [5]int32
}

type Dirent struct {
	Ino          uint64
	Off          int64
	Reclen       uint16
	Type         uint8
	Name         [256]int8
	Pad_godefs_0 [1]byte
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
	Data   [14]int8
}

type RawSockaddrAny struct {
	Addr RawSockaddr
	Pad  [96]int8
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

type Inet6Pktinfo struct {
	Addr    [16]byte /* in6_addr */
	Ifindex uint32
}

type Ucred struct {
	Pid int32
	Uid uint32
	Gid uint32
}

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

type SockFilter struct {
	Code uint16
	Jt   uint8
	Jf   uint8
	K    uint32
}

type SockFprog struct {
	Len          uint16
	Pad_godefs_0 [2]byte
	Filter       *SockFilter
}

type InotifyEvent struct {
	Wd     int32
	Mask   uint32
	Cookie uint32
	Len    uint32
}

type PtraceRegs struct {
	Ebx      int32
	Ecx      int32
	Edx      int32
	Esi      int32
	Edi      int32
	Ebp      int32
	Eax      int32
	Xds      int32
	Xes      int32
	Xfs      int32
	Xgs      int32
	Orig_eax int32
	Eip      int32
	Xcs      int32
	Eflags   int32
	Esp      int32
	Xss      int32
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
	X_f       [8]int8
}

type Utsname struct {
	Sysname    [65]int8
	Nodename   [65]int8
	Release    [65]int8
	Version    [65]int8
	Machine    [65]int8
	Domainname [65]int8
}

type Ustat_t struct {
	Tfree  int32
	Tinode uint32
	Fname  [6]int8
	Fpack  [6]int8
}

type EpollEvent struct {
	Events uint32
	Fd     int32
	Pad    int32
}
