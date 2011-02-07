// godefs -gsyscall types_linux.c

// MACHINE GENERATED - DO NOT EDIT.

// Manual corrections: TODO(rsc): need to fix godefs
//	remove duplicate PtraceRegs type
//	change RawSockaddrUnix field to Path [108]int8 (was uint8)
//  add padding to EpollEvent

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
	SizeofLinger            = 0x8
	SizeofIpMreq            = 0x8
	SizeofMsghdr            = 0x1c
	SizeofCmsghdr           = 0xc
	SizeofUcred             = 0xc
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
	Pad0      int32
	Pad1      int32
	Pad2      int32
	Pad3      int32
	Pad4      int32
	Pad5      int32
	Pad6      int32
	Pad7      int32
	Pad8      int32
	Pad9      int32
	Pad10     int32
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
	Pad0      [2]byte
	X__st_ino uint32
	Mode      uint32
	Nlink     uint32
	Uid       uint32
	Gid       uint32
	Rdev      uint64
	X__pad2   uint16
	Pad1      [6]byte
	Size      int64
	Blksize   int32
	Pad2      [4]byte
	Blocks    int64
	Atim      Timespec
	Mtim      Timespec
	Ctim      Timespec
	Ino       uint64
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
	Pad0    [4]byte
}

type Dirent struct {
	Ino    uint64
	Off    int64
	Reclen uint16
	Type   uint8
	Name   [256]uint8
	Pad0   [5]byte
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

type IpMreq struct {
	Multiaddr [4]byte /* in_addr */
	Interface [4]byte /* in_addr */
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

type Ucred struct {
	Pid int32
	Uid uint32
	Gid uint32
}

type InotifyEvent struct {
	Wd     int32
	Mask   uint32
	Cookie uint32
	Len    uint32
}

type PtraceRegs struct{}

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
