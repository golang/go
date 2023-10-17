// Copyright 2020 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

//go:build zos && s390x
// +build zos,s390x

// Hand edited based on ztypes_linux_s390x.go
// TODO: auto-generate.

package unix

const (
	SizeofPtr      = 0x8
	SizeofShort    = 0x2
	SizeofInt      = 0x4
	SizeofLong     = 0x8
	SizeofLongLong = 0x8
	PathMax        = 0x1000
)

const (
	SizeofSockaddrAny   = 128
	SizeofCmsghdr       = 12
	SizeofIPMreq        = 8
	SizeofIPv6Mreq      = 20
	SizeofICMPv6Filter  = 32
	SizeofIPv6MTUInfo   = 32
	SizeofLinger        = 8
	SizeofSockaddrInet4 = 16
	SizeofSockaddrInet6 = 28
	SizeofTCPInfo       = 0x68
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

type timeval_zos struct { //correct (with padding and all)
	Sec  int64
	_    [4]byte // pad
	Usec int32
}

type Tms struct { //clock_t is 4-byte unsigned int in zos
	Utime  uint32
	Stime  uint32
	Cutime uint32
	Cstime uint32
}

type Time_t int64

type Utimbuf struct {
	Actime  int64
	Modtime int64
}

type Utsname struct {
	Sysname    [65]byte
	Nodename   [65]byte
	Release    [65]byte
	Version    [65]byte
	Machine    [65]byte
	Domainname [65]byte
}

type RawSockaddrInet4 struct {
	Len    uint8
	Family uint8
	Port   uint16
	Addr   [4]byte /* in_addr */
	Zero   [8]uint8
}

type RawSockaddrInet6 struct {
	Len      uint8
	Family   uint8
	Port     uint16
	Flowinfo uint32
	Addr     [16]byte /* in6_addr */
	Scope_id uint32
}

type RawSockaddrUnix struct {
	Len    uint8
	Family uint8
	Path   [108]int8
}

type RawSockaddr struct {
	Len    uint8
	Family uint8
	Data   [14]uint8
}

type RawSockaddrAny struct {
	Addr RawSockaddr
	_    [112]uint8 // pad
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

type IPv6Mreq struct {
	Multiaddr [16]byte /* in6_addr */
	Interface uint32
}

type Msghdr struct {
	Name       *byte
	Iov        *Iovec
	Control    *byte
	Flags      int32
	Namelen    int32
	Iovlen     int32
	Controllen int32
}

type Cmsghdr struct {
	Len   int32
	Level int32
	Type  int32
}

type Inet4Pktinfo struct {
	Addr    [4]byte /* in_addr */
	Ifindex uint32
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

type _Gid_t uint32

type rusage_zos struct {
	Utime timeval_zos
	Stime timeval_zos
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

// { int, short, short } in poll.h
type PollFd struct {
	Fd      int32
	Events  int16
	Revents int16
}

type Stat_t struct { //Linux Definition
	Dev     uint64
	Ino     uint64
	Nlink   uint64
	Mode    uint32
	Uid     uint32
	Gid     uint32
	_       int32
	Rdev    uint64
	Size    int64
	Atim    Timespec
	Mtim    Timespec
	Ctim    Timespec
	Blksize int64
	Blocks  int64
	_       [3]int64
}

type Stat_LE_t struct {
	_            [4]byte // eye catcher
	Length       uint16
	Version      uint16
	Mode         int32
	Ino          uint32
	Dev          uint32
	Nlink        int32
	Uid          int32
	Gid          int32
	Size         int64
	Atim31       [4]byte
	Mtim31       [4]byte
	Ctim31       [4]byte
	Rdev         uint32
	Auditoraudit uint32
	Useraudit    uint32
	Blksize      int32
	Creatim31    [4]byte
	AuditID      [16]byte
	_            [4]byte // rsrvd1
	File_tag     struct {
		Ccsid   uint16
		Txtflag uint16 // aggregating Txflag:1 deferred:1 rsvflags:14
	}
	CharsetID [8]byte
	Blocks    int64
	Genvalue  uint32
	Reftim31  [4]byte
	Fid       [8]byte
	Filefmt   byte
	Fspflag2  byte
	_         [2]byte // rsrvd2
	Ctimemsec int32
	Seclabel  [8]byte
	_         [4]byte // rsrvd3
	_         [4]byte // rsrvd4
	Atim      Time_t
	Mtim      Time_t
	Ctim      Time_t
	Creatim   Time_t
	Reftim    Time_t
	_         [24]byte // rsrvd5
}

type Statvfs_t struct {
	ID          [4]byte
	Len         int32
	Bsize       uint64
	Blocks      uint64
	Usedspace   uint64
	Bavail      uint64
	Flag        uint64
	Maxfilesize int64
	_           [16]byte
	Frsize      uint64
	Bfree       uint64
	Files       uint32
	Ffree       uint32
	Favail      uint32
	Namemax31   uint32
	Invarsec    uint32
	_           [4]byte
	Fsid        uint64
	Namemax     uint64
}

type Statfs_t struct {
	Type    uint32
	Bsize   uint64
	Blocks  uint64
	Bfree   uint64
	Bavail  uint64
	Files   uint32
	Ffree   uint32
	Fsid    uint64
	Namelen uint64
	Frsize  uint64
	Flags   uint64
}

type direntLE struct {
	Reclen uint16
	Namlen uint16
	Ino    uint32
	Extra  uintptr
	Name   [256]byte
}

type Dirent struct {
	Ino    uint64
	Off    int64
	Reclen uint16
	Type   uint8
	Name   [256]uint8
	_      [5]byte
}

type FdSet struct {
	Bits [64]int32
}

// This struct is packed on z/OS so it can't be used directly.
type Flock_t struct {
	Type   int16
	Whence int16
	Start  int64
	Len    int64
	Pid    int32
}

type Termios struct {
	Cflag uint32
	Iflag uint32
	Lflag uint32
	Oflag uint32
	Cc    [11]uint8
}

type Winsize struct {
	Row    uint16
	Col    uint16
	Xpixel uint16
	Ypixel uint16
}

type W_Mnth struct {
	Hid   [4]byte
	Size  int32
	Cur1  int32 //32bit pointer
	Cur2  int32 //^
	Devno uint32
	_     [4]byte
}

type W_Mntent struct {
	Fstype       uint32
	Mode         uint32
	Dev          uint32
	Parentdev    uint32
	Rootino      uint32
	Status       byte
	Ddname       [9]byte
	Fstname      [9]byte
	Fsname       [45]byte
	Pathlen      uint32
	Mountpoint   [1024]byte
	Jobname      [8]byte
	PID          int32
	Parmoffset   int32
	Parmlen      int16
	Owner        [8]byte
	Quiesceowner [8]byte
	_            [38]byte
}
