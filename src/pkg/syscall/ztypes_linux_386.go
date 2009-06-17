// godefs -gsyscall -f-m32 types_linux.c types_linux_386.c

// MACHINE GENERATED - DO NOT EDIT.

package syscall

// Constants
const (
	sizeofPtr = 0x4;
	sizeofShort = 0x2;
	sizeofInt = 0x4;
	sizeofLong = 0x4;
	sizeofLongLong = 0x8;
	PathMax = 0x1000;
	O_RDONLY = 0;
	O_WRONLY = 0x1;
	O_RDWR = 0x2;
	O_APPEND = 0x400;
	O_ASYNC = 0x2000;
	O_CREAT = 0x40;
	O_NOCTTY = 0x100;
	O_NONBLOCK = 0x800;
	O_SYNC = 0x1000;
	O_TRUNC = 0x200;
	O_CLOEXEC = 0;
	F_GETFD = 0x1;
	F_SETFD = 0x2;
	F_GETFL = 0x3;
	F_SETFL = 0x4;
	FD_CLOEXEC = 0x1;
	NAME_MAX = 0xff;
	S_IFMT = 0xf000;
	S_IFIFO = 0x1000;
	S_IFCHR = 0x2000;
	S_IFDIR = 0x4000;
	S_IFBLK = 0x6000;
	S_IFREG = 0x8000;
	S_IFLNK = 0xa000;
	S_IFSOCK = 0xc000;
	S_ISUID = 0x800;
	S_ISGID = 0x400;
	S_ISVTX = 0x200;
	S_IRUSR = 0x100;
	S_IWUSR = 0x80;
	S_IXUSR = 0x40;
	WNOHANG = 0x1;
	WUNTRACED = 0x2;
	WEXITED = 0x4;
	WSTOPPED = 0x2;
	WCONTINUED = 0x8;
	WNOWAIT = 0x1000000;
	AF_UNIX = 0x1;
	AF_INET = 0x2;
	AF_INET6 = 0xa;
	SOCK_STREAM = 0x1;
	SOCK_DGRAM = 0x2;
	SOCK_RAW = 0x3;
	SOCK_SEQPACKET = 0x5;
	SOL_SOCKET = 0x1;
	SO_REUSEADDR = 0x2;
	SO_KEEPALIVE = 0x9;
	SO_DONTROUTE = 0x5;
	SO_BROADCAST = 0x6;
	SO_LINGER = 0xd;
	SO_SNDBUF = 0x7;
	SO_RCVBUF = 0x8;
	SO_SNDTIMEO = 0x15;
	SO_RCVTIMEO = 0x14;
	IPPROTO_TCP = 0x6;
	IPPROTO_UDP = 0x11;
	TCP_NODELAY = 0x1;
	SOMAXCONN = 0x80;
	SizeofSockaddrInet4 = 0x10;
	SizeofSockaddrInet6 = 0x1c;
	SizeofSockaddrAny = 0x1c;
	SizeofSockaddrUnix = 0x6e;
	EPOLLIN = 0x1;
	EPOLLRDHUP = 0x2000;
	EPOLLOUT = 0x4;
	EPOLLONESHOT = 0x40000000;
	EPOLL_CTL_MOD = 0x3;
	EPOLL_CTL_ADD = 0x1;
	EPOLL_CTL_DEL = 0x2;
)

// Types

type Timespec struct {
	Sec int32;
	Nsec int32;
}

type Timeval struct {
	Sec int32;
	Usec int32;
}

type Timex struct {
	Modes uint32;
	Offset int32;
	Freq int32;
	Maxerror int32;
	Esterror int32;
	Status int32;
	Constant int32;
	Precision int32;
	Tolerance int32;
	Time Timeval;
	Tick int32;
	Ppsfreq int32;
	Jitter int32;
	Shift int32;
	Stabil int32;
	Jitcnt int32;
	Calcnt int32;
	Errcnt int32;
	Stbcnt int32;
	 int32;
	 int32;
	 int32;
	 int32;
	 int32;
	 int32;
	 int32;
	 int32;
	 int32;
	 int32;
	 int32;
	 int32;
}

type Time_t int32

type Tms struct {
	Utime int32;
	Stime int32;
	Cutime int32;
	Cstime int32;
}

type Utimbuf struct {
	Actime int32;
	Modtime int32;
}

type Rusage struct {
	Utime Timeval;
	Stime Timeval;
	Maxrss int32;
	Ixrss int32;
	Idrss int32;
	Isrss int32;
	Minflt int32;
	Majflt int32;
	Nswap int32;
	Inblock int32;
	Oublock int32;
	Msgsnd int32;
	Msgrcv int32;
	Nsignals int32;
	Nvcsw int32;
	Nivcsw int32;
}

type Rlimit struct {
	Cur uint64;
	Max uint64;
}

type _C_int int32

type _Gid_t uint32

type Stat_t struct {
	Dev uint64;
	__pad1 uint16;
	Pad0 [2]byte;
	__st_ino uint32;
	Mode uint32;
	Nlink uint32;
	Uid uint32;
	Gid uint32;
	Rdev uint64;
	__pad2 uint16;
	Pad1 [2]byte;
	Size int64;
	Blksize int32;
	Blocks int64;
	Atim Timespec;
	Mtim Timespec;
	Ctim Timespec;
	Ino uint64;
}

type Statfs_t struct {
	Type int32;
	Bsize int32;
	Blocks uint64;
	Bfree uint64;
	Bavail uint64;
	Files uint64;
	Ffree uint64;
	Fsid [8]byte /* __fsid_t */;
	Namelen int32;
	Frsize int32;
	Spare [5]int32;
}

type Dirent struct {
	Ino uint64;
	Off int64;
	Reclen uint16;
	Type uint8;
	Name [256]int8;
	Pad0 [1]byte;
}

type RawSockaddrInet4 struct {
	Family uint16;
	Port uint16;
	Addr [4]byte /* in_addr */;
	Zero [8]uint8;
}

type RawSockaddrInet6 struct {
	Family uint16;
	Port uint16;
	Flowinfo uint32;
	Addr [16]byte /* in6_addr */;
	Scope_id uint32;
}

type RawSockaddrUnix struct {
	Family uint16;
	Path [108]int8;
}

type RawSockaddr struct {
	Family uint16;
	Data [14]int8;
}

type RawSockaddrAny struct {
	Addr RawSockaddr;
	Pad [12]int8;
}

type _Socklen uint32

type Linger struct {
	Onoff int32;
	Linger int32;
}

type FdSet struct {
	Bits [32]int32;
}

type Sysinfo_t struct {
	Uptime int32;
	Loads [3]uint32;
	Totalram uint32;
	Freeram uint32;
	Sharedram uint32;
	Bufferram uint32;
	Totalswap uint32;
	Freeswap uint32;
	Procs uint16;
	Pad uint16;
	Totalhigh uint32;
	Freehigh uint32;
	Unit uint32;
	_f [8]int8;
}

type Utsname struct {
	Sysname [65]int8;
	Nodename [65]int8;
	Release [65]int8;
	Version [65]int8;
	Machine [65]int8;
	Domainname [65]int8;
}

type Ustat_t struct {
	Tfree int32;
	Tinode uint32;
	Fname [6]int8;
	Fpack [6]int8;
}

type EpollEvent struct {
	Events uint32;
	Fd int32;
	Pad int32;
}
