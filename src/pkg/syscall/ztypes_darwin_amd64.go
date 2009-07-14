// godefs -gsyscall -f-m64 types_darwin.c types_darwin_amd64.c

// MACHINE GENERATED - DO NOT EDIT.

package syscall

// Constants
const (
	sizeofPtr = 0x8;
	sizeofShort = 0x2;
	sizeofInt = 0x4;
	sizeofLong = 0x8;
	sizeofLongLong = 0x8;
	O_RDONLY = 0;
	O_WRONLY = 0x1;
	O_RDWR = 0x2;
	O_APPEND = 0x8;
	O_ASYNC = 0x40;
	O_CREAT = 0x200;
	O_NOCTTY = 0x20000;
	O_NONBLOCK = 0x4;
	O_SYNC = 0x80;
	O_TRUNC = 0x400;
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
	S_IFWHT = 0xe000;
	S_ISUID = 0x800;
	S_ISGID = 0x400;
	S_ISVTX = 0x200;
	S_IRUSR = 0x100;
	S_IWUSR = 0x80;
	S_IXUSR = 0x40;
	WNOHANG = 0x1;
	WUNTRACED = 0x2;
	WEXITED = 0x4;
	WSTOPPED = 0x7f;
	WCONTINUED = 0x10;
	WNOWAIT = 0x20;
	AF_UNIX = 0x1;
	AF_INET = 0x2;
	AF_DATAKIT = 0x9;
	AF_INET6 = 0x1e;
	SOCK_STREAM = 0x1;
	SOCK_DGRAM = 0x2;
	SOCK_RAW = 0x3;
	SOCK_SEQPACKET = 0x5;
	SOL_SOCKET = 0xffff;
	SO_REUSEADDR = 0x4;
	SO_KEEPALIVE = 0x8;
	SO_DONTROUTE = 0x10;
	SO_BROADCAST = 0x20;
	SO_USELOOPBACK = 0x40;
	SO_LINGER = 0x80;
	SO_REUSEPORT = 0x200;
	SO_SNDBUF = 0x1001;
	SO_RCVBUF = 0x1002;
	SO_SNDTIMEO = 0x1005;
	SO_RCVTIMEO = 0x1006;
	SO_NOSIGPIPE = 0x1022;
	IPPROTO_TCP = 0x6;
	IPPROTO_UDP = 0x11;
	TCP_NODELAY = 0x1;
	SOMAXCONN = 0x80;
	SizeofSockaddrInet4 = 0x10;
	SizeofSockaddrInet6 = 0x1c;
	SizeofSockaddrAny = 0x1c;
	SizeofSockaddrUnix = 0x6a;
	_PTRACE_TRACEME = 0;
	_PTRACE_CONT = 0x7;
	_PTRACE_KILL = 0x8;
	EVFILT_READ = -0x1;
	EVFILT_WRITE = -0x2;
	EVFILT_AIO = -0x3;
	EVFILT_VNODE = -0x4;
	EVFILT_PROC = -0x5;
	EVFILT_SIGNAL = -0x6;
	EVFILT_TIMER = -0x7;
	EVFILT_MACHPORT = -0x8;
	EVFILT_FS = -0x9;
	EVFILT_SYSCOUNT = 0x9;
	EV_ADD = 0x1;
	EV_DELETE = 0x2;
	EV_DISABLE = 0x8;
	EV_RECEIPT = 0x40;
	EV_ONESHOT = 0x10;
	EV_CLEAR = 0x20;
	EV_SYSFLAGS = 0xf000;
	EV_FLAG0 = 0x1000;
	EV_FLAG1 = 0x2000;
	EV_EOF = 0x8000;
	EV_ERROR = 0x4000;
)

// Types

type _C_short int16

type _C_int int32

type _C_long int64

type _C_long_long int64

type Timespec struct {
	Sec int64;
	Nsec int64;
}

type Timeval struct {
	Sec int64;
	Usec int32;
	Pad0 [4]byte;
}

type Rusage struct {
	Utime Timeval;
	Stime Timeval;
	Maxrss int64;
	Ixrss int64;
	Idrss int64;
	Isrss int64;
	Minflt int64;
	Majflt int64;
	Nswap int64;
	Inblock int64;
	Oublock int64;
	Msgsnd int64;
	Msgrcv int64;
	Nsignals int64;
	Nvcsw int64;
	Nivcsw int64;
}

type Rlimit struct {
	Cur uint64;
	Max uint64;
}

type _Gid_t uint32

type Stat_t struct {
	Dev int32;
	Mode uint16;
	Nlink uint16;
	Ino uint64;
	Uid uint32;
	Gid uint32;
	Rdev int32;
	Pad0 [4]byte;
	Atimespec Timespec;
	Mtimespec Timespec;
	Ctimespec Timespec;
	Birthtimespec Timespec;
	Size int64;
	Blocks int64;
	Blksize int32;
	Flags uint32;
	Gen uint32;
	Lspare int32;
	Qspare [2]int64;
}

type Statfs_t struct {
	Bsize uint32;
	Iosize int32;
	Blocks uint64;
	Bfree uint64;
	Bavail uint64;
	Files uint64;
	Ffree uint64;
	Fsid [8]byte /* fsid */;
	Owner uint32;
	Type uint32;
	Flags uint32;
	Fssubtype uint32;
	Fstypename [16]int8;
	Mntonname [1024]int8;
	Mntfromname [1024]int8;
	Reserved [8]uint32;
}

type Dirent struct {
	Ino uint64;
	Seekoff uint64;
	Reclen uint16;
	Namlen uint16;
	Type uint8;
	Name [1024]int8;
	Pad0 [3]byte;
}

type RawSockaddrInet4 struct {
	Len uint8;
	Family uint8;
	Port uint16;
	Addr [4]byte /* in_addr */;
	Zero [8]int8;
}

type RawSockaddrInet6 struct {
	Len uint8;
	Family uint8;
	Port uint16;
	Flowinfo uint32;
	Addr [16]byte /* in6_addr */;
	Scope_id uint32;
}

type RawSockaddrUnix struct {
	Len uint8;
	Family uint8;
	Path [104]int8;
}

type RawSockaddr struct {
	Len uint8;
	Family uint8;
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

type Kevent_t struct {
	Ident uint64;
	Filter int16;
	Flags uint16;
	Fflags uint32;
	Data int64;
	Udata *byte;
}

type FdSet struct {
	Bits [32]int32;
}
