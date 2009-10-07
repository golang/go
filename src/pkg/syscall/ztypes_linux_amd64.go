// godefs -gsyscall -f-m64 types_linux.c

// MACHINE GENERATED - DO NOT EDIT.

package syscall

// Constants
const (
	sizeofPtr		= 0x8;
	sizeofShort		= 0x2;
	sizeofInt		= 0x4;
	sizeofLong		= 0x8;
	sizeofLongLong		= 0x8;
	PathMax			= 0x1000;
	O_RDONLY		= 0;
	O_WRONLY		= 0x1;
	O_RDWR			= 0x2;
	O_APPEND		= 0x400;
	O_ASYNC			= 0x2000;
	O_CREAT			= 0x40;
	O_NOCTTY		= 0x100;
	O_NONBLOCK		= 0x800;
	O_SYNC			= 0x1000;
	O_TRUNC			= 0x200;
	O_CLOEXEC		= 0;
	F_GETFD			= 0x1;
	F_SETFD			= 0x2;
	F_GETFL			= 0x3;
	F_SETFL			= 0x4;
	FD_CLOEXEC		= 0x1;
	NAME_MAX		= 0xff;
	S_IFMT			= 0xf000;
	S_IFIFO			= 0x1000;
	S_IFCHR			= 0x2000;
	S_IFDIR			= 0x4000;
	S_IFBLK			= 0x6000;
	S_IFREG			= 0x8000;
	S_IFLNK			= 0xa000;
	S_IFSOCK		= 0xc000;
	S_ISUID			= 0x800;
	S_ISGID			= 0x400;
	S_ISVTX			= 0x200;
	S_IRUSR			= 0x100;
	S_IWUSR			= 0x80;
	S_IXUSR			= 0x40;
	WNOHANG			= 0x1;
	WUNTRACED		= 0x2;
	WEXITED			= 0x4;
	WSTOPPED		= 0x2;
	WCONTINUED		= 0x8;
	WNOWAIT			= 0x1000000;
	WCLONE			= 0x80000000;
	WALL			= 0x40000000;
	WNOTHREAD		= 0x20000000;
	AF_UNIX			= 0x1;
	AF_INET			= 0x2;
	AF_INET6		= 0xa;
	SOCK_STREAM		= 0x1;
	SOCK_DGRAM		= 0x2;
	SOCK_RAW		= 0x3;
	SOCK_SEQPACKET		= 0x5;
	SOL_SOCKET		= 0x1;
	SO_REUSEADDR		= 0x2;
	SO_KEEPALIVE		= 0x9;
	SO_DONTROUTE		= 0x5;
	SO_BROADCAST		= 0x6;
	SO_LINGER		= 0xd;
	SO_SNDBUF		= 0x7;
	SO_RCVBUF		= 0x8;
	SO_SNDTIMEO		= 0x15;
	SO_RCVTIMEO		= 0x14;
	IPPROTO_TCP		= 0x6;
	IPPROTO_UDP		= 0x11;
	TCP_NODELAY		= 0x1;
	SOMAXCONN		= 0x80;
	SizeofSockaddrInet4	= 0x10;
	SizeofSockaddrInet6	= 0x1c;
	SizeofSockaddrAny	= 0x1c;
	SizeofSockaddrUnix	= 0x6e;
	_PTRACE_TRACEME		= 0;
	_PTRACE_PEEKTEXT	= 0x1;
	_PTRACE_PEEKDATA	= 0x2;
	_PTRACE_PEEKUSER	= 0x3;
	_PTRACE_POKETEXT	= 0x4;
	_PTRACE_POKEDATA	= 0x5;
	_PTRACE_POKEUSER	= 0x6;
	_PTRACE_CONT		= 0x7;
	_PTRACE_KILL		= 0x8;
	_PTRACE_SINGLESTEP	= 0x9;
	_PTRACE_GETREGS		= 0xc;
	_PTRACE_SETREGS		= 0xd;
	_PTRACE_GETFPREGS	= 0xe;
	_PTRACE_SETFPREGS	= 0xf;
	_PTRACE_ATTACH		= 0x10;
	_PTRACE_DETACH		= 0x11;
	_PTRACE_GETFPXREGS	= 0x12;
	_PTRACE_SETFPXREGS	= 0x13;
	_PTRACE_SYSCALL		= 0x18;
	_PTRACE_SETOPTIONS	= 0x4200;
	_PTRACE_GETEVENTMSG	= 0x4201;
	_PTRACE_GETSIGINFO	= 0x4202;
	_PTRACE_SETSIGINFO	= 0x4203;
	PTRACE_O_TRACESYSGOOD	= 0x1;
	PTRACE_O_TRACEFORK	= 0x2;
	PTRACE_O_TRACEVFORK	= 0x4;
	PTRACE_O_TRACECLONE	= 0x8;
	PTRACE_O_TRACEEXEC	= 0x10;
	PTRACE_O_TRACEVFORKDONE	= 0x20;
	PTRACE_O_TRACEEXIT	= 0x40;
	PTRACE_O_MASK		= 0x7f;
	PTRACE_EVENT_FORK	= 0x1;
	PTRACE_EVENT_VFORK	= 0x2;
	PTRACE_EVENT_CLONE	= 0x3;
	PTRACE_EVENT_EXEC	= 0x4;
	PTRACE_EVENT_VFORK_DONE	= 0x5;
	PTRACE_EVENT_EXIT	= 0x6;
	EPOLLIN			= 0x1;
	EPOLLRDHUP		= 0x2000;
	EPOLLOUT		= 0x4;
	EPOLLONESHOT		= 0x40000000;
	EPOLL_CTL_MOD		= 0x3;
	EPOLL_CTL_ADD		= 0x1;
	EPOLL_CTL_DEL		= 0x2;
)

// Types

type _C_short int16

type _C_int int32

type _C_long int64

type _C_long_long int64

type Timespec struct {
	Sec	int64;
	Nsec	int64;
}

type Timeval struct {
	Sec	int64;
	Usec	int64;
}

type Timex struct {
	Modes		uint32;
	Pad0		[4]byte;
	Offset		int64;
	Freq		int64;
	Maxerror	int64;
	Esterror	int64;
	Status		int32;
	Pad1		[4]byte;
	Constant	int64;
	Precision	int64;
	Tolerance	int64;
	Time		Timeval;
	Tick		int64;
	Ppsfreq		int64;
	Jitter		int64;
	Shift		int32;
	Pad2		[4]byte;
	Stabil		int64;
	Jitcnt		int64;
	Calcnt		int64;
	Errcnt		int64;
	Stbcnt		int64;
	Pad3		int32;
	Pad4		int32;
	Pad5		int32;
	Pad6		int32;
	Pad7		int32;
	Pad8		int32;
	Pad9		int32;
	Pad10		int32;
	Pad11		int32;
	Pad12		int32;
	Pad13		int32;
	Pad14		int32;
}

type Time_t int64

type Tms struct {
	Utime	int64;
	Stime	int64;
	Cutime	int64;
	Cstime	int64;
}

type Utimbuf struct {
	Actime	int64;
	Modtime	int64;
}

type Rusage struct {
	Utime		Timeval;
	Stime		Timeval;
	Maxrss		int64;
	Ixrss		int64;
	Idrss		int64;
	Isrss		int64;
	Minflt		int64;
	Majflt		int64;
	Nswap		int64;
	Inblock		int64;
	Oublock		int64;
	Msgsnd		int64;
	Msgrcv		int64;
	Nsignals	int64;
	Nvcsw		int64;
	Nivcsw		int64;
}

type Rlimit struct {
	Cur	uint64;
	Max	uint64;
}

type _Gid_t uint32

type Stat_t struct {
	Dev		uint64;
	Ino		uint64;
	Nlink		uint64;
	Mode		uint32;
	Uid		uint32;
	Gid		uint32;
	Pad0		int32;
	Rdev		uint64;
	Size		int64;
	Blksize		int64;
	Blocks		int64;
	Atim		Timespec;
	Mtim		Timespec;
	Ctim		Timespec;
	__unused	[3]int64;
}

type Statfs_t struct {
	Type	int64;
	Bsize	int64;
	Blocks	uint64;
	Bfree	uint64;
	Bavail	uint64;
	Files	uint64;
	Ffree	uint64;
	Fsid	[8]byte;	/* __fsid_t */
	Namelen	int64;
	Frsize	int64;
	Spare	[5]int64;
}

type Dirent struct {
	Ino	uint64;
	Off	int64;
	Reclen	uint16;
	Type	uint8;
	Name	[256]int8;
	Pad0	[5]byte;
}

type RawSockaddrInet4 struct {
	Family	uint16;
	Port	uint16;
	Addr	[4]byte;	/* in_addr */
	Zero	[8]uint8;
}

type RawSockaddrInet6 struct {
	Family		uint16;
	Port		uint16;
	Flowinfo	uint32;
	Addr		[16]byte;	/* in6_addr */
	Scope_id	uint32;
}

type RawSockaddrUnix struct {
	Family	uint16;
	Path	[108]int8;
}

type RawSockaddr struct {
	Family	uint16;
	Data	[14]int8;
}

type RawSockaddrAny struct {
	Addr	RawSockaddr;
	Pad	[12]int8;
}

type _Socklen uint32

type Linger struct {
	Onoff	int32;
	Linger	int32;
}

type PtraceRegs struct {
	R15		uint64;
	R14		uint64;
	R13		uint64;
	R12		uint64;
	Rbp		uint64;
	Rbx		uint64;
	R11		uint64;
	R10		uint64;
	R9		uint64;
	R8		uint64;
	Rax		uint64;
	Rcx		uint64;
	Rdx		uint64;
	Rsi		uint64;
	Rdi		uint64;
	Orig_rax	uint64;
	Rip		uint64;
	Cs		uint64;
	Eflags		uint64;
	Rsp		uint64;
	Ss		uint64;
	Fs_base		uint64;
	Gs_base		uint64;
	Ds		uint64;
	Es		uint64;
	Fs		uint64;
	Gs		uint64;
}

type FdSet struct {
	Bits [16]int64;
}

type Sysinfo_t struct {
	Uptime		int64;
	Loads		[3]uint64;
	Totalram	uint64;
	Freeram		uint64;
	Sharedram	uint64;
	Bufferram	uint64;
	Totalswap	uint64;
	Freeswap	uint64;
	Procs		uint16;
	Pad		uint16;
	Pad0		[4]byte;
	Totalhigh	uint64;
	Freehigh	uint64;
	Unit		uint32;
	_f		[2]int8;
	Pad1		[4]byte;
}

type Utsname struct {
	Sysname		[65]int8;
	Nodename	[65]int8;
	Release		[65]int8;
	Version		[65]int8;
	Machine		[65]int8;
	Domainname	[65]int8;
}

type Ustat_t struct {
	Tfree	int32;
	Pad0	[4]byte;
	Tinode	uint64;
	Fname	[6]int8;
	Fpack	[6]int8;
	Pad1	[4]byte;
}

type EpollEvent struct {
	Events	uint32;
	Fd	int32;
	Pad	int32;
}
