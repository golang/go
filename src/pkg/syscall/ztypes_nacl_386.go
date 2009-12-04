// godefs -gsyscall -f-m32 -f-I/home/rsc/pub/nacl/native_client/src/third_party/nacl_sdk/linux/sdk/nacl-sdk/nacl/include -f-I/home/rsc/pub/nacl/native_client types_nacl.c

// MACHINE GENERATED - DO NOT EDIT.

package syscall

// Constants
const (
	sizeofPtr	= 0x4;
	sizeofShort	= 0x2;
	sizeofInt	= 0x4;
	sizeofLong	= 0x4;
	sizeofLongLong	= 0x8;
	PROT_READ	= 0x1;
	PROT_WRITE	= 0x2;
	MAP_SHARED	= 0x1;
	SYS_FORK	= 0;
	SYS_PTRACE	= 0;
	SYS_CHDIR	= 0;
	SYS_DUP2	= 0;
	SYS_FCNTL	= 0;
	SYS_EXECVE	= 0;
	O_RDONLY	= 0;
	O_WRONLY	= 0x1;
	O_RDWR		= 0x2;
	O_APPEND	= 0x400;
	O_ASYNC		= 0x2000;
	O_CREAT		= 0x40;
	O_NOCTTY	= 0;
	O_NONBLOCK	= 0x800;
	O_SYNC		= 0x1000;
	O_TRUNC		= 0x200;
	O_CLOEXEC	= 0;
	O_EXCL		= 0;
	F_GETFD		= 0x1;
	F_SETFD		= 0x2;
	F_GETFL		= 0x3;
	F_SETFL		= 0x4;
	FD_CLOEXEC	= 0;
	S_IFMT		= 0x1f000;
	S_IFIFO		= 0x1000;
	S_IFCHR		= 0x2000;
	S_IFDIR		= 0x4000;
	S_IFBLK		= 0x6000;
	S_IFREG		= 0x8000;
	S_IFLNK		= 0xa000;
	S_IFSOCK	= 0xc000;
	S_ISUID		= 0x800;
	S_ISGID		= 0x400;
	S_ISVTX		= 0x200;
	S_IRUSR		= 0x100;
	S_IWUSR		= 0x80;
	S_IXUSR		= 0x40;
)

// Types

type _C_short int16

type _C_int int32

type _C_long int32

type _C_long_long int64

type _C_off_t int32

type Timespec struct {
	Sec	int32;
	Nsec	int32;
}

type Timeval struct {
	Sec	int32;
	Usec	int32;
}

type Time_t int32

type _Gid_t uint32

type Stat_t struct {
	Dev		int64;
	Ino		uint32;
	Mode		uint32;
	Nlink		uint32;
	Uid		uint32;
	Gid		uint32;
	__padding	int32;
	Rdev		int64;
	Size		int32;
	Blksize		int32;
	Blocks		int32;
	Atime		int32;
	Mtime		int32;
	Ctime		int32;
}

type Dirent struct {
	Ino	uint32;
	Off	int32;
	Reclen	uint16;
	Name	[256]int8;
	Pad0	[2]byte;
}
