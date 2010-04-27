// godefs -gsyscall -f-m32 types_linux.c

// MACHINE GENERATED - DO NOT EDIT.

package syscall

// TODO(brainman): autogenerate types in ztypes_mingw_386.go

//import "unsafe"

// Constants
const (
	sizeofPtr           = 0x4
	sizeofShort         = 0x2
	sizeofInt           = 0x4
	sizeofLong          = 0x4
	sizeofLongLong      = 0x8
	PathMax             = 0x1000
	SizeofSockaddrInet4 = 0x10
	SizeofSockaddrInet6 = 0x1c
	SizeofSockaddrAny   = 0x70
	SizeofSockaddrUnix  = 0x6e
	SizeofLinger        = 0x8
	SizeofMsghdr        = 0x1c
	SizeofCmsghdr       = 0xc
)

const (
	// Invented values to support what package os expects.
	O_RDONLY   = 0x00000
	O_WRONLY   = 0x00001
	O_RDWR     = 0x00002
	O_CREAT    = 0x00040
	O_EXCL     = 0x00080
	O_NOCTTY   = 0x00100
	O_TRUNC    = 0x00200
	O_NONBLOCK = 0x00800
	O_APPEND   = 0x00400
	O_SYNC     = 0x01000
	O_ASYNC    = 0x02000
	O_CLOEXEC  = 0x80000
)

const (
	GENERIC_READ    = 0x80000000
	GENERIC_WRITE   = 0x40000000
	GENERIC_EXECUTE = 0x20000000
	GENERIC_ALL     = 0x10000000

	FILE_SHARE_READ          = 0x00000001
	FILE_SHARE_WRITE         = 0x00000002
	FILE_SHARE_DELETE        = 0x00000004
	FILE_ATTRIBUTE_READONLY  = 0x00000001
	FILE_ATTRIBUTE_HIDDEN    = 0x00000002
	FILE_ATTRIBUTE_SYSTEM    = 0x00000004
	FILE_ATTRIBUTE_DIRECTORY = 0x00000010
	FILE_ATTRIBUTE_ARCHIVE   = 0x00000020
	FILE_ATTRIBUTE_NORMAL    = 0x00000080

	CREATE_NEW        = 1
	CREATE_ALWAYS     = 2
	OPEN_EXISTING     = 3
	OPEN_ALWAYS       = 4
	TRUNCATE_EXISTING = 5

	STD_INPUT_HANDLE  = -10
	STD_OUTPUT_HANDLE = -11
	STD_ERROR_HANDLE  = -12

	FILE_BEGIN   = 0
	FILE_CURRENT = 1
	FILE_END     = 2

	FORMAT_MESSAGE_ALLOCATE_BUFFER = 256
	FORMAT_MESSAGE_IGNORE_INSERTS  = 512
	FORMAT_MESSAGE_FROM_STRING     = 1024
	FORMAT_MESSAGE_FROM_HMODULE    = 2048
	FORMAT_MESSAGE_FROM_SYSTEM     = 4096
	FORMAT_MESSAGE_ARGUMENT_ARRAY  = 8192
	FORMAT_MESSAGE_MAX_WIDTH_MASK  = 255

	MAX_PATH = 260

	MAX_COMPUTERNAME_LENGTH = 15
)

// Types

type _C_short int16

type _C_int int32

type _C_long int32

type _C_long_long int64

// Invented values to support what package os expects.
type Timeval struct {
	Sec  int32
	Usec int32
}

type Overlapped struct {
	Internal     uint32
	InternalHigh uint32
	Offset       uint32
	OffsetHigh   uint32
	HEvent       *byte
}

type Filetime struct {
	LowDateTime  uint32
	HighDateTime uint32
}

type Win32finddata struct {
	FileAttributes    uint32
	CreationTime      Filetime
	LastAccessTime    Filetime
	LastWriteTime     Filetime
	FileSizeHigh      uint32
	FileSizeLow       uint32
	Reserved0         uint32
	Reserved1         uint32
	FileName          [MAX_PATH - 1]uint16
	AlternateFileName [13]uint16
}

type ByHandleFileInformation struct {
	FileAttributes     uint32
	CreationTime       Filetime
	LastAccessTime     Filetime
	LastWriteTime      Filetime
	VolumeSerialNumber uint32
	FileSizeHigh       uint32
	FileSizeLow        uint32
	NumberOfLinks      uint32
	FileIndexHigh      uint32
	FileIndexLow       uint32
}

// Invented values to support what package os expects.
type Stat_t struct {
	Windata Win32finddata
	Mode    uint32
}

// TODO(brainman): fix all needed for os

const (
	PROT_READ  = 0x1
	PROT_WRITE = 0x2
	MAP_SHARED = 0x1
	SYS_FORK   = 0
	SYS_PTRACE = 0
	SYS_CHDIR  = 0
	SYS_DUP2   = 0
	SYS_FCNTL  = 0
	SYS_EXECVE = 0
	F_GETFD    = 0x1
	F_SETFD    = 0x2
	F_GETFL    = 0x3
	F_SETFL    = 0x4
	FD_CLOEXEC = 0
	S_IFMT     = 0x1f000
	S_IFIFO    = 0x1000
	S_IFCHR    = 0x2000
	S_IFDIR    = 0x4000
	S_IFBLK    = 0x6000
	S_IFREG    = 0x8000
	S_IFLNK    = 0xa000
	S_IFSOCK   = 0xc000
	S_ISUID    = 0x800
	S_ISGID    = 0x400
	S_ISVTX    = 0x200
	S_IRUSR    = 0x100
	S_IWUSR    = 0x80
	S_IXUSR    = 0x40
)
