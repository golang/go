// godefs -gsyscall -f-m64 types_freebsd.c

// MACHINE GENERATED - DO NOT EDIT.

package syscall

// Constants
const (
	sizeofPtr           = 0x8
	sizeofShort         = 0x2
	sizeofInt           = 0x4
	sizeofLong          = 0x8
	sizeofLongLong      = 0x8
	O_CLOEXEC           = 0
	S_IFMT              = 0xf000
	S_IFIFO             = 0x1000
	S_IFCHR             = 0x2000
	S_IFDIR             = 0x4000
	S_IFBLK             = 0x6000
	S_IFREG             = 0x8000
	S_IFLNK             = 0xa000
	S_IFSOCK            = 0xc000
	S_ISUID             = 0x800
	S_ISGID             = 0x400
	S_ISVTX             = 0x200
	S_IRUSR             = 0x100
	S_IWUSR             = 0x80
	S_IXUSR             = 0x40
	SizeofSockaddrInet4 = 0x10
	SizeofSockaddrInet6 = 0x1c
	SizeofSockaddrAny   = 0x6c
	SizeofSockaddrUnix  = 0x6a
	SizeofLinger        = 0x8
	SizeofMsghdr        = 0x30
	SizeofCmsghdr       = 0xc
	PTRACE_TRACEME      = 0
	PTRACE_CONT         = 0x7
	PTRACE_KILL         = 0x8
)

// Types

type _C_short int16

type _C_int int32

type _C_long int64

type _C_long_long int64

type Timespec struct {
	Sec  int64
	Nsec int64
}

type Timeval struct {
	Sec  int64
	Usec int64
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
	Cur int64
	Max int64
}

type _Gid_t uint32

type Stat_t struct {
	Dev           uint32
	Ino           uint32
	Mode          uint16
	Nlink         uint16
	Uid           uint32
	Gid           uint32
	Rdev          uint32
	Atimespec     Timespec
	Mtimespec     Timespec
	Ctimespec     Timespec
	Size          int64
	Blocks        int64
	Blksize       uint32
	Flags         uint32
	Gen           uint32
	Lspare        int32
	Birthtimespec Timespec
	Pad0          uint8
	Pad1          uint8
}

type Statfs_t struct {
	Version     uint32
	Type        uint32
	Flags       uint64
	Bsize       uint64
	Iosize      uint64
	Blocks      uint64
	Bfree       uint64
	Bavail      int64
	Files       uint64
	Ffree       int64
	Syncwrites  uint64
	Asyncwrites uint64
	Syncreads   uint64
	Asyncreads  uint64
	Spare       [10]uint64
	Namemax     uint32
	Owner       uint32
	Fsid        [8]byte /* fsid */
	Charspare   [80]int8
	Fstypename  [16]int8
	Mntfromname [88]int8
	Mntonname   [88]int8
}

type Flock_t struct {
	Start  int64
	Len    int64
	Pid    int32
	Type   int16
	Whence int16
	Sysid  int32
	Pad0   [4]byte
}

type Dirent struct {
	Fileno uint32
	Reclen uint16
	Type   uint8
	Namlen uint8
	Name   [256]int8
}

type RawSockaddrInet4 struct {
	Len    uint8
	Family uint8
	Port   uint16
	Addr   [4]byte /* in_addr */
	Zero   [8]int8
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
	Path   [104]int8
}

type RawSockaddr struct {
	Len    uint8
	Family uint8
	Data   [14]int8
}

type RawSockaddrAny struct {
	Addr RawSockaddr
	Pad  [92]int8
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

type Msghdr struct {
	Name       *byte
	Namelen    uint32
	Pad0       [4]byte
	Iov        *Iovec
	Iovlen     int32
	Pad1       [4]byte
	Control    *byte
	Controllen uint32
	Flags      int32
}

type Cmsghdr struct {
	Len   uint32
	Level int32
	Type  int32
}

type Kevent_t struct {
	Ident  uint64
	Filter int16
	Flags  uint16
	Fflags uint32
	Data   int64
	Udata  *byte
}

type FdSet struct {
	X__fds_bits [16]uint64
}
