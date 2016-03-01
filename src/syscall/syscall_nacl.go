// Copyright 2013 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package syscall

import (
	"sync"
	"unsafe"
)

//sys	naclClose(fd int) (err error) = sys_close
//sys	Exit(code int) (err error)
//sys	naclFstat(fd int, stat *Stat_t) (err error) = sys_fstat
//sys	naclRead(fd int, b []byte) (n int, err error) = sys_read
//sys	naclSeek(fd int, off *int64, whence int) (err error) = sys_lseek
//sys	naclGetRandomBytes(b []byte) (err error) = sys_get_random_bytes

const direntSize = 8 + 8 + 2 + 256

// native_client/src/trusted/service_runtime/include/sys/dirent.h
type Dirent struct {
	Ino    int64
	Off    int64
	Reclen uint16
	Name   [256]byte
}

func ParseDirent(buf []byte, max int, names []string) (consumed int, count int, newnames []string) {
	origlen := len(buf)
	count = 0
	for max != 0 && len(buf) > 0 {
		dirent := (*Dirent)(unsafe.Pointer(&buf[0]))
		buf = buf[dirent.Reclen:]
		if dirent.Ino == 0 { // File absent in directory.
			continue
		}
		bytes := (*[512 + PathMax]byte)(unsafe.Pointer(&dirent.Name[0]))
		var name = string(bytes[0:clen(bytes[:])])
		if name == "." || name == ".." { // Useless names
			continue
		}
		max--
		count++
		names = append(names, name)
	}
	return origlen - len(buf), count, names
}

func clen(n []byte) int {
	for i := 0; i < len(n); i++ {
		if n[i] == 0 {
			return i
		}
	}
	return len(n)
}

const PathMax = 256

// An Errno is an unsigned number describing an error condition.
// It implements the error interface. The zero Errno is by convention
// a non-error, so code to convert from Errno to error should use:
//	err = nil
//	if errno != 0 {
//		err = errno
//	}
type Errno uintptr

func (e Errno) Error() string {
	if 0 <= int(e) && int(e) < len(errorstr) {
		s := errorstr[e]
		if s != "" {
			return s
		}
	}
	return "errno " + itoa(int(e))
}

func (e Errno) Temporary() bool {
	return e == EINTR || e == EMFILE || e.Timeout()
}

func (e Errno) Timeout() bool {
	return e == EAGAIN || e == EWOULDBLOCK || e == ETIMEDOUT
}

// A Signal is a number describing a process signal.
// It implements the os.Signal interface.
type Signal int

const (
	_ Signal = iota
	SIGCHLD
	SIGINT
	SIGKILL
	SIGTRAP
	SIGQUIT
)

func (s Signal) Signal() {}

func (s Signal) String() string {
	if 0 <= s && int(s) < len(signals) {
		str := signals[s]
		if str != "" {
			return str
		}
	}
	return "signal " + itoa(int(s))
}

var signals = [...]string{}

// File system

const (
	Stdin  = 0
	Stdout = 1
	Stderr = 2
)

// native_client/src/trusted/service_runtime/include/sys/fcntl.h
const (
	O_RDONLY  = 0
	O_WRONLY  = 1
	O_RDWR    = 2
	O_ACCMODE = 3

	O_CREAT    = 0100
	O_CREATE   = O_CREAT // for ken
	O_TRUNC    = 01000
	O_APPEND   = 02000
	O_EXCL     = 0200
	O_NONBLOCK = 04000
	O_NDELAY   = O_NONBLOCK
	O_SYNC     = 010000
	O_FSYNC    = O_SYNC
	O_ASYNC    = 020000

	O_CLOEXEC = 0

	FD_CLOEXEC = 1
)

// native_client/src/trusted/service_runtime/include/sys/fcntl.h
const (
	F_DUPFD   = 0
	F_GETFD   = 1
	F_SETFD   = 2
	F_GETFL   = 3
	F_SETFL   = 4
	F_GETOWN  = 5
	F_SETOWN  = 6
	F_GETLK   = 7
	F_SETLK   = 8
	F_SETLKW  = 9
	F_RGETLK  = 10
	F_RSETLK  = 11
	F_CNVT    = 12
	F_RSETLKW = 13

	F_RDLCK   = 1
	F_WRLCK   = 2
	F_UNLCK   = 3
	F_UNLKSYS = 4
)

// native_client/src/trusted/service_runtime/include/bits/stat.h
const (
	S_IFMT        = 0000370000
	S_IFSHM_SYSV  = 0000300000
	S_IFSEMA      = 0000270000
	S_IFCOND      = 0000260000
	S_IFMUTEX     = 0000250000
	S_IFSHM       = 0000240000
	S_IFBOUNDSOCK = 0000230000
	S_IFSOCKADDR  = 0000220000
	S_IFDSOCK     = 0000210000

	S_IFSOCK = 0000140000
	S_IFLNK  = 0000120000
	S_IFREG  = 0000100000
	S_IFBLK  = 0000060000
	S_IFDIR  = 0000040000
	S_IFCHR  = 0000020000
	S_IFIFO  = 0000010000

	S_UNSUP = 0000370000

	S_ISUID = 0004000
	S_ISGID = 0002000
	S_ISVTX = 0001000

	S_IREAD  = 0400
	S_IWRITE = 0200
	S_IEXEC  = 0100

	S_IRWXU = 0700
	S_IRUSR = 0400
	S_IWUSR = 0200
	S_IXUSR = 0100

	S_IRWXG = 070
	S_IRGRP = 040
	S_IWGRP = 020
	S_IXGRP = 010

	S_IRWXO = 07
	S_IROTH = 04
	S_IWOTH = 02
	S_IXOTH = 01
)

// native_client/src/trusted/service_runtime/include/sys/stat.h
// native_client/src/trusted/service_runtime/include/machine/_types.h
type Stat_t struct {
	Dev       int64
	Ino       uint64
	Mode      uint32
	Nlink     uint32
	Uid       uint32
	Gid       uint32
	Rdev      int64
	Size      int64
	Blksize   int32
	Blocks    int32
	Atime     int64
	AtimeNsec int64
	Mtime     int64
	MtimeNsec int64
	Ctime     int64
	CtimeNsec int64
}

// Processes
// Not supported on NaCl - just enough for package os.

var ForkLock sync.RWMutex

type WaitStatus uint32

func (w WaitStatus) Exited() bool       { return false }
func (w WaitStatus) ExitStatus() int    { return 0 }
func (w WaitStatus) Signaled() bool     { return false }
func (w WaitStatus) Signal() Signal     { return 0 }
func (w WaitStatus) CoreDump() bool     { return false }
func (w WaitStatus) Stopped() bool      { return false }
func (w WaitStatus) Continued() bool    { return false }
func (w WaitStatus) StopSignal() Signal { return 0 }
func (w WaitStatus) TrapCause() int     { return 0 }

// XXX made up
type Rusage struct {
	Utime Timeval
	Stime Timeval
}

// XXX made up
type ProcAttr struct {
	Dir   string
	Env   []string
	Files []uintptr
	Sys   *SysProcAttr
}

type SysProcAttr struct {
}

// System

func Syscall(trap, a1, a2, a3 uintptr) (r1, r2 uintptr, err Errno)
func Syscall6(trap, a1, a2, a3, a4, a5, a6 uintptr) (r1, r2 uintptr, err Errno) { return 0, 0, ENOSYS }
func RawSyscall(trap, a1, a2, a3 uintptr) (r1, r2 uintptr, err Errno)           { return 0, 0, ENOSYS }
func RawSyscall6(trap, a1, a2, a3, a4, a5, a6 uintptr) (r1, r2 uintptr, err Errno) {
	return 0, 0, ENOSYS
}

func Sysctl(key string) (string, error) {
	if key == "kern.hostname" {
		return "naclbox", nil
	}
	return "", ENOSYS
}

// Unimplemented Unix midden heap.

const ImplementsGetwd = false

func Getwd() (wd string, err error)     { return "", ENOSYS }
func Getegid() int                      { return 1 }
func Geteuid() int                      { return 1 }
func Getgid() int                       { return 1 }
func Getgroups() ([]int, error)         { return []int{1}, nil }
func Getpagesize() int                  { return 65536 }
func Getppid() int                      { return 2 }
func Getpid() int                       { return 3 }
func Getuid() int                       { return 1 }
func Kill(pid int, signum Signal) error { return ENOSYS }
func Sendfile(outfd int, infd int, offset *int64, count int) (written int, err error) {
	return 0, ENOSYS
}
func StartProcess(argv0 string, argv []string, attr *ProcAttr) (pid int, handle uintptr, err error) {
	return 0, 0, ENOSYS
}
func Wait4(pid int, wstatus *WaitStatus, options int, rusage *Rusage) (wpid int, err error) {
	return 0, ENOSYS
}
func RouteRIB(facility, param int) ([]byte, error)                { return nil, ENOSYS }
func ParseRoutingMessage(b []byte) ([]RoutingMessage, error)      { return nil, ENOSYS }
func ParseRoutingSockaddr(msg RoutingMessage) ([]Sockaddr, error) { return nil, ENOSYS }
func SysctlUint32(name string) (value uint32, err error)          { return 0, ENOSYS }
