// Copyright 2023 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

//go:build wasip1

package syscall

import (
	"errors"
	"internal/itoa"
	"internal/oserror"
	"unsafe"
)

type Dircookie = uint64

type Filetype = uint8

const (
	FILETYPE_UNKNOWN Filetype = iota
	FILETYPE_BLOCK_DEVICE
	FILETYPE_CHARACTER_DEVICE
	FILETYPE_DIRECTORY
	FILETYPE_REGULAR_FILE
	FILETYPE_SOCKET_DGRAM
	FILETYPE_SOCKET_STREAM
	FILETYPE_SYMBOLIC_LINK
)

type Dirent struct {
	// The offset of the next directory entry stored in this directory.
	Next Dircookie
	// The serial number of the file referred to by this directory entry.
	Ino uint64
	// The length of the name of the directory entry.
	Namlen uint32
	// The type of the file referred to by this directory entry.
	Type Filetype
	// Name of the directory entry.
	Name *byte
}

func direntIno(buf []byte) (uint64, bool) {
	return readInt(buf, unsafe.Offsetof(Dirent{}.Ino), unsafe.Sizeof(Dirent{}.Ino))
}

func direntReclen(buf []byte) (uint64, bool) {
	namelen, ok := direntNamlen(buf)
	return 24 + namelen, ok
}

func direntNamlen(buf []byte) (uint64, bool) {
	return readInt(buf, unsafe.Offsetof(Dirent{}.Namlen), unsafe.Sizeof(Dirent{}.Namlen))
}

// An Errno is an unsigned number describing an error condition.
// It implements the error interface. The zero Errno is by convention
// a non-error, so code to convert from Errno to error should use:
//
//	var err = nil
//	if errno != 0 {
//		err = errno
//	}
type Errno uint32

func (e Errno) Error() string {
	if 0 <= int(e) && int(e) < len(errorstr) {
		s := errorstr[e]
		if s != "" {
			return s
		}
	}
	return "errno " + itoa.Itoa(int(e))
}

func (e Errno) Is(target error) bool {
	switch target {
	case oserror.ErrPermission:
		return e == EACCES || e == EPERM
	case oserror.ErrExist:
		return e == EEXIST || e == ENOTEMPTY
	case oserror.ErrNotExist:
		return e == ENOENT
	case errors.ErrUnsupported:
		return e == ENOSYS
	}
	return false
}

func (e Errno) Temporary() bool {
	return e == EINTR || e == EMFILE || e.Timeout()
}

func (e Errno) Timeout() bool {
	return e == EAGAIN || e == ETIMEDOUT
}

// A Signal is a number describing a process signal.
// It implements the [os.Signal] interface.
type Signal uint8

const (
	SIGNONE Signal = iota
	SIGHUP
	SIGINT
	SIGQUIT
	SIGILL
	SIGTRAP
	SIGABRT
	SIGBUS
	SIGFPE
	SIGKILL
	SIGUSR1
	SIGSEGV
	SIGUSR2
	SIGPIPE
	SIGALRM
	SIGTERM
	SIGCHLD
	SIGCONT
	SIGSTOP
	SIGTSTP
	SIGTTIN
	SIGTTOU
	SIGURG
	SIGXCPU
	SIGXFSZ
	SIGVTARLM
	SIGPROF
	SIGWINCH
	SIGPOLL
	SIGPWR
	SIGSYS
)

func (s Signal) Signal() {}

func (s Signal) String() string {
	switch s {
	case SIGNONE:
		return "no signal"
	case SIGHUP:
		return "hangup"
	case SIGINT:
		return "interrupt"
	case SIGQUIT:
		return "quit"
	case SIGILL:
		return "illegal instruction"
	case SIGTRAP:
		return "trace/breakpoint trap"
	case SIGABRT:
		return "abort"
	case SIGBUS:
		return "bus error"
	case SIGFPE:
		return "floating point exception"
	case SIGKILL:
		return "killed"
	case SIGUSR1:
		return "user defined signal 1"
	case SIGSEGV:
		return "segmentation fault"
	case SIGUSR2:
		return "user defined signal 2"
	case SIGPIPE:
		return "broken pipe"
	case SIGALRM:
		return "alarm clock"
	case SIGTERM:
		return "terminated"
	case SIGCHLD:
		return "child exited"
	case SIGCONT:
		return "continued"
	case SIGSTOP:
		return "stopped (signal)"
	case SIGTSTP:
		return "stopped"
	case SIGTTIN:
		return "stopped (tty input)"
	case SIGTTOU:
		return "stopped (tty output)"
	case SIGURG:
		return "urgent I/O condition"
	case SIGXCPU:
		return "CPU time limit exceeded"
	case SIGXFSZ:
		return "file size limit exceeded"
	case SIGVTARLM:
		return "virtual timer expired"
	case SIGPROF:
		return "profiling timer expired"
	case SIGWINCH:
		return "window changed"
	case SIGPOLL:
		return "I/O possible"
	case SIGPWR:
		return "power failure"
	case SIGSYS:
		return "bad system call"
	default:
		return "signal " + itoa.Itoa(int(s))
	}
}

const (
	Stdin  = 0
	Stdout = 1
	Stderr = 2
)

const (
	O_RDONLY = 0
	O_WRONLY = 1
	O_RDWR   = 2

	O_CREAT     = 0100
	O_CREATE    = O_CREAT
	O_TRUNC     = 01000
	O_APPEND    = 02000
	O_EXCL      = 0200
	O_SYNC      = 010000
	O_DIRECTORY = 020000

	O_CLOEXEC = 0
)

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

// Rusage is a placeholder to allow compilation of the [os/exec] package
// because we need Go programs to be portable across platforms. WASI does
// not have a mechanism to spawn processes so there is no reason for an
// application to take a dependency on this type.
type Rusage struct {
	Utime Timeval
	Stime Timeval
}

// ProcAttr is a placeholder to allow compilation of the [os/exec] package
// because we need Go programs to be portable across platforms. WASI does
// not have a mechanism to spawn processes so there is no reason for an
// application to take a dependency on this type.
type ProcAttr struct {
	Dir   string
	Env   []string
	Files []uintptr
	Sys   *SysProcAttr
}

type SysProcAttr struct {
}

func Syscall(trap, a1, a2, a3 uintptr) (r1, r2 uintptr, err Errno) {
	return 0, 0, ENOSYS
}

func Syscall6(trap, a1, a2, a3, a4, a5, a6 uintptr) (r1, r2 uintptr, err Errno) {
	return 0, 0, ENOSYS
}

func RawSyscall(trap, a1, a2, a3 uintptr) (r1, r2 uintptr, err Errno) {
	return 0, 0, ENOSYS
}

func RawSyscall6(trap, a1, a2, a3, a4, a5, a6 uintptr) (r1, r2 uintptr, err Errno) {
	return 0, 0, ENOSYS
}

func Sysctl(key string) (string, error) {
	if key == "kern.hostname" {
		return "wasip1", nil
	}
	return "", ENOSYS
}

func Getuid() int {
	return 1
}

func Getgid() int {
	return 1
}

func Geteuid() int {
	return 1
}

func Getegid() int {
	return 1
}

func Getgroups() ([]int, error) {
	return []int{1}, nil
}

func Getpid() int {
	return 3
}

func Getppid() int {
	return 2
}

func Gettimeofday(tv *Timeval) error {
	var time timestamp
	if errno := clock_time_get(clockRealtime, 1e3, unsafe.Pointer(&time)); errno != 0 {
		return errno
	}
	tv.setTimestamp(time)
	return nil
}

func Kill(pid int, signum Signal) error {
	// WASI does not have the notion of processes nor signal handlers.
	//
	// Any signal that the application raises to the process itself will
	// be interpreted as being cause for termination.
	if pid > 0 && pid != Getpid() {
		return ESRCH
	}
	ProcExit(128 + int32(signum))
	return nil
}

func Sendfile(outfd int, infd int, offset *int64, count int) (written int, err error) {
	return 0, ENOSYS
}

func StartProcess(argv0 string, argv []string, attr *ProcAttr) (pid int, handle uintptr, err error) {
	return 0, 0, ENOSYS
}

func Wait4(pid int, wstatus *WaitStatus, options int, rusage *Rusage) (wpid int, err error) {
	return 0, ENOSYS
}

func Umask(mask int) int {
	return 0
}

type Timespec struct {
	Sec  int64
	Nsec int64
}

func (ts *Timespec) timestamp() timestamp {
	return timestamp(ts.Sec*1e9) + timestamp(ts.Nsec)
}

func (ts *Timespec) setTimestamp(t timestamp) {
	ts.Sec = int64(t / 1e9)
	ts.Nsec = int64(t % 1e9)
}

type Timeval struct {
	Sec  int64
	Usec int64
}

func (tv *Timeval) timestamp() timestamp {
	return timestamp(tv.Sec*1e9) + timestamp(tv.Usec*1e3)
}

func (tv *Timeval) setTimestamp(t timestamp) {
	tv.Sec = int64(t / 1e9)
	tv.Usec = int64((t % 1e9) / 1e3)
}

func setTimespec(sec, nsec int64) Timespec {
	return Timespec{Sec: sec, Nsec: nsec}
}

func setTimeval(sec, usec int64) Timeval {
	return Timeval{Sec: sec, Usec: usec}
}

type clockid = uint32

const (
	clockRealtime clockid = iota
	clockMonotonic
	clockProcessCPUTimeID
	clockThreadCPUTimeID
)

//go:wasmimport wasi_snapshot_preview1 clock_time_get
//go:noescape
func clock_time_get(id clockid, precision timestamp, time unsafe.Pointer) Errno

func SetNonblock(fd int, nonblocking bool) error {
	flags, err := fd_fdstat_get_flags(fd)
	if err != nil {
		return err
	}
	if nonblocking {
		flags |= FDFLAG_NONBLOCK
	} else {
		flags &^= FDFLAG_NONBLOCK
	}
	errno := fd_fdstat_set_flags(int32(fd), flags)
	return errnoErr(errno)
}

type Rlimit struct {
	Cur uint64
	Max uint64
}

const (
	RLIMIT_NOFILE = iota
)

func Getrlimit(which int, lim *Rlimit) error {
	return ENOSYS
}
