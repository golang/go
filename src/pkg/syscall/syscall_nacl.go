// Copyright 2009 The Go Authors.  All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// Native Client system calls.

package syscall

const OS = "nacl"

// Auto-generated

//sys	Chmod(path string, mode int) (errno int)
//sys	Clock() (clock int)
//sys	Close(fd int) (errno int)
//sys	Exit(code int)
//sys	Fstat(fd int, stat *Stat_t) (errno int)
//sys	Getdents(fd int, buf []byte) (n int, errno int)
//sys	Getpid() (pid int)
//sys	Gettimeofday(tv *Timeval) (errno int)
//sys	Open(path string, mode int, perm int) (fd int, errno int)
//sys	Read(fd int, p []byte) (n int, errno int)
//sys	read(fd int, buf *byte, nbuf int) (n int, errno int)
//sys	Stat(path string, stat *Stat_t) (errno int)
//sys	Write(fd int, p []byte) (n int, errno int)

// Hand-written

func Seek(fd int, offset int64, whence int) (newoffset int64, errno int) {
	// Offset passed to system call is 32 bits.  Failure of vision by NaCl.
	if int64(int32(offset)) != offset {
		return 0, ERANGE
	}
	o, _, e := Syscall(SYS_LSEEK, uintptr(fd), uintptr(offset), uintptr(whence));
	return int64(o), int(e);
}

// Implemented in NaCl but not here:
//	SYS_IOCTL
//	SYS_SYSBRK
//	SYS_MMAP
//	SYS_MUNMAP
//	SYS_MULTIMEDIA_*
//	SYS_VIDEO_*
//	SYS_AUDIO_*
//	SYS_IMC_*
//	SYS_MUTEX_*
//	SYS_COND_*
//	SYS_THREAD_*
//	SYS_TLS_*
//	SYS_SRPC_*
//	SYS_SEM_*
//	SYS_SCHED_YIELD
//	SYS_SYSCONF

// Not implemented in NaCl but needed to compile other packages.

const (
	SIGTRAP = 5;
)

func Pipe(p []int) (errno int) {
	return ENACL;
}

func fcntl(fd, cmd, arg int) (val int, errno int) {
	return 0, ENACL;
}

func Pread(fd int, p []byte, offset int64) (n int, errno int) {
	return 0, ENACL;
}

func Pwrite(fd int, p []byte, offset int64) (n int, errno int) {
	return 0, ENACL;
}

func Mkdir(path string, mode int) (errno int) {
	return ENACL;
}

func Lstat(path string, stat *Stat_t) (errno int) {
	return ENACL;
}

func Chdir(path string) (errno int) {
	return ENACL;
}

func Fchdir(fd int) (errno int) {
	return ENACL;
}

func Unlink(path string) (errno int) {
	return ENACL;
}

func Rmdir(path string) (errno int) {
	return ENACL;
}

func Link(oldpath, newpath string) (errno int) {
	return ENACL;
}

func Symlink(path, link string) (errno int) {
	return ENACL;
}

func Readlink(path string, buf []byte) (n int, errno int) {
	return 0, ENACL;
}

func Fchmod(fd int, mode int) (errno int) {
	return ENACL;
}

func Chown(path string, uid int, gid int) (errno int) {
	return ENACL;
}

func Lchown(path string, uid int, gid int) (errno int) {
	return ENACL;
}

func Fchown(fd int, uid int, gid int) (errno int) {
	return ENACL;
}

func Truncate(name string, size int64) (errno int) {
	return ENACL;
}

func Ftruncate(fd int, length int64) (errno int) {
	return ENACL;
}

// TODO(rsc): There must be a way to sleep, perhaps
// via the multimedia system calls.

func Sleep(ns int64) (errno int) {
	return ENACL;
}

// NaCL doesn't actually implement Getwd, but it also
// don't implement Chdir, so the fallback algorithm
// fails worse than calling Getwd does.

const ImplementsGetwd = true;

func Getwd() (wd string, errno int) {
	return "", ENACL;
}

func Getuid() (uid int) {
	return -1
}

func Geteuid() (euid int) {
	return -1
}

func Getgid() (gid int) {
	return -1
}

func Getegid() (egid int) {
	return -1
}

func Getppid() (ppid int) {
	return -1
}

func Getgroups() (gids []int, errno int) {
	return nil, ENACL
}

type Sockaddr interface {
	sockaddr()
}

type SockaddrInet4 struct {
	Port int;
	Addr [4]byte;
}

func (*SockaddrInet4) sockaddr() {
}

type SockaddrInet6 struct {
	Port int;
	Addr [16]byte;
}

func (*SockaddrInet6) sockaddr() {
}

type SockaddrUnix struct {
	Name string;
}

func (*SockaddrUnix) sockaddr() {
}

const (
	AF_INET = 1+iota;
	AF_INET6;
	AF_UNIX;
	IPPROTO_TCP;
	SOCK_DGRAM;
	SOCK_STREAM;
	SOL_SOCKET;
	SOMAXCONN;
	SO_DONTROUTE;
	SO_KEEPALIVE;
	SO_LINGER;
	SO_RCVBUF;
	SO_REUSEADDR;
	SO_SNDBUF;
	TCP_NODELAY;
	_PTRACE_TRACEME;
)

func Accept(fd int) (nfd int, sa Sockaddr, errno int) {
	return 0, nil, ENACL;
}

func Getsockname(fd int) (sa Sockaddr, errno int) {
	return nil, ENACL;
}

func Getpeername(fd int) (sa Sockaddr, errno int) {
	return nil, ENACL;
}

func Bind(fd int, sa Sockaddr) (errno int) {
	return ENACL;
}

func Connect(fd int, sa Sockaddr) (errno int) {
	return ENACL;
}

func Socket(domain, typ, proto int) (fd, errno int) {
	return 0, ENACL;
}

func SetsockoptInt(fd, level, opt int, value int) (errno int) {
	return ENACL;
}

func SetsockoptTimeval(fd, level, opt int, tv *Timeval) (errno int) {
	return ENACL;
}

type Linger struct {
	Onoff int32;
	Linger int32;
}

func SetsockoptLinger(fd, level, opt int, l *Linger) (errno int) {
	return ENACL;
}

func Listen(s int, n int) (errno int) {
	return ENACL;
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

func Wait4(pid int, wstatus *WaitStatus, options int, rusage *Rusage) (wpid int, errno int) {
	return 0, ENACL;
}

type WaitStatus uint32

func (WaitStatus) Exited() bool {
	return false
}

func (WaitStatus) ExitStatus() int {
	return -1
}

func (WaitStatus) Signal() int {
	return -1
}

func (WaitStatus) CoreDump() bool {
	return false
}

func (WaitStatus) Stopped() bool {
	return false
}

func (WaitStatus) Continued() bool {
	return false
}

func (WaitStatus) StopSignal() int {
	return -1
}

func (WaitStatus) Signaled() bool {
	return false
}

func (WaitStatus) TrapCause() int {
	return -1
}
