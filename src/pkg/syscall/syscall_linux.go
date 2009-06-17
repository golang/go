// Copyright 2009 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// Linux system calls.
// This file is compiled as ordinary Go code,
// but it is also input to mksyscall,
// which parses the //sys lines and generates system call stubs.
// Note that sometimes we use a lowercase //sys name and
// wrap it in our own nicer implementation.

package syscall

import (
	"syscall";
	"unsafe";
)

const OS = "linux"

/*
 * Wrapped
 */

//sys	pipe(p *[2]_C_int) (errno int)
func Pipe(p []int) (errno int) {
	if len(p) != 2 {
		return EINVAL;
	}
	var pp [2]_C_int;
	errno = pipe(&pp);
	p[0] = int(pp[0]);
	p[1] = int(pp[1]);
	return;
}

//sys	utimes(path string, times *[2]Timeval) (errno int)
func Utimes(path string, tv []Timeval) (errno int) {
	if len(tv) != 2 {
		return EINVAL;
	}
	return utimes(path, (*[2]Timeval)(unsafe.Pointer(&tv[0])));
}

//sys	futimesat(dirfd int, path string, times *[2]Timeval) (errno int)
func Futimesat(dirfd int, path string, tv []Timeval) (errno int) {
	if len(tv) != 2 {
		return EINVAL;
	}
	return futimesat(dirfd, path, (*[2]Timeval)(unsafe.Pointer(&tv[0])));
}

const ImplementsGetwd = true;

//sys	Getcwd(buf []byte) (n int, errno int)
func Getwd() (wd string, errno int) {
	var buf [PathMax]byte;
	n, err := Getcwd(&buf);
	if err != 0 {
		return "", err;
	}
	// Getcwd returns the number of bytes written to buf, including the NUL.
	if n < 1|| n > len(buf) || buf[n-1] != 0 {
		return "", EINVAL;
	}
	return string(buf[0:n-1]), 0
}

func Getgroups() (gids []int, errno int) {
	n, err := getgroups(0, nil);
	if err != 0 {
		return nil, errno;
	}
	if n == 0 {
		return nil, 0;
	}

	// Sanity check group count.  Max is 1<<16 on Linux.
	if n < 0 || n > 1<<20 {
		return nil, EINVAL;
	}

	a := make([]_Gid_t, n);
	n, err = getgroups(n, &a[0]);
	if err != 0 {
		return nil, errno;
	}
	gids = make([]int, n);
	for i, v := range a[0:n] {
		gids[i] = int(v);
	}
	return;
}

func Setgroups(gids []int) (errno int) {
	if len(gids) == 0 {
		return setgroups(0, nil);
	}

	a := make([]_Gid_t, len(gids));
	for i, v := range gids {
		a[i] = _Gid_t(v);
	}
	return setgroups(len(a), &a[0]);
}

type WaitStatus uint32

// Wait status is 7 bits at bottom, either 0 (exited),
// 0x7F (stopped), or a signal number that caused an exit.
// The 0x80 bit is whether there was a core dump.
// An extra number (exit code, signal causing a stop)
// is in the high bits.  At least that's the idea.
// There are various irregularities.  For example, the
// "continued" status is 0xFFFF, distinguishing itself
// from stopped via the core dump bit.

const (
	mask = 0x7F;
	core = 0x80;
	exited = 0x00;
	stopped = 0x7F;
	shift = 8;
)

func (w WaitStatus) Exited() bool {
	return w&mask == exited;
}

func (w WaitStatus) Signaled() bool {
	return w&mask != stopped && w&mask != exited;
}

func (w WaitStatus) Stopped() bool {
	return w&0xFF == stopped;
}

func (w WaitStatus) Continued() bool {
	return w == 0xFFFF;
}

func (w WaitStatus) CoreDump() bool {
	return w.Signaled() && w&core != 0;
}

func (w WaitStatus) ExitStatus() int {
	if !w.Exited() {
		return -1;
	}
	return int(w >> shift) & 0xFF;
}

func (w WaitStatus) Signal() int {
	if !w.Signaled() {
		return -1;
	}
	return int(w & mask);
}

func (w WaitStatus) StopSignal() int {
	if !w.Stopped() {
		return -1;
	}
	return int(w >> shift) & 0xFF;
}

//sys	wait4(pid int, wstatus *_C_int, options int, rusage *Rusage) (wpid int, errno int)
func Wait4(pid int, wstatus *WaitStatus, options int, rusage *Rusage) (wpid int, errno int) {
	var status _C_int;
	wpid, errno = wait4(pid, &status, options, rusage);
	if wstatus != nil {
		*wstatus = WaitStatus(status);
	}
	return;
}

func Sleep(nsec int64) (errno int) {
	tv := NsecToTimeval(nsec);
	n, err := Select(0, nil, nil, nil, &tv);
	return err;
}

// Implemented in syscall_linux_*.go
func accept(s int, rsa *RawSockaddrAny, addrlen *_Socklen) (fd int, errno int)
func bind(s int, addr uintptr, addrlen _Socklen) (errno int)
func connect(s int, addr uintptr, addrlen _Socklen) (errno int)
func socket(domain int, typ int, proto int) (fd int, errno int)
func setsockopt(s int, level int, name int, val uintptr, vallen int) (errno int)
func Listen(s int, n int) (errno int)

// For testing: clients can set this flag to force
// creation of IPv6 sockets to return EAFNOSUPPORT.
var SocketDisableIPv6 bool

type Sockaddr interface {
	sockaddr() (ptr uintptr, len _Socklen, errno int);	// lowercase; only we can define Sockaddrs
}

type SockaddrInet4 struct {
	Port int;
	Addr [4]byte;
	raw RawSockaddrInet4;
}

func (sa *SockaddrInet4) sockaddr() (uintptr, _Socklen, int) {
	if sa.Port < 0 || sa.Port > 0xFFFF {
		return 0, 0, EINVAL;
	}
	sa.raw.Family = AF_INET;
	p := (*[2]byte)(unsafe.Pointer(&sa.raw.Port));
	p[0] = byte(sa.Port>>8);
	p[1] = byte(sa.Port);
	for i := 0; i < len(sa.Addr); i++ {
		sa.raw.Addr[i] = sa.Addr[i];
	}
	return uintptr(unsafe.Pointer(&sa.raw)), SizeofSockaddrInet4, 0;
}

type SockaddrInet6 struct {
	Port int;
	Addr [16]byte;
	raw RawSockaddrInet6;
}

func (sa *SockaddrInet6) sockaddr() (uintptr, _Socklen, int) {
	if sa.Port < 0 || sa.Port > 0xFFFF {
		return 0, 0, EINVAL;
	}
	sa.raw.Family = AF_INET6;
	p := (*[2]byte)(unsafe.Pointer(&sa.raw.Port));
	p[0] = byte(sa.Port>>8);
	p[1] = byte(sa.Port);
	for i := 0; i < len(sa.Addr); i++ {
		sa.raw.Addr[i] = sa.Addr[i];
	}
	return uintptr(unsafe.Pointer(&sa.raw)), SizeofSockaddrInet6, 0;
}

type SockaddrUnix struct {
	Name string;
	raw RawSockaddrUnix;
}

func (sa *SockaddrUnix) sockaddr() (uintptr, _Socklen, int) {
	name := sa.Name;
	n := len(name);
	if n >= len(sa.raw.Path) || n == 0 {
		return 0, 0, EINVAL;
	}
	sa.raw.Family = AF_UNIX;
	for i := 0; i < n; i++ {
		sa.raw.Path[i] = int8(name[i]);
	}
	if sa.raw.Path[0] == '@' {
		sa.raw.Path[0] = 0;
	}

	// length is family, name, NUL.
	return uintptr(unsafe.Pointer(&sa.raw)), 1 + _Socklen(n) + 1, 0;
}

func anyToSockaddr(rsa *RawSockaddrAny) (Sockaddr, int) {
	switch rsa.Addr.Family {
	case AF_UNIX:
		pp := (*RawSockaddrUnix)(unsafe.Pointer(rsa));
		sa := new(SockaddrUnix);
		if pp.Path[0] == 0 {
			// "Abstract" Unix domain socket.
			// Rewrite leading NUL as @ for textual display.
			// (This is the standard convention.)
			// Not friendly to overwrite in place,
			// but the callers below don't care.
			pp.Path[0] = '@';
		}

		// Assume path ends at NUL.
		// This is not technically the Linux semantics for
		// abstract Unix domain sockets--they are supposed
		// to be uninterpreted fixed-size binary blobs--but
		// everyone uses this convention.
		n := 0;
		for n < len(pp.Path) && pp.Path[n] != 0 {
			n++;
		}
		bytes := (*[len(pp.Path)]byte)(unsafe.Pointer(&pp.Path[0]));
		sa.Name = string(bytes[0:n]);
		return sa, 0;

	case AF_INET:
		pp := (*RawSockaddrInet4)(unsafe.Pointer(rsa));
		sa := new(SockaddrInet4);
		p := (*[2]byte)(unsafe.Pointer(&pp.Port));
		sa.Port = int(p[0])<<8 + int(p[1]);
		for i := 0; i < len(sa.Addr); i++ {
			sa.Addr[i] = pp.Addr[i];
		}
		return sa, 0;

	case AF_INET6:
		pp := (*RawSockaddrInet6)(unsafe.Pointer(rsa));
		sa := new(SockaddrInet6);
		p := (*[2]byte)(unsafe.Pointer(&pp.Port));
		sa.Port = int(p[0])<<8 + int(p[1]);
		for i := 0; i < len(sa.Addr); i++ {
			sa.Addr[i] = pp.Addr[i];
		}
		return sa, 0;
	}
	return nil, EAFNOSUPPORT;
}

func Accept(fd int) (nfd int, sa Sockaddr, errno int) {
	var rsa RawSockaddrAny;
	var len _Socklen = SizeofSockaddrAny;
	nfd, errno = accept(fd, &rsa, &len);
	if errno != 0 {
		return;
	}
	sa, errno = anyToSockaddr(&rsa);
	if errno != 0 {
		Close(nfd);
		nfd = 0;
	}
	return;
}

func Bind(fd int, sa Sockaddr) (errno int) {
	ptr, n, err := sa.sockaddr();
	if err != 0 {
		return err;
	}
	return bind(fd, ptr, n);
}

func Connect(fd int, sa Sockaddr) (errno int) {
	ptr, n, err := sa.sockaddr();
	if err != 0 {
		return err;
	}
	return connect(fd, ptr, n);
}

func Socket(domain, typ, proto int) (fd, errno int) {
	if domain == AF_INET6 && SocketDisableIPv6 {
		return -1, EAFNOSUPPORT
	}
	fd, errno = socket(domain, typ, proto);
	return;
}

func SetsockoptInt(fd, level, opt int, value int) (errno int) {
	var n = int32(value);
	return setsockopt(fd, level, opt, uintptr(unsafe.Pointer(&n)), 4);
}

func SetsockoptTimeval(fd, level, opt int, tv *Timeval) (errno int) {
	return setsockopt(fd, level, opt, uintptr(unsafe.Pointer(tv)), unsafe.Sizeof(*tv));
}

func SetsockoptLinger(fd, level, opt int, l *Linger) (errno int) {
	return setsockopt(fd, level, opt, uintptr(unsafe.Pointer(l)), unsafe.Sizeof(*l));
}

// Sendto
// Recvfrom
// Sendmsg
// Recvmsg
// Getsockname
// Getpeername
// Socketpair
// Getsockopt

/*
 * Direct access
 */
//sys	Access(path string, mode int) (errno int)
//sys	Acct(path string) (errno int)
//sys	Adjtimex(buf *Timex) (state int, errno int)
//sys	Chdir(path string) (errno int)
//sys	Chmod(path string, mode int) (errno int)
//sys	Chroot(path string) (errno int)
//sys	Close(fd int) (errno int)
//sys	Creat(path string, mode int) (fd int, errno int)
//sys	Dup(oldfd int) (fd int, errno int)
//sys	Dup2(oldfd int, newfd int) (fd int, errno int)
//sys	EpollCreate(size int) (fd int, errno int)
//sys	EpollCtl(epfd int, op int, fd int, event *EpollEvent) (errno int)
//sys	EpollWait(epfd int, events []EpollEvent, msec int) (n int, errno int)
//sys	Exit(code int) = SYS_EXIT_GROUP
//sys	Faccessat(dirfd int, path string, mode int, flags int) (errno int)
//sys	Fallocate(fd int, mode int, off int64, len int64) (errno int)
//sys	Fchdir(fd int) (errno int)
//sys	Fchmod(fd int, mode int) (errno int)
//sys	Fchmodat(dirfd int, path string, mode int, flags int) (errno int)
//sys	Fchownat(dirfd int, path string, uid int, gid int, flags int) (errno int)
//sys	fcntl(fd int, cmd int, arg int) (val int, errno int)
//sys	Fdatasync(fd int) (errno int)
//sys	Fsync(fd int) (errno int)
//sys	Ftruncate(fd int, length int64) (errno int)
//sys	Getdents(fd int, buf []byte) (n int, errno int) = SYS_GETDENTS64
//sys	Getpgid(pid int) (pgid int, errno int)
//sys	Getpgrp() (pid int)
//sys	Getpid() (pid int)
//sys	Getppid() (ppid int)
//sys	Getrlimit(resource int, rlim *Rlimit) (errno int)
//sys	Getrusage(who int, rusage *Rusage) (errno int)
//sys	Gettid() (tid int)
//sys	Gettimeofday(tv *Timeval) (errno int)
//sys	Ioperm(from int, num int, on int) (errno int)
//sys	Iopl(level int) (errno int)
//sys	Kill(pid int, sig int) (errno int)
//sys	Klogctl(typ int, buf []byte) (n int, errno int) = SYS_SYSLOG
//sys	Link(oldpath string, newpath string) (errno int)
//sys	Mkdir(path string, mode int) (errno int)
//sys	Mkdirat(dirfd int, path string, mode int) (errno int)
//sys	Mknod(path string, mode int, dev int) (errno int)
//sys	Mknodat(dirfd int, path string, mode int, dev int) (errno int)
//sys	Nanosleep(time *Timespec, leftover *Timespec) (errno int)
//sys	Open(path string, mode int, perm int) (fd int, errno int)
//sys	Openat(dirfd int, path string, flags int, mode int) (fd int, errno int)
//sys	Pause() (errno int)
//sys	PivotRoot(newroot string, putold string) (errno int) = SYS_PIVOT_ROOT
//sys	Pread(fd int, p []byte, offset int64) (n int, errno int) = SYS_PREAD64
//sys	Pwrite(fd int, p []byte, offset int64) (n int, errno int) = SYS_PWRITE64
//sys	Read(fd int, p []byte) (n int, errno int)
//sys	Readlink(path string, buf []byte) (n int, errno int)
//sys	Rename(oldpath string, newpath string) (errno int)
//sys	Renameat(olddirfd int, oldpath string, newdirfd int, newpath string) (errno int)
//sys	Rmdir(path string) (errno int)
//sys	Setdomainname(p []byte) (errno int)
//sys	Sethostname(p []byte) (errno int)
//sys	Setpgid(pid int, pgid int) (errno int)
//sys	Setrlimit(resource int, rlim *Rlimit) (errno int)
//sys	Setsid() (pid int)
//sys	Settimeofday(tv *Timeval) (errno int)
//sys	Setuid(uid int) (errno int)
//sys	Splice(rfd int, roff *int64, wfd int, woff *int64, len int, flags int) (n int64, errno int)
//sys	Symlink(oldpath string, newpath string) (errno int)
//sys	Sync()
//sys	SyncFileRange(fd int, off int64, n int64, flags int) (errno int)
//sys	Sysinfo(info *Sysinfo_t) (errno int)
//sys	Tee(rfd int, wfd int, len int, flags int) (n int64, errno int)
//sys	Tgkill(tgid int, tid int, sig int) (errno int)
//sys	Time(t *Time_t) (tt Time_t, errno int)
//sys	Times(tms *Tms) (ticks uintptr, errno int)
//sys	Truncate(path string, length int64) (errno int)
//sys	Umask(mask int) (oldmask int)
//sys	Uname(buf *Utsname) (errno int)
//sys	Unlink(path string) (errno int)
//sys	Unlinkat(dirfd int, path string) (errno int)
//sys	Unshare(flags int) (errno int)
//sys	Ustat(dev int, ubuf *Ustat_t) (errno int)
//sys	Utime(path string, buf *Utimbuf) (errno int)
//sys	Write(fd int, p []byte) (n int, errno int)
//sys	exitThread(code int) (errno int) = SYS_EXIT
//sys	read(fd int, p *byte, np int) (n int, errno int)
//sys	write(fd int, p *byte, np int) (n int, errno int)

/*
 * Unimplemented
 */
// AddKey
// AfsSyscall
// Alarm
// ArchPrctl
// Brk
// Capget
// Capset
// ClockGetres
// ClockGettime
// ClockNanosleep
// ClockSettime
// Clone
// CreateModule
// DeleteModule
// EpollCtlOld
// EpollPwait
// EpollWaitOld
// Eventfd
// Execve
// Fadvise64
// Fgetxattr
// Flistxattr
// Flock
// Fork
// Fremovexattr
// Fsetxattr
// Futex
// GetKernelSyms
// GetMempolicy
// GetRobustList
// GetThreadArea
// Getitimer
// Getpmsg
// Getpriority
// Getxattr
// InotifyAddWatch
// InotifyInit
// InotifyRmWatch
// IoCancel
// IoDestroy
// IoGetevents
// IoSetup
// IoSubmit
// Ioctl
// IoprioGet
// IoprioSet
// KexecLoad
// Keyctl
// Lgetxattr
// Listxattr
// Llistxattr
// LookupDcookie
// Lremovexattr
// Lsetxattr
// Madvise
// Mbind
// MigratePages
// Mincore
// Mlock
// Mmap
// ModifyLdt
// Mount
// MovePages
// Mprotect
// MqGetsetattr
// MqNotify
// MqOpen
// MqTimedreceive
// MqTimedsend
// MqUnlink
// Mremap
// Msgctl
// Msgget
// Msgrcv
// Msgsnd
// Msync
// Munlock
// Munlockall
// Munmap
// Newfstatat
// Nfsservctl
// Personality
// Poll
// Ppoll
// Prctl
// Pselect6
// Ptrace
// Putpmsg
// QueryModule
// Quotactl
// Readahead
// Readv
// Reboot
// RemapFilePages
// Removexattr
// RequestKey
// RestartSyscall
// RtSigaction
// RtSigpending
// RtSigprocmask
// RtSigqueueinfo
// RtSigreturn
// RtSigsuspend
// RtSigtimedwait
// SchedGetPriorityMax
// SchedGetPriorityMin
// SchedGetaffinity
// SchedGetparam
// SchedGetscheduler
// SchedRrGetInterval
// SchedSetaffinity
// SchedSetparam
// SchedYield
// Security
// Semctl
// Semget
// Semop
// Semtimedop
// Sendfile
// SetMempolicy
// SetRobustList
// SetThreadArea
// SetTidAddress
// Setpriority
// Setxattr
// Shmat
// Shmctl
// Shmdt
// Shmget
// Sigaltstack
// Signalfd
// Swapoff
// Swapon
// Sysfs
// TimerCreate
// TimerDelete
// TimerGetoverrun
// TimerGettime
// TimerSettime
// Timerfd
// Tkill (obsolete)
// Tuxcall
// Umount2
// Uselib
// Utimensat
// Vfork
// Vhangup
// Vmsplice
// Vserver
// Waitid
// Writev
// _Sysctl
