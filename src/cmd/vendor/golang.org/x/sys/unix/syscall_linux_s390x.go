// Copyright 2016 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// +build s390x,linux

package unix

import (
	"unsafe"
)

//sys	Dup2(oldfd int, newfd int) (err error)
//sysnb	EpollCreate(size int) (fd int, err error)
//sys	EpollWait(epfd int, events []EpollEvent, msec int) (n int, err error)
//sys	Fadvise(fd int, offset int64, length int64, advice int) (err error) = SYS_FADVISE64
//sys	Fchown(fd int, uid int, gid int) (err error)
//sys	Fstat(fd int, stat *Stat_t) (err error)
//sys	Fstatat(dirfd int, path string, stat *Stat_t, flags int) (err error) = SYS_NEWFSTATAT
//sys	Fstatfs(fd int, buf *Statfs_t) (err error)
//sys	Ftruncate(fd int, length int64) (err error)
//sysnb	Getegid() (egid int)
//sysnb	Geteuid() (euid int)
//sysnb	Getgid() (gid int)
//sysnb	Getrlimit(resource int, rlim *Rlimit) (err error)
//sysnb	Getuid() (uid int)
//sysnb	InotifyInit() (fd int, err error)
//sys	Lchown(path string, uid int, gid int) (err error)
//sys	Lstat(path string, stat *Stat_t) (err error)
//sys	Pause() (err error)
//sys	Pread(fd int, p []byte, offset int64) (n int, err error) = SYS_PREAD64
//sys	Pwrite(fd int, p []byte, offset int64) (n int, err error) = SYS_PWRITE64
//sys	Renameat(olddirfd int, oldpath string, newdirfd int, newpath string) (err error)
//sys	Seek(fd int, offset int64, whence int) (off int64, err error) = SYS_LSEEK
//sys	Select(nfd int, r *FdSet, w *FdSet, e *FdSet, timeout *Timeval) (n int, err error)
//sys	sendfile(outfd int, infd int, offset *int64, count int) (written int, err error)
//sys	setfsgid(gid int) (prev int, err error)
//sys	setfsuid(uid int) (prev int, err error)
//sysnb	Setregid(rgid int, egid int) (err error)
//sysnb	Setresgid(rgid int, egid int, sgid int) (err error)
//sysnb	Setresuid(ruid int, euid int, suid int) (err error)
//sysnb	Setrlimit(resource int, rlim *Rlimit) (err error)
//sysnb	Setreuid(ruid int, euid int) (err error)
//sys	Splice(rfd int, roff *int64, wfd int, woff *int64, len int, flags int) (n int64, err error)
//sys	Stat(path string, stat *Stat_t) (err error)
//sys	Statfs(path string, buf *Statfs_t) (err error)
//sys	SyncFileRange(fd int, off int64, n int64, flags int) (err error)
//sys	Truncate(path string, length int64) (err error)
//sys	Ustat(dev int, ubuf *Ustat_t) (err error)
//sysnb	getgroups(n int, list *_Gid_t) (nn int, err error)
//sysnb	setgroups(n int, list *_Gid_t) (err error)

//sys	futimesat(dirfd int, path string, times *[2]Timeval) (err error)
//sysnb	Gettimeofday(tv *Timeval) (err error)

func Time(t *Time_t) (tt Time_t, err error) {
	var tv Timeval
	err = Gettimeofday(&tv)
	if err != nil {
		return 0, err
	}
	if t != nil {
		*t = Time_t(tv.Sec)
	}
	return Time_t(tv.Sec), nil
}

//sys	Utime(path string, buf *Utimbuf) (err error)
//sys	utimes(path string, times *[2]Timeval) (err error)

func setTimespec(sec, nsec int64) Timespec {
	return Timespec{Sec: sec, Nsec: nsec}
}

func setTimeval(sec, usec int64) Timeval {
	return Timeval{Sec: sec, Usec: usec}
}

//sysnb pipe2(p *[2]_C_int, flags int) (err error)

func Pipe(p []int) (err error) {
	if len(p) != 2 {
		return EINVAL
	}
	var pp [2]_C_int
	err = pipe2(&pp, 0) // pipe2 is the same as pipe when flags are set to 0.
	p[0] = int(pp[0])
	p[1] = int(pp[1])
	return
}

func Pipe2(p []int, flags int) (err error) {
	if len(p) != 2 {
		return EINVAL
	}
	var pp [2]_C_int
	err = pipe2(&pp, flags)
	p[0] = int(pp[0])
	p[1] = int(pp[1])
	return
}

func Ioperm(from int, num int, on int) (err error) {
	return ENOSYS
}

func Iopl(level int) (err error) {
	return ENOSYS
}

func (r *PtraceRegs) PC() uint64 { return r.Psw.Addr }

func (r *PtraceRegs) SetPC(pc uint64) { r.Psw.Addr = pc }

func (iov *Iovec) SetLen(length int) {
	iov.Len = uint64(length)
}

func (msghdr *Msghdr) SetControllen(length int) {
	msghdr.Controllen = uint64(length)
}

func (msghdr *Msghdr) SetIovlen(length int) {
	msghdr.Iovlen = uint64(length)
}

func (cmsg *Cmsghdr) SetLen(length int) {
	cmsg.Len = uint64(length)
}

// Linux on s390x uses the old mmap interface, which requires arguments to be passed in a struct.
// mmap2 also requires arguments to be passed in a struct; it is currently not exposed in <asm/unistd.h>.
func mmap(addr uintptr, length uintptr, prot int, flags int, fd int, offset int64) (xaddr uintptr, err error) {
	mmap_args := [6]uintptr{addr, length, uintptr(prot), uintptr(flags), uintptr(fd), uintptr(offset)}
	r0, _, e1 := Syscall(SYS_MMAP, uintptr(unsafe.Pointer(&mmap_args[0])), 0, 0)
	xaddr = uintptr(r0)
	if e1 != 0 {
		err = errnoErr(e1)
	}
	return
}

// On s390x Linux, all the socket calls go through an extra indirection.
// The arguments to the underlying system call (SYS_SOCKETCALL) are the
// number below and a pointer to an array of uintptr.
const (
	// see linux/net.h
	netSocket      = 1
	netBind        = 2
	netConnect     = 3
	netListen      = 4
	netAccept      = 5
	netGetSockName = 6
	netGetPeerName = 7
	netSocketPair  = 8
	netSend        = 9
	netRecv        = 10
	netSendTo      = 11
	netRecvFrom    = 12
	netShutdown    = 13
	netSetSockOpt  = 14
	netGetSockOpt  = 15
	netSendMsg     = 16
	netRecvMsg     = 17
	netAccept4     = 18
	netRecvMMsg    = 19
	netSendMMsg    = 20
)

func accept(s int, rsa *RawSockaddrAny, addrlen *_Socklen) (int, error) {
	args := [3]uintptr{uintptr(s), uintptr(unsafe.Pointer(rsa)), uintptr(unsafe.Pointer(addrlen))}
	fd, _, err := Syscall(SYS_SOCKETCALL, netAccept, uintptr(unsafe.Pointer(&args)), 0)
	if err != 0 {
		return 0, err
	}
	return int(fd), nil
}

func accept4(s int, rsa *RawSockaddrAny, addrlen *_Socklen, flags int) (int, error) {
	args := [4]uintptr{uintptr(s), uintptr(unsafe.Pointer(rsa)), uintptr(unsafe.Pointer(addrlen)), uintptr(flags)}
	fd, _, err := Syscall(SYS_SOCKETCALL, netAccept4, uintptr(unsafe.Pointer(&args)), 0)
	if err != 0 {
		return 0, err
	}
	return int(fd), nil
}

func getsockname(s int, rsa *RawSockaddrAny, addrlen *_Socklen) error {
	args := [3]uintptr{uintptr(s), uintptr(unsafe.Pointer(rsa)), uintptr(unsafe.Pointer(addrlen))}
	_, _, err := RawSyscall(SYS_SOCKETCALL, netGetSockName, uintptr(unsafe.Pointer(&args)), 0)
	if err != 0 {
		return err
	}
	return nil
}

func getpeername(s int, rsa *RawSockaddrAny, addrlen *_Socklen) error {
	args := [3]uintptr{uintptr(s), uintptr(unsafe.Pointer(rsa)), uintptr(unsafe.Pointer(addrlen))}
	_, _, err := RawSyscall(SYS_SOCKETCALL, netGetPeerName, uintptr(unsafe.Pointer(&args)), 0)
	if err != 0 {
		return err
	}
	return nil
}

func socketpair(domain int, typ int, flags int, fd *[2]int32) error {
	args := [4]uintptr{uintptr(domain), uintptr(typ), uintptr(flags), uintptr(unsafe.Pointer(fd))}
	_, _, err := RawSyscall(SYS_SOCKETCALL, netSocketPair, uintptr(unsafe.Pointer(&args)), 0)
	if err != 0 {
		return err
	}
	return nil
}

func bind(s int, addr unsafe.Pointer, addrlen _Socklen) error {
	args := [3]uintptr{uintptr(s), uintptr(addr), uintptr(addrlen)}
	_, _, err := Syscall(SYS_SOCKETCALL, netBind, uintptr(unsafe.Pointer(&args)), 0)
	if err != 0 {
		return err
	}
	return nil
}

func connect(s int, addr unsafe.Pointer, addrlen _Socklen) error {
	args := [3]uintptr{uintptr(s), uintptr(addr), uintptr(addrlen)}
	_, _, err := Syscall(SYS_SOCKETCALL, netConnect, uintptr(unsafe.Pointer(&args)), 0)
	if err != 0 {
		return err
	}
	return nil
}

func socket(domain int, typ int, proto int) (int, error) {
	args := [3]uintptr{uintptr(domain), uintptr(typ), uintptr(proto)}
	fd, _, err := RawSyscall(SYS_SOCKETCALL, netSocket, uintptr(unsafe.Pointer(&args)), 0)
	if err != 0 {
		return 0, err
	}
	return int(fd), nil
}

func getsockopt(s int, level int, name int, val unsafe.Pointer, vallen *_Socklen) error {
	args := [5]uintptr{uintptr(s), uintptr(level), uintptr(name), uintptr(val), uintptr(unsafe.Pointer(vallen))}
	_, _, err := Syscall(SYS_SOCKETCALL, netGetSockOpt, uintptr(unsafe.Pointer(&args)), 0)
	if err != 0 {
		return err
	}
	return nil
}

func setsockopt(s int, level int, name int, val unsafe.Pointer, vallen uintptr) error {
	args := [4]uintptr{uintptr(s), uintptr(level), uintptr(name), uintptr(val)}
	_, _, err := Syscall(SYS_SOCKETCALL, netSetSockOpt, uintptr(unsafe.Pointer(&args)), 0)
	if err != 0 {
		return err
	}
	return nil
}

func recvfrom(s int, p []byte, flags int, from *RawSockaddrAny, fromlen *_Socklen) (int, error) {
	var base uintptr
	if len(p) > 0 {
		base = uintptr(unsafe.Pointer(&p[0]))
	}
	args := [6]uintptr{uintptr(s), base, uintptr(len(p)), uintptr(flags), uintptr(unsafe.Pointer(from)), uintptr(unsafe.Pointer(fromlen))}
	n, _, err := Syscall(SYS_SOCKETCALL, netRecvFrom, uintptr(unsafe.Pointer(&args)), 0)
	if err != 0 {
		return 0, err
	}
	return int(n), nil
}

func sendto(s int, p []byte, flags int, to unsafe.Pointer, addrlen _Socklen) error {
	var base uintptr
	if len(p) > 0 {
		base = uintptr(unsafe.Pointer(&p[0]))
	}
	args := [6]uintptr{uintptr(s), base, uintptr(len(p)), uintptr(flags), uintptr(to), uintptr(addrlen)}
	_, _, err := Syscall(SYS_SOCKETCALL, netSendTo, uintptr(unsafe.Pointer(&args)), 0)
	if err != 0 {
		return err
	}
	return nil
}

func recvmsg(s int, msg *Msghdr, flags int) (int, error) {
	args := [3]uintptr{uintptr(s), uintptr(unsafe.Pointer(msg)), uintptr(flags)}
	n, _, err := Syscall(SYS_SOCKETCALL, netRecvMsg, uintptr(unsafe.Pointer(&args)), 0)
	if err != 0 {
		return 0, err
	}
	return int(n), nil
}

func sendmsg(s int, msg *Msghdr, flags int) (int, error) {
	args := [3]uintptr{uintptr(s), uintptr(unsafe.Pointer(msg)), uintptr(flags)}
	n, _, err := Syscall(SYS_SOCKETCALL, netSendMsg, uintptr(unsafe.Pointer(&args)), 0)
	if err != 0 {
		return 0, err
	}
	return int(n), nil
}

func Listen(s int, n int) error {
	args := [2]uintptr{uintptr(s), uintptr(n)}
	_, _, err := Syscall(SYS_SOCKETCALL, netListen, uintptr(unsafe.Pointer(&args)), 0)
	if err != 0 {
		return err
	}
	return nil
}

func Shutdown(s, how int) error {
	args := [2]uintptr{uintptr(s), uintptr(how)}
	_, _, err := Syscall(SYS_SOCKETCALL, netShutdown, uintptr(unsafe.Pointer(&args)), 0)
	if err != 0 {
		return err
	}
	return nil
}

//sys	poll(fds *PollFd, nfds int, timeout int) (n int, err error)

func Poll(fds []PollFd, timeout int) (n int, err error) {
	if len(fds) == 0 {
		return poll(nil, 0, timeout)
	}
	return poll(&fds[0], len(fds), timeout)
}

//sys	kexecFileLoad(kernelFd int, initrdFd int, cmdlineLen int, cmdline string, flags int) (err error)

func KexecFileLoad(kernelFd int, initrdFd int, cmdline string, flags int) error {
	cmdlineLen := len(cmdline)
	if cmdlineLen > 0 {
		// Account for the additional NULL byte added by
		// BytePtrFromString in kexecFileLoad. The kexec_file_load
		// syscall expects a NULL-terminated string.
		cmdlineLen++
	}
	return kexecFileLoad(kernelFd, initrdFd, cmdlineLen, cmdline, flags)
}
