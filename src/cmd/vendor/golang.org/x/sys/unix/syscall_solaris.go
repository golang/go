// Copyright 2009 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// Solaris system calls.
// This file is compiled as ordinary Go code,
// but it is also input to mksyscall,
// which parses the //sys lines and generates system call stubs.
// Note that sometimes we use a lowercase //sys name and wrap
// it in our own nicer implementation, either here or in
// syscall_solaris.go or syscall_unix.go.

package unix

import (
	"fmt"
	"os"
	"runtime"
	"sync"
	"syscall"
	"unsafe"
)

// Implemented in runtime/syscall_solaris.go.
type syscallFunc uintptr

func rawSysvicall6(trap, nargs, a1, a2, a3, a4, a5, a6 uintptr) (r1, r2 uintptr, err syscall.Errno)
func sysvicall6(trap, nargs, a1, a2, a3, a4, a5, a6 uintptr) (r1, r2 uintptr, err syscall.Errno)

// SockaddrDatalink implements the Sockaddr interface for AF_LINK type sockets.
type SockaddrDatalink struct {
	Family uint16
	Index  uint16
	Type   uint8
	Nlen   uint8
	Alen   uint8
	Slen   uint8
	Data   [244]int8
	raw    RawSockaddrDatalink
}

func direntIno(buf []byte) (uint64, bool) {
	return readInt(buf, unsafe.Offsetof(Dirent{}.Ino), unsafe.Sizeof(Dirent{}.Ino))
}

func direntReclen(buf []byte) (uint64, bool) {
	return readInt(buf, unsafe.Offsetof(Dirent{}.Reclen), unsafe.Sizeof(Dirent{}.Reclen))
}

func direntNamlen(buf []byte) (uint64, bool) {
	reclen, ok := direntReclen(buf)
	if !ok {
		return 0, false
	}
	return reclen - uint64(unsafe.Offsetof(Dirent{}.Name)), true
}

//sysnb	pipe(p *[2]_C_int) (n int, err error)

func Pipe(p []int) (err error) {
	if len(p) != 2 {
		return EINVAL
	}
	var pp [2]_C_int
	n, err := pipe(&pp)
	if n != 0 {
		return err
	}
	if err == nil {
		p[0] = int(pp[0])
		p[1] = int(pp[1])
	}
	return nil
}

//sysnb	pipe2(p *[2]_C_int, flags int) (err error)

func Pipe2(p []int, flags int) error {
	if len(p) != 2 {
		return EINVAL
	}
	var pp [2]_C_int
	err := pipe2(&pp, flags)
	if err == nil {
		p[0] = int(pp[0])
		p[1] = int(pp[1])
	}
	return err
}

func (sa *SockaddrInet4) sockaddr() (unsafe.Pointer, _Socklen, error) {
	if sa.Port < 0 || sa.Port > 0xFFFF {
		return nil, 0, EINVAL
	}
	sa.raw.Family = AF_INET
	p := (*[2]byte)(unsafe.Pointer(&sa.raw.Port))
	p[0] = byte(sa.Port >> 8)
	p[1] = byte(sa.Port)
	sa.raw.Addr = sa.Addr
	return unsafe.Pointer(&sa.raw), SizeofSockaddrInet4, nil
}

func (sa *SockaddrInet6) sockaddr() (unsafe.Pointer, _Socklen, error) {
	if sa.Port < 0 || sa.Port > 0xFFFF {
		return nil, 0, EINVAL
	}
	sa.raw.Family = AF_INET6
	p := (*[2]byte)(unsafe.Pointer(&sa.raw.Port))
	p[0] = byte(sa.Port >> 8)
	p[1] = byte(sa.Port)
	sa.raw.Scope_id = sa.ZoneId
	sa.raw.Addr = sa.Addr
	return unsafe.Pointer(&sa.raw), SizeofSockaddrInet6, nil
}

func (sa *SockaddrUnix) sockaddr() (unsafe.Pointer, _Socklen, error) {
	name := sa.Name
	n := len(name)
	if n >= len(sa.raw.Path) {
		return nil, 0, EINVAL
	}
	sa.raw.Family = AF_UNIX
	for i := 0; i < n; i++ {
		sa.raw.Path[i] = int8(name[i])
	}
	// length is family (uint16), name, NUL.
	sl := _Socklen(2)
	if n > 0 {
		sl += _Socklen(n) + 1
	}
	if sa.raw.Path[0] == '@' {
		sa.raw.Path[0] = 0
		// Don't count trailing NUL for abstract address.
		sl--
	}

	return unsafe.Pointer(&sa.raw), sl, nil
}

//sys	getsockname(fd int, rsa *RawSockaddrAny, addrlen *_Socklen) (err error) = libsocket.getsockname

func Getsockname(fd int) (sa Sockaddr, err error) {
	var rsa RawSockaddrAny
	var len _Socklen = SizeofSockaddrAny
	if err = getsockname(fd, &rsa, &len); err != nil {
		return
	}
	return anyToSockaddr(fd, &rsa)
}

// GetsockoptString returns the string value of the socket option opt for the
// socket associated with fd at the given socket level.
func GetsockoptString(fd, level, opt int) (string, error) {
	buf := make([]byte, 256)
	vallen := _Socklen(len(buf))
	err := getsockopt(fd, level, opt, unsafe.Pointer(&buf[0]), &vallen)
	if err != nil {
		return "", err
	}
	return string(buf[:vallen-1]), nil
}

const ImplementsGetwd = true

//sys	Getcwd(buf []byte) (n int, err error)

func Getwd() (wd string, err error) {
	var buf [PathMax]byte
	// Getcwd will return an error if it failed for any reason.
	_, err = Getcwd(buf[0:])
	if err != nil {
		return "", err
	}
	n := clen(buf[:])
	if n < 1 {
		return "", EINVAL
	}
	return string(buf[:n]), nil
}

/*
 * Wrapped
 */

//sysnb	getgroups(ngid int, gid *_Gid_t) (n int, err error)
//sysnb	setgroups(ngid int, gid *_Gid_t) (err error)

func Getgroups() (gids []int, err error) {
	n, err := getgroups(0, nil)
	// Check for error and sanity check group count. Newer versions of
	// Solaris allow up to 1024 (NGROUPS_MAX).
	if n < 0 || n > 1024 {
		if err != nil {
			return nil, err
		}
		return nil, EINVAL
	} else if n == 0 {
		return nil, nil
	}

	a := make([]_Gid_t, n)
	n, err = getgroups(n, &a[0])
	if n == -1 {
		return nil, err
	}
	gids = make([]int, n)
	for i, v := range a[0:n] {
		gids[i] = int(v)
	}
	return
}

func Setgroups(gids []int) (err error) {
	if len(gids) == 0 {
		return setgroups(0, nil)
	}

	a := make([]_Gid_t, len(gids))
	for i, v := range gids {
		a[i] = _Gid_t(v)
	}
	return setgroups(len(a), &a[0])
}

// ReadDirent reads directory entries from fd and writes them into buf.
func ReadDirent(fd int, buf []byte) (n int, err error) {
	// Final argument is (basep *uintptr) and the syscall doesn't take nil.
	// TODO(rsc): Can we use a single global basep for all calls?
	return Getdents(fd, buf, new(uintptr))
}

// Wait status is 7 bits at bottom, either 0 (exited),
// 0x7F (stopped), or a signal number that caused an exit.
// The 0x80 bit is whether there was a core dump.
// An extra number (exit code, signal causing a stop)
// is in the high bits.

type WaitStatus uint32

const (
	mask  = 0x7F
	core  = 0x80
	shift = 8

	exited  = 0
	stopped = 0x7F
)

func (w WaitStatus) Exited() bool { return w&mask == exited }

func (w WaitStatus) ExitStatus() int {
	if w&mask != exited {
		return -1
	}
	return int(w >> shift)
}

func (w WaitStatus) Signaled() bool { return w&mask != stopped && w&mask != 0 }

func (w WaitStatus) Signal() syscall.Signal {
	sig := syscall.Signal(w & mask)
	if sig == stopped || sig == 0 {
		return -1
	}
	return sig
}

func (w WaitStatus) CoreDump() bool { return w.Signaled() && w&core != 0 }

func (w WaitStatus) Stopped() bool { return w&mask == stopped && syscall.Signal(w>>shift) != SIGSTOP }

func (w WaitStatus) Continued() bool { return w&mask == stopped && syscall.Signal(w>>shift) == SIGSTOP }

func (w WaitStatus) StopSignal() syscall.Signal {
	if !w.Stopped() {
		return -1
	}
	return syscall.Signal(w>>shift) & 0xFF
}

func (w WaitStatus) TrapCause() int { return -1 }

//sys	wait4(pid int32, statusp *_C_int, options int, rusage *Rusage) (wpid int32, err error)

func Wait4(pid int, wstatus *WaitStatus, options int, rusage *Rusage) (int, error) {
	var status _C_int
	rpid, err := wait4(int32(pid), &status, options, rusage)
	wpid := int(rpid)
	if wpid == -1 {
		return wpid, err
	}
	if wstatus != nil {
		*wstatus = WaitStatus(status)
	}
	return wpid, nil
}

//sys	gethostname(buf []byte) (n int, err error)

func Gethostname() (name string, err error) {
	var buf [MaxHostNameLen]byte
	n, err := gethostname(buf[:])
	if n != 0 {
		return "", err
	}
	n = clen(buf[:])
	if n < 1 {
		return "", EFAULT
	}
	return string(buf[:n]), nil
}

//sys	utimes(path string, times *[2]Timeval) (err error)

func Utimes(path string, tv []Timeval) (err error) {
	if tv == nil {
		return utimes(path, nil)
	}
	if len(tv) != 2 {
		return EINVAL
	}
	return utimes(path, (*[2]Timeval)(unsafe.Pointer(&tv[0])))
}

//sys	utimensat(fd int, path string, times *[2]Timespec, flag int) (err error)

func UtimesNano(path string, ts []Timespec) error {
	if ts == nil {
		return utimensat(AT_FDCWD, path, nil, 0)
	}
	if len(ts) != 2 {
		return EINVAL
	}
	return utimensat(AT_FDCWD, path, (*[2]Timespec)(unsafe.Pointer(&ts[0])), 0)
}

func UtimesNanoAt(dirfd int, path string, ts []Timespec, flags int) error {
	if ts == nil {
		return utimensat(dirfd, path, nil, flags)
	}
	if len(ts) != 2 {
		return EINVAL
	}
	return utimensat(dirfd, path, (*[2]Timespec)(unsafe.Pointer(&ts[0])), flags)
}

//sys	fcntl(fd int, cmd int, arg int) (val int, err error)

// FcntlInt performs a fcntl syscall on fd with the provided command and argument.
func FcntlInt(fd uintptr, cmd, arg int) (int, error) {
	valptr, _, errno := sysvicall6(uintptr(unsafe.Pointer(&procfcntl)), 3, uintptr(fd), uintptr(cmd), uintptr(arg), 0, 0, 0)
	var err error
	if errno != 0 {
		err = errno
	}
	return int(valptr), err
}

// FcntlFlock performs a fcntl syscall for the F_GETLK, F_SETLK or F_SETLKW command.
func FcntlFlock(fd uintptr, cmd int, lk *Flock_t) error {
	_, _, e1 := sysvicall6(uintptr(unsafe.Pointer(&procfcntl)), 3, uintptr(fd), uintptr(cmd), uintptr(unsafe.Pointer(lk)), 0, 0, 0)
	if e1 != 0 {
		return e1
	}
	return nil
}

//sys	futimesat(fildes int, path *byte, times *[2]Timeval) (err error)

func Futimesat(dirfd int, path string, tv []Timeval) error {
	pathp, err := BytePtrFromString(path)
	if err != nil {
		return err
	}
	if tv == nil {
		return futimesat(dirfd, pathp, nil)
	}
	if len(tv) != 2 {
		return EINVAL
	}
	return futimesat(dirfd, pathp, (*[2]Timeval)(unsafe.Pointer(&tv[0])))
}

// Solaris doesn't have an futimes function because it allows NULL to be
// specified as the path for futimesat. However, Go doesn't like
// NULL-style string interfaces, so this simple wrapper is provided.
func Futimes(fd int, tv []Timeval) error {
	if tv == nil {
		return futimesat(fd, nil, nil)
	}
	if len(tv) != 2 {
		return EINVAL
	}
	return futimesat(fd, nil, (*[2]Timeval)(unsafe.Pointer(&tv[0])))
}

func anyToSockaddr(fd int, rsa *RawSockaddrAny) (Sockaddr, error) {
	switch rsa.Addr.Family {
	case AF_UNIX:
		pp := (*RawSockaddrUnix)(unsafe.Pointer(rsa))
		sa := new(SockaddrUnix)
		// Assume path ends at NUL.
		// This is not technically the Solaris semantics for
		// abstract Unix domain sockets -- they are supposed
		// to be uninterpreted fixed-size binary blobs -- but
		// everyone uses this convention.
		n := 0
		for n < len(pp.Path) && pp.Path[n] != 0 {
			n++
		}
		bytes := (*[len(pp.Path)]byte)(unsafe.Pointer(&pp.Path[0]))[0:n]
		sa.Name = string(bytes)
		return sa, nil

	case AF_INET:
		pp := (*RawSockaddrInet4)(unsafe.Pointer(rsa))
		sa := new(SockaddrInet4)
		p := (*[2]byte)(unsafe.Pointer(&pp.Port))
		sa.Port = int(p[0])<<8 + int(p[1])
		sa.Addr = pp.Addr
		return sa, nil

	case AF_INET6:
		pp := (*RawSockaddrInet6)(unsafe.Pointer(rsa))
		sa := new(SockaddrInet6)
		p := (*[2]byte)(unsafe.Pointer(&pp.Port))
		sa.Port = int(p[0])<<8 + int(p[1])
		sa.ZoneId = pp.Scope_id
		sa.Addr = pp.Addr
		return sa, nil
	}
	return nil, EAFNOSUPPORT
}

//sys	accept(s int, rsa *RawSockaddrAny, addrlen *_Socklen) (fd int, err error) = libsocket.accept

func Accept(fd int) (nfd int, sa Sockaddr, err error) {
	var rsa RawSockaddrAny
	var len _Socklen = SizeofSockaddrAny
	nfd, err = accept(fd, &rsa, &len)
	if nfd == -1 {
		return
	}
	sa, err = anyToSockaddr(fd, &rsa)
	if err != nil {
		Close(nfd)
		nfd = 0
	}
	return
}

//sys	recvmsg(s int, msg *Msghdr, flags int) (n int, err error) = libsocket.__xnet_recvmsg

func recvmsgRaw(fd int, iov []Iovec, oob []byte, flags int, rsa *RawSockaddrAny) (n, oobn int, recvflags int, err error) {
	var msg Msghdr
	msg.Name = (*byte)(unsafe.Pointer(rsa))
	msg.Namelen = uint32(SizeofSockaddrAny)
	var dummy byte
	if len(oob) > 0 {
		// receive at least one normal byte
		if emptyIovecs(iov) {
			var iova [1]Iovec
			iova[0].Base = &dummy
			iova[0].SetLen(1)
			iov = iova[:]
		}
		msg.Accrightslen = int32(len(oob))
	}
	if len(iov) > 0 {
		msg.Iov = &iov[0]
		msg.SetIovlen(len(iov))
	}
	if n, err = recvmsg(fd, &msg, flags); n == -1 {
		return
	}
	oobn = int(msg.Accrightslen)
	return
}

//sys	sendmsg(s int, msg *Msghdr, flags int) (n int, err error) = libsocket.__xnet_sendmsg

func sendmsgN(fd int, iov []Iovec, oob []byte, ptr unsafe.Pointer, salen _Socklen, flags int) (n int, err error) {
	var msg Msghdr
	msg.Name = (*byte)(unsafe.Pointer(ptr))
	msg.Namelen = uint32(salen)
	var dummy byte
	var empty bool
	if len(oob) > 0 {
		// send at least one normal byte
		empty = emptyIovecs(iov)
		if empty {
			var iova [1]Iovec
			iova[0].Base = &dummy
			iova[0].SetLen(1)
			iov = iova[:]
		}
		msg.Accrightslen = int32(len(oob))
	}
	if len(iov) > 0 {
		msg.Iov = &iov[0]
		msg.SetIovlen(len(iov))
	}
	if n, err = sendmsg(fd, &msg, flags); err != nil {
		return 0, err
	}
	if len(oob) > 0 && empty {
		n = 0
	}
	return n, nil
}

//sys	acct(path *byte) (err error)

func Acct(path string) (err error) {
	if len(path) == 0 {
		// Assume caller wants to disable accounting.
		return acct(nil)
	}

	pathp, err := BytePtrFromString(path)
	if err != nil {
		return err
	}
	return acct(pathp)
}

//sys	__makedev(version int, major uint, minor uint) (val uint64)

func Mkdev(major, minor uint32) uint64 {
	return __makedev(NEWDEV, uint(major), uint(minor))
}

//sys	__major(version int, dev uint64) (val uint)

func Major(dev uint64) uint32 {
	return uint32(__major(NEWDEV, dev))
}

//sys	__minor(version int, dev uint64) (val uint)

func Minor(dev uint64) uint32 {
	return uint32(__minor(NEWDEV, dev))
}

/*
 * Expose the ioctl function
 */

//sys	ioctlRet(fd int, req uint, arg uintptr) (ret int, err error) = libc.ioctl

func ioctl(fd int, req uint, arg uintptr) (err error) {
	_, err = ioctlRet(fd, req, arg)
	return err
}

func IoctlSetTermio(fd int, req uint, value *Termio) error {
	err := ioctl(fd, req, uintptr(unsafe.Pointer(value)))
	runtime.KeepAlive(value)
	return err
}

func IoctlGetTermio(fd int, req uint) (*Termio, error) {
	var value Termio
	err := ioctl(fd, req, uintptr(unsafe.Pointer(&value)))
	return &value, err
}

//sys	poll(fds *PollFd, nfds int, timeout int) (n int, err error)

func Poll(fds []PollFd, timeout int) (n int, err error) {
	if len(fds) == 0 {
		return poll(nil, 0, timeout)
	}
	return poll(&fds[0], len(fds), timeout)
}

func Sendfile(outfd int, infd int, offset *int64, count int) (written int, err error) {
	if raceenabled {
		raceReleaseMerge(unsafe.Pointer(&ioSync))
	}
	return sendfile(outfd, infd, offset, count)
}

/*
 * Exposed directly
 */
//sys	Access(path string, mode uint32) (err error)
//sys	Adjtime(delta *Timeval, olddelta *Timeval) (err error)
//sys	Chdir(path string) (err error)
//sys	Chmod(path string, mode uint32) (err error)
//sys	Chown(path string, uid int, gid int) (err error)
//sys	Chroot(path string) (err error)
//sys	Close(fd int) (err error)
//sys	Creat(path string, mode uint32) (fd int, err error)
//sys	Dup(fd int) (nfd int, err error)
//sys	Dup2(oldfd int, newfd int) (err error)
//sys	Exit(code int)
//sys	Faccessat(dirfd int, path string, mode uint32, flags int) (err error)
//sys	Fchdir(fd int) (err error)
//sys	Fchmod(fd int, mode uint32) (err error)
//sys	Fchmodat(dirfd int, path string, mode uint32, flags int) (err error)
//sys	Fchown(fd int, uid int, gid int) (err error)
//sys	Fchownat(dirfd int, path string, uid int, gid int, flags int) (err error)
//sys	Fdatasync(fd int) (err error)
//sys	Flock(fd int, how int) (err error)
//sys	Fpathconf(fd int, name int) (val int, err error)
//sys	Fstat(fd int, stat *Stat_t) (err error)
//sys	Fstatat(fd int, path string, stat *Stat_t, flags int) (err error)
//sys	Fstatvfs(fd int, vfsstat *Statvfs_t) (err error)
//sys	Getdents(fd int, buf []byte, basep *uintptr) (n int, err error)
//sysnb	Getgid() (gid int)
//sysnb	Getpid() (pid int)
//sysnb	Getpgid(pid int) (pgid int, err error)
//sysnb	Getpgrp() (pgid int, err error)
//sys	Geteuid() (euid int)
//sys	Getegid() (egid int)
//sys	Getppid() (ppid int)
//sys	Getpriority(which int, who int) (n int, err error)
//sysnb	Getrlimit(which int, lim *Rlimit) (err error)
//sysnb	Getrusage(who int, rusage *Rusage) (err error)
//sysnb	Getsid(pid int) (sid int, err error)
//sysnb	Gettimeofday(tv *Timeval) (err error)
//sysnb	Getuid() (uid int)
//sys	Kill(pid int, signum syscall.Signal) (err error)
//sys	Lchown(path string, uid int, gid int) (err error)
//sys	Link(path string, link string) (err error)
//sys	Listen(s int, backlog int) (err error) = libsocket.__xnet_llisten
//sys	Lstat(path string, stat *Stat_t) (err error)
//sys	Madvise(b []byte, advice int) (err error)
//sys	Mkdir(path string, mode uint32) (err error)
//sys	Mkdirat(dirfd int, path string, mode uint32) (err error)
//sys	Mkfifo(path string, mode uint32) (err error)
//sys	Mkfifoat(dirfd int, path string, mode uint32) (err error)
//sys	Mknod(path string, mode uint32, dev int) (err error)
//sys	Mknodat(dirfd int, path string, mode uint32, dev int) (err error)
//sys	Mlock(b []byte) (err error)
//sys	Mlockall(flags int) (err error)
//sys	Mprotect(b []byte, prot int) (err error)
//sys	Msync(b []byte, flags int) (err error)
//sys	Munlock(b []byte) (err error)
//sys	Munlockall() (err error)
//sys	Nanosleep(time *Timespec, leftover *Timespec) (err error)
//sys	Open(path string, mode int, perm uint32) (fd int, err error)
//sys	Openat(dirfd int, path string, flags int, mode uint32) (fd int, err error)
//sys	Pathconf(path string, name int) (val int, err error)
//sys	Pause() (err error)
//sys	pread(fd int, p []byte, offset int64) (n int, err error)
//sys	pwrite(fd int, p []byte, offset int64) (n int, err error)
//sys	read(fd int, p []byte) (n int, err error)
//sys	Readlink(path string, buf []byte) (n int, err error)
//sys	Rename(from string, to string) (err error)
//sys	Renameat(olddirfd int, oldpath string, newdirfd int, newpath string) (err error)
//sys	Rmdir(path string) (err error)
//sys	Seek(fd int, offset int64, whence int) (newoffset int64, err error) = lseek
//sys	Select(nfd int, r *FdSet, w *FdSet, e *FdSet, timeout *Timeval) (n int, err error)
//sysnb	Setegid(egid int) (err error)
//sysnb	Seteuid(euid int) (err error)
//sysnb	Setgid(gid int) (err error)
//sys	Sethostname(p []byte) (err error)
//sysnb	Setpgid(pid int, pgid int) (err error)
//sys	Setpriority(which int, who int, prio int) (err error)
//sysnb	Setregid(rgid int, egid int) (err error)
//sysnb	Setreuid(ruid int, euid int) (err error)
//sysnb	Setrlimit(which int, lim *Rlimit) (err error)
//sysnb	Setsid() (pid int, err error)
//sysnb	Setuid(uid int) (err error)
//sys	Shutdown(s int, how int) (err error) = libsocket.shutdown
//sys	Stat(path string, stat *Stat_t) (err error)
//sys	Statvfs(path string, vfsstat *Statvfs_t) (err error)
//sys	Symlink(path string, link string) (err error)
//sys	Sync() (err error)
//sys	Sysconf(which int) (n int64, err error)
//sysnb	Times(tms *Tms) (ticks uintptr, err error)
//sys	Truncate(path string, length int64) (err error)
//sys	Fsync(fd int) (err error)
//sys	Ftruncate(fd int, length int64) (err error)
//sys	Umask(mask int) (oldmask int)
//sysnb	Uname(buf *Utsname) (err error)
//sys	Unmount(target string, flags int) (err error) = libc.umount
//sys	Unlink(path string) (err error)
//sys	Unlinkat(dirfd int, path string, flags int) (err error)
//sys	Ustat(dev int, ubuf *Ustat_t) (err error)
//sys	Utime(path string, buf *Utimbuf) (err error)
//sys	bind(s int, addr unsafe.Pointer, addrlen _Socklen) (err error) = libsocket.__xnet_bind
//sys	connect(s int, addr unsafe.Pointer, addrlen _Socklen) (err error) = libsocket.__xnet_connect
//sys	mmap(addr uintptr, length uintptr, prot int, flag int, fd int, pos int64) (ret uintptr, err error)
//sys	munmap(addr uintptr, length uintptr) (err error)
//sys	sendfile(outfd int, infd int, offset *int64, count int) (written int, err error) = libsendfile.sendfile
//sys	sendto(s int, buf []byte, flags int, to unsafe.Pointer, addrlen _Socklen) (err error) = libsocket.__xnet_sendto
//sys	socket(domain int, typ int, proto int) (fd int, err error) = libsocket.__xnet_socket
//sysnb	socketpair(domain int, typ int, proto int, fd *[2]int32) (err error) = libsocket.__xnet_socketpair
//sys	write(fd int, p []byte) (n int, err error)
//sys	getsockopt(s int, level int, name int, val unsafe.Pointer, vallen *_Socklen) (err error) = libsocket.__xnet_getsockopt
//sysnb	getpeername(fd int, rsa *RawSockaddrAny, addrlen *_Socklen) (err error) = libsocket.getpeername
//sys	setsockopt(s int, level int, name int, val unsafe.Pointer, vallen uintptr) (err error) = libsocket.setsockopt
//sys	recvfrom(fd int, p []byte, flags int, from *RawSockaddrAny, fromlen *_Socklen) (n int, err error) = libsocket.recvfrom

func readlen(fd int, buf *byte, nbuf int) (n int, err error) {
	r0, _, e1 := sysvicall6(uintptr(unsafe.Pointer(&procread)), 3, uintptr(fd), uintptr(unsafe.Pointer(buf)), uintptr(nbuf), 0, 0, 0)
	n = int(r0)
	if e1 != 0 {
		err = e1
	}
	return
}

func writelen(fd int, buf *byte, nbuf int) (n int, err error) {
	r0, _, e1 := sysvicall6(uintptr(unsafe.Pointer(&procwrite)), 3, uintptr(fd), uintptr(unsafe.Pointer(buf)), uintptr(nbuf), 0, 0, 0)
	n = int(r0)
	if e1 != 0 {
		err = e1
	}
	return
}

var mapper = &mmapper{
	active: make(map[*byte][]byte),
	mmap:   mmap,
	munmap: munmap,
}

func Mmap(fd int, offset int64, length int, prot int, flags int) (data []byte, err error) {
	return mapper.Mmap(fd, offset, length, prot, flags)
}

func Munmap(b []byte) (err error) {
	return mapper.Munmap(b)
}

// Event Ports

type fileObjCookie struct {
	fobj   *fileObj
	cookie interface{}
}

// EventPort provides a safe abstraction on top of Solaris/illumos Event Ports.
type EventPort struct {
	port  int
	mu    sync.Mutex
	fds   map[uintptr]*fileObjCookie
	paths map[string]*fileObjCookie
	// The user cookie presents an interesting challenge from a memory management perspective.
	// There are two paths by which we can discover that it is no longer in use:
	// 1. The user calls port_dissociate before any events fire
	// 2. An event fires and we return it to the user
	// The tricky situation is if the event has fired in the kernel but
	// the user hasn't requested/received it yet.
	// If the user wants to port_dissociate before the event has been processed,
	// we should handle things gracefully. To do so, we need to keep an extra
	// reference to the cookie around until the event is processed
	// thus the otherwise seemingly extraneous "cookies" map
	// The key of this map is a pointer to the corresponding fCookie
	cookies map[*fileObjCookie]struct{}
}

// PortEvent is an abstraction of the port_event C struct.
// Compare Source against PORT_SOURCE_FILE or PORT_SOURCE_FD
// to see if Path or Fd was the event source. The other will be
// uninitialized.
type PortEvent struct {
	Cookie interface{}
	Events int32
	Fd     uintptr
	Path   string
	Source uint16
	fobj   *fileObj
}

// NewEventPort creates a new EventPort including the
// underlying call to port_create(3c).
func NewEventPort() (*EventPort, error) {
	port, err := port_create()
	if err != nil {
		return nil, err
	}
	e := &EventPort{
		port:    port,
		fds:     make(map[uintptr]*fileObjCookie),
		paths:   make(map[string]*fileObjCookie),
		cookies: make(map[*fileObjCookie]struct{}),
	}
	return e, nil
}

//sys	port_create() (n int, err error)
//sys	port_associate(port int, source int, object uintptr, events int, user *byte) (n int, err error)
//sys	port_dissociate(port int, source int, object uintptr) (n int, err error)
//sys	port_get(port int, pe *portEvent, timeout *Timespec) (n int, err error)
//sys	port_getn(port int, pe *portEvent, max uint32, nget *uint32, timeout *Timespec) (n int, err error)

// Close closes the event port.
func (e *EventPort) Close() error {
	e.mu.Lock()
	defer e.mu.Unlock()
	err := Close(e.port)
	if err != nil {
		return err
	}
	e.fds = nil
	e.paths = nil
	e.cookies = nil
	return nil
}

// PathIsWatched checks to see if path is associated with this EventPort.
func (e *EventPort) PathIsWatched(path string) bool {
	e.mu.Lock()
	defer e.mu.Unlock()
	_, found := e.paths[path]
	return found
}

// FdIsWatched checks to see if fd is associated with this EventPort.
func (e *EventPort) FdIsWatched(fd uintptr) bool {
	e.mu.Lock()
	defer e.mu.Unlock()
	_, found := e.fds[fd]
	return found
}

// AssociatePath wraps port_associate(3c) for a filesystem path including
// creating the necessary file_obj from the provided stat information.
func (e *EventPort) AssociatePath(path string, stat os.FileInfo, events int, cookie interface{}) error {
	e.mu.Lock()
	defer e.mu.Unlock()
	if _, found := e.paths[path]; found {
		return fmt.Errorf("%v is already associated with this Event Port", path)
	}
	fCookie, err := createFileObjCookie(path, stat, cookie)
	if err != nil {
		return err
	}
	_, err = port_associate(e.port, PORT_SOURCE_FILE, uintptr(unsafe.Pointer(fCookie.fobj)), events, (*byte)(unsafe.Pointer(fCookie)))
	if err != nil {
		return err
	}
	e.paths[path] = fCookie
	e.cookies[fCookie] = struct{}{}
	return nil
}

// DissociatePath wraps port_dissociate(3c) for a filesystem path.
func (e *EventPort) DissociatePath(path string) error {
	e.mu.Lock()
	defer e.mu.Unlock()
	f, ok := e.paths[path]
	if !ok {
		return fmt.Errorf("%v is not associated with this Event Port", path)
	}
	_, err := port_dissociate(e.port, PORT_SOURCE_FILE, uintptr(unsafe.Pointer(f.fobj)))
	// If the path is no longer associated with this event port (ENOENT)
	// we should delete it from our map. We can still return ENOENT to the caller.
	// But we need to save the cookie
	if err != nil && err != ENOENT {
		return err
	}
	if err == nil {
		// dissociate was successful, safe to delete the cookie
		fCookie := e.paths[path]
		delete(e.cookies, fCookie)
	}
	delete(e.paths, path)
	return err
}

// AssociateFd wraps calls to port_associate(3c) on file descriptors.
func (e *EventPort) AssociateFd(fd uintptr, events int, cookie interface{}) error {
	e.mu.Lock()
	defer e.mu.Unlock()
	if _, found := e.fds[fd]; found {
		return fmt.Errorf("%v is already associated with this Event Port", fd)
	}
	fCookie, err := createFileObjCookie("", nil, cookie)
	if err != nil {
		return err
	}
	_, err = port_associate(e.port, PORT_SOURCE_FD, fd, events, (*byte)(unsafe.Pointer(fCookie)))
	if err != nil {
		return err
	}
	e.fds[fd] = fCookie
	e.cookies[fCookie] = struct{}{}
	return nil
}

// DissociateFd wraps calls to port_dissociate(3c) on file descriptors.
func (e *EventPort) DissociateFd(fd uintptr) error {
	e.mu.Lock()
	defer e.mu.Unlock()
	_, ok := e.fds[fd]
	if !ok {
		return fmt.Errorf("%v is not associated with this Event Port", fd)
	}
	_, err := port_dissociate(e.port, PORT_SOURCE_FD, fd)
	if err != nil && err != ENOENT {
		return err
	}
	if err == nil {
		// dissociate was successful, safe to delete the cookie
		fCookie := e.fds[fd]
		delete(e.cookies, fCookie)
	}
	delete(e.fds, fd)
	return err
}

func createFileObjCookie(name string, stat os.FileInfo, cookie interface{}) (*fileObjCookie, error) {
	fCookie := new(fileObjCookie)
	fCookie.cookie = cookie
	if name != "" && stat != nil {
		fCookie.fobj = new(fileObj)
		bs, err := ByteSliceFromString(name)
		if err != nil {
			return nil, err
		}
		fCookie.fobj.Name = (*int8)(unsafe.Pointer(&bs[0]))
		s := stat.Sys().(*syscall.Stat_t)
		fCookie.fobj.Atim.Sec = s.Atim.Sec
		fCookie.fobj.Atim.Nsec = s.Atim.Nsec
		fCookie.fobj.Mtim.Sec = s.Mtim.Sec
		fCookie.fobj.Mtim.Nsec = s.Mtim.Nsec
		fCookie.fobj.Ctim.Sec = s.Ctim.Sec
		fCookie.fobj.Ctim.Nsec = s.Ctim.Nsec
	}
	return fCookie, nil
}

// GetOne wraps port_get(3c) and returns a single PortEvent.
func (e *EventPort) GetOne(t *Timespec) (*PortEvent, error) {
	pe := new(portEvent)
	_, err := port_get(e.port, pe, t)
	if err != nil {
		return nil, err
	}
	p := new(PortEvent)
	e.mu.Lock()
	defer e.mu.Unlock()
	err = e.peIntToExt(pe, p)
	if err != nil {
		return nil, err
	}
	return p, nil
}

// peIntToExt converts a cgo portEvent struct into the friendlier PortEvent
// NOTE: Always call this function while holding the e.mu mutex
func (e *EventPort) peIntToExt(peInt *portEvent, peExt *PortEvent) error {
	if e.cookies == nil {
		return fmt.Errorf("this EventPort is already closed")
	}
	peExt.Events = peInt.Events
	peExt.Source = peInt.Source
	fCookie := (*fileObjCookie)(unsafe.Pointer(peInt.User))
	_, found := e.cookies[fCookie]

	if !found {
		panic("unexpected event port address; may be due to kernel bug; see https://go.dev/issue/54254")
	}
	peExt.Cookie = fCookie.cookie
	delete(e.cookies, fCookie)

	switch peInt.Source {
	case PORT_SOURCE_FD:
		peExt.Fd = uintptr(peInt.Object)
		// Only remove the fds entry if it exists and this cookie matches
		if fobj, ok := e.fds[peExt.Fd]; ok {
			if fobj == fCookie {
				delete(e.fds, peExt.Fd)
			}
		}
	case PORT_SOURCE_FILE:
		peExt.fobj = fCookie.fobj
		peExt.Path = BytePtrToString((*byte)(unsafe.Pointer(peExt.fobj.Name)))
		// Only remove the paths entry if it exists and this cookie matches
		if fobj, ok := e.paths[peExt.Path]; ok {
			if fobj == fCookie {
				delete(e.paths, peExt.Path)
			}
		}
	}
	return nil
}

// Pending wraps port_getn(3c) and returns how many events are pending.
func (e *EventPort) Pending() (int, error) {
	var n uint32 = 0
	_, err := port_getn(e.port, nil, 0, &n, nil)
	return int(n), err
}

// Get wraps port_getn(3c) and fills a slice of PortEvent.
// It will block until either min events have been received
// or the timeout has been exceeded. It will return how many
// events were actually received along with any error information.
func (e *EventPort) Get(s []PortEvent, min int, timeout *Timespec) (int, error) {
	if min == 0 {
		return 0, fmt.Errorf("need to request at least one event or use Pending() instead")
	}
	if len(s) < min {
		return 0, fmt.Errorf("len(s) (%d) is less than min events requested (%d)", len(s), min)
	}
	got := uint32(min)
	max := uint32(len(s))
	var err error
	ps := make([]portEvent, max)
	_, err = port_getn(e.port, &ps[0], max, &got, timeout)
	// got will be trustworthy with ETIME, but not any other error.
	if err != nil && err != ETIME {
		return 0, err
	}
	e.mu.Lock()
	defer e.mu.Unlock()
	valid := 0
	for i := 0; i < int(got); i++ {
		err2 := e.peIntToExt(&ps[i], &s[i])
		if err2 != nil {
			if valid == 0 && err == nil {
				// If err2 is the only error and there are no valid events
				// to return, return it to the caller.
				err = err2
			}
			break
		}
		valid = i + 1
	}
	return valid, err
}

//sys	putmsg(fd int, clptr *strbuf, dataptr *strbuf, flags int) (err error)

func Putmsg(fd int, cl []byte, data []byte, flags int) (err error) {
	var clp, datap *strbuf
	if len(cl) > 0 {
		clp = &strbuf{
			Len: int32(len(cl)),
			Buf: (*int8)(unsafe.Pointer(&cl[0])),
		}
	}
	if len(data) > 0 {
		datap = &strbuf{
			Len: int32(len(data)),
			Buf: (*int8)(unsafe.Pointer(&data[0])),
		}
	}
	return putmsg(fd, clp, datap, flags)
}

//sys	getmsg(fd int, clptr *strbuf, dataptr *strbuf, flags *int) (err error)

func Getmsg(fd int, cl []byte, data []byte) (retCl []byte, retData []byte, flags int, err error) {
	var clp, datap *strbuf
	if len(cl) > 0 {
		clp = &strbuf{
			Maxlen: int32(len(cl)),
			Buf:    (*int8)(unsafe.Pointer(&cl[0])),
		}
	}
	if len(data) > 0 {
		datap = &strbuf{
			Maxlen: int32(len(data)),
			Buf:    (*int8)(unsafe.Pointer(&data[0])),
		}
	}

	if err = getmsg(fd, clp, datap, &flags); err != nil {
		return nil, nil, 0, err
	}

	if len(cl) > 0 {
		retCl = cl[:clp.Len]
	}
	if len(data) > 0 {
		retData = data[:datap.Len]
	}
	return retCl, retData, flags, nil
}

func IoctlSetIntRetInt(fd int, req uint, arg int) (int, error) {
	return ioctlRet(fd, req, uintptr(arg))
}

func IoctlSetString(fd int, req uint, val string) error {
	bs := make([]byte, len(val)+1)
	copy(bs[:len(bs)-1], val)
	err := ioctl(fd, req, uintptr(unsafe.Pointer(&bs[0])))
	runtime.KeepAlive(&bs[0])
	return err
}

// Lifreq Helpers

func (l *Lifreq) SetName(name string) error {
	if len(name) >= len(l.Name) {
		return fmt.Errorf("name cannot be more than %d characters", len(l.Name)-1)
	}
	for i := range name {
		l.Name[i] = int8(name[i])
	}
	return nil
}

func (l *Lifreq) SetLifruInt(d int) {
	*(*int)(unsafe.Pointer(&l.Lifru[0])) = d
}

func (l *Lifreq) GetLifruInt() int {
	return *(*int)(unsafe.Pointer(&l.Lifru[0]))
}

func (l *Lifreq) SetLifruUint(d uint) {
	*(*uint)(unsafe.Pointer(&l.Lifru[0])) = d
}

func (l *Lifreq) GetLifruUint() uint {
	return *(*uint)(unsafe.Pointer(&l.Lifru[0]))
}

func IoctlLifreq(fd int, req uint, l *Lifreq) error {
	return ioctl(fd, req, uintptr(unsafe.Pointer(l)))
}

// Strioctl Helpers

func (s *Strioctl) SetInt(i int) {
	s.Len = int32(unsafe.Sizeof(i))
	s.Dp = (*int8)(unsafe.Pointer(&i))
}

func IoctlSetStrioctlRetInt(fd int, req uint, s *Strioctl) (int, error) {
	return ioctlRet(fd, req, uintptr(unsafe.Pointer(s)))
}
