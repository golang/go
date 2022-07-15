// Copyright 2009 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package syscall

import "unsafe"

const (
	_SYS_setgroups  = SYS_SETGROUPS32
	_SYS_clone3     = 435
	_SYS_faccessat2 = 439
)

func setTimespec(sec, nsec int64) Timespec {
	return Timespec{Sec: int32(sec), Nsec: int32(nsec)}
}

func setTimeval(sec, usec int64) Timeval {
	return Timeval{Sec: int32(sec), Usec: int32(usec)}
}

// 64-bit file system and 32-bit uid calls
// (386 default is 32-bit file system and 16-bit uid).
//sys	Dup2(oldfd int, newfd int) (err error)
//sys	Fchown(fd int, uid int, gid int) (err error) = SYS_FCHOWN32
//sys	Fstat(fd int, stat *Stat_t) (err error) = SYS_FSTAT64
//sys	fstatat(dirfd int, path string, stat *Stat_t, flags int) (err error) = SYS_FSTATAT64
//sys	Ftruncate(fd int, length int64) (err error) = SYS_FTRUNCATE64
//sysnb	Getegid() (egid int) = SYS_GETEGID32
//sysnb	Geteuid() (euid int) = SYS_GETEUID32
//sysnb	Getgid() (gid int) = SYS_GETGID32
//sysnb	Getuid() (uid int) = SYS_GETUID32
//sysnb	InotifyInit() (fd int, err error)
//sys	Ioperm(from int, num int, on int) (err error)
//sys	Iopl(level int) (err error)
//sys	Pause() (err error)
//sys	pread(fd int, p []byte, offset int64) (n int, err error) = SYS_PREAD64
//sys	pwrite(fd int, p []byte, offset int64) (n int, err error) = SYS_PWRITE64
//sys	Renameat(olddirfd int, oldpath string, newdirfd int, newpath string) (err error)
//sys	sendfile(outfd int, infd int, offset *int64, count int) (written int, err error) = SYS_SENDFILE64
//sys	Setfsgid(gid int) (err error) = SYS_SETFSGID32
//sys	Setfsuid(uid int) (err error) = SYS_SETFSUID32
//sys	Splice(rfd int, roff *int64, wfd int, woff *int64, len int, flags int) (n int, err error)
//sys	SyncFileRange(fd int, off int64, n int64, flags int) (err error)
//sys	Truncate(path string, length int64) (err error) = SYS_TRUNCATE64
//sys	Ustat(dev int, ubuf *Ustat_t) (err error)
//sysnb	getgroups(n int, list *_Gid_t) (nn int, err error) = SYS_GETGROUPS32
//sys	Select(nfd int, r *FdSet, w *FdSet, e *FdSet, timeout *Timeval) (n int, err error) = SYS__NEWSELECT

//sys	mmap2(addr uintptr, length uintptr, prot int, flags int, fd int, pageOffset uintptr) (xaddr uintptr, err error)
//sys	EpollWait(epfd int, events []EpollEvent, msec int) (n int, err error)

func Stat(path string, stat *Stat_t) (err error) {
	return fstatat(_AT_FDCWD, path, stat, 0)
}

func Lchown(path string, uid int, gid int) (err error) {
	return Fchownat(_AT_FDCWD, path, uid, gid, _AT_SYMLINK_NOFOLLOW)
}

func Lstat(path string, stat *Stat_t) (err error) {
	return fstatat(_AT_FDCWD, path, stat, _AT_SYMLINK_NOFOLLOW)
}

func mmap(addr uintptr, length uintptr, prot int, flags int, fd int, offset int64) (xaddr uintptr, err error) {
	page := uintptr(offset / 4096)
	if offset != int64(page)*4096 {
		return 0, EINVAL
	}
	return mmap2(addr, length, prot, flags, fd, page)
}

type rlimit32 struct {
	Cur uint32
	Max uint32
}

//sysnb getrlimit(resource int, rlim *rlimit32) (err error) = SYS_GETRLIMIT

const rlimInf32 = ^uint32(0)
const rlimInf64 = ^uint64(0)

func Getrlimit(resource int, rlim *Rlimit) (err error) {
	err = prlimit(0, resource, nil, rlim)
	if err != ENOSYS {
		return err
	}

	rl := rlimit32{}
	err = getrlimit(resource, &rl)
	if err != nil {
		return
	}

	if rl.Cur == rlimInf32 {
		rlim.Cur = rlimInf64
	} else {
		rlim.Cur = uint64(rl.Cur)
	}

	if rl.Max == rlimInf32 {
		rlim.Max = rlimInf64
	} else {
		rlim.Max = uint64(rl.Max)
	}
	return
}

//sysnb setrlimit(resource int, rlim *rlimit32) (err error) = SYS_SETRLIMIT

func Setrlimit(resource int, rlim *Rlimit) (err error) {
	err = prlimit(0, resource, rlim, nil)
	if err != ENOSYS {
		return err
	}

	rl := rlimit32{}
	if rlim.Cur == rlimInf64 {
		rl.Cur = rlimInf32
	} else if rlim.Cur < uint64(rlimInf32) {
		rl.Cur = uint32(rlim.Cur)
	} else {
		return EINVAL
	}
	if rlim.Max == rlimInf64 {
		rl.Max = rlimInf32
	} else if rlim.Max < uint64(rlimInf32) {
		rl.Max = uint32(rlim.Max)
	} else {
		return EINVAL
	}

	return setrlimit(resource, &rl)
}

// Underlying system call writes to newoffset via pointer.
// Implemented in assembly to avoid allocation.
func seek(fd int, offset int64, whence int) (newoffset int64, err Errno)

func Seek(fd int, offset int64, whence int) (newoffset int64, err error) {
	newoffset, errno := seek(fd, offset, whence)
	if errno != 0 {
		return 0, errno
	}
	return newoffset, nil
}

//sys	futimesat(dirfd int, path string, times *[2]Timeval) (err error)
//sysnb	Gettimeofday(tv *Timeval) (err error)
//sysnb	Time(t *Time_t) (tt Time_t, err error)
//sys	Utime(path string, buf *Utimbuf) (err error)
//sys	utimes(path string, times *[2]Timeval) (err error)

// On x86 Linux, all the socket calls go through an extra indirection,
// I think because the 5-register system call interface can't handle
// the 6-argument calls like sendto and recvfrom. Instead the
// arguments to the underlying system call are the number below
// and a pointer to an array of uintptr. We hide the pointer in the
// socketcall assembly to avoid allocation on every system call.

const (
	// see linux/net.h
	_SOCKET      = 1
	_BIND        = 2
	_CONNECT     = 3
	_LISTEN      = 4
	_ACCEPT      = 5
	_GETSOCKNAME = 6
	_GETPEERNAME = 7
	_SOCKETPAIR  = 8
	_SEND        = 9
	_RECV        = 10
	_SENDTO      = 11
	_RECVFROM    = 12
	_SHUTDOWN    = 13
	_SETSOCKOPT  = 14
	_GETSOCKOPT  = 15
	_SENDMSG     = 16
	_RECVMSG     = 17
	_ACCEPT4     = 18
	_RECVMMSG    = 19
	_SENDMMSG    = 20
)

func socketcall(call int, a0, a1, a2, a3, a4, a5 uintptr) (n int, err Errno)
func rawsocketcall(call int, a0, a1, a2, a3, a4, a5 uintptr) (n int, err Errno)

func accept4(s int, rsa *RawSockaddrAny, addrlen *_Socklen, flags int) (fd int, err error) {
	fd, e := socketcall(_ACCEPT4, uintptr(s), uintptr(unsafe.Pointer(rsa)), uintptr(unsafe.Pointer(addrlen)), uintptr(flags), 0, 0)
	if e != 0 {
		err = e
	}
	return
}

func getsockname(s int, rsa *RawSockaddrAny, addrlen *_Socklen) (err error) {
	_, e := rawsocketcall(_GETSOCKNAME, uintptr(s), uintptr(unsafe.Pointer(rsa)), uintptr(unsafe.Pointer(addrlen)), 0, 0, 0)
	if e != 0 {
		err = e
	}
	return
}

func getpeername(s int, rsa *RawSockaddrAny, addrlen *_Socklen) (err error) {
	_, e := rawsocketcall(_GETPEERNAME, uintptr(s), uintptr(unsafe.Pointer(rsa)), uintptr(unsafe.Pointer(addrlen)), 0, 0, 0)
	if e != 0 {
		err = e
	}
	return
}

func socketpair(domain int, typ int, flags int, fd *[2]int32) (err error) {
	_, e := rawsocketcall(_SOCKETPAIR, uintptr(domain), uintptr(typ), uintptr(flags), uintptr(unsafe.Pointer(fd)), 0, 0)
	if e != 0 {
		err = e
	}
	return
}

func bind(s int, addr unsafe.Pointer, addrlen _Socklen) (err error) {
	_, e := socketcall(_BIND, uintptr(s), uintptr(addr), uintptr(addrlen), 0, 0, 0)
	if e != 0 {
		err = e
	}
	return
}

func connect(s int, addr unsafe.Pointer, addrlen _Socklen) (err error) {
	_, e := socketcall(_CONNECT, uintptr(s), uintptr(addr), uintptr(addrlen), 0, 0, 0)
	if e != 0 {
		err = e
	}
	return
}

func socket(domain int, typ int, proto int) (fd int, err error) {
	fd, e := rawsocketcall(_SOCKET, uintptr(domain), uintptr(typ), uintptr(proto), 0, 0, 0)
	if e != 0 {
		err = e
	}
	return
}

func getsockopt(s int, level int, name int, val unsafe.Pointer, vallen *_Socklen) (err error) {
	_, e := socketcall(_GETSOCKOPT, uintptr(s), uintptr(level), uintptr(name), uintptr(val), uintptr(unsafe.Pointer(vallen)), 0)
	if e != 0 {
		err = e
	}
	return
}

func setsockopt(s int, level int, name int, val unsafe.Pointer, vallen uintptr) (err error) {
	_, e := socketcall(_SETSOCKOPT, uintptr(s), uintptr(level), uintptr(name), uintptr(val), vallen, 0)
	if e != 0 {
		err = e
	}
	return
}

func recvfrom(s int, p []byte, flags int, from *RawSockaddrAny, fromlen *_Socklen) (n int, err error) {
	var base uintptr
	if len(p) > 0 {
		base = uintptr(unsafe.Pointer(&p[0]))
	}
	n, e := socketcall(_RECVFROM, uintptr(s), base, uintptr(len(p)), uintptr(flags), uintptr(unsafe.Pointer(from)), uintptr(unsafe.Pointer(fromlen)))
	if e != 0 {
		err = e
	}
	return
}

func sendto(s int, p []byte, flags int, to unsafe.Pointer, addrlen _Socklen) (err error) {
	var base uintptr
	if len(p) > 0 {
		base = uintptr(unsafe.Pointer(&p[0]))
	}
	_, e := socketcall(_SENDTO, uintptr(s), base, uintptr(len(p)), uintptr(flags), uintptr(to), uintptr(addrlen))
	if e != 0 {
		err = e
	}
	return
}

func recvmsg(s int, msg *Msghdr, flags int) (n int, err error) {
	n, e := socketcall(_RECVMSG, uintptr(s), uintptr(unsafe.Pointer(msg)), uintptr(flags), 0, 0, 0)
	if e != 0 {
		err = e
	}
	return
}

func sendmsg(s int, msg *Msghdr, flags int) (n int, err error) {
	n, e := socketcall(_SENDMSG, uintptr(s), uintptr(unsafe.Pointer(msg)), uintptr(flags), 0, 0, 0)
	if e != 0 {
		err = e
	}
	return
}

func Listen(s int, n int) (err error) {
	_, e := socketcall(_LISTEN, uintptr(s), uintptr(n), 0, 0, 0, 0)
	if e != 0 {
		err = e
	}
	return
}

func Shutdown(s, how int) (err error) {
	_, e := socketcall(_SHUTDOWN, uintptr(s), uintptr(how), 0, 0, 0, 0)
	if e != 0 {
		err = e
	}
	return
}

func Fstatfs(fd int, buf *Statfs_t) (err error) {
	_, _, e := Syscall(SYS_FSTATFS64, uintptr(fd), unsafe.Sizeof(*buf), uintptr(unsafe.Pointer(buf)))
	if e != 0 {
		err = e
	}
	return
}

func Statfs(path string, buf *Statfs_t) (err error) {
	pathp, err := BytePtrFromString(path)
	if err != nil {
		return err
	}
	_, _, e := Syscall(SYS_STATFS64, uintptr(unsafe.Pointer(pathp)), unsafe.Sizeof(*buf), uintptr(unsafe.Pointer(buf)))
	if e != 0 {
		err = e
	}
	return
}

func (r *PtraceRegs) PC() uint64 { return uint64(uint32(r.Eip)) }

func (r *PtraceRegs) SetPC(pc uint64) { r.Eip = int32(pc) }

func (iov *Iovec) SetLen(length int) {
	iov.Len = uint32(length)
}

func (msghdr *Msghdr) SetControllen(length int) {
	msghdr.Controllen = uint32(length)
}

func (cmsg *Cmsghdr) SetLen(length int) {
	cmsg.Len = uint32(length)
}
