// Copyright 2009 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package syscall

import "unsafe"

func Getpagesize() int {
	return 4096
}

func TimespecToNsec(ts Timespec) int64 {
	return int64(ts.Sec)*1e9 + int64(ts.Nsec);
}

func NsecToTimespec(nsec int64) (ts Timespec) {
	ts.Sec = int32(nsec / 1e9);
	ts.Nsec = int32(nsec % 1e9);
	return;
}

func TimevalToNsec(tv Timeval) int64 {
	return int64(tv.Sec)*1e9 + int64(tv.Usec)*1e3;
}

func NsecToTimeval(nsec int64) (tv Timeval) {
	nsec += 999;	// round up to microsecond
	tv.Sec = int32(nsec/1e9);
	tv.Usec = int32(nsec%1e9 / 1e3);
	return;
}

// 64-bit file system and 32-bit uid calls
// (386 default is 32-bit file system and 16-bit uid).
//sys	Chown(path string, uid int, gid int) (errno int) = SYS_CHOWN32
//sys	Fchown(fd int, uid int, gid int) (errno int) = SYS_FCHOWN32
//sys	Fstat(fd int, stat *Stat_t) (errno int) = SYS_FSTAT64
//sys	Fstatfs(fd int, buf *Statfs_t) (errno int) = SYS_FSTATFS64
//sys	Getegid() (egid int) = SYS_GETEGID32
//sys	Geteuid() (euid int) = SYS_GETEUID32
//sys	Getgid() (gid int) = SYS_GETGID32
//sys	Getuid() (uid int) = SYS_GETUID32
//sys	Ioperm(from int, num int, on int) (errno int)
//sys	Iopl(level int) (errno int)
//sys	Lchown(path string, uid int, gid int) (errno int) = SYS_LCHOWN32
//sys	Lstat(path string, stat *Stat_t) (errno int) = SYS_LSTAT64
//sys	Setfsgid(gid int) (errno int) = SYS_SETFSGID32
//sys	Setfsuid(uid int) (errno int) = SYS_SETFSUID32
//sys	Setgid(gid int) (errno int) = SYS_SETGID32
//sys	Setregid(rgid int, egid int) (errno int) = SYS_SETREGID32
//sys	Setresgid(rgid int, egid int, sgid int) (errno int) = SYS_SETRESGID32
//sys	Setresuid(ruid int, euid int, suid int) (errno int) = SYS_SETRESUID32
//sys	Setreuid(ruid int, euid int) (errno int) = SYS_SETREUID32
//sys	Stat(path string, stat *Stat_t) (errno int) = SYS_STAT64
//sys	Statfs(path string, buf *Statfs_t) (errno int) = SYS_STATFS64
//sys	SyncFileRange(fd int, off int64, n int64, flags int) (errno int)
//sys	getgroups(n int, list *_Gid_t) (nn int, errno int) = SYS_GETGROUPS32
//sys	setgroups(n int, list *_Gid_t) (errno int) = SYS_SETGROUPS32

//sys	Select(nfd int, r *FdSet, w *FdSet, e *FdSet, timeout *Timeval) (n int, errno int) = SYS__NEWSELECT

// Underlying system call writes to newoffset via pointer.
// Implemented in assembly to avoid allocation.
func Seek(fd int, offset int64, whence int) (newoffset int64, errno int)

// On x86 Linux, all the socket calls go through an extra indirection,
// I think because the 5-register system call interface can't handle
// the 6-argument calls like sendto and recvfrom.  Instead the
// arguments to the underlying system call are the number below
// and a pointer to an array of uintptr.  We hide the pointer in the
// socketcall assembly to avoid allocation on every system call.

const (
	// see linux/net.h
	_SOCKET = 1;
	_BIND = 2;
	_CONNECT = 3;
	_LISTEN = 4;
	_ACCEPT = 5;
	_GETSOCKNAME = 6;
	_GETPEERNAME = 7;
	_SOCKETPAIR = 8;
	_SEND = 9;
	_RECV = 10;
	_SENDTO = 11;
	_RECVFROM = 12;
	_SHUTDOWN = 13;
	_SETSOCKOPT = 14;
	_GETSOCKOPT = 15;
	_SENDMSG = 16;
	_RECVMSG = 17;
)

func socketcall(call int, a0, a1, a2, a3, a4, a5 uintptr) (n int, errno int)

func accept(s int, rsa *RawSockaddrAny, addrlen *_Socklen) (fd int, errno int) {
	fd, errno = socketcall(_ACCEPT, uintptr(s), uintptr(unsafe.Pointer(rsa)), uintptr(unsafe.Pointer(addrlen)), 0, 0, 0);
	return;
}

func getsockname(s int, rsa *RawSockaddrAny, addrlen *_Socklen) (errno int) {
	var _ int;
	_, errno = socketcall(_GETSOCKNAME, uintptr(s), uintptr(unsafe.Pointer(rsa)), uintptr(unsafe.Pointer(addrlen)), 0, 0, 0);
	return;
}

func getpeername(s int, rsa *RawSockaddrAny, addrlen *_Socklen) (errno int) {
	var _ int;
	_, errno = socketcall(_GETPEERNAME, uintptr(s), uintptr(unsafe.Pointer(rsa)), uintptr(unsafe.Pointer(addrlen)), 0, 0, 0);
	return;
}

func bind(s int, addr uintptr, addrlen _Socklen) (errno int) {
	var _ int;
	_, errno = socketcall(_BIND, uintptr(s), uintptr(addr), uintptr(addrlen), 0, 0, 0);
	return;
}

func connect(s int, addr uintptr, addrlen _Socklen) (errno int) {
	var _ int;
	_, errno = socketcall(_CONNECT, uintptr(s), uintptr(addr), uintptr(addrlen), 0, 0, 0);
	return;
}

func socket(domain int, typ int, proto int) (fd int, errno int) {
	fd, errno = socketcall(_SOCKET, uintptr(domain), uintptr(typ), uintptr(proto), 0, 0, 0);
	return;
}

func setsockopt(s int, level int, name int, val uintptr, vallen int) (errno int) {
	var _ int;
	_, errno = socketcall(_SETSOCKOPT, uintptr(s), uintptr(level), uintptr(name), uintptr(val), uintptr(vallen), 0);
	return;
}

func Listen(s int, n int) (errno int) {
	var _ int;
	_, errno = socketcall(_LISTEN, uintptr(s), uintptr(n), 0, 0, 0, 0);
	return;
}

func (r *PtraceRegs) PC() uint64 {
	return uint64(uint32(r.Eip));
}

func (r *PtraceRegs) SetPC(pc uint64) {
	r.Eip = int32(pc);
}
