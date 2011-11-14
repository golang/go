// Copyright 2009 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package syscall

import "unsafe"

func Getpagesize() int { return 4096 }

func TimespecToNsec(ts Timespec) int64 { return int64(ts.Sec)*1e9 + int64(ts.Nsec) }

func NsecToTimespec(nsec int64) (ts Timespec) {
	ts.Sec = int32(nsec / 1e9)
	ts.Nsec = int32(nsec % 1e9)
	return
}

func NsecToTimeval(nsec int64) (tv Timeval) {
	nsec += 999 // round up to microsecond
	tv.Sec = int32(nsec / 1e9)
	tv.Usec = int32(nsec % 1e9 / 1e3)
	return
}

// Pread and Pwrite are special: they insert padding before the int64.

func Pread(fd int, p []byte, offset int64) (n int, err error) {
	var _p0 unsafe.Pointer
	if len(p) > 0 {
		_p0 = unsafe.Pointer(&p[0])
	}
	r0, _, e1 := Syscall6(SYS_PREAD64, uintptr(fd), uintptr(_p0), uintptr(len(p)), 0, uintptr(offset), uintptr(offset>>32))
	n = int(r0)
	errno = int(e1)
	return
}

func Pwrite(fd int, p []byte, offset int64) (n int, err error) {
	var _p0 unsafe.Pointer
	if len(p) > 0 {
		_p0 = unsafe.Pointer(&p[0])
	}
	r0, _, e1 := Syscall6(SYS_PWRITE64, uintptr(fd), uintptr(_p0), uintptr(len(p)), 0, uintptr(offset), uintptr(offset>>32))
	n = int(r0)
	errno = int(e1)
	return
}

func Ftruncate(fd int, length int64) (err error) {
	// ARM EABI requires 64-bit arguments should be put in a pair
	// of registers from an even register number.
	_, _, e1 := Syscall6(SYS_FTRUNCATE64, uintptr(fd), 0, uintptr(length), uintptr(length>>32), 0, 0)
	errno = int(e1)
	return
}

func Truncate(path string, length int64) (err error) {
	_, _, e1 := Syscall6(SYS_TRUNCATE64, uintptr(unsafe.Pointer(StringBytePtr(path))), 0, uintptr(length), uintptr(length>>32), 0, 0)
	errno = int(e1)
	return
}

// Seek is defined in assembly.

func Seek(fd int, offset int64, whence int) (newoffset int64, err error)

//sys	accept(s int, rsa *RawSockaddrAny, addrlen *_Socklen) (fd int, err error)
//sys	bind(s int, addr uintptr, addrlen _Socklen) (err error)
//sys	connect(s int, addr uintptr, addrlen _Socklen) (err error)
//sysnb	getgroups(n int, list *_Gid_t) (nn int, err error) = SYS_GETGROUPS32
//sysnb	setgroups(n int, list *_Gid_t) (err error) = SYS_SETGROUPS32
//sys	getsockopt(s int, level int, name int, val uintptr, vallen *_Socklen) (err error)
//sys	setsockopt(s int, level int, name int, val uintptr, vallen uintptr) (err error)
//sysnb	socket(domain int, typ int, proto int) (fd int, err error)
//sysnb	getpeername(fd int, rsa *RawSockaddrAny, addrlen *_Socklen) (err error)
//sysnb	getsockname(fd int, rsa *RawSockaddrAny, addrlen *_Socklen) (err error)
//sys	recvfrom(fd int, p []byte, flags int, from *RawSockaddrAny, fromlen *_Socklen) (n int, err error)
//sys	sendto(s int, buf []byte, flags int, to uintptr, addrlen _Socklen) (err error)
//sysnb	socketpair(domain int, typ int, flags int, fd *[2]int) (err error)
//sys	recvmsg(s int, msg *Msghdr, flags int) (n int, err error)
//sys	sendmsg(s int, msg *Msghdr, flags int) (err error)

//sys	Chown(path string, uid int, gid int) (err error)
//sys	Fchown(fd int, uid int, gid int) (err error)
//sys	Fstat(fd int, stat *Stat_t) (err error) = SYS_FSTAT64
//sys	Fstatfs(fd int, buf *Statfs_t) (err error) = SYS_FSTATFS64
//sysnb	Getegid() (egid int)
//sysnb	Geteuid() (euid int)
//sysnb	Getgid() (gid int)
//sysnb	Getuid() (uid int)
//sys	Lchown(path string, uid int, gid int) (err error)
//sys	Listen(s int, n int) (err error)
//sys	Lstat(path string, stat *Stat_t) (err error) = SYS_LSTAT64
//sys	Sendfile(outfd int, infd int, offset *int64, count int) (written int, err error) = SYS_SENDFILE64
//sys	Select(nfd int, r *FdSet, w *FdSet, e *FdSet, timeout *Timeval) (n int, err error) = SYS__NEWSELECT
//sys	Setfsgid(gid int) (err error)
//sys	Setfsuid(uid int) (err error)
//sysnb	Setgid(gid int) (err error)
//sysnb	Setregid(rgid int, egid int) (err error)
//sysnb	Setresgid(rgid int, egid int, sgid int) (err error)
//sysnb	Setresuid(ruid int, euid int, suid int) (err error)
//sysnb	Setreuid(ruid int, euid int) (err error)
//sys	Shutdown(fd int, how int) (err error)
//sys	Splice(rfd int, roff *int64, wfd int, woff *int64, len int, flags int) (n int, err error)
//sys	Stat(path string, stat *Stat_t) (err error) = SYS_STAT64
//sys	Statfs(path string, buf *Statfs_t) (err error) = SYS_STATFS64

// Vsyscalls on amd64.
//sysnb	Gettimeofday(tv *Timeval) (err error)
//sysnb	Time(t *Time_t) (tt Time_t, err error)

//sys	mmap2(addr uintptr, length uintptr, prot int, flags int, fd int, pageOffset uintptr) (xaddr uintptr, err error)

func mmap(addr uintptr, length uintptr, prot int, flags int, fd int, offset int64) (xaddr uintptr, err error) {
	page := uintptr(offset / 4096)
	if offset != int64(page)*4096 {
		return 0, EINVAL
	}
	return mmap2(addr, length, prot, flags, fd, page)
}

// TODO(kaib): add support for tracing
func (r *PtraceRegs) PC() uint64 { return 0 }

func (r *PtraceRegs) SetPC(pc uint64) {}

func (iov *Iovec) SetLen(length int) {
	iov.Len = uint32(length)
}

func (msghdr *Msghdr) SetControllen(length int) {
	msghdr.Controllen = uint32(length)
}

func (cmsg *Cmsghdr) SetLen(length int) {
	cmsg.Len = uint32(length)
}
