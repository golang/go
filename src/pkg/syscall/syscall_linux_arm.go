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
// (Ftruncate and truncate are not; go figure.)

func Pread(fd int, p []byte, offset int64) (n int, errno int) {
	var _p0 unsafe.Pointer
	if len(p) > 0 {
		_p0 = unsafe.Pointer(&p[0])
	}
	r0, _, e1 := Syscall6(SYS_PREAD64, uintptr(fd), uintptr(_p0), uintptr(len(p)), 0, uintptr(offset), uintptr(offset>>32))
	n = int(r0)
	errno = int(e1)
	return
}

func Pwrite(fd int, p []byte, offset int64) (n int, errno int) {
	var _p0 unsafe.Pointer
	if len(p) > 0 {
		_p0 = unsafe.Pointer(&p[0])
	}
	r0, _, e1 := Syscall6(SYS_PWRITE64, uintptr(fd), uintptr(_p0), uintptr(len(p)), 0, uintptr(offset), uintptr(offset>>32))
	n = int(r0)
	errno = int(e1)
	return
}

// Seek is defined in assembly.

func Seek(fd int, offset int64, whence int) (newoffset int64, errno int)

//sys	accept(s int, rsa *RawSockaddrAny, addrlen *_Socklen) (fd int, errno int)
//sys	bind(s int, addr uintptr, addrlen _Socklen) (errno int)
//sys	connect(s int, addr uintptr, addrlen _Socklen) (errno int)
//sysnb	getgroups(n int, list *_Gid_t) (nn int, errno int) = SYS_GETGROUPS32
//sysnb	setgroups(n int, list *_Gid_t) (errno int) = SYS_SETGROUPS32
//sys	setsockopt(s int, level int, name int, val uintptr, vallen int) (errno int)
//sysnb	socket(domain int, typ int, proto int) (fd int, errno int)
//sysnb	getpeername(fd int, rsa *RawSockaddrAny, addrlen *_Socklen) (errno int)
//sysnb	getsockname(fd int, rsa *RawSockaddrAny, addrlen *_Socklen) (errno int)
//sys	recvfrom(fd int, p []byte, flags int, from *RawSockaddrAny, fromlen *_Socklen) (n int, errno int)
//sys	sendto(s int, buf []byte, flags int, to uintptr, addrlen _Socklen) (errno int)
//sysnb	socketpair(domain int, typ int, flags int, fd *[2]int) (errno int)
//sys	recvmsg(s int, msg *Msghdr, flags int) (n int, errno int)
//sys	sendmsg(s int, msg *Msghdr, flags int) (errno int)

//sys	Chown(path string, uid int, gid int) (errno int)
//sys	Fchown(fd int, uid int, gid int) (errno int)
//sys	Fstat(fd int, stat *Stat_t) (errno int) = SYS_FSTAT64
//sys	Fstatfs(fd int, buf *Statfs_t) (errno int) = SYS_FSTATFS64
//sys	Ftruncate(fd int, length int64) (errno int) = SYS_FTRUNCATE64
//sysnb	Getegid() (egid int)
//sysnb	Geteuid() (euid int)
//sysnb	Getgid() (gid int)
//sysnb	Getuid() (uid int)
//sys	Lchown(path string, uid int, gid int) (errno int)
//sys	Listen(s int, n int) (errno int)
//sys	Lstat(path string, stat *Stat_t) (errno int) = SYS_LSTAT64
//sys	Select(nfd int, r *FdSet, w *FdSet, e *FdSet, timeout *Timeval) (n int, errno int) = SYS__NEWSELECT
//sys	Setfsgid(gid int) (errno int)
//sys	Setfsuid(uid int) (errno int)
//sysnb	Setgid(gid int) (errno int)
//sysnb	Setregid(rgid int, egid int) (errno int)
//sysnb	Setresgid(rgid int, egid int, sgid int) (errno int)
//sysnb	Setresuid(ruid int, euid int, suid int) (errno int)
//sysnb	Setreuid(ruid int, euid int) (errno int)
//sys	Shutdown(fd int, how int) (errno int)
//sys	Splice(rfd int, roff *int64, wfd int, woff *int64, len int, flags int) (n int, errno int)
//sys	Stat(path string, stat *Stat_t) (errno int) = SYS_STAT64
//sys	Statfs(path string, buf *Statfs_t) (errno int) = SYS_STATFS64
//sys	Truncate(path string, length int64) (errno int) = SYS_TRUNCATE64

// Vsyscalls on amd64.
//sysnb	Gettimeofday(tv *Timeval) (errno int)
//sysnb	Time(t *Time_t) (tt Time_t, errno int)

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
