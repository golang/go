// Copyright 2009 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package syscall

import "syscall"

//sys	Chown(path string, uid int, gid int) (errno int)
//sys	Fchown(fd int, uid int, gid int) (errno int)
//sys	Fstat(fd int, stat *Stat_t) (errno int)
//sys	Fstatfs(fd int, buf *Statfs_t) (errno int)
//sys	Getegid() (egid int)
//sys	Geteuid() (euid int)
//sys	Getgid() (gid int)
//sys	Getuid() (uid int)
//sys	Lchown(path string, uid int, gid int) (errno int)
//sys	Listen(s int, n int) (errno int)
//sys	Lstat(path string, stat *Stat_t) (errno int)
//sys	Seek(fd int, offset int64, whence int) (off int64, errno int) = SYS_LSEEK
//sys	Select(nfd int, r *FdSet, w *FdSet, e *FdSet, timeout *Timeval) (n int, errno int)
//sys	Setfsgid(gid int) (errno int)
//sys	Setfsuid(uid int) (errno int)
//sys	Setgid(gid int) (errno int)
//sys	Setregid(rgid int, egid int) (errno int)
//sys	Setresgid(rgid int, egid int, sgid int) (errno int)
//sys	Setresuid(ruid int, euid int, suid int) (errno int)
//sys	Setreuid(ruid int, euid int) (errno int)
//sys	Shutdown(fd int, how int) (errno int)
//sys	Stat(path string, stat *Stat_t) (errno int)
//sys	Statfs(path string, buf *Statfs_t) (errno int)
//sys	accept(s int, rsa *RawSockaddrAny, addrlen *_Socklen) (fd int, errno int)
//sys	bind(s int, addr uintptr, addrlen _Socklen) (errno int)
//sys	connect(s int, addr uintptr, addrlen _Socklen) (errno int)
//sys	getgroups(n int, list *_Gid_t) (nn int, errno int)
//sys	setgroups(n int, list *_Gid_t) (errno int)
//sys	setsockopt(s int, level int, name int, val uintptr, vallen int) (errno int)
//sys	socket(domain int, typ int, proto int) (fd int, errno int)
//sys	getpeername(fd int, rsa *RawSockaddrAny, addrlen *_Socklen) (errno int)
//sys	getsockname(fd int, rsa *RawSockaddrAny, addrlen *_Socklen) (errno int)

func Getpagesize() int {
	return 4096
}

func TimespecToNsec(ts Timespec) int64 {
	return int64(ts.Sec)*1e9 + int64(ts.Nsec);
}

func NsecToTimespec(nsec int64) (ts Timespec) {
	ts.Sec = nsec / 1e9;
	ts.Nsec = nsec % 1e9;
	return;
}

func TimevalToNsec(tv Timeval) int64 {
	return int64(tv.Sec)*1e9 + int64(tv.Usec)*1e3;
}

func NsecToTimeval(nsec int64) (tv Timeval) {
	nsec += 999;	// round up to microsecond
	tv.Sec = nsec/1e9;
	tv.Usec = nsec%1e9 / 1e3;
	return;
}

