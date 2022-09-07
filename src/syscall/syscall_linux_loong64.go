// Copyright 2022 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package syscall

import "unsafe"

const (
	_SYS_setgroups  = SYS_SETGROUPS
	_SYS_faccessat2 = 439
)

//sys	EpollWait(epfd int, events []EpollEvent, msec int) (n int, err error) = SYS_EPOLL_PWAIT
//sys	Fchown(fd int, uid int, gid int) (err error)
//sys	Fstatfs(fd int, buf *Statfs_t) (err error)
//sys	Ftruncate(fd int, length int64) (err error)
//sysnb	Getegid() (egid int)
//sysnb	Geteuid() (euid int)
//sysnb	Getgid() (gid int)
//sysnb	Getuid() (uid int)
//sys	Listen(s int, n int) (err error)
//sys	pread(fd int, p []byte, offset int64) (n int, err error) = SYS_PREAD64
//sys	pwrite(fd int, p []byte, offset int64) (n int, err error) = SYS_PWRITE64
//sys	Renameat(olddirfd int, oldpath string, newdirfd int, newpath string) (err error) = SYS_RENAMEAT2
//sys	Seek(fd int, offset int64, whence int) (off int64, err error) = SYS_LSEEK
//sys	sendfile(outfd int, infd int, offset *int64, count int) (written int, err error)
//sys	Setfsgid(gid int) (err error)
//sys	Setfsuid(uid int) (err error)
//sys	Shutdown(fd int, how int) (err error)
//sys	Splice(rfd int, roff *int64, wfd int, woff *int64, len int, flags int) (n int64, err error)
//sys	statx(dirfd int, path string, flags int, mask int, stat *statx_t) (err error)

// makedev makes C dev_t from major and minor numbers the glibc way:
// 0xMMMM_MMMM 0xmmmm_mmmm -> 0xMMMM_Mmmm_mmmM_MMmm
func makedev(major uint32, minor uint32) uint64 {
	majorH := uint64(major >> 12)
	majorL := uint64(major & 0xfff)
	minorH := uint64(minor >> 8)
	minorL := uint64(minor & 0xff)
	return (majorH << 44) | (minorH << 20) | (majorL << 8) | minorL
}

func timespecFromStatxTimestamp(x statxTimestamp) Timespec {
	return Timespec{
		Sec:  x.Sec,
		Nsec: int64(x.Nsec),
	}
}

func fstatat(dirfd int, path string, stat *Stat_t, flags int) (err error) {
	var r statx_t
	// Do it the glibc way, add AT_NO_AUTOMOUNT.
	if err = statx(dirfd, path, _AT_NO_AUTOMOUNT|flags, _STATX_BASIC_STATS, &r); err != nil {
		return err
	}

	stat.Dev = makedev(r.Dev_major, r.Dev_minor)
	stat.Ino = r.Ino
	stat.Mode = uint32(r.Mode)
	stat.Nlink = r.Nlink
	stat.Uid = r.Uid
	stat.Gid = r.Gid
	stat.Rdev = makedev(r.Rdev_major, r.Rdev_minor)
	// hope we don't get to process files so large to overflow these size
	// fields...
	stat.Size = int64(r.Size)
	stat.Blksize = int32(r.Blksize)
	stat.Blocks = int64(r.Blocks)
	stat.Atim = timespecFromStatxTimestamp(r.Atime)
	stat.Mtim = timespecFromStatxTimestamp(r.Mtime)
	stat.Ctim = timespecFromStatxTimestamp(r.Ctime)

	return nil
}

func Fstatat(fd int, path string, stat *Stat_t, flags int) (err error) {
	return fstatat(fd, path, stat, flags)
}

func Fstat(fd int, stat *Stat_t) (err error) {
	return fstatat(fd, "", stat, _AT_EMPTY_PATH)
}

func Stat(path string, stat *Stat_t) (err error) {
	return fstatat(_AT_FDCWD, path, stat, 0)
}

func Lchown(path string, uid int, gid int) (err error) {
	return Fchownat(_AT_FDCWD, path, uid, gid, _AT_SYMLINK_NOFOLLOW)
}

func Lstat(path string, stat *Stat_t) (err error) {
	return fstatat(_AT_FDCWD, path, stat, _AT_SYMLINK_NOFOLLOW)
}

//sys	Statfs(path string, buf *Statfs_t) (err error)
//sys	SyncFileRange(fd int, off int64, n int64, flags int) (err error)
//sys	Truncate(path string, length int64) (err error)
//sys	accept4(s int, rsa *RawSockaddrAny, addrlen *_Socklen, flags int) (fd int, err error)
//sys	bind(s int, addr unsafe.Pointer, addrlen _Socklen) (err error)
//sys	connect(s int, addr unsafe.Pointer, addrlen _Socklen) (err error)
//sysnb	getgroups(n int, list *_Gid_t) (nn int, err error)
//sys	getsockopt(s int, level int, name int, val unsafe.Pointer, vallen *_Socklen) (err error)
//sys	setsockopt(s int, level int, name int, val unsafe.Pointer, vallen uintptr) (err error)
//sysnb	socket(domain int, typ int, proto int) (fd int, err error)
//sysnb	socketpair(domain int, typ int, proto int, fd *[2]int32) (err error)
//sysnb	getpeername(fd int, rsa *RawSockaddrAny, addrlen *_Socklen) (err error)
//sysnb	getsockname(fd int, rsa *RawSockaddrAny, addrlen *_Socklen) (err error)
//sys	recvfrom(fd int, p []byte, flags int, from *RawSockaddrAny, fromlen *_Socklen) (n int, err error)
//sys	sendto(s int, buf []byte, flags int, to unsafe.Pointer, addrlen _Socklen) (err error)
//sys	recvmsg(s int, msg *Msghdr, flags int) (n int, err error)
//sys	sendmsg(s int, msg *Msghdr, flags int) (n int, err error)
//sys	mmap(addr uintptr, length uintptr, prot int, flags int, fd int, offset int64) (xaddr uintptr, err error)

type sigset_t struct {
	X__val [16]uint64
}

//sys	pselect(nfd int, r *FdSet, w *FdSet, e *FdSet, timeout *Timespec, sigmask *sigset_t) (n int, err error) = SYS_PSELECT6

func Select(nfd int, r *FdSet, w *FdSet, e *FdSet, timeout *Timeval) (n int, err error) {
	var ts *Timespec
	if timeout != nil {
		ts = &Timespec{Sec: timeout.Sec, Nsec: timeout.Usec * 1000}
	}
	return pselect(nfd, r, w, e, ts, nil)
}

//sysnb	Gettimeofday(tv *Timeval) (err error)

func setTimespec(sec, nsec int64) Timespec {
	return Timespec{Sec: sec, Nsec: nsec}
}

func setTimeval(sec, usec int64) Timeval {
	return Timeval{Sec: sec, Usec: usec}
}

func futimesat(dirfd int, path string, tv *[2]Timeval) (err error) {
	if tv == nil {
		return utimensat(dirfd, path, nil, 0)
	}

	ts := []Timespec{
		NsecToTimespec(TimevalToNsec(tv[0])),
		NsecToTimespec(TimevalToNsec(tv[1])),
	}
	return utimensat(dirfd, path, (*[2]Timespec)(unsafe.Pointer(&ts[0])), 0)
}

func Time(t *Time_t) (Time_t, error) {
	var tv Timeval
	err := Gettimeofday(&tv)
	if err != nil {
		return 0, err
	}
	if t != nil {
		*t = Time_t(tv.Sec)
	}
	return Time_t(tv.Sec), nil
}

func Utime(path string, buf *Utimbuf) error {
	tv := []Timeval{
		{Sec: buf.Actime},
		{Sec: buf.Modtime},
	}
	return Utimes(path, tv)
}

func utimes(path string, tv *[2]Timeval) (err error) {
	if tv == nil {
		return utimensat(_AT_FDCWD, path, nil, 0)
	}

	ts := []Timespec{
		NsecToTimespec(TimevalToNsec(tv[0])),
		NsecToTimespec(TimevalToNsec(tv[1])),
	}
	return utimensat(_AT_FDCWD, path, (*[2]Timespec)(unsafe.Pointer(&ts[0])), 0)
}

// Getrlimit prefers the prlimit64 system call.
func Getrlimit(resource int, rlim *Rlimit) error {
	return prlimit(0, resource, nil, rlim)
}

// Setrlimit prefers the prlimit64 system call.
func Setrlimit(resource int, rlim *Rlimit) error {
	return prlimit(0, resource, rlim, nil)
}

func (r *PtraceRegs) GetEra() uint64 { return r.Era }

func (r *PtraceRegs) SetEra(era uint64) { r.Era = era }

func (iov *Iovec) SetLen(length int) {
	iov.Len = uint64(length)
}

func (msghdr *Msghdr) SetControllen(length int) {
	msghdr.Controllen = uint64(length)
}

func (cmsg *Cmsghdr) SetLen(length int) {
	cmsg.Len = uint64(length)
}

func InotifyInit() (fd int, err error) {
	return InotifyInit1(0)
}

//sys	ppoll(fds *pollFd, nfds int, timeout *Timespec, sigmask *sigset_t) (n int, err error)

func Pause() error {
	_, err := ppoll(nil, 0, nil, nil)
	return err
}

func rawVforkSyscall(trap, a1 uintptr) (r1 uintptr, err Errno)
