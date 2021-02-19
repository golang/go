// Copyright 2015 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// +build linux
// +build mips64 mips64le

package unix

//sys	dup2(oldfd int, newfd int) (err error)
//sysnb	EpollCreate(size int) (fd int, err error)
//sys	EpollWait(epfd int, events []EpollEvent, msec int) (n int, err error)
//sys	Fadvise(fd int, offset int64, length int64, advice int) (err error) = SYS_FADVISE64
//sys	Fchown(fd int, uid int, gid int) (err error)
//sys	Fstatfs(fd int, buf *Statfs_t) (err error)
//sys	Ftruncate(fd int, length int64) (err error)
//sysnb	Getegid() (egid int)
//sysnb	Geteuid() (euid int)
//sysnb	Getgid() (gid int)
//sysnb	Getrlimit(resource int, rlim *Rlimit) (err error)
//sysnb	Getuid() (uid int)
//sys	Lchown(path string, uid int, gid int) (err error)
//sys	Listen(s int, n int) (err error)
//sys	Pause() (err error)
//sys	Pread(fd int, p []byte, offset int64) (n int, err error) = SYS_PREAD64
//sys	Pwrite(fd int, p []byte, offset int64) (n int, err error) = SYS_PWRITE64
//sys	Renameat(olddirfd int, oldpath string, newdirfd int, newpath string) (err error)
//sys	Seek(fd int, offset int64, whence int) (off int64, err error) = SYS_LSEEK

func Select(nfd int, r *FdSet, w *FdSet, e *FdSet, timeout *Timeval) (n int, err error) {
	var ts *Timespec
	if timeout != nil {
		ts = &Timespec{Sec: timeout.Sec, Nsec: timeout.Usec * 1000}
	}
	return Pselect(nfd, r, w, e, ts, nil)
}

//sys	sendfile(outfd int, infd int, offset *int64, count int) (written int, err error)
//sys	setfsgid(gid int) (prev int, err error)
//sys	setfsuid(uid int) (prev int, err error)
//sysnb	Setregid(rgid int, egid int) (err error)
//sysnb	Setresgid(rgid int, egid int, sgid int) (err error)
//sysnb	Setresuid(ruid int, euid int, suid int) (err error)
//sysnb	Setrlimit(resource int, rlim *Rlimit) (err error)
//sysnb	Setreuid(ruid int, euid int) (err error)
//sys	Shutdown(fd int, how int) (err error)
//sys	Splice(rfd int, roff *int64, wfd int, woff *int64, len int, flags int) (n int64, err error)
//sys	Statfs(path string, buf *Statfs_t) (err error)
//sys	SyncFileRange(fd int, off int64, n int64, flags int) (err error)
//sys	Truncate(path string, length int64) (err error)
//sys	Ustat(dev int, ubuf *Ustat_t) (err error)
//sys	accept(s int, rsa *RawSockaddrAny, addrlen *_Socklen) (fd int, err error)
//sys	accept4(s int, rsa *RawSockaddrAny, addrlen *_Socklen, flags int) (fd int, err error)
//sys	bind(s int, addr unsafe.Pointer, addrlen _Socklen) (err error)
//sys	connect(s int, addr unsafe.Pointer, addrlen _Socklen) (err error)
//sysnb	getgroups(n int, list *_Gid_t) (nn int, err error)
//sysnb	setgroups(n int, list *_Gid_t) (err error)
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

func Pipe(p []int) (err error) {
	if len(p) != 2 {
		return EINVAL
	}
	var pp [2]_C_int
	err = pipe2(&pp, 0)
	p[0] = int(pp[0])
	p[1] = int(pp[1])
	return
}

//sysnb	pipe2(p *[2]_C_int, flags int) (err error)

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

type stat_t struct {
	Dev        uint32
	Pad0       [3]int32
	Ino        uint64
	Mode       uint32
	Nlink      uint32
	Uid        uint32
	Gid        uint32
	Rdev       uint32
	Pad1       [3]uint32
	Size       int64
	Atime      uint32
	Atime_nsec uint32
	Mtime      uint32
	Mtime_nsec uint32
	Ctime      uint32
	Ctime_nsec uint32
	Blksize    uint32
	Pad2       uint32
	Blocks     int64
}

//sys	fstat(fd int, st *stat_t) (err error)
//sys	fstatat(dirfd int, path string, st *stat_t, flags int) (err error) = SYS_NEWFSTATAT
//sys	lstat(path string, st *stat_t) (err error)
//sys	stat(path string, st *stat_t) (err error)

func Fstat(fd int, s *Stat_t) (err error) {
	st := &stat_t{}
	err = fstat(fd, st)
	fillStat_t(s, st)
	return
}

func Fstatat(dirfd int, path string, s *Stat_t, flags int) (err error) {
	st := &stat_t{}
	err = fstatat(dirfd, path, st, flags)
	fillStat_t(s, st)
	return
}

func Lstat(path string, s *Stat_t) (err error) {
	st := &stat_t{}
	err = lstat(path, st)
	fillStat_t(s, st)
	return
}

func Stat(path string, s *Stat_t) (err error) {
	st := &stat_t{}
	err = stat(path, st)
	fillStat_t(s, st)
	return
}

func fillStat_t(s *Stat_t, st *stat_t) {
	s.Dev = st.Dev
	s.Ino = st.Ino
	s.Mode = st.Mode
	s.Nlink = st.Nlink
	s.Uid = st.Uid
	s.Gid = st.Gid
	s.Rdev = st.Rdev
	s.Size = st.Size
	s.Atim = Timespec{int64(st.Atime), int64(st.Atime_nsec)}
	s.Mtim = Timespec{int64(st.Mtime), int64(st.Mtime_nsec)}
	s.Ctim = Timespec{int64(st.Ctime), int64(st.Ctime_nsec)}
	s.Blksize = st.Blksize
	s.Blocks = st.Blocks
}

func (r *PtraceRegs) PC() uint64 { return r.Epc }

func (r *PtraceRegs) SetPC(pc uint64) { r.Epc = pc }

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

func InotifyInit() (fd int, err error) {
	return InotifyInit1(0)
}

//sys	poll(fds *PollFd, nfds int, timeout int) (n int, err error)

func Poll(fds []PollFd, timeout int) (n int, err error) {
	if len(fds) == 0 {
		return poll(nil, 0, timeout)
	}
	return poll(&fds[0], len(fds), timeout)
}
