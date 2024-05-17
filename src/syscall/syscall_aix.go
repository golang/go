// Copyright 2018 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// Aix system calls.
// This file is compiled as ordinary Go code,
// but it is also input to mksyscall,
// which parses the //sys lines and generates system call stubs.
// Note that sometimes we use a lowercase //sys name and
// wrap it in our own nicer implementation.

package syscall

import (
	"internal/byteorder"
	"unsafe"
)

func Syscall(trap, a1, a2, a3 uintptr) (r1, r2 uintptr, err Errno)
func Syscall6(trap, a1, a2, a3, a4, a5, a6 uintptr) (r1, r2 uintptr, err Errno)
func RawSyscall(trap, a1, a2, a3 uintptr) (r1, r2 uintptr, err Errno)
func RawSyscall6(trap, a1, a2, a3, a4, a5, a6 uintptr) (r1, r2 uintptr, err Errno)

// Implemented in runtime/syscall_aix.go.
func rawSyscall6(trap, nargs, a1, a2, a3, a4, a5, a6 uintptr) (r1, r2 uintptr, err Errno)
func syscall6(trap, nargs, a1, a2, a3, a4, a5, a6 uintptr) (r1, r2 uintptr, err Errno)

// Constant expected by package but not supported
const (
	_ = iota
	TIOCSCTTY
	SYS_EXECVE
	SYS_FCNTL
)

const (
	F_DUPFD_CLOEXEC = 0
	// AF_LOCAL doesn't exist on AIX
	AF_LOCAL = AF_UNIX

	_F_DUP2FD_CLOEXEC = 0
)

func (ts *StTimespec_t) Unix() (sec int64, nsec int64) {
	return int64(ts.Sec), int64(ts.Nsec)
}

func (ts *StTimespec_t) Nano() int64 {
	return int64(ts.Sec)*1e9 + int64(ts.Nsec)
}

/*
 * Wrapped
 */

func Access(path string, mode uint32) (err error) {
	return Faccessat(_AT_FDCWD, path, mode, 0)
}

// fcntl must never be called with cmd=F_DUP2FD because it doesn't work on AIX
// There is no way to create a custom fcntl and to keep //sys fcntl easily,
// because we need fcntl name for its libc symbol. This is linked with the script.
// But, as fcntl is currently not exported and isn't called with F_DUP2FD,
// it doesn't matter.
//sys	fcntl(fd int, cmd int, arg int) (val int, err error)
//sys	Dup2(old int, new int) (err error)

//sysnb pipe(p *[2]_C_int) (err error)

func Pipe(p []int) (err error) {
	if len(p) != 2 {
		return EINVAL
	}
	var pp [2]_C_int
	err = pipe(&pp)
	if err == nil {
		p[0] = int(pp[0])
		p[1] = int(pp[1])
	}
	return
}

//sys	readlink(path string, buf []byte, bufSize uint64) (n int, err error)

func Readlink(path string, buf []byte) (n int, err error) {
	s := uint64(len(buf))
	return readlink(path, buf, s)
}

//sys	utimes(path string, times *[2]Timeval) (err error)

func Utimes(path string, tv []Timeval) error {
	if len(tv) != 2 {
		return EINVAL
	}
	return utimes(path, (*[2]Timeval)(unsafe.Pointer(&tv[0])))
}

//sys	utimensat(dirfd int, path string, times *[2]Timespec, flag int) (err error)

func UtimesNano(path string, ts []Timespec) error {
	if len(ts) != 2 {
		return EINVAL
	}
	return utimensat(_AT_FDCWD, path, (*[2]Timespec)(unsafe.Pointer(&ts[0])), 0)
}

//sys	unlinkat(dirfd int, path string, flags int) (err error)

func Unlinkat(dirfd int, path string) (err error) {
	return unlinkat(dirfd, path, 0)
}

//sys	getcwd(buf *byte, size uint64) (err error)

const ImplementsGetwd = true

func Getwd() (ret string, err error) {
	for len := uint64(4096); ; len *= 2 {
		b := make([]byte, len)
		err := getcwd(&b[0], len)
		if err == nil {
			i := 0
			for b[i] != 0 {
				i++
			}
			return string(b[0:i]), nil
		}
		if err != ERANGE {
			return "", err
		}
	}
}

func Getcwd(buf []byte) (n int, err error) {
	err = getcwd(&buf[0], uint64(len(buf)))
	if err == nil {
		i := 0
		for buf[i] != 0 {
			i++
		}
		n = i + 1
	}
	return
}

//sysnb	getgroups(ngid int, gid *_Gid_t) (n int, err error)
//sysnb	setgroups(ngid int, gid *_Gid_t) (err error)

func Getgroups() (gids []int, err error) {
	n, err := getgroups(0, nil)
	if err != nil {
		return nil, err
	}
	if n == 0 {
		return nil, nil
	}

	// Sanity check group count. Max is 16 on BSD.
	if n < 0 || n > 1000 {
		return nil, EINVAL
	}

	a := make([]_Gid_t, n)
	n, err = getgroups(n, &a[0])
	if err != nil {
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

func direntIno(buf []byte) (uint64, bool) {
	return byteorder.ReadInt(buf, unsafe.Offsetof(Dirent{}.Ino), unsafe.Sizeof(Dirent{}.Ino))
}

func direntReclen(buf []byte) (uint64, bool) {
	return byteorder.ReadInt(buf, unsafe.Offsetof(Dirent{}.Reclen), unsafe.Sizeof(Dirent{}.Reclen))
}

func direntNamlen(buf []byte) (uint64, bool) {
	reclen, ok := direntReclen(buf)
	if !ok {
		return 0, false
	}
	return reclen - uint64(unsafe.Offsetof(Dirent{}.Name)), true
}

func Gettimeofday(tv *Timeval) (err error) {
	err = gettimeofday(tv, nil)
	return
}

// TODO
func sendfile(outfd int, infd int, offset *int64, count int) (written int, err error) {
	return -1, ENOSYS
}

//sys	getdirent(fd int, buf []byte) (n int, err error)

func ReadDirent(fd int, buf []byte) (n int, err error) {
	return getdirent(fd, buf)
}

//sys  wait4(pid _Pid_t, status *_C_int, options int, rusage *Rusage) (wpid _Pid_t, err error)

func Wait4(pid int, wstatus *WaitStatus, options int, rusage *Rusage) (wpid int, err error) {
	var status _C_int
	var r _Pid_t
	err = ERESTART
	// AIX wait4 may return with ERESTART errno, while the process is still
	// active.
	for err == ERESTART {
		r, err = wait4(_Pid_t(pid), &status, options, rusage)
	}
	wpid = int(r)
	if wstatus != nil {
		*wstatus = WaitStatus(status)
	}
	return
}

//sys	fsyncRange(fd int, how int, start int64, length int64) (err error) = fsync_range

func Fsync(fd int) error {
	return fsyncRange(fd, O_SYNC, 0, 0)
}

/*
 * Socket
 */
//sys	bind(s int, addr unsafe.Pointer, addrlen _Socklen) (err error)
//sys	connect(s int, addr unsafe.Pointer, addrlen _Socklen) (err error)
//sys   Getkerninfo(op int32, where uintptr, size uintptr, arg int64) (i int32, err error)
//sys	getsockopt(s int, level int, name int, val unsafe.Pointer, vallen *_Socklen) (err error)
//sys	Listen(s int, backlog int) (err error)
//sys	setsockopt(s int, level int, name int, val unsafe.Pointer, vallen uintptr) (err error)
//sys	socket(domain int, typ int, proto int) (fd int, err error)
//sysnb	socketpair(domain int, typ int, proto int, fd *[2]int32) (err error)
//sysnb	getpeername(fd int, rsa *RawSockaddrAny, addrlen *_Socklen) (err error)
//sys	getsockname(fd int, rsa *RawSockaddrAny, addrlen *_Socklen) (err error)
//sys	recvfrom(fd int, p []byte, flags int, from *RawSockaddrAny, fromlen *_Socklen) (n int, err error)
//sys	sendto(s int, buf []byte, flags int, to unsafe.Pointer, addrlen _Socklen) (err error)
//sys	Shutdown(s int, how int) (err error)

// In order to use msghdr structure with Control, Controllen in golang.org/x/net,
// nrecvmsg and nsendmsg must be used.
//sys	recvmsg(s int, msg *Msghdr, flags int) (n int, err error) = nrecvmsg
//sys	sendmsg(s int, msg *Msghdr, flags int) (n int, err error) = nsendmsg

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

func (sa *RawSockaddrUnix) setLen(n int) {
	sa.Len = uint8(3 + n) // 2 for Family, Len; 1 for NUL.
}

func (sa *SockaddrUnix) sockaddr() (unsafe.Pointer, _Socklen, error) {
	name := sa.Name
	n := len(name)
	if n > len(sa.raw.Path) {
		return nil, 0, EINVAL
	}
	sa.raw.Family = AF_UNIX
	sa.raw.setLen(n)
	for i := 0; i < n; i++ {
		sa.raw.Path[i] = uint8(name[i])
	}
	// length is family (uint16), name, NUL.
	sl := _Socklen(2)
	if n > 0 {
		sl += _Socklen(n) + 1
	}

	return unsafe.Pointer(&sa.raw), sl, nil
}

func Getsockname(fd int) (sa Sockaddr, err error) {
	var rsa RawSockaddrAny
	var len _Socklen = SizeofSockaddrAny
	if err = getsockname(fd, &rsa, &len); err != nil {
		return
	}
	return anyToSockaddr(&rsa)
}

//sys	accept(s int, rsa *RawSockaddrAny, addrlen *_Socklen) (fd int, err error)

func Accept(fd int) (nfd int, sa Sockaddr, err error) {
	var rsa RawSockaddrAny
	var len _Socklen = SizeofSockaddrAny
	nfd, err = accept(fd, &rsa, &len)
	if err != nil {
		return
	}
	sa, err = anyToSockaddr(&rsa)
	if err != nil {
		Close(nfd)
		nfd = 0
	}
	return
}

func recvmsgRaw(fd int, p, oob []byte, flags int, rsa *RawSockaddrAny) (n, oobn int, recvflags int, err error) {
	var msg Msghdr
	msg.Name = (*byte)(unsafe.Pointer(rsa))
	msg.Namelen = uint32(SizeofSockaddrAny)
	var iov Iovec
	if len(p) > 0 {
		iov.Base = &p[0]
		iov.SetLen(len(p))
	}
	var dummy byte
	if len(oob) > 0 {
		var sockType int
		sockType, err = GetsockoptInt(fd, SOL_SOCKET, SO_TYPE)
		if err != nil {
			return
		}
		// receive at least one normal byte
		if sockType != SOCK_DGRAM && len(p) == 0 {
			iov.Base = &dummy
			iov.SetLen(1)
		}
		msg.Control = &oob[0]
		msg.SetControllen(len(oob))
	}
	msg.Iov = &iov
	msg.Iovlen = 1
	if n, err = recvmsg(fd, &msg, flags); err != nil {
		return
	}
	oobn = int(msg.Controllen)
	recvflags = int(msg.Flags)
	return
}

func sendmsgN(fd int, p, oob []byte, ptr unsafe.Pointer, salen _Socklen, flags int) (n int, err error) {
	var msg Msghdr
	msg.Name = (*byte)(ptr)
	msg.Namelen = uint32(salen)
	var iov Iovec
	if len(p) > 0 {
		iov.Base = &p[0]
		iov.SetLen(len(p))
	}
	var dummy byte
	if len(oob) > 0 {
		var sockType int
		sockType, err = GetsockoptInt(fd, SOL_SOCKET, SO_TYPE)
		if err != nil {
			return 0, err
		}
		// send at least one normal byte
		if sockType != SOCK_DGRAM && len(p) == 0 {
			iov.Base = &dummy
			iov.SetLen(1)
		}
		msg.Control = &oob[0]
		msg.SetControllen(len(oob))
	}
	msg.Iov = &iov
	msg.Iovlen = 1
	if n, err = sendmsg(fd, &msg, flags); err != nil {
		return 0, err
	}
	if len(oob) > 0 && len(p) == 0 {
		n = 0
	}
	return n, nil
}

func (sa *RawSockaddrUnix) getLen() (int, error) {
	// Some versions of AIX have a bug in getsockname (see IV78655).
	// We can't rely on sa.Len being set correctly.
	n := SizeofSockaddrUnix - 3 // subtract leading Family, Len, terminating NUL.
	for i := 0; i < n; i++ {
		if sa.Path[i] == 0 {
			n = i
			break
		}
	}
	return n, nil
}

func anyToSockaddr(rsa *RawSockaddrAny) (Sockaddr, error) {
	switch rsa.Addr.Family {
	case AF_UNIX:
		pp := (*RawSockaddrUnix)(unsafe.Pointer(rsa))
		sa := new(SockaddrUnix)
		n, err := pp.getLen()
		if err != nil {
			return nil, err
		}
		sa.Name = string(unsafe.Slice((*byte)(unsafe.Pointer(&pp.Path[0])), n))
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
		sa.Addr = pp.Addr
		return sa, nil
	}
	return nil, EAFNOSUPPORT
}

type SockaddrDatalink struct {
	Len    uint8
	Family uint8
	Index  uint16
	Type   uint8
	Nlen   uint8
	Alen   uint8
	Slen   uint8
	Data   [120]uint8
	raw    RawSockaddrDatalink
}

/*
 * Wait
 */

type WaitStatus uint32

func (w WaitStatus) Stopped() bool { return w&0x40 != 0 }
func (w WaitStatus) StopSignal() Signal {
	if !w.Stopped() {
		return -1
	}
	return Signal(w>>8) & 0xFF
}

func (w WaitStatus) Exited() bool { return w&0xFF == 0 }
func (w WaitStatus) ExitStatus() int {
	if !w.Exited() {
		return -1
	}
	return int((w >> 8) & 0xFF)
}

func (w WaitStatus) Signaled() bool { return w&0x40 == 0 && w&0xFF != 0 }
func (w WaitStatus) Signal() Signal {
	if !w.Signaled() {
		return -1
	}
	return Signal(w>>16) & 0xFF
}

func (w WaitStatus) Continued() bool { return w&0x01000000 != 0 }

func (w WaitStatus) CoreDump() bool { return w&0x80 == 0x80 }

func (w WaitStatus) TrapCause() int { return -1 }

/*
 * ptrace
 */

//sys	Openat(dirfd int, path string, flags int, mode uint32) (fd int, err error)
//sys	ptrace64(request int, id int64, addr int64, data int, buff uintptr) (err error)
//sys	ptrace64Ptr(request int, id int64, addr int64, data int, buff unsafe.Pointer) (err error) = ptrace64

func raw_ptrace(request int, pid int, addr *byte, data *byte) Errno {
	if request == PTRACE_TRACEME {
		// Convert to AIX ptrace call.
		err := ptrace64(PT_TRACE_ME, 0, 0, 0, 0)
		if err != nil {
			return err.(Errno)
		}
		return 0
	}
	return ENOSYS
}

func ptracePeek(pid int, addr uintptr, out []byte) (count int, err error) {
	n := 0
	for len(out) > 0 {
		bsize := len(out)
		if bsize > 1024 {
			bsize = 1024
		}
		err = ptrace64Ptr(PT_READ_BLOCK, int64(pid), int64(addr), bsize, unsafe.Pointer(&out[0]))
		if err != nil {
			return 0, err
		}
		addr += uintptr(bsize)
		n += bsize
		out = out[n:]
	}
	return n, nil
}

func PtracePeekText(pid int, addr uintptr, out []byte) (count int, err error) {
	return ptracePeek(pid, addr, out)
}

func PtracePeekData(pid int, addr uintptr, out []byte) (count int, err error) {
	return ptracePeek(pid, addr, out)
}

func ptracePoke(pid int, addr uintptr, data []byte) (count int, err error) {
	n := 0
	for len(data) > 0 {
		bsize := len(data)
		if bsize > 1024 {
			bsize = 1024
		}
		err = ptrace64Ptr(PT_WRITE_BLOCK, int64(pid), int64(addr), bsize, unsafe.Pointer(&data[0]))
		if err != nil {
			return 0, err
		}
		addr += uintptr(bsize)
		n += bsize
		data = data[n:]
	}
	return n, nil
}

func PtracePokeText(pid int, addr uintptr, data []byte) (count int, err error) {
	return ptracePoke(pid, addr, data)
}

func PtracePokeData(pid int, addr uintptr, data []byte) (count int, err error) {
	return ptracePoke(pid, addr, data)
}

func PtraceCont(pid int, signal int) (err error) {
	return ptrace64(PT_CONTINUE, int64(pid), 1, signal, 0)
}

func PtraceSingleStep(pid int) (err error) { return ptrace64(PT_STEP, int64(pid), 1, 0, 0) }

func PtraceAttach(pid int) (err error) { return ptrace64(PT_ATTACH, int64(pid), 0, 0, 0) }

func PtraceDetach(pid int) (err error) { return ptrace64(PT_DETACH, int64(pid), 0, 0, 0) }

/*
 * Direct access
 */

//sys	Acct(path string) (err error)
//sys	Chdir(path string) (err error)
//sys	Chmod(path string, mode uint32) (err error)
//sys	Chown(path string, uid int, gid int) (err error)
//sys	Chroot(path string) (err error)
//sys	Close(fd int) (err error)
//sys	Dup(fd int) (nfd int, err error)
//sys	Faccessat(dirfd int, path string, mode uint32, flags int) (err error)
//sys	Fchdir(fd int) (err error)
//sys	Fchmod(fd int, mode uint32) (err error)
//sys	Fchmodat(dirfd int, path string, mode uint32, flags int) (err error)
//sys	Fchown(fd int, uid int, gid int) (err error)
//sys	Fchownat(dirfd int, path string, uid int, gid int, flags int) (err error)
//sys	Fpathconf(fd int, name int) (val int, err error)
//sys	Fstat(fd int, stat *Stat_t) (err error)
//sys	Fstatfs(fd int, buf *Statfs_t) (err error)
//sys	Ftruncate(fd int, length int64) (err error)
//sysnb	Getgid() (gid int)
//sysnb	Getpid() (pid int)
//sys	Geteuid() (euid int)
//sys	Getegid() (egid int)
//sys	Getppid() (ppid int)
//sys	Getpriority(which int, who int) (n int, err error)
//sysnb	Getrlimit(which int, lim *Rlimit) (err error)
//sysnb	Getrusage(who int, rusage *Rusage) (err error)
//sysnb	Getuid() (uid int)
//sys	Kill(pid int, signum Signal) (err error)
//sys	Lchown(path string, uid int, gid int) (err error)
//sys	Link(path string, link string) (err error)
//sys	Lstat(path string, stat *Stat_t) (err error)
//sys	Mkdir(path string, mode uint32) (err error)
//sys	Mkdirat(dirfd int, path string, mode uint32) (err error)
//sys	Mknodat(dirfd int, path string, mode uint32, dev int) (err error)
//sys	Open(path string, mode int, perm uint32) (fd int, err error)
//sys	pread(fd int, p []byte, offset int64) (n int, err error)
//sys	pwrite(fd int, p []byte, offset int64) (n int, err error)
//sys	read(fd int, p []byte) (n int, err error)
//sys	Reboot(how int) (err error)
//sys	Rename(from string, to string) (err error)
//sys	Renameat(olddirfd int, oldpath string, newdirfd int, newpath string) (err error)
//sys	Rmdir(path string) (err error)
//sys	Seek(fd int, offset int64, whence int) (newoffset int64, err error) = lseek
//sysnb	Setegid(egid int) (err error)
//sysnb	Seteuid(euid int) (err error)
//sysnb	Setgid(gid int) (err error)
//sysnb	Setpgid(pid int, pgid int) (err error)
//sys	Setpriority(which int, who int, prio int) (err error)
//sysnb	Setregid(rgid int, egid int) (err error)
//sysnb	Setreuid(ruid int, euid int) (err error)
//sysnb	setrlimit(which int, lim *Rlimit) (err error)
//sys	Stat(path string, stat *Stat_t) (err error)
//sys	Statfs(path string, buf *Statfs_t) (err error)
//sys	Symlink(path string, link string) (err error)
//sys	Truncate(path string, length int64) (err error)
//sys	Umask(newmask int) (oldmask int)
//sys	Unlink(path string) (err error)
//sysnb	Uname(buf *Utsname) (err error)
//sys	write(fd int, p []byte) (n int, err error)
//sys	writev(fd int, iovecs []Iovec) (n uintptr, err error)

//sys	gettimeofday(tv *Timeval, tzp *Timezone) (err error)

func setTimespec(sec, nsec int64) Timespec {
	return Timespec{Sec: sec, Nsec: nsec}
}

func setTimeval(sec, usec int64) Timeval {
	return Timeval{Sec: sec, Usec: int32(usec)}
}

func readlen(fd int, buf *byte, nbuf int) (n int, err error) {
	r0, _, e1 := syscall6(uintptr(unsafe.Pointer(&libc_read)), 3, uintptr(fd), uintptr(unsafe.Pointer(buf)), uintptr(nbuf), 0, 0, 0)
	n = int(r0)
	if e1 != 0 {
		err = e1
	}
	return
}

/*
 * Map
 */

var mapper = &mmapper{
	active: make(map[*byte][]byte),
	mmap:   mmap,
	munmap: munmap,
}

//sys	mmap(addr uintptr, length uintptr, prot int, flag int, fd int, pos int64) (ret uintptr, err error)
//sys	munmap(addr uintptr, length uintptr) (err error)

func Mmap(fd int, offset int64, length int, prot int, flags int) (data []byte, err error) {
	return mapper.Mmap(fd, offset, length, prot, flags)
}

func Munmap(b []byte) (err error) {
	return mapper.Munmap(b)
}
