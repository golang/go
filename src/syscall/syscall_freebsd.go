// Copyright 2009,2010 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// FreeBSD system calls.
// This file is compiled as ordinary Go code,
// but it is also input to mksyscall,
// which parses the //sys lines and generates system call stubs.
// Note that sometimes we use a lowercase //sys name and wrap
// it in our own nicer implementation, either here or in
// syscall_bsd.go or syscall_unix.go.

package syscall

import (
	"sync"
	"unsafe"
)

const (
	_SYS_FSTAT_FREEBSD12         = 551 // { int fstat(int fd, _Out_ struct stat *sb); }
	_SYS_FSTATAT_FREEBSD12       = 552 // { int fstatat(int fd, _In_z_ char *path, _Out_ struct stat *buf, int flag); }
	_SYS_GETDIRENTRIES_FREEBSD12 = 554 // { ssize_t getdirentries(int fd, _Out_writes_bytes_(count) char *buf, size_t count, _Out_ off_t *basep); }
	_SYS_STATFS_FREEBSD12        = 555 // { int statfs(_In_z_ char *path, _Out_ struct statfs *buf); }
	_SYS_FSTATFS_FREEBSD12       = 556 // { int fstatfs(int fd, _Out_ struct statfs *buf); }
	_SYS_GETFSSTAT_FREEBSD12     = 557 // { int getfsstat(_Out_writes_bytes_opt_(bufsize) struct statfs *buf, long bufsize, int mode); }
	_SYS_MKNODAT_FREEBSD12       = 559 // { int mknodat(int fd, _In_z_ char *path, mode_t mode, dev_t dev); }
)

// See https://www.freebsd.org/doc/en_US.ISO8859-1/books/porters-handbook/versions.html.
var (
	osreldateOnce sync.Once
	osreldate     uint32
)

// INO64_FIRST from /usr/src/lib/libc/sys/compat-ino64.h
const _ino64First = 1200031

func supportsABI(ver uint32) bool {
	osreldateOnce.Do(func() { osreldate, _ = SysctlUint32("kern.osreldate") })
	return osreldate >= ver
}

type SockaddrDatalink struct {
	Len    uint8
	Family uint8
	Index  uint16
	Type   uint8
	Nlen   uint8
	Alen   uint8
	Slen   uint8
	Data   [46]int8
	raw    RawSockaddrDatalink
}

// Translate "kern.hostname" to []_C_int{0,1,2,3}.
func nametomib(name string) (mib []_C_int, err error) {
	const siz = unsafe.Sizeof(mib[0])

	// NOTE(rsc): It seems strange to set the buffer to have
	// size CTL_MAXNAME+2 but use only CTL_MAXNAME
	// as the size. I don't know why the +2 is here, but the
	// kernel uses +2 for its own implementation of this function.
	// I am scared that if we don't include the +2 here, the kernel
	// will silently write 2 words farther than we specify
	// and we'll get memory corruption.
	var buf [CTL_MAXNAME + 2]_C_int
	n := uintptr(CTL_MAXNAME) * siz

	p := (*byte)(unsafe.Pointer(&buf[0]))
	bytes, err := ByteSliceFromString(name)
	if err != nil {
		return nil, err
	}

	// Magic sysctl: "setting" 0.3 to a string name
	// lets you read back the array of integers form.
	if err = sysctl([]_C_int{0, 3}, p, &n, &bytes[0], uintptr(len(name))); err != nil {
		return nil, err
	}
	return buf[0 : n/siz], nil
}

func direntIno(buf []byte) (uint64, bool) {
	return readInt(buf, unsafe.Offsetof(Dirent{}.Fileno), unsafe.Sizeof(Dirent{}.Fileno))
}

func direntReclen(buf []byte) (uint64, bool) {
	return readInt(buf, unsafe.Offsetof(Dirent{}.Reclen), unsafe.Sizeof(Dirent{}.Reclen))
}

func direntNamlen(buf []byte) (uint64, bool) {
	return readInt(buf, unsafe.Offsetof(Dirent{}.Namlen), unsafe.Sizeof(Dirent{}.Namlen))
}

func Pipe(p []int) error {
	return Pipe2(p, 0)
}

//sysnb pipe2(p *[2]_C_int, flags int) (err error)

func Pipe2(p []int, flags int) error {
	if len(p) != 2 {
		return EINVAL
	}
	var pp [2]_C_int
	err := pipe2(&pp, flags)
	p[0] = int(pp[0])
	p[1] = int(pp[1])
	return err
}

func GetsockoptIPMreqn(fd, level, opt int) (*IPMreqn, error) {
	var value IPMreqn
	vallen := _Socklen(SizeofIPMreqn)
	errno := getsockopt(fd, level, opt, unsafe.Pointer(&value), &vallen)
	return &value, errno
}

func SetsockoptIPMreqn(fd, level, opt int, mreq *IPMreqn) (err error) {
	return setsockopt(fd, level, opt, unsafe.Pointer(mreq), unsafe.Sizeof(*mreq))
}

func Accept4(fd, flags int) (nfd int, sa Sockaddr, err error) {
	var rsa RawSockaddrAny
	var len _Socklen = SizeofSockaddrAny
	nfd, err = accept4(fd, &rsa, &len, flags)
	if err != nil {
		return
	}
	if len > SizeofSockaddrAny {
		panic("RawSockaddrAny too small")
	}
	sa, err = anyToSockaddr(&rsa)
	if err != nil {
		Close(nfd)
		nfd = 0
	}
	return
}

func Getfsstat(buf []Statfs_t, flags int) (n int, err error) {
	var (
		_p0          unsafe.Pointer
		bufsize      uintptr
		oldBuf       []statfs_freebsd11_t
		needsConvert bool
	)

	if len(buf) > 0 {
		if supportsABI(_ino64First) {
			_p0 = unsafe.Pointer(&buf[0])
			bufsize = unsafe.Sizeof(Statfs_t{}) * uintptr(len(buf))
		} else {
			n := len(buf)
			oldBuf = make([]statfs_freebsd11_t, n)
			_p0 = unsafe.Pointer(&oldBuf[0])
			bufsize = unsafe.Sizeof(statfs_freebsd11_t{}) * uintptr(n)
			needsConvert = true
		}
	}
	var sysno uintptr = SYS_GETFSSTAT
	if supportsABI(_ino64First) {
		sysno = _SYS_GETFSSTAT_FREEBSD12
	}
	r0, _, e1 := Syscall(sysno, uintptr(_p0), bufsize, uintptr(flags))
	n = int(r0)
	if e1 != 0 {
		err = e1
	}
	if e1 == 0 && needsConvert {
		for i := range oldBuf {
			buf[i].convertFrom(&oldBuf[i])
		}
	}
	return
}

func setattrlistTimes(path string, times []Timespec) error {
	// used on Darwin for UtimesNano
	return ENOSYS
}

func Stat(path string, st *Stat_t) (err error) {
	var oldStat stat_freebsd11_t
	if supportsABI(_ino64First) {
		return fstatat_freebsd12(_AT_FDCWD, path, st, 0)
	}
	err = stat(path, &oldStat)
	if err != nil {
		return err
	}

	st.convertFrom(&oldStat)
	return nil
}

func Lstat(path string, st *Stat_t) (err error) {
	var oldStat stat_freebsd11_t
	if supportsABI(_ino64First) {
		return fstatat_freebsd12(_AT_FDCWD, path, st, _AT_SYMLINK_NOFOLLOW)
	}
	err = lstat(path, &oldStat)
	if err != nil {
		return err
	}

	st.convertFrom(&oldStat)
	return nil
}

func Fstat(fd int, st *Stat_t) (err error) {
	var oldStat stat_freebsd11_t
	if supportsABI(_ino64First) {
		return fstat_freebsd12(fd, st)
	}
	err = fstat(fd, &oldStat)
	if err != nil {
		return err
	}

	st.convertFrom(&oldStat)
	return nil
}

func Fstatat(fd int, path string, st *Stat_t, flags int) (err error) {
	var oldStat stat_freebsd11_t
	if supportsABI(_ino64First) {
		return fstatat_freebsd12(fd, path, st, flags)
	}
	err = fstatat(fd, path, &oldStat, flags)
	if err != nil {
		return err
	}

	st.convertFrom(&oldStat)
	return nil
}

func Statfs(path string, st *Statfs_t) (err error) {
	var oldStatfs statfs_freebsd11_t
	if supportsABI(_ino64First) {
		return statfs_freebsd12(path, st)
	}
	err = statfs(path, &oldStatfs)
	if err != nil {
		return err
	}

	st.convertFrom(&oldStatfs)
	return nil
}

func Fstatfs(fd int, st *Statfs_t) (err error) {
	var oldStatfs statfs_freebsd11_t
	if supportsABI(_ino64First) {
		return fstatfs_freebsd12(fd, st)
	}
	err = fstatfs(fd, &oldStatfs)
	if err != nil {
		return err
	}

	st.convertFrom(&oldStatfs)
	return nil
}

func Getdirentries(fd int, buf []byte, basep *uintptr) (n int, err error) {
	if supportsABI(_ino64First) {
		if basep == nil || unsafe.Sizeof(*basep) == 8 {
			return getdirentries_freebsd12(fd, buf, (*uint64)(unsafe.Pointer(basep)))
		}
		// The freebsd12 syscall needs a 64-bit base. On 32-bit machines
		// we can't just use the basep passed in. See #32498.
		var base uint64 = uint64(*basep)
		n, err = getdirentries_freebsd12(fd, buf, &base)
		*basep = uintptr(base)
		if base>>32 != 0 {
			// We can't stuff the base back into a uintptr, so any
			// future calls would be suspect. Generate an error.
			// EIO is allowed by getdirentries.
			err = EIO
		}
		return
	}

	// The old syscall entries are smaller than the new. Use 1/4 of the original
	// buffer size rounded up to DIRBLKSIZ (see /usr/src/lib/libc/sys/getdirentries.c).
	oldBufLen := roundup(len(buf)/4, _dirblksiz)
	oldBuf := make([]byte, oldBufLen)
	n, err = getdirentries(fd, oldBuf, basep)
	if err == nil && n > 0 {
		n = convertFromDirents11(buf, oldBuf[:n])
	}
	return
}

func Mknod(path string, mode uint32, dev uint64) (err error) {
	var oldDev int
	if supportsABI(_ino64First) {
		return mknodat_freebsd12(_AT_FDCWD, path, mode, dev)
	}
	oldDev = int(dev)
	return mknod(path, mode, oldDev)
}

// round x to the nearest multiple of y, larger or equal to x.
//
// from /usr/include/sys/param.h Macros for counting and rounding.
// #define roundup(x, y)   ((((x)+((y)-1))/(y))*(y))
func roundup(x, y int) int {
	return ((x + y - 1) / y) * y
}

func (s *Stat_t) convertFrom(old *stat_freebsd11_t) {
	*s = Stat_t{
		Dev:           uint64(old.Dev),
		Ino:           uint64(old.Ino),
		Nlink:         uint64(old.Nlink),
		Mode:          old.Mode,
		Uid:           old.Uid,
		Gid:           old.Gid,
		Rdev:          uint64(old.Rdev),
		Atimespec:     old.Atimespec,
		Mtimespec:     old.Mtimespec,
		Ctimespec:     old.Ctimespec,
		Birthtimespec: old.Birthtimespec,
		Size:          old.Size,
		Blocks:        old.Blocks,
		Blksize:       old.Blksize,
		Flags:         old.Flags,
		Gen:           uint64(old.Gen),
	}
}

func (s *Statfs_t) convertFrom(old *statfs_freebsd11_t) {
	*s = Statfs_t{
		Version:     _statfsVersion,
		Type:        old.Type,
		Flags:       old.Flags,
		Bsize:       old.Bsize,
		Iosize:      old.Iosize,
		Blocks:      old.Blocks,
		Bfree:       old.Bfree,
		Bavail:      old.Bavail,
		Files:       old.Files,
		Ffree:       old.Ffree,
		Syncwrites:  old.Syncwrites,
		Asyncwrites: old.Asyncwrites,
		Syncreads:   old.Syncreads,
		Asyncreads:  old.Asyncreads,
		// Spare
		Namemax: old.Namemax,
		Owner:   old.Owner,
		Fsid:    old.Fsid,
		// Charspare
		// Fstypename
		// Mntfromname
		// Mntonname
	}

	sl := old.Fstypename[:]
	n := clen(*(*[]byte)(unsafe.Pointer(&sl)))
	copy(s.Fstypename[:], old.Fstypename[:n])

	sl = old.Mntfromname[:]
	n = clen(*(*[]byte)(unsafe.Pointer(&sl)))
	copy(s.Mntfromname[:], old.Mntfromname[:n])

	sl = old.Mntonname[:]
	n = clen(*(*[]byte)(unsafe.Pointer(&sl)))
	copy(s.Mntonname[:], old.Mntonname[:n])
}

func convertFromDirents11(buf []byte, old []byte) int {
	const (
		fixedSize    = int(unsafe.Offsetof(Dirent{}.Name))
		oldFixedSize = int(unsafe.Offsetof(dirent_freebsd11{}.Name))
	)

	dstPos := 0
	srcPos := 0
	for dstPos+fixedSize < len(buf) && srcPos+oldFixedSize < len(old) {
		var dstDirent Dirent
		var srcDirent dirent_freebsd11

		// If multiple direntries are written, sometimes when we reach the final one,
		// we may have cap of old less than size of dirent_freebsd11.
		copy((*[unsafe.Sizeof(srcDirent)]byte)(unsafe.Pointer(&srcDirent))[:], old[srcPos:])

		reclen := roundup(fixedSize+int(srcDirent.Namlen)+1, 8)
		if dstPos+reclen > len(buf) {
			break
		}

		dstDirent.Fileno = uint64(srcDirent.Fileno)
		dstDirent.Off = 0
		dstDirent.Reclen = uint16(reclen)
		dstDirent.Type = srcDirent.Type
		dstDirent.Pad0 = 0
		dstDirent.Namlen = uint16(srcDirent.Namlen)
		dstDirent.Pad1 = 0

		copy(dstDirent.Name[:], srcDirent.Name[:srcDirent.Namlen])
		copy(buf[dstPos:], (*[unsafe.Sizeof(dstDirent)]byte)(unsafe.Pointer(&dstDirent))[:])
		padding := buf[dstPos+fixedSize+int(dstDirent.Namlen) : dstPos+reclen]
		for i := range padding {
			padding[i] = 0
		}

		dstPos += int(dstDirent.Reclen)
		srcPos += int(srcDirent.Reclen)
	}

	return dstPos
}

/*
 * Exposed directly
 */
//sys	Access(path string, mode uint32) (err error)
//sys	Adjtime(delta *Timeval, olddelta *Timeval) (err error)
//sys	Chdir(path string) (err error)
//sys	Chflags(path string, flags int) (err error)
//sys	Chmod(path string, mode uint32) (err error)
//sys	Chown(path string, uid int, gid int) (err error)
//sys	Chroot(path string) (err error)
//sys	Close(fd int) (err error)
//sys	Dup(fd int) (nfd int, err error)
//sys	Dup2(from int, to int) (err error)
//sys	Fchdir(fd int) (err error)
//sys	Fchflags(fd int, flags int) (err error)
//sys	Fchmod(fd int, mode uint32) (err error)
//sys	Fchown(fd int, uid int, gid int) (err error)
//sys	Flock(fd int, how int) (err error)
//sys	Fpathconf(fd int, name int) (val int, err error)
//sys	fstat(fd int, stat *stat_freebsd11_t) (err error)
//sys	fstat_freebsd12(fd int, stat *Stat_t) (err error) = _SYS_FSTAT_FREEBSD12
//sys	fstatat(fd int, path string, stat *stat_freebsd11_t, flags int) (err error)
//sys	fstatat_freebsd12(fd int, path string, stat *Stat_t, flags int) (err error) = _SYS_FSTATAT_FREEBSD12
//sys	fstatfs(fd int, stat *statfs_freebsd11_t) (err error)
//sys	fstatfs_freebsd12(fd int, stat *Statfs_t) (err error) = _SYS_FSTATFS_FREEBSD12
//sys	Fsync(fd int) (err error)
//sys	Ftruncate(fd int, length int64) (err error)
//sys	getdirentries(fd int, buf []byte, basep *uintptr) (n int, err error)
//sys	getdirentries_freebsd12(fd int, buf []byte, basep *uint64) (n int, err error) = _SYS_GETDIRENTRIES_FREEBSD12
//sys	Getdtablesize() (size int)
//sysnb	Getegid() (egid int)
//sysnb	Geteuid() (uid int)
//sysnb	Getgid() (gid int)
//sysnb	Getpgid(pid int) (pgid int, err error)
//sysnb	Getpgrp() (pgrp int)
//sysnb	Getpid() (pid int)
//sysnb	Getppid() (ppid int)
//sys	Getpriority(which int, who int) (prio int, err error)
//sysnb	Getrlimit(which int, lim *Rlimit) (err error)
//sysnb	Getrusage(who int, rusage *Rusage) (err error)
//sysnb	Getsid(pid int) (sid int, err error)
//sysnb	Gettimeofday(tv *Timeval) (err error)
//sysnb	Getuid() (uid int)
//sys	Issetugid() (tainted bool)
//sys	Kill(pid int, signum Signal) (err error)
//sys	Kqueue() (fd int, err error)
//sys	Lchown(path string, uid int, gid int) (err error)
//sys	Link(path string, link string) (err error)
//sys	Listen(s int, backlog int) (err error)
//sys	lstat(path string, stat *stat_freebsd11_t) (err error)
//sys	Mkdir(path string, mode uint32) (err error)
//sys	Mkfifo(path string, mode uint32) (err error)
//sys	mknod(path string, mode uint32, dev int) (err error)
//sys	mknodat_freebsd12(fd int, path string, mode uint32, dev uint64) (err error) = _SYS_MKNODAT_FREEBSD12
//sys	Nanosleep(time *Timespec, leftover *Timespec) (err error)
//sys	Open(path string, mode int, perm uint32) (fd int, err error)
//sys	Pathconf(path string, name int) (val int, err error)
//sys	Pread(fd int, p []byte, offset int64) (n int, err error)
//sys	Pwrite(fd int, p []byte, offset int64) (n int, err error)
//sys	read(fd int, p []byte) (n int, err error)
//sys	Readlink(path string, buf []byte) (n int, err error)
//sys	Rename(from string, to string) (err error)
//sys	Revoke(path string) (err error)
//sys	Rmdir(path string) (err error)
//sys	Seek(fd int, offset int64, whence int) (newoffset int64, err error) = SYS_LSEEK
//sys	Select(n int, r *FdSet, w *FdSet, e *FdSet, timeout *Timeval) (err error)
//sysnb	Setegid(egid int) (err error)
//sysnb	Seteuid(euid int) (err error)
//sysnb	Setgid(gid int) (err error)
//sys	Setlogin(name string) (err error)
//sysnb	Setpgid(pid int, pgid int) (err error)
//sys	Setpriority(which int, who int, prio int) (err error)
//sysnb	Setregid(rgid int, egid int) (err error)
//sysnb	Setreuid(ruid int, euid int) (err error)
//sysnb	Setrlimit(which int, lim *Rlimit) (err error)
//sysnb	Setsid() (pid int, err error)
//sysnb	Settimeofday(tp *Timeval) (err error)
//sysnb	Setuid(uid int) (err error)
//sys	stat(path string, stat *stat_freebsd11_t) (err error)
//sys	statfs(path string, stat *statfs_freebsd11_t) (err error)
//sys	statfs_freebsd12(path string, stat *Statfs_t) (err error) = _SYS_STATFS_FREEBSD12
//sys	Symlink(path string, link string) (err error)
//sys	Sync() (err error)
//sys	Truncate(path string, length int64) (err error)
//sys	Umask(newmask int) (oldmask int)
//sys	Undelete(path string) (err error)
//sys	Unlink(path string) (err error)
//sys	Unmount(path string, flags int) (err error)
//sys	write(fd int, p []byte) (n int, err error)
//sys   mmap(addr uintptr, length uintptr, prot int, flag int, fd int, pos int64) (ret uintptr, err error)
//sys   munmap(addr uintptr, length uintptr) (err error)
//sys	readlen(fd int, buf *byte, nbuf int) (n int, err error) = SYS_READ
//sys	writelen(fd int, buf *byte, nbuf int) (n int, err error) = SYS_WRITE
//sys	accept4(fd int, rsa *RawSockaddrAny, addrlen *_Socklen, flags int) (nfd int, err error)
//sys	utimensat(dirfd int, path string, times *[2]Timespec, flag int) (err error)
//sys	getcwd(buf []byte) (n int, err error) = SYS___GETCWD
//sys	sysctl(mib []_C_int, old *byte, oldlen *uintptr, new *byte, newlen uintptr) (err error) = SYS___SYSCTL
