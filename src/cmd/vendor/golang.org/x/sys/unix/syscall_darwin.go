// Copyright 2009,2010 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// Darwin system calls.
// This file is compiled as ordinary Go code,
// but it is also input to mksyscall,
// which parses the //sys lines and generates system call stubs.
// Note that sometimes we use a lowercase //sys name and wrap
// it in our own nicer implementation, either here or in
// syscall_bsd.go or syscall_unix.go.

package unix

import (
	"errors"
	"syscall"
	"unsafe"
)

const ImplementsGetwd = true

func Getwd() (string, error) {
	buf := make([]byte, 2048)
	attrs, err := getAttrList(".", attrList{CommonAttr: attrCmnFullpath}, buf, 0)
	if err == nil && len(attrs) == 1 && len(attrs[0]) >= 2 {
		wd := string(attrs[0])
		// Sanity check that it's an absolute path and ends
		// in a null byte, which we then strip.
		if wd[0] == '/' && wd[len(wd)-1] == 0 {
			return wd[:len(wd)-1], nil
		}
	}
	// If pkg/os/getwd.go gets ENOTSUP, it will fall back to the
	// slow algorithm.
	return "", ENOTSUP
}

// SockaddrDatalink implements the Sockaddr interface for AF_LINK type sockets.
type SockaddrDatalink struct {
	Len    uint8
	Family uint8
	Index  uint16
	Type   uint8
	Nlen   uint8
	Alen   uint8
	Slen   uint8
	Data   [12]int8
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

//sys   ptrace(request int, pid int, addr uintptr, data uintptr) (err error)
func PtraceAttach(pid int) (err error) { return ptrace(PT_ATTACH, pid, 0, 0) }
func PtraceDetach(pid int) (err error) { return ptrace(PT_DETACH, pid, 0, 0) }

const (
	attrBitMapCount = 5
	attrCmnFullpath = 0x08000000
)

type attrList struct {
	bitmapCount uint16
	_           uint16
	CommonAttr  uint32
	VolAttr     uint32
	DirAttr     uint32
	FileAttr    uint32
	Forkattr    uint32
}

func getAttrList(path string, attrList attrList, attrBuf []byte, options uint) (attrs [][]byte, err error) {
	if len(attrBuf) < 4 {
		return nil, errors.New("attrBuf too small")
	}
	attrList.bitmapCount = attrBitMapCount

	var _p0 *byte
	_p0, err = BytePtrFromString(path)
	if err != nil {
		return nil, err
	}

	_, _, e1 := Syscall6(
		SYS_GETATTRLIST,
		uintptr(unsafe.Pointer(_p0)),
		uintptr(unsafe.Pointer(&attrList)),
		uintptr(unsafe.Pointer(&attrBuf[0])),
		uintptr(len(attrBuf)),
		uintptr(options),
		0,
	)
	if e1 != 0 {
		return nil, e1
	}
	size := *(*uint32)(unsafe.Pointer(&attrBuf[0]))

	// dat is the section of attrBuf that contains valid data,
	// without the 4 byte length header. All attribute offsets
	// are relative to dat.
	dat := attrBuf
	if int(size) < len(attrBuf) {
		dat = dat[:size]
	}
	dat = dat[4:] // remove length prefix

	for i := uint32(0); int(i) < len(dat); {
		header := dat[i:]
		if len(header) < 8 {
			return attrs, errors.New("truncated attribute header")
		}
		datOff := *(*int32)(unsafe.Pointer(&header[0]))
		attrLen := *(*uint32)(unsafe.Pointer(&header[4]))
		if datOff < 0 || uint32(datOff)+attrLen > uint32(len(dat)) {
			return attrs, errors.New("truncated results; attrBuf too small")
		}
		end := uint32(datOff) + attrLen
		attrs = append(attrs, dat[datOff:end])
		i = end
		if r := i % 4; r != 0 {
			i += (4 - r)
		}
	}
	return
}

//sysnb pipe() (r int, w int, err error)

func Pipe(p []int) (err error) {
	if len(p) != 2 {
		return EINVAL
	}
	p[0], p[1], err = pipe()
	return
}

func Getfsstat(buf []Statfs_t, flags int) (n int, err error) {
	var _p0 unsafe.Pointer
	var bufsize uintptr
	if len(buf) > 0 {
		_p0 = unsafe.Pointer(&buf[0])
		bufsize = unsafe.Sizeof(Statfs_t{}) * uintptr(len(buf))
	}
	r0, _, e1 := Syscall(SYS_GETFSSTAT64, uintptr(_p0), bufsize, uintptr(flags))
	n = int(r0)
	if e1 != 0 {
		err = e1
	}
	return
}

func xattrPointer(dest []byte) *byte {
	// It's only when dest is set to NULL that the OS X implementations of
	// getxattr() and listxattr() return the current sizes of the named attributes.
	// An empty byte array is not sufficient. To maintain the same behaviour as the
	// linux implementation, we wrap around the system calls and pass in NULL when
	// dest is empty.
	var destp *byte
	if len(dest) > 0 {
		destp = &dest[0]
	}
	return destp
}

//sys	getxattr(path string, attr string, dest *byte, size int, position uint32, options int) (sz int, err error)

func Getxattr(path string, attr string, dest []byte) (sz int, err error) {
	return getxattr(path, attr, xattrPointer(dest), len(dest), 0, 0)
}

func Lgetxattr(link string, attr string, dest []byte) (sz int, err error) {
	return getxattr(link, attr, xattrPointer(dest), len(dest), 0, XATTR_NOFOLLOW)
}

//sys	fgetxattr(fd int, attr string, dest *byte, size int, position uint32, options int) (sz int, err error)

func Fgetxattr(fd int, attr string, dest []byte) (sz int, err error) {
	return fgetxattr(fd, attr, xattrPointer(dest), len(dest), 0, 0)
}

//sys	setxattr(path string, attr string, data *byte, size int, position uint32, options int) (err error)

func Setxattr(path string, attr string, data []byte, flags int) (err error) {
	// The parameters for the OS X implementation vary slightly compared to the
	// linux system call, specifically the position parameter:
	//
	//  linux:
	//      int setxattr(
	//          const char *path,
	//          const char *name,
	//          const void *value,
	//          size_t size,
	//          int flags
	//      );
	//
	//  darwin:
	//      int setxattr(
	//          const char *path,
	//          const char *name,
	//          void *value,
	//          size_t size,
	//          u_int32_t position,
	//          int options
	//      );
	//
	// position specifies the offset within the extended attribute. In the
	// current implementation, only the resource fork extended attribute makes
	// use of this argument. For all others, position is reserved. We simply
	// default to setting it to zero.
	return setxattr(path, attr, xattrPointer(data), len(data), 0, flags)
}

func Lsetxattr(link string, attr string, data []byte, flags int) (err error) {
	return setxattr(link, attr, xattrPointer(data), len(data), 0, flags|XATTR_NOFOLLOW)
}

//sys	fsetxattr(fd int, attr string, data *byte, size int, position uint32, options int) (err error)

func Fsetxattr(fd int, attr string, data []byte, flags int) (err error) {
	return fsetxattr(fd, attr, xattrPointer(data), len(data), 0, 0)
}

//sys	removexattr(path string, attr string, options int) (err error)

func Removexattr(path string, attr string) (err error) {
	// We wrap around and explicitly zero out the options provided to the OS X
	// implementation of removexattr, we do so for interoperability with the
	// linux variant.
	return removexattr(path, attr, 0)
}

func Lremovexattr(link string, attr string) (err error) {
	return removexattr(link, attr, XATTR_NOFOLLOW)
}

//sys	fremovexattr(fd int, attr string, options int) (err error)

func Fremovexattr(fd int, attr string) (err error) {
	return fremovexattr(fd, attr, 0)
}

//sys	listxattr(path string, dest *byte, size int, options int) (sz int, err error)

func Listxattr(path string, dest []byte) (sz int, err error) {
	return listxattr(path, xattrPointer(dest), len(dest), 0)
}

func Llistxattr(link string, dest []byte) (sz int, err error) {
	return listxattr(link, xattrPointer(dest), len(dest), XATTR_NOFOLLOW)
}

//sys	flistxattr(fd int, dest *byte, size int, options int) (sz int, err error)

func Flistxattr(fd int, dest []byte) (sz int, err error) {
	return flistxattr(fd, xattrPointer(dest), len(dest), 0)
}

func setattrlistTimes(path string, times []Timespec, flags int) error {
	_p0, err := BytePtrFromString(path)
	if err != nil {
		return err
	}

	var attrList attrList
	attrList.bitmapCount = ATTR_BIT_MAP_COUNT
	attrList.CommonAttr = ATTR_CMN_MODTIME | ATTR_CMN_ACCTIME

	// order is mtime, atime: the opposite of Chtimes
	attributes := [2]Timespec{times[1], times[0]}
	options := 0
	if flags&AT_SYMLINK_NOFOLLOW != 0 {
		options |= FSOPT_NOFOLLOW
	}
	_, _, e1 := Syscall6(
		SYS_SETATTRLIST,
		uintptr(unsafe.Pointer(_p0)),
		uintptr(unsafe.Pointer(&attrList)),
		uintptr(unsafe.Pointer(&attributes)),
		uintptr(unsafe.Sizeof(attributes)),
		uintptr(options),
		0,
	)
	if e1 != 0 {
		return e1
	}
	return nil
}

func utimensat(dirfd int, path string, times *[2]Timespec, flags int) error {
	// Darwin doesn't support SYS_UTIMENSAT
	return ENOSYS
}

/*
 * Wrapped
 */

//sys	kill(pid int, signum int, posix int) (err error)

func Kill(pid int, signum syscall.Signal) (err error) { return kill(pid, int(signum), 1) }

//sys	ioctl(fd int, req uint, arg uintptr) (err error)

// ioctl itself should not be exposed directly, but additional get/set
// functions for specific types are permissible.

// IoctlSetInt performs an ioctl operation which sets an integer value
// on fd, using the specified request number.
func IoctlSetInt(fd int, req uint, value int) error {
	return ioctl(fd, req, uintptr(value))
}

func ioctlSetWinsize(fd int, req uint, value *Winsize) error {
	return ioctl(fd, req, uintptr(unsafe.Pointer(value)))
}

func ioctlSetTermios(fd int, req uint, value *Termios) error {
	return ioctl(fd, req, uintptr(unsafe.Pointer(value)))
}

// IoctlGetInt performs an ioctl operation which gets an integer value
// from fd, using the specified request number.
func IoctlGetInt(fd int, req uint) (int, error) {
	var value int
	err := ioctl(fd, req, uintptr(unsafe.Pointer(&value)))
	return value, err
}

func IoctlGetWinsize(fd int, req uint) (*Winsize, error) {
	var value Winsize
	err := ioctl(fd, req, uintptr(unsafe.Pointer(&value)))
	return &value, err
}

func IoctlGetTermios(fd int, req uint) (*Termios, error) {
	var value Termios
	err := ioctl(fd, req, uintptr(unsafe.Pointer(&value)))
	return &value, err
}

func Uname(uname *Utsname) error {
	mib := []_C_int{CTL_KERN, KERN_OSTYPE}
	n := unsafe.Sizeof(uname.Sysname)
	if err := sysctl(mib, &uname.Sysname[0], &n, nil, 0); err != nil {
		return err
	}

	mib = []_C_int{CTL_KERN, KERN_HOSTNAME}
	n = unsafe.Sizeof(uname.Nodename)
	if err := sysctl(mib, &uname.Nodename[0], &n, nil, 0); err != nil {
		return err
	}

	mib = []_C_int{CTL_KERN, KERN_OSRELEASE}
	n = unsafe.Sizeof(uname.Release)
	if err := sysctl(mib, &uname.Release[0], &n, nil, 0); err != nil {
		return err
	}

	mib = []_C_int{CTL_KERN, KERN_VERSION}
	n = unsafe.Sizeof(uname.Version)
	if err := sysctl(mib, &uname.Version[0], &n, nil, 0); err != nil {
		return err
	}

	// The version might have newlines or tabs in it, convert them to
	// spaces.
	for i, b := range uname.Version {
		if b == '\n' || b == '\t' {
			if i == len(uname.Version)-1 {
				uname.Version[i] = 0
			} else {
				uname.Version[i] = ' '
			}
		}
	}

	mib = []_C_int{CTL_HW, HW_MACHINE}
	n = unsafe.Sizeof(uname.Machine)
	if err := sysctl(mib, &uname.Machine[0], &n, nil, 0); err != nil {
		return err
	}

	return nil
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
//sys	Exchangedata(path1 string, path2 string, options int) (err error)
//sys	Exit(code int)
//sys	Faccessat(dirfd int, path string, mode uint32, flags int) (err error)
//sys	Fchdir(fd int) (err error)
//sys	Fchflags(fd int, flags int) (err error)
//sys	Fchmod(fd int, mode uint32) (err error)
//sys	Fchmodat(dirfd int, path string, mode uint32, flags int) (err error)
//sys	Fchown(fd int, uid int, gid int) (err error)
//sys	Fchownat(dirfd int, path string, uid int, gid int, flags int) (err error)
//sys	Flock(fd int, how int) (err error)
//sys	Fpathconf(fd int, name int) (val int, err error)
//sys	Fstat(fd int, stat *Stat_t) (err error) = SYS_FSTAT64
//sys	Fstatat(fd int, path string, stat *Stat_t, flags int) (err error) = SYS_FSTATAT64
//sys	Fstatfs(fd int, stat *Statfs_t) (err error) = SYS_FSTATFS64
//sys	Fsync(fd int) (err error)
//sys	Ftruncate(fd int, length int64) (err error)
//sys	Getdirentries(fd int, buf []byte, basep *uintptr) (n int, err error) = SYS_GETDIRENTRIES64
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
//sysnb	Getuid() (uid int)
//sysnb	Issetugid() (tainted bool)
//sys	Kqueue() (fd int, err error)
//sys	Lchown(path string, uid int, gid int) (err error)
//sys	Link(path string, link string) (err error)
//sys	Linkat(pathfd int, path string, linkfd int, link string, flags int) (err error)
//sys	Listen(s int, backlog int) (err error)
//sys	Lstat(path string, stat *Stat_t) (err error) = SYS_LSTAT64
//sys	Mkdir(path string, mode uint32) (err error)
//sys	Mkdirat(dirfd int, path string, mode uint32) (err error)
//sys	Mkfifo(path string, mode uint32) (err error)
//sys	Mknod(path string, mode uint32, dev int) (err error)
//sys	Open(path string, mode int, perm uint32) (fd int, err error)
//sys	Openat(dirfd int, path string, mode int, perm uint32) (fd int, err error)
//sys	Pathconf(path string, name int) (val int, err error)
//sys	Pread(fd int, p []byte, offset int64) (n int, err error)
//sys	Pwrite(fd int, p []byte, offset int64) (n int, err error)
//sys	read(fd int, p []byte) (n int, err error)
//sys	Readlink(path string, buf []byte) (n int, err error)
//sys	Readlinkat(dirfd int, path string, buf []byte) (n int, err error)
//sys	Rename(from string, to string) (err error)
//sys	Renameat(fromfd int, from string, tofd int, to string) (err error)
//sys	Revoke(path string) (err error)
//sys	Rmdir(path string) (err error)
//sys	Seek(fd int, offset int64, whence int) (newoffset int64, err error) = SYS_LSEEK
//sys	Select(n int, r *FdSet, w *FdSet, e *FdSet, timeout *Timeval) (err error)
//sys	Setegid(egid int) (err error)
//sysnb	Seteuid(euid int) (err error)
//sysnb	Setgid(gid int) (err error)
//sys	Setlogin(name string) (err error)
//sysnb	Setpgid(pid int, pgid int) (err error)
//sys	Setpriority(which int, who int, prio int) (err error)
//sys	Setprivexec(flag int) (err error)
//sysnb	Setregid(rgid int, egid int) (err error)
//sysnb	Setreuid(ruid int, euid int) (err error)
//sysnb	Setrlimit(which int, lim *Rlimit) (err error)
//sysnb	Setsid() (pid int, err error)
//sysnb	Settimeofday(tp *Timeval) (err error)
//sysnb	Setuid(uid int) (err error)
//sys	Stat(path string, stat *Stat_t) (err error) = SYS_STAT64
//sys	Statfs(path string, stat *Statfs_t) (err error) = SYS_STATFS64
//sys	Symlink(path string, link string) (err error)
//sys	Symlinkat(oldpath string, newdirfd int, newpath string) (err error)
//sys	Sync() (err error)
//sys	Truncate(path string, length int64) (err error)
//sys	Umask(newmask int) (oldmask int)
//sys	Undelete(path string) (err error)
//sys	Unlink(path string) (err error)
//sys	Unlinkat(dirfd int, path string, flags int) (err error)
//sys	Unmount(path string, flags int) (err error)
//sys	write(fd int, p []byte) (n int, err error)
//sys   mmap(addr uintptr, length uintptr, prot int, flag int, fd int, pos int64) (ret uintptr, err error)
//sys   munmap(addr uintptr, length uintptr) (err error)
//sys	readlen(fd int, buf *byte, nbuf int) (n int, err error) = SYS_READ
//sys	writelen(fd int, buf *byte, nbuf int) (n int, err error) = SYS_WRITE

/*
 * Unimplemented
 */
// Profil
// Sigaction
// Sigprocmask
// Getlogin
// Sigpending
// Sigaltstack
// Ioctl
// Reboot
// Execve
// Vfork
// Sbrk
// Sstk
// Ovadvise
// Mincore
// Setitimer
// Swapon
// Select
// Sigsuspend
// Readv
// Writev
// Nfssvc
// Getfh
// Quotactl
// Mount
// Csops
// Waitid
// Add_profil
// Kdebug_trace
// Sigreturn
// Atsocket
// Kqueue_from_portset_np
// Kqueue_portset
// Getattrlist
// Setattrlist
// Getdirentriesattr
// Searchfs
// Delete
// Copyfile
// Watchevent
// Waitevent
// Modwatch
// Fsctl
// Initgroups
// Posix_spawn
// Nfsclnt
// Fhopen
// Minherit
// Semsys
// Msgsys
// Shmsys
// Semctl
// Semget
// Semop
// Msgctl
// Msgget
// Msgsnd
// Msgrcv
// Shmat
// Shmctl
// Shmdt
// Shmget
// Shm_open
// Shm_unlink
// Sem_open
// Sem_close
// Sem_unlink
// Sem_wait
// Sem_trywait
// Sem_post
// Sem_getvalue
// Sem_init
// Sem_destroy
// Open_extended
// Umask_extended
// Stat_extended
// Lstat_extended
// Fstat_extended
// Chmod_extended
// Fchmod_extended
// Access_extended
// Settid
// Gettid
// Setsgroups
// Getsgroups
// Setwgroups
// Getwgroups
// Mkfifo_extended
// Mkdir_extended
// Identitysvc
// Shared_region_check_np
// Shared_region_map_np
// __pthread_mutex_destroy
// __pthread_mutex_init
// __pthread_mutex_lock
// __pthread_mutex_trylock
// __pthread_mutex_unlock
// __pthread_cond_init
// __pthread_cond_destroy
// __pthread_cond_broadcast
// __pthread_cond_signal
// Setsid_with_pid
// __pthread_cond_timedwait
// Aio_fsync
// Aio_return
// Aio_suspend
// Aio_cancel
// Aio_error
// Aio_read
// Aio_write
// Lio_listio
// __pthread_cond_wait
// Iopolicysys
// __pthread_kill
// __pthread_sigmask
// __sigwait
// __disable_threadsignal
// __pthread_markcancel
// __pthread_canceled
// __semwait_signal
// Proc_info
// sendfile
// Stat64_extended
// Lstat64_extended
// Fstat64_extended
// __pthread_chdir
// __pthread_fchdir
// Audit
// Auditon
// Getauid
// Setauid
// Getaudit
// Setaudit
// Getaudit_addr
// Setaudit_addr
// Auditctl
// Bsdthread_create
// Bsdthread_terminate
// Stack_snapshot
// Bsdthread_register
// Workq_open
// Workq_ops
// __mac_execve
// __mac_syscall
// __mac_get_file
// __mac_set_file
// __mac_get_link
// __mac_set_link
// __mac_get_proc
// __mac_set_proc
// __mac_get_fd
// __mac_set_fd
// __mac_get_pid
// __mac_get_lcid
// __mac_get_lctx
// __mac_set_lctx
// Setlcid
// Read_nocancel
// Write_nocancel
// Open_nocancel
// Close_nocancel
// Wait4_nocancel
// Recvmsg_nocancel
// Sendmsg_nocancel
// Recvfrom_nocancel
// Accept_nocancel
// Fcntl_nocancel
// Select_nocancel
// Fsync_nocancel
// Connect_nocancel
// Sigsuspend_nocancel
// Readv_nocancel
// Writev_nocancel
// Sendto_nocancel
// Pread_nocancel
// Pwrite_nocancel
// Waitid_nocancel
// Poll_nocancel
// Msgsnd_nocancel
// Msgrcv_nocancel
// Sem_wait_nocancel
// Aio_suspend_nocancel
// __sigwait_nocancel
// __semwait_signal_nocancel
// __mac_mount
// __mac_get_mount
// __mac_getfsstat
