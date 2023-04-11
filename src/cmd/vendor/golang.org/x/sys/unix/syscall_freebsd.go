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

package unix

import (
	"sync"
	"unsafe"
)

// See https://www.freebsd.org/doc/en_US.ISO8859-1/books/porters-handbook/versions.html.
var (
	osreldateOnce sync.Once
	osreldate     uint32
)

func supportsABI(ver uint32) bool {
	osreldateOnce.Do(func() { osreldate, _ = SysctlUint32("kern.osreldate") })
	return osreldate >= ver
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
	Data   [46]int8
	raw    RawSockaddrDatalink
}

func anyToSockaddrGOOS(fd int, rsa *RawSockaddrAny) (Sockaddr, error) {
	return nil, EAFNOSUPPORT
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

func Pipe(p []int) (err error) {
	return Pipe2(p, 0)
}

//sysnb	pipe2(p *[2]_C_int, flags int) (err error)

func Pipe2(p []int, flags int) error {
	if len(p) != 2 {
		return EINVAL
	}
	var pp [2]_C_int
	err := pipe2(&pp, flags)
	if err == nil {
		p[0] = int(pp[0])
		p[1] = int(pp[1])
	}
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

// GetsockoptXucred is a getsockopt wrapper that returns an Xucred struct.
// The usual level and opt are SOL_LOCAL and LOCAL_PEERCRED, respectively.
func GetsockoptXucred(fd, level, opt int) (*Xucred, error) {
	x := new(Xucred)
	vallen := _Socklen(SizeofXucred)
	err := getsockopt(fd, level, opt, unsafe.Pointer(x), &vallen)
	return x, err
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
	sa, err = anyToSockaddr(fd, &rsa)
	if err != nil {
		Close(nfd)
		nfd = 0
	}
	return
}

//sys	Getcwd(buf []byte) (n int, err error) = SYS___GETCWD

func Getfsstat(buf []Statfs_t, flags int) (n int, err error) {
	var (
		_p0     unsafe.Pointer
		bufsize uintptr
	)
	if len(buf) > 0 {
		_p0 = unsafe.Pointer(&buf[0])
		bufsize = unsafe.Sizeof(Statfs_t{}) * uintptr(len(buf))
	}
	r0, _, e1 := Syscall(SYS_GETFSSTAT, uintptr(_p0), bufsize, uintptr(flags))
	n = int(r0)
	if e1 != 0 {
		err = e1
	}
	return
}

//sys	ioctl(fd int, req uint, arg uintptr) (err error) = SYS_IOCTL
//sys	ioctlPtr(fd int, req uint, arg unsafe.Pointer) (err error) = SYS_IOCTL

//sys	sysctl(mib []_C_int, old *byte, oldlen *uintptr, new *byte, newlen uintptr) (err error) = SYS___SYSCTL

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

func Stat(path string, st *Stat_t) (err error) {
	return Fstatat(AT_FDCWD, path, st, 0)
}

func Lstat(path string, st *Stat_t) (err error) {
	return Fstatat(AT_FDCWD, path, st, AT_SYMLINK_NOFOLLOW)
}

func Getdents(fd int, buf []byte) (n int, err error) {
	return Getdirentries(fd, buf, nil)
}

func Getdirentries(fd int, buf []byte, basep *uintptr) (n int, err error) {
	if basep == nil || unsafe.Sizeof(*basep) == 8 {
		return getdirentries(fd, buf, (*uint64)(unsafe.Pointer(basep)))
	}
	// The syscall needs a 64-bit base. On 32-bit machines
	// we can't just use the basep passed in. See #32498.
	var base uint64 = uint64(*basep)
	n, err = getdirentries(fd, buf, &base)
	*basep = uintptr(base)
	if base>>32 != 0 {
		// We can't stuff the base back into a uintptr, so any
		// future calls would be suspect. Generate an error.
		// EIO is allowed by getdirentries.
		err = EIO
	}
	return
}

func Mknod(path string, mode uint32, dev uint64) (err error) {
	return Mknodat(AT_FDCWD, path, mode, dev)
}

func Sendfile(outfd int, infd int, offset *int64, count int) (written int, err error) {
	if raceenabled {
		raceReleaseMerge(unsafe.Pointer(&ioSync))
	}
	return sendfile(outfd, infd, offset, count)
}

//sys	ptrace(request int, pid int, addr uintptr, data int) (err error)
//sys	ptracePtr(request int, pid int, addr unsafe.Pointer, data int) (err error) = SYS_PTRACE

func PtraceAttach(pid int) (err error) {
	return ptrace(PT_ATTACH, pid, 0, 0)
}

func PtraceCont(pid int, signal int) (err error) {
	return ptrace(PT_CONTINUE, pid, 1, signal)
}

func PtraceDetach(pid int) (err error) {
	return ptrace(PT_DETACH, pid, 1, 0)
}

func PtraceGetFpRegs(pid int, fpregsout *FpReg) (err error) {
	return ptracePtr(PT_GETFPREGS, pid, unsafe.Pointer(fpregsout), 0)
}

func PtraceGetRegs(pid int, regsout *Reg) (err error) {
	return ptracePtr(PT_GETREGS, pid, unsafe.Pointer(regsout), 0)
}

func PtraceIO(req int, pid int, offs uintptr, out []byte, countin int) (count int, err error) {
	ioDesc := PtraceIoDesc{
		Op:   int32(req),
		Offs: offs,
	}
	if countin > 0 {
		_ = out[:countin] // check bounds
		ioDesc.Addr = &out[0]
	} else if out != nil {
		ioDesc.Addr = (*byte)(unsafe.Pointer(&_zero))
	}
	ioDesc.SetLen(countin)

	err = ptracePtr(PT_IO, pid, unsafe.Pointer(&ioDesc), 0)
	return int(ioDesc.Len), err
}

func PtraceLwpEvents(pid int, enable int) (err error) {
	return ptrace(PT_LWP_EVENTS, pid, 0, enable)
}

func PtraceLwpInfo(pid int, info *PtraceLwpInfoStruct) (err error) {
	return ptracePtr(PT_LWPINFO, pid, unsafe.Pointer(info), int(unsafe.Sizeof(*info)))
}

func PtracePeekData(pid int, addr uintptr, out []byte) (count int, err error) {
	return PtraceIO(PIOD_READ_D, pid, addr, out, SizeofLong)
}

func PtracePeekText(pid int, addr uintptr, out []byte) (count int, err error) {
	return PtraceIO(PIOD_READ_I, pid, addr, out, SizeofLong)
}

func PtracePokeData(pid int, addr uintptr, data []byte) (count int, err error) {
	return PtraceIO(PIOD_WRITE_D, pid, addr, data, SizeofLong)
}

func PtracePokeText(pid int, addr uintptr, data []byte) (count int, err error) {
	return PtraceIO(PIOD_WRITE_I, pid, addr, data, SizeofLong)
}

func PtraceSetRegs(pid int, regs *Reg) (err error) {
	return ptracePtr(PT_SETREGS, pid, unsafe.Pointer(regs), 0)
}

func PtraceSingleStep(pid int) (err error) {
	return ptrace(PT_STEP, pid, 1, 0)
}

func Dup3(oldfd, newfd, flags int) error {
	if oldfd == newfd || flags&^O_CLOEXEC != 0 {
		return EINVAL
	}
	how := F_DUP2FD
	if flags&O_CLOEXEC != 0 {
		how = F_DUP2FD_CLOEXEC
	}
	_, err := fcntl(oldfd, how, newfd)
	return err
}

/*
 * Exposed directly
 */
//sys	Access(path string, mode uint32) (err error)
//sys	Adjtime(delta *Timeval, olddelta *Timeval) (err error)
//sys	CapEnter() (err error)
//sys	capRightsGet(version int, fd int, rightsp *CapRights) (err error) = SYS___CAP_RIGHTS_GET
//sys	capRightsLimit(fd int, rightsp *CapRights) (err error)
//sys	Chdir(path string) (err error)
//sys	Chflags(path string, flags int) (err error)
//sys	Chmod(path string, mode uint32) (err error)
//sys	Chown(path string, uid int, gid int) (err error)
//sys	Chroot(path string) (err error)
//sys	ClockGettime(clockid int32, time *Timespec) (err error)
//sys	Close(fd int) (err error)
//sys	Dup(fd int) (nfd int, err error)
//sys	Dup2(from int, to int) (err error)
//sys	Exit(code int)
//sys	ExtattrGetFd(fd int, attrnamespace int, attrname string, data uintptr, nbytes int) (ret int, err error)
//sys	ExtattrSetFd(fd int, attrnamespace int, attrname string, data uintptr, nbytes int) (ret int, err error)
//sys	ExtattrDeleteFd(fd int, attrnamespace int, attrname string) (err error)
//sys	ExtattrListFd(fd int, attrnamespace int, data uintptr, nbytes int) (ret int, err error)
//sys	ExtattrGetFile(file string, attrnamespace int, attrname string, data uintptr, nbytes int) (ret int, err error)
//sys	ExtattrSetFile(file string, attrnamespace int, attrname string, data uintptr, nbytes int) (ret int, err error)
//sys	ExtattrDeleteFile(file string, attrnamespace int, attrname string) (err error)
//sys	ExtattrListFile(file string, attrnamespace int, data uintptr, nbytes int) (ret int, err error)
//sys	ExtattrGetLink(link string, attrnamespace int, attrname string, data uintptr, nbytes int) (ret int, err error)
//sys	ExtattrSetLink(link string, attrnamespace int, attrname string, data uintptr, nbytes int) (ret int, err error)
//sys	ExtattrDeleteLink(link string, attrnamespace int, attrname string) (err error)
//sys	ExtattrListLink(link string, attrnamespace int, data uintptr, nbytes int) (ret int, err error)
//sys	Fadvise(fd int, offset int64, length int64, advice int) (err error) = SYS_POSIX_FADVISE
//sys	Faccessat(dirfd int, path string, mode uint32, flags int) (err error)
//sys	Fchdir(fd int) (err error)
//sys	Fchflags(fd int, flags int) (err error)
//sys	Fchmod(fd int, mode uint32) (err error)
//sys	Fchmodat(dirfd int, path string, mode uint32, flags int) (err error)
//sys	Fchown(fd int, uid int, gid int) (err error)
//sys	Fchownat(dirfd int, path string, uid int, gid int, flags int) (err error)
//sys	Flock(fd int, how int) (err error)
//sys	Fpathconf(fd int, name int) (val int, err error)
//sys	Fstat(fd int, stat *Stat_t) (err error)
//sys	Fstatat(fd int, path string, stat *Stat_t, flags int) (err error)
//sys	Fstatfs(fd int, stat *Statfs_t) (err error)
//sys	Fsync(fd int) (err error)
//sys	Ftruncate(fd int, length int64) (err error)
//sys	getdirentries(fd int, buf []byte, basep *uint64) (n int, err error)
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
//sys	Kill(pid int, signum syscall.Signal) (err error)
//sys	Kqueue() (fd int, err error)
//sys	Lchown(path string, uid int, gid int) (err error)
//sys	Link(path string, link string) (err error)
//sys	Linkat(pathfd int, path string, linkfd int, link string, flags int) (err error)
//sys	Listen(s int, backlog int) (err error)
//sys	Mkdir(path string, mode uint32) (err error)
//sys	Mkdirat(dirfd int, path string, mode uint32) (err error)
//sys	Mkfifo(path string, mode uint32) (err error)
//sys	Mknodat(fd int, path string, mode uint32, dev uint64) (err error)
//sys	Nanosleep(time *Timespec, leftover *Timespec) (err error)
//sys	Open(path string, mode int, perm uint32) (fd int, err error)
//sys	Openat(fdat int, path string, mode int, perm uint32) (fd int, err error)
//sys	Pathconf(path string, name int) (val int, err error)
//sys	pread(fd int, p []byte, offset int64) (n int, err error)
//sys	pwrite(fd int, p []byte, offset int64) (n int, err error)
//sys	read(fd int, p []byte) (n int, err error)
//sys	Readlink(path string, buf []byte) (n int, err error)
//sys	Readlinkat(dirfd int, path string, buf []byte) (n int, err error)
//sys	Rename(from string, to string) (err error)
//sys	Renameat(fromfd int, from string, tofd int, to string) (err error)
//sys	Revoke(path string) (err error)
//sys	Rmdir(path string) (err error)
//sys	Seek(fd int, offset int64, whence int) (newoffset int64, err error) = SYS_LSEEK
//sys	Select(nfd int, r *FdSet, w *FdSet, e *FdSet, timeout *Timeval) (n int, err error)
//sysnb	Setegid(egid int) (err error)
//sysnb	Seteuid(euid int) (err error)
//sysnb	Setgid(gid int) (err error)
//sys	Setlogin(name string) (err error)
//sysnb	Setpgid(pid int, pgid int) (err error)
//sys	Setpriority(which int, who int, prio int) (err error)
//sysnb	Setregid(rgid int, egid int) (err error)
//sysnb	Setreuid(ruid int, euid int) (err error)
//sysnb	Setresgid(rgid int, egid int, sgid int) (err error)
//sysnb	Setresuid(ruid int, euid int, suid int) (err error)
//sysnb	Setsid() (pid int, err error)
//sysnb	Settimeofday(tp *Timeval) (err error)
//sysnb	Setuid(uid int) (err error)
//sys	Statfs(path string, stat *Statfs_t) (err error)
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
//sys	mmap(addr uintptr, length uintptr, prot int, flag int, fd int, pos int64) (ret uintptr, err error)
//sys	munmap(addr uintptr, length uintptr) (err error)
//sys	readlen(fd int, buf *byte, nbuf int) (n int, err error) = SYS_READ
//sys	writelen(fd int, buf *byte, nbuf int) (n int, err error) = SYS_WRITE
//sys	accept4(fd int, rsa *RawSockaddrAny, addrlen *_Socklen, flags int) (nfd int, err error)
//sys	utimensat(dirfd int, path string, times *[2]Timespec, flags int) (err error)

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
// Getdents
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
