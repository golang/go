// Copyright 2009,2010 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// NetBSD system calls.
// This file is compiled as ordinary Go code,
// but it is also input to mksyscall,
// which parses the //sys lines and generates system call stubs.
// Note that sometimes we use a lowercase //sys name and wrap
// it in our own nicer implementation, either here or in
// syscall_bsd.go or syscall_unix.go.

package syscall

import "unsafe"

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

func Syscall9(trap, a1, a2, a3, a4, a5, a6, a7, a8, a9 uintptr) (r1, r2 uintptr, err Errno)

func sysctlNodes(mib []_C_int) (nodes []Sysctlnode, err error) {
	var olen uintptr

	// Get a list of all sysctl nodes below the given MIB by performing
	// a sysctl for the given MIB with CTL_QUERY appended.
	mib = append(mib, CTL_QUERY)
	qnode := Sysctlnode{Flags: SYSCTL_VERS_1}
	qp := (*byte)(unsafe.Pointer(&qnode))
	sz := unsafe.Sizeof(qnode)
	if err = sysctl(mib, nil, &olen, qp, sz); err != nil {
		return nil, err
	}

	// Now that we know the size, get the actual nodes.
	nodes = make([]Sysctlnode, olen/sz)
	np := (*byte)(unsafe.Pointer(&nodes[0]))
	if err = sysctl(mib, np, &olen, qp, sz); err != nil {
		return nil, err
	}

	return nodes, nil
}

func nametomib(name string) (mib []_C_int, err error) {

	// Split name into components.
	var parts []string
	last := 0
	for i := 0; i < len(name); i++ {
		if name[i] == '.' {
			parts = append(parts, name[last:i])
			last = i + 1
		}
	}
	parts = append(parts, name[last:])

	// Discover the nodes and construct the MIB OID.
	for partno, part := range parts {
		nodes, err := sysctlNodes(mib)
		if err != nil {
			return nil, err
		}
		for _, node := range nodes {
			n := make([]byte, 0)
			for i := range node.Name {
				if node.Name[i] != 0 {
					n = append(n, byte(node.Name[i]))
				}
			}
			if string(n) == part {
				mib = append(mib, _C_int(node.Num))
				break
			}
		}
		if len(mib) != partno+1 {
			return nil, EINVAL
		}
	}

	return mib, nil
}

// ParseDirent parses up to max directory entries in buf,
// appending the names to names. It returns the number
// bytes consumed from buf, the number of entries added
// to names, and the new names slice.
func ParseDirent(buf []byte, max int, names []string) (consumed int, count int, newnames []string) {
	origlen := len(buf)
	for max != 0 && len(buf) > 0 {
		dirent := (*Dirent)(unsafe.Pointer(&buf[0]))
		if dirent.Reclen == 0 {
			buf = nil
			break
		}
		buf = buf[dirent.Reclen:]
		if dirent.Fileno == 0 { // File absent in directory.
			continue
		}
		bytes := (*[10000]byte)(unsafe.Pointer(&dirent.Name[0]))
		var name = string(bytes[0:dirent.Namlen])
		if name == "." || name == ".." { // Useless names
			continue
		}
		max--
		count++
		names = append(names, name)
	}
	return origlen - len(buf), count, names
}

//sysnb pipe() (fd1 int, fd2 int, err error)
func Pipe(p []int) (err error) {
	if len(p) != 2 {
		return EINVAL
	}
	p[0], p[1], err = pipe()
	return
}

//sys getdents(fd int, buf []byte) (n int, err error)
func Getdirentries(fd int, buf []byte, basep *uintptr) (n int, err error) {
	return getdents(fd, buf)
}

// TODO
func sendfile(outfd int, infd int, offset *int64, count int) (written int, err error) {
	return -1, ENOSYS
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
//sysnb	Dup(fd int) (nfd int, err error)
//sysnb	Dup2(from int, to int) (err error)
//sys	Exit(code int)
//sys	Fchdir(fd int) (err error)
//sys	Fchflags(fd int, flags int) (err error)
//sys	Fchmod(fd int, mode uint32) (err error)
//sys	Fchown(fd int, uid int, gid int) (err error)
//sys	Flock(fd int, how int) (err error)
//sys	Fpathconf(fd int, name int) (val int, err error)
//sys	Fstat(fd int, stat *Stat_t) (err error)
//sys	Fsync(fd int) (err error)
//sys	Ftruncate(fd int, length int64) (err error)
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
//sys	Lstat(path string, stat *Stat_t) (err error)
//sys	Mkdir(path string, mode uint32) (err error)
//sys	Mkfifo(path string, mode uint32) (err error)
//sys	Mknod(path string, mode uint32, dev int) (err error)
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
//sysnb	Setpgid(pid int, pgid int) (err error)
//sys	Setpriority(which int, who int, prio int) (err error)
//sysnb	Setregid(rgid int, egid int) (err error)
//sysnb	Setreuid(ruid int, euid int) (err error)
//sysnb	Setrlimit(which int, lim *Rlimit) (err error)
//sysnb	Setsid() (pid int, err error)
//sysnb	Settimeofday(tp *Timeval) (err error)
//sysnb	Setuid(uid int) (err error)
//sys	Stat(path string, stat *Stat_t) (err error)
//sys	Symlink(path string, link string) (err error)
//sys	Sync() (err error)
//sys	Truncate(path string, length int64) (err error)
//sys	Umask(newmask int) (oldmask int)
//sys	Unlink(path string) (err error)
//sys	Unmount(path string, flags int) (err error)
//sys	write(fd int, p []byte) (n int, err error)
//sys	mmap(addr uintptr, length uintptr, prot int, flag int, fd int, pos int64) (ret uintptr, err error)
//sys	munmap(addr uintptr, length uintptr) (err error)
//sys	readlen(fd int, buf *byte, nbuf int) (n int, err error) = SYS_READ
//sys	writelen(fd int, buf *byte, nbuf int) (n int, err error) = SYS_WRITE

/*
 * Unimplemented
 */
// ____semctl13
// __clone
// __fhopen40
// __fhstat40
// __fhstatvfs140
// __fstat30
// __getcwd
// __getfh30
// __getlogin
// __lstat30
// __mount50
// __msgctl13
// __msync13
// __ntp_gettime30
// __posix_chown
// __posix_fadvise50
// __posix_fchown
// __posix_lchown
// __posix_rename
// __setlogin
// __shmctl13
// __sigaction_sigtramp
// __sigaltstack14
// __sigpending14
// __sigprocmask14
// __sigsuspend14
// __sigtimedwait
// __stat30
// __syscall
// __vfork14
// _ksem_close
// _ksem_destroy
// _ksem_getvalue
// _ksem_init
// _ksem_open
// _ksem_post
// _ksem_trywait
// _ksem_unlink
// _ksem_wait
// _lwp_continue
// _lwp_create
// _lwp_ctl
// _lwp_detach
// _lwp_exit
// _lwp_getname
// _lwp_getprivate
// _lwp_kill
// _lwp_park
// _lwp_self
// _lwp_setname
// _lwp_setprivate
// _lwp_suspend
// _lwp_unpark
// _lwp_unpark_all
// _lwp_wait
// _lwp_wakeup
// _pset_bind
// _sched_getaffinity
// _sched_getparam
// _sched_setaffinity
// _sched_setparam
// acct
// aio_cancel
// aio_error
// aio_fsync
// aio_read
// aio_return
// aio_suspend
// aio_write
// break
// clock_getres
// clock_gettime
// clock_settime
// compat_09_ogetdomainname
// compat_09_osetdomainname
// compat_09_ouname
// compat_10_omsgsys
// compat_10_osemsys
// compat_10_oshmsys
// compat_12_fstat12
// compat_12_getdirentries
// compat_12_lstat12
// compat_12_msync
// compat_12_oreboot
// compat_12_oswapon
// compat_12_stat12
// compat_13_sigaction13
// compat_13_sigaltstack13
// compat_13_sigpending13
// compat_13_sigprocmask13
// compat_13_sigreturn13
// compat_13_sigsuspend13
// compat_14___semctl
// compat_14_msgctl
// compat_14_shmctl
// compat_16___sigaction14
// compat_16___sigreturn14
// compat_20_fhstatfs
// compat_20_fstatfs
// compat_20_getfsstat
// compat_20_statfs
// compat_30___fhstat30
// compat_30___fstat13
// compat_30___lstat13
// compat_30___stat13
// compat_30_fhopen
// compat_30_fhstat
// compat_30_fhstatvfs1
// compat_30_getdents
// compat_30_getfh
// compat_30_ntp_gettime
// compat_30_socket
// compat_40_mount
// compat_43_fstat43
// compat_43_lstat43
// compat_43_oaccept
// compat_43_ocreat
// compat_43_oftruncate
// compat_43_ogetdirentries
// compat_43_ogetdtablesize
// compat_43_ogethostid
// compat_43_ogethostname
// compat_43_ogetkerninfo
// compat_43_ogetpagesize
// compat_43_ogetpeername
// compat_43_ogetrlimit
// compat_43_ogetsockname
// compat_43_okillpg
// compat_43_olseek
// compat_43_ommap
// compat_43_oquota
// compat_43_orecv
// compat_43_orecvfrom
// compat_43_orecvmsg
// compat_43_osend
// compat_43_osendmsg
// compat_43_osethostid
// compat_43_osethostname
// compat_43_osetrlimit
// compat_43_osigblock
// compat_43_osigsetmask
// compat_43_osigstack
// compat_43_osigvec
// compat_43_otruncate
// compat_43_owait
// compat_43_stat43
// execve
// extattr_delete_fd
// extattr_delete_file
// extattr_delete_link
// extattr_get_fd
// extattr_get_file
// extattr_get_link
// extattr_list_fd
// extattr_list_file
// extattr_list_link
// extattr_set_fd
// extattr_set_file
// extattr_set_link
// extattrctl
// fchroot
// fdatasync
// fgetxattr
// fktrace
// flistxattr
// fork
// fremovexattr
// fsetxattr
// fstatvfs1
// fsync_range
// getcontext
// getitimer
// getvfsstat
// getxattr
// ioctl
// ktrace
// lchflags
// lchmod
// lfs_bmapv
// lfs_markv
// lfs_segclean
// lfs_segwait
// lgetxattr
// lio_listio
// listxattr
// llistxattr
// lremovexattr
// lseek
// lsetxattr
// lutimes
// madvise
// mincore
// minherit
// mlock
// mlockall
// modctl
// mprotect
// mq_close
// mq_getattr
// mq_notify
// mq_open
// mq_receive
// mq_send
// mq_setattr
// mq_timedreceive
// mq_timedsend
// mq_unlink
// mremap
// msgget
// msgrcv
// msgsnd
// munlock
// munlockall
// nfssvc
// ntp_adjtime
// pmc_control
// pmc_get_info
// poll
// pollts
// preadv
// profil
// pselect
// pset_assign
// pset_create
// pset_destroy
// ptrace
// pwritev
// quotactl
// rasctl
// readv
// reboot
// removexattr
// sa_enable
// sa_preempt
// sa_register
// sa_setconcurrency
// sa_stacks
// sa_yield
// sbrk
// sched_yield
// semconfig
// semget
// semop
// setcontext
// setitimer
// setxattr
// shmat
// shmdt
// shmget
// sstk
// statvfs1
// swapctl
// sysarch
// syscall
// timer_create
// timer_delete
// timer_getoverrun
// timer_gettime
// timer_settime
// undelete
// utrace
// uuidgen
// vadvise
// vfork
// writev
