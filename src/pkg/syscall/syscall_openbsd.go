// Copyright 2009,2010 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// OpenBSD system calls.
// This file is compiled as ordinary Go code,
// but it is also input to mksyscall,
// which parses the //sys lines and generates system call stubs.
// Note that sometimes we use a lowercase //sys name and wrap
// it in our own nicer implementation, either here or in
// syscall_bsd.go or syscall_unix.go.

package syscall

import "unsafe"

const OS = "openbsd"

type SockaddrDatalink struct {
	Len    uint8
	Family uint8
	Index  uint16
	Type   uint8
	Nlen   uint8
	Alen   uint8
	Slen   uint8
	Data   [24]int8
	raw    RawSockaddrDatalink
}

func Syscall9(trap, a1, a2, a3, a4, a5, a6, a7, a8, a9 uintptr) (r1, r2, err uintptr)

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

//sysnb pipe(p *[2]_C_int) (errno int)
func Pipe(p []int) (errno int) {
	if len(p) != 2 {
		return EINVAL
	}
	var pp [2]_C_int
	errno = pipe(&pp)
	p[0] = int(pp[0])
	p[1] = int(pp[1])
	return
}

// TODO
func Sendfile(outfd int, infd int, offset *int64, count int) (written int, errno int) {
	return -1, ENOSYS
}

/*
 * Exposed directly
 */
//sys	Access(path string, mode uint32) (errno int)
//sys	Adjtime(delta *Timeval, olddelta *Timeval) (errno int)
//sys	Chdir(path string) (errno int)
//sys	Chflags(path string, flags int) (errno int)
//sys	Chmod(path string, mode uint32) (errno int)
//sys	Chown(path string, uid int, gid int) (errno int)
//sys	Chroot(path string) (errno int)
//sys	Close(fd int) (errno int)
//sysnb	Dup(fd int) (nfd int, errno int)
//sysnb	Dup2(from int, to int) (errno int)
//sys	Exit(code int)
//sys	Fchdir(fd int) (errno int)
//sys	Fchflags(path string, flags int) (errno int)
//sys	Fchmod(fd int, mode uint32) (errno int)
//sys	Fchown(fd int, uid int, gid int) (errno int)
//sys	Flock(fd int, how int) (errno int)
//sys	Fpathconf(fd int, name int) (val int, errno int)
//sys	Fstat(fd int, stat *Stat_t) (errno int)
//sys	Fstatfs(fd int, stat *Statfs_t) (errno int)
//sys	Fsync(fd int) (errno int)
//sys	Ftruncate(fd int, length int64) (errno int)
//sys	Getdirentries(fd int, buf []byte, basep *uintptr) (n int, errno int)
//sysnb	Getegid() (egid int)
//sysnb	Geteuid() (uid int)
//sys	Getfsstat(buf []Statfs_t, flags int) (n int, errno int)
//sysnb	Getgid() (gid int)
//sysnb	Getpgid(pid int) (pgid int, errno int)
//sysnb	Getpgrp() (pgrp int)
//sysnb	Getpid() (pid int)
//sysnb	Getppid() (ppid int)
//sys	Getpriority(which int, who int) (prio int, errno int)
//sysnb	Getrlimit(which int, lim *Rlimit) (errno int)
//sysnb	Getrusage(who int, rusage *Rusage) (errno int)
//sysnb	Getsid(pid int) (sid int, errno int)
//sysnb	Gettimeofday(tv *Timeval) (errno int)
//sysnb	Getuid() (uid int)
//sys	Issetugid() (tainted bool)
//sys	Kill(pid int, signum int) (errno int)
//sys	Kqueue() (fd int, errno int)
//sys	Lchown(path string, uid int, gid int) (errno int)
//sys	Link(path string, link string) (errno int)
//sys	Listen(s int, backlog int) (errno int)
//sys	Lstat(path string, stat *Stat_t) (errno int)
//sys	Mkdir(path string, mode uint32) (errno int)
//sys	Mkfifo(path string, mode uint32) (errno int)
//sys	Mknod(path string, mode uint32, dev int) (errno int)
//sys	Nanosleep(time *Timespec, leftover *Timespec) (errno int)
//sys	Open(path string, mode int, perm uint32) (fd int, errno int)
//sys	Pathconf(path string, name int) (val int, errno int)
//sys	Pread(fd int, p []byte, offset int64) (n int, errno int)
//sys	Pwrite(fd int, p []byte, offset int64) (n int, errno int)
//sys	Read(fd int, p []byte) (n int, errno int)
//sys	Readlink(path string, buf []byte) (n int, errno int)
//sys	Rename(from string, to string) (errno int)
//sys	Revoke(path string) (errno int)
//sys	Rmdir(path string) (errno int)
//sys	Seek(fd int, offset int64, whence int) (newoffset int64, errno int) = SYS_LSEEK
//sys	Select(n int, r *FdSet, w *FdSet, e *FdSet, timeout *Timeval) (errno int)
//sysnb	Setegid(egid int) (errno int)
//sysnb	Seteuid(euid int) (errno int)
//sysnb	Setgid(gid int) (errno int)
//sys	Setlogin(name string) (errno int)
//sysnb	Setpgid(pid int, pgid int) (errno int)
//sys	Setpriority(which int, who int, prio int) (errno int)
//sysnb	Setregid(rgid int, egid int) (errno int)
//sysnb	Setreuid(ruid int, euid int) (errno int)
//sysnb	Setrlimit(which int, lim *Rlimit) (errno int)
//sysnb	Setsid() (pid int, errno int)
//sysnb	Settimeofday(tp *Timeval) (errno int)
//sysnb	Setuid(uid int) (errno int)
//sys	Stat(path string, stat *Stat_t) (errno int)
//sys	Statfs(path string, stat *Statfs_t) (errno int)
//sys	Symlink(path string, link string) (errno int)
//sys	Sync() (errno int)
//sys	Truncate(path string, length int64) (errno int)
//sys	Umask(newmask int) (oldmask int)
//sys	Unlink(path string) (errno int)
//sys	Unmount(path string, flags int) (errno int)
//sys	Write(fd int, p []byte) (n int, errno int)
//sys	mmap(addr uintptr, length uintptr, prot int, flag int, fd int, pos int64) (ret uintptr, errno int)
//sys	munmap(addr uintptr, length uintptr) (errno int)
//sys	read(fd int, buf *byte, nbuf int) (n int, errno int)
//sys	write(fd int, buf *byte, nbuf int) (n int, errno int)

/*
 * Unimplemented
 */
// __getcwd
// __semctl
// __syscall
// __sysctl
// adjfreq
// break
// clock_getres
// clock_gettime
// clock_settime
// closefrom
// execve
// faccessat
// fchmodat
// fchownat
// fcntl
// fhopen
// fhstat
// fhstatfs
// fork
// fstatat
// futimens
// getfh
// getgid
// getitimer
// getlogin
// getresgid
// getresuid
// getrtable
// getthrid
// ioctl
// ktrace
// lfs_bmapv
// lfs_markv
// lfs_segclean
// lfs_segwait
// linkat
// mincore
// minherit
// mkdirat
// mkfifoat
// mknodat
// mlock
// mlockall
// mount
// mquery
// msgctl
// msgget
// msgrcv
// msgsnd
// munlock
// munlockall
// nfssvc
// nnpfspioctl
// openat
// poll
// preadv
// profil
// pwritev
// quotactl
// readlinkat
// readv
// reboot
// renameat
// rfork
// sched_yield
// semget
// semop
// setgroups
// setitimer
// setresgid
// setresuid
// setrtable
// setsockopt
// shmat
// shmctl
// shmdt
// shmget
// sigaction
// sigaltstack
// sigpending
// sigprocmask
// sigreturn
// sigsuspend
// symlinkat
// sysarch
// syscall
// threxit
// thrsigdivert
// thrsleep
// thrwakeup
// unlinkat
// utimensat
// vfork
// writev
