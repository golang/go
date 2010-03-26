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

package syscall

const OS = "darwin"

/*
 * Wrapped
 */

//sys	kill(pid int, signum int, posix int) (errno int)

func Kill(pid int, signum int) (errno int) { return kill(pid, signum, 1) }

/*
 * Exposed directly
 */
//sys	Access(path string, flags int) (errno int)
//sys	Adjtime(delta *Timeval, olddelta *Timeval) (errno int)
//sys	Chdir(path string) (errno int)
//sys	Chflags(path string, flags int) (errno int)
//sys	Chmod(path string, mode int) (errno int)
//sys	Chown(path string, uid int, gid int) (errno int)
//sys	Chroot(path string) (errno int)
//sys	Close(fd int) (errno int)
//sys	Dup(fd int) (nfd int, errno int)
//sys	Dup2(from int, to int) (errno int)
//sys	Exchangedata(path1 string, path2 string, options int) (errno int)
//sys	Exit(code int)
//sys	Fchdir(fd int) (errno int)
//sys	Fchflags(path string, flags int) (errno int)
//sys	Fchmod(fd int, mode int) (errno int)
//sys	Fchown(fd int, uid int, gid int) (errno int)
//sys	Flock(fd int, how int) (errno int)
//sys	Fpathconf(fd int, name int) (val int, errno int)
//sys	Fstat(fd int, stat *Stat_t) (errno int) = SYS_FSTAT64
//sys	Fstatfs(fd int, stat *Statfs_t) (errno int) = SYS_FSTATFS64
//sys	Fsync(fd int) (errno int)
//sys	Ftruncate(fd int, length int64) (errno int)
//sys	Getdirentries(fd int, buf []byte, basep *uintptr) (n int, errno int) = SYS_GETDIRENTRIES64
//sys	Getdtablesize() (size int)
//sys	Getegid() (egid int)
//sys	Geteuid() (uid int)
//sys	Getfsstat(buf []Statfs_t, flags int) (n int, errno int) = SYS_GETFSSTAT64
//sys	Getgid() (gid int)
//sys	Getpgid(pid int) (pgid int, errno int)
//sys	Getpgrp() (pgrp int)
//sys	Getpid() (pid int)
//sys	Getppid() (ppid int)
//sys	Getpriority(which int, who int) (prio int, errno int)
//sys	Getrlimit(which int, lim *Rlimit) (errno int)
//sys	Getrusage(who int, rusage *Rusage) (errno int)
//sys	Getsid(pid int) (sid int, errno int)
//sys	Getuid() (uid int)
//sys	Issetugid() (tainted bool)
//sys	Kqueue() (fd int, errno int)
//sys	Lchown(path string, uid int, gid int) (errno int)
//sys	Link(path string, link string) (errno int)
//sys	Listen(s int, backlog int) (errno int)
//sys	Lstat(path string, stat *Stat_t) (errno int) = SYS_LSTAT64
//sys	Mkdir(path string, mode int) (errno int)
//sys	Mkfifo(path string, mode int) (errno int)
//sys	Mknod(path string, mode int, dev int) (errno int)
//sys	Open(path string, mode int, perm int) (fd int, errno int)
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
//sys	Setegid(egid int) (errno int)
//sys	Seteuid(euid int) (errno int)
//sys	Setgid(gid int) (errno int)
//sys	Setlogin(name string) (errno int)
//sys	Setpgid(pid int, pgid int) (errno int)
//sys	Setpriority(which int, who int, prio int) (errno int)
//sys	Setprivexec(flag int) (errno int)
//sys	Setregid(rgid int, egid int) (errno int)
//sys	Setreuid(ruid int, euid int) (errno int)
//sys	Setrlimit(which int, lim *Rlimit) (errno int)
//sys	Setsid() (pid int, errno int)
//sys	Settimeofday(tp *Timeval) (errno int)
//sys	Setuid(uid int) (errno int)
//sys	Stat(path string, stat *Stat_t) (errno int) = SYS_STAT64
//sys	Statfs(path string, stat *Statfs_t) (errno int) = SYS_STATFS64
//sys	Symlink(path string, link string) (errno int)
//sys	Sync() (errno int)
//sys	Truncate(path string, length int64) (errno int)
//sys	Umask(newmask int) (oldmask int)
//sys	Undelete(path string) (errno int)
//sys	Unlink(path string) (errno int)
//sys	Unmount(path string, flags int) (errno int)
//sys	Write(fd int, p []byte) (n int, errno int)
//sys	read(fd int, buf *byte, nbuf int) (n int, errno int)
//sys	write(fd int, buf *byte, nbuf int) (n int, errno int)


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
// Mmap
// Mlock
// Munlock
// Atsocket
// Kqueue_from_portset_np
// Kqueue_portset
// Getattrlist
// Setattrlist
// Getdirentriesattr
// Searchfs
// Delete
// Copyfile
// Poll
// Watchevent
// Waitevent
// Modwatch
// Getxattr
// Fgetxattr
// Setxattr
// Fsetxattr
// Removexattr
// Fremovexattr
// Listxattr
// Flistxattr
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
// Mlockall
// Munlockall
// __pthread_kill
// __pthread_sigmask
// __sigwait
// __disable_threadsignal
// __pthread_markcancel
// __pthread_canceled
// __semwait_signal
// Proc_info
// Sendfile
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
// Msync_nocancel
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
