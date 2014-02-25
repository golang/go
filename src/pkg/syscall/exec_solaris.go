// Copyright 2011 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package syscall

import (
	"unsafe"
)

type SysProcAttr struct {
	Chroot     string      // Chroot.
	Credential *Credential // Credential.
	Setsid     bool        // Create session.
	Setpgid    bool        // Set process group ID to new pid (SYSV setpgrp)
	Setctty    bool        // Set controlling terminal to fd 0
	Noctty     bool        // Detach fd 0 from controlling terminal
}

// Implemented in runtime package.
func runtime_BeforeFork()
func runtime_AfterFork()

func chdir(path uintptr) (err Errno)
func chroot1(path uintptr) (err Errno)
func close(fd uintptr) (err Errno)
func execve(path uintptr, argv uintptr, envp uintptr) (err Errno)
func exit(code uintptr)
func fcntl1(fd uintptr, cmd uintptr, arg uintptr) (val uintptr, err Errno)
func forkx(flags uintptr) (pid uintptr, err Errno)
func ioctl(fd uintptr, req uintptr, arg uintptr) (err Errno)
func setgid(gid uintptr) (err Errno)
func setgroups1(ngid uintptr, gid uintptr) (err Errno)
func setsid() (pid uintptr, err Errno)
func setuid(uid uintptr) (err Errno)
func setpgid(pid uintptr, pgid uintptr) (err Errno)
func write1(fd uintptr, buf uintptr, nbyte uintptr) (n uintptr, err Errno)

// Fork, dup fd onto 0..len(fd), and exec(argv0, argvv, envv) in child.
// If a dup or exec fails, write the errno error to pipe.
// (Pipe is close-on-exec so if exec succeeds, it will be closed.)
// In the child, this function must not acquire any locks, because
// they might have been locked at the time of the fork.  This means
// no rescheduling, no malloc calls, and no new stack segments.
//
// We call hand-crafted syscalls, implemented in
// ../runtime/syscall_solaris.goc, rather than generated libc wrappers
// because we need to avoid lazy-loading the functions (might malloc,
// split the stack, or acquire mutexes). We can't call RawSyscall
// because it's not safe even for BSD-subsystem calls.
func forkAndExecInChild(argv0 *byte, argv, envv []*byte, chroot, dir *byte, attr *ProcAttr, sys *SysProcAttr, pipe int) (pid int, err Errno) {
	// Declare all variables at top in case any
	// declarations require heap allocation (e.g., err1).
	var (
		r1     uintptr
		err1   Errno
		nextfd int
		i      int
	)

	// guard against side effects of shuffling fds below.
	// Make sure that nextfd is beyond any currently open files so
	// that we can't run the risk of overwriting any of them.
	fd := make([]int, len(attr.Files))
	nextfd = len(attr.Files)
	for i, ufd := range attr.Files {
		if nextfd < int(ufd) {
			nextfd = int(ufd)
		}
		fd[i] = int(ufd)
	}
	nextfd++

	// About to call fork.
	// No more allocation or calls of non-assembly functions.
	runtime_BeforeFork()
	r1, err1 = forkx(0x1) // FORK_NOSIGCHLD
	if err1 != 0 {
		runtime_AfterFork()
		return 0, err1
	}

	if r1 != 0 {
		// parent; return PID
		runtime_AfterFork()
		return int(r1), 0
	}

	// Fork succeeded, now in child.

	// Session ID
	if sys.Setsid {
		_, err1 = setsid()
		if err1 != 0 {
			goto childerror
		}
	}

	// Set process group
	if sys.Setpgid {
		err1 = setpgid(0, 0)
		if err1 != 0 {
			goto childerror
		}
	}

	// Chroot
	if chroot != nil {
		err1 = chroot1(uintptr(unsafe.Pointer(chroot)))
		if err1 != 0 {
			goto childerror
		}
	}

	// User and groups
	if cred := sys.Credential; cred != nil {
		ngroups := uintptr(len(cred.Groups))
		groups := uintptr(0)
		if ngroups > 0 {
			groups = uintptr(unsafe.Pointer(&cred.Groups[0]))
		}
		err1 = setgroups1(ngroups, groups)
		if err1 != 0 {
			goto childerror
		}
		err1 = setgid(uintptr(cred.Gid))
		if err1 != 0 {
			goto childerror
		}
		err1 = setuid(uintptr(cred.Uid))
		if err1 != 0 {
			goto childerror
		}
	}

	// Chdir
	if dir != nil {
		err1 = chdir(uintptr(unsafe.Pointer(dir)))
		if err1 != 0 {
			goto childerror
		}
	}

	// Pass 1: look for fd[i] < i and move those up above len(fd)
	// so that pass 2 won't stomp on an fd it needs later.
	if pipe < nextfd {
		_, err1 = fcntl1(uintptr(pipe), F_DUP2FD, uintptr(nextfd))
		if err1 != 0 {
			goto childerror
		}
		fcntl1(uintptr(nextfd), F_SETFD, FD_CLOEXEC)
		pipe = nextfd
		nextfd++
	}
	for i = 0; i < len(fd); i++ {
		if fd[i] >= 0 && fd[i] < int(i) {
			_, err1 = fcntl1(uintptr(fd[i]), F_DUP2FD, uintptr(nextfd))
			if err1 != 0 {
				goto childerror
			}
			fcntl1(uintptr(nextfd), F_SETFD, FD_CLOEXEC)
			fd[i] = nextfd
			nextfd++
			if nextfd == pipe { // don't stomp on pipe
				nextfd++
			}
		}
	}

	// Pass 2: dup fd[i] down onto i.
	for i = 0; i < len(fd); i++ {
		if fd[i] == -1 {
			close(uintptr(i))
			continue
		}
		if fd[i] == int(i) {
			// dup2(i, i) won't clear close-on-exec flag on Linux,
			// probably not elsewhere either.
			_, err1 = fcntl1(uintptr(fd[i]), F_SETFD, 0)
			if err1 != 0 {
				goto childerror
			}
			continue
		}
		// The new fd is created NOT close-on-exec,
		// which is exactly what we want.
		_, err1 = fcntl1(uintptr(fd[i]), F_DUP2FD, uintptr(i))
		if err1 != 0 {
			goto childerror
		}
	}

	// By convention, we don't close-on-exec the fds we are
	// started with, so if len(fd) < 3, close 0, 1, 2 as needed.
	// Programs that know they inherit fds >= 3 will need
	// to set them close-on-exec.
	for i = len(fd); i < 3; i++ {
		close(uintptr(i))
	}

	// Detach fd 0 from tty
	if sys.Noctty {
		err1 = ioctl(0, uintptr(TIOCNOTTY), 0)
		if err1 != 0 {
			goto childerror
		}
	}

	// Make fd 0 the tty
	if sys.Setctty {
		err1 = ioctl(0, uintptr(TIOCSCTTY), 0)
		if err1 != 0 {
			goto childerror
		}
	}

	// Time to exec.
	err1 = execve(
		uintptr(unsafe.Pointer(argv0)),
		uintptr(unsafe.Pointer(&argv[0])),
		uintptr(unsafe.Pointer(&envv[0])))

childerror:
	// send error code on pipe
	write1(uintptr(pipe), uintptr(unsafe.Pointer(&err1)), unsafe.Sizeof(err1))
	for {
		exit(253)
	}
}

// Try to open a pipe with O_CLOEXEC set on both file descriptors.
func forkExecPipe(p []int) error {
	err := Pipe(p)
	if err != nil {
		return err
	}
	_, err = fcntl(p[0], F_SETFD, FD_CLOEXEC)
	if err != nil {
		return err
	}
	_, err = fcntl(p[1], F_SETFD, FD_CLOEXEC)
	return err
}
