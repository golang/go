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
	Ptrace     bool        // Enable tracing.
	Setsid     bool        // Create session.
	Setpgid    bool        // Set process group ID to Pgid, or, if Pgid == 0, to new pid.
	Setctty    bool        // Set controlling terminal to fd Ctty
	Noctty     bool        // Detach fd 0 from controlling terminal
	Ctty       int         // Controlling TTY fd
	Foreground bool        // Place child's process group in foreground. (Implies Setpgid. Uses Ctty as fd of controlling TTY)
	Pgid       int         // Child's process group ID if Setpgid.
}

// Implemented in runtime package.
func runtime_BeforeFork()
func runtime_AfterFork()
func runtime_AfterForkInChild()

// Fork, dup fd onto 0..len(fd), and exec(argv0, argvv, envv) in child.
// If a dup or exec fails, write the errno error to pipe.
// (Pipe is close-on-exec so if exec succeeds, it will be closed.)
// In the child, this function must not acquire any locks, because
// they might have been locked at the time of the fork. This means
// no rescheduling, no malloc calls, and no new stack segments.
// For the same reason compiler does not race instrument it.
// The calls to rawSyscall are okay because they are assembly
// functions that do not grow the stack.
//go:norace
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
	r1, _, err1 = rawSyscall(funcPC(libc_fork_trampoline), 0, 0, 0)
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

	runtime_AfterForkInChild()

	// Enable tracing if requested.
	if sys.Ptrace {
		if err := ptrace(PTRACE_TRACEME, 0, 0, 0); err != nil {
			err1 = err.(Errno)
			goto childerror
		}
	}

	// Session ID
	if sys.Setsid {
		_, _, err1 = rawSyscall(funcPC(libc_setsid_trampoline), 0, 0, 0)
		if err1 != 0 {
			goto childerror
		}
	}

	// Set process group
	if sys.Setpgid || sys.Foreground {
		// Place child in process group.
		_, _, err1 = rawSyscall(funcPC(libc_setpgid_trampoline), 0, uintptr(sys.Pgid), 0)
		if err1 != 0 {
			goto childerror
		}
	}

	if sys.Foreground {
		pgrp := sys.Pgid
		if pgrp == 0 {
			r1, _, err1 = rawSyscall(funcPC(libc_getpid_trampoline), 0, 0, 0)
			if err1 != 0 {
				goto childerror
			}

			pgrp = int(r1)
		}

		// Place process group in foreground.
		_, _, err1 = rawSyscall(funcPC(libc_ioctl_trampoline), uintptr(sys.Ctty), uintptr(TIOCSPGRP), uintptr(unsafe.Pointer(&pgrp)))
		if err1 != 0 {
			goto childerror
		}
	}

	// Chroot
	if chroot != nil {
		_, _, err1 = rawSyscall(funcPC(libc_chroot_trampoline), uintptr(unsafe.Pointer(chroot)), 0, 0)
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
		if !cred.NoSetGroups {
			_, _, err1 = rawSyscall(funcPC(libc_setgroups_trampoline), ngroups, groups, 0)
			if err1 != 0 {
				goto childerror
			}
		}
		_, _, err1 = rawSyscall(funcPC(libc_setgid_trampoline), uintptr(cred.Gid), 0, 0)
		if err1 != 0 {
			goto childerror
		}
		_, _, err1 = rawSyscall(funcPC(libc_setuid_trampoline), uintptr(cred.Uid), 0, 0)
		if err1 != 0 {
			goto childerror
		}
	}

	// Chdir
	if dir != nil {
		_, _, err1 = rawSyscall(funcPC(libc_chdir_trampoline), uintptr(unsafe.Pointer(dir)), 0, 0)
		if err1 != 0 {
			goto childerror
		}
	}

	// Pass 1: look for fd[i] < i and move those up above len(fd)
	// so that pass 2 won't stomp on an fd it needs later.
	if pipe < nextfd {
		_, _, err1 = rawSyscall(funcPC(libc_dup2_trampoline), uintptr(pipe), uintptr(nextfd), 0)
		if err1 != 0 {
			goto childerror
		}
		rawSyscall(funcPC(libc_fcntl_trampoline), uintptr(nextfd), F_SETFD, FD_CLOEXEC)
		pipe = nextfd
		nextfd++
	}
	for i = 0; i < len(fd); i++ {
		if fd[i] >= 0 && fd[i] < int(i) {
			if nextfd == pipe { // don't stomp on pipe
				nextfd++
			}
			_, _, err1 = rawSyscall(funcPC(libc_dup2_trampoline), uintptr(fd[i]), uintptr(nextfd), 0)
			if err1 != 0 {
				goto childerror
			}
			rawSyscall(funcPC(libc_fcntl_trampoline), uintptr(nextfd), F_SETFD, FD_CLOEXEC)
			fd[i] = nextfd
			nextfd++
		}
	}

	// Pass 2: dup fd[i] down onto i.
	for i = 0; i < len(fd); i++ {
		if fd[i] == -1 {
			rawSyscall(funcPC(libc_close_trampoline), uintptr(i), 0, 0)
			continue
		}
		if fd[i] == int(i) {
			// dup2(i, i) won't clear close-on-exec flag on Linux,
			// probably not elsewhere either.
			_, _, err1 = rawSyscall(funcPC(libc_fcntl_trampoline), uintptr(fd[i]), F_SETFD, 0)
			if err1 != 0 {
				goto childerror
			}
			continue
		}
		// The new fd is created NOT close-on-exec,
		// which is exactly what we want.
		_, _, err1 = rawSyscall(funcPC(libc_dup2_trampoline), uintptr(fd[i]), uintptr(i), 0)
		if err1 != 0 {
			goto childerror
		}
	}

	// By convention, we don't close-on-exec the fds we are
	// started with, so if len(fd) < 3, close 0, 1, 2 as needed.
	// Programs that know they inherit fds >= 3 will need
	// to set them close-on-exec.
	for i = len(fd); i < 3; i++ {
		rawSyscall(funcPC(libc_close_trampoline), uintptr(i), 0, 0)
	}

	// Detach fd 0 from tty
	if sys.Noctty {
		_, _, err1 = rawSyscall(funcPC(libc_ioctl_trampoline), 0, uintptr(TIOCNOTTY), 0)
		if err1 != 0 {
			goto childerror
		}
	}

	// Set the controlling TTY to Ctty
	if sys.Setctty {
		_, _, err1 = rawSyscall(funcPC(libc_ioctl_trampoline), uintptr(sys.Ctty), uintptr(TIOCSCTTY), 0)
		if err1 != 0 {
			goto childerror
		}
	}

	// Time to exec.
	_, _, err1 = rawSyscall(funcPC(libc_execve_trampoline),
		uintptr(unsafe.Pointer(argv0)),
		uintptr(unsafe.Pointer(&argv[0])),
		uintptr(unsafe.Pointer(&envv[0])))

childerror:
	// send error code on pipe
	rawSyscall(funcPC(libc_write_trampoline), uintptr(pipe), uintptr(unsafe.Pointer(&err1)), unsafe.Sizeof(err1))
	for {
		rawSyscall(funcPC(libc_exit_trampoline), 253, 0, 0)
	}
}
