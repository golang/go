// Copyright 2011 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

//go:build dragonfly || netbsd || (openbsd && mips64)

package syscall

import (
	"runtime"
	"unsafe"
)

type SysProcAttr struct {
	Chroot     string      // Chroot.
	Credential *Credential // Credential.
	Ptrace     bool        // Enable tracing.
	Setsid     bool        // Create session.
	// Setpgid sets the process group ID of the child to Pgid,
	// or, if Pgid == 0, to the new child's process ID.
	Setpgid bool
	// Setctty sets the controlling terminal of the child to
	// file descriptor Ctty. Ctty must be a descriptor number
	// in the child process: an index into ProcAttr.Files.
	// This is only meaningful if Setsid is true.
	Setctty bool
	Noctty  bool // Detach fd 0 from controlling terminal
	Ctty    int  // Controlling TTY fd
	// Foreground places the child process group in the foreground.
	// This implies Setpgid. The Ctty field must be set to
	// the descriptor of the controlling TTY.
	// Unlike Setctty, in this case Ctty must be a descriptor
	// number in the parent process.
	Foreground bool
	Pgid       int // Child's process group ID if Setpgid.
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
// The calls to RawSyscall are okay because they are assembly
// functions that do not grow the stack.
//
//go:norace
func forkAndExecInChild(argv0 *byte, argv, envv []*byte, chroot, dir *byte, attr *ProcAttr, sys *SysProcAttr, pipe int) (pid int, err Errno) {
	// Declare all variables at top in case any
	// declarations require heap allocation (e.g., err1).
	var (
		r1              uintptr
		err1            Errno
		nextfd          int
		i               int
		pgrp            _C_int
		cred            *Credential
		ngroups, groups uintptr
	)

	rlim := origRlimitNofile.Load()

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
	r1, _, err1 = RawSyscall(SYS_FORK, 0, 0, 0)
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

	// Enable tracing if requested.
	if sys.Ptrace {
		_, _, err1 = RawSyscall(SYS_PTRACE, uintptr(PTRACE_TRACEME), 0, 0)
		if err1 != 0 {
			goto childerror
		}
	}

	// Session ID
	if sys.Setsid {
		_, _, err1 = RawSyscall(SYS_SETSID, 0, 0, 0)
		if err1 != 0 {
			goto childerror
		}
	}

	// Set process group
	if sys.Setpgid || sys.Foreground {
		// Place child in process group.
		_, _, err1 = RawSyscall(SYS_SETPGID, 0, uintptr(sys.Pgid), 0)
		if err1 != 0 {
			goto childerror
		}
	}

	if sys.Foreground {
		// This should really be pid_t, however _C_int (aka int32) is
		// generally equivalent.
		pgrp = _C_int(sys.Pgid)
		if pgrp == 0 {
			r1, _, err1 = RawSyscall(SYS_GETPID, 0, 0, 0)
			if err1 != 0 {
				goto childerror
			}

			pgrp = _C_int(r1)
		}

		// Place process group in foreground.
		_, _, err1 = RawSyscall(SYS_IOCTL, uintptr(sys.Ctty), uintptr(TIOCSPGRP), uintptr(unsafe.Pointer(&pgrp)))
		if err1 != 0 {
			goto childerror
		}
	}

	// Restore the signal mask. We do this after TIOCSPGRP to avoid
	// having the kernel send a SIGTTOU signal to the process group.
	runtime_AfterForkInChild()

	// Chroot
	if chroot != nil {
		_, _, err1 = RawSyscall(SYS_CHROOT, uintptr(unsafe.Pointer(chroot)), 0, 0)
		if err1 != 0 {
			goto childerror
		}
	}

	// User and groups
	if cred = sys.Credential; cred != nil {
		ngroups = uintptr(len(cred.Groups))
		groups = uintptr(0)
		if ngroups > 0 {
			groups = uintptr(unsafe.Pointer(&cred.Groups[0]))
		}
		if !cred.NoSetGroups {
			_, _, err1 = RawSyscall(SYS_SETGROUPS, ngroups, groups, 0)
			if err1 != 0 {
				goto childerror
			}
		}
		_, _, err1 = RawSyscall(SYS_SETGID, uintptr(cred.Gid), 0, 0)
		if err1 != 0 {
			goto childerror
		}
		_, _, err1 = RawSyscall(SYS_SETUID, uintptr(cred.Uid), 0, 0)
		if err1 != 0 {
			goto childerror
		}
	}

	// Chdir
	if dir != nil {
		_, _, err1 = RawSyscall(SYS_CHDIR, uintptr(unsafe.Pointer(dir)), 0, 0)
		if err1 != 0 {
			goto childerror
		}
	}

	// Pass 1: look for fd[i] < i and move those up above len(fd)
	// so that pass 2 won't stomp on an fd it needs later.
	if pipe < nextfd {
		if runtime.GOOS == "netbsd" || (runtime.GOOS == "openbsd" && runtime.GOARCH == "mips64") {
			_, _, err1 = RawSyscall(_SYS_DUP3, uintptr(pipe), uintptr(nextfd), O_CLOEXEC)
		} else if runtime.GOOS == "dragonfly" {
			_, _, err1 = RawSyscall(SYS_FCNTL, uintptr(pipe), _F_DUP2FD_CLOEXEC, uintptr(nextfd))
		} else {
			_, _, err1 = RawSyscall(SYS_DUP2, uintptr(pipe), uintptr(nextfd), 0)
			if err1 != 0 {
				goto childerror
			}
			_, _, err1 = RawSyscall(SYS_FCNTL, uintptr(nextfd), F_SETFD, FD_CLOEXEC)
		}
		if err1 != 0 {
			goto childerror
		}
		pipe = nextfd
		nextfd++
	}
	for i = 0; i < len(fd); i++ {
		if fd[i] >= 0 && fd[i] < i {
			if nextfd == pipe { // don't stomp on pipe
				nextfd++
			}
			if runtime.GOOS == "netbsd" || (runtime.GOOS == "openbsd" && runtime.GOARCH == "mips64") {
				_, _, err1 = RawSyscall(_SYS_DUP3, uintptr(fd[i]), uintptr(nextfd), O_CLOEXEC)
			} else if runtime.GOOS == "dragonfly" {
				_, _, err1 = RawSyscall(SYS_FCNTL, uintptr(fd[i]), _F_DUP2FD_CLOEXEC, uintptr(nextfd))
			} else {
				_, _, err1 = RawSyscall(SYS_DUP2, uintptr(fd[i]), uintptr(nextfd), 0)
				if err1 != 0 {
					goto childerror
				}
				_, _, err1 = RawSyscall(SYS_FCNTL, uintptr(nextfd), F_SETFD, FD_CLOEXEC)
			}
			if err1 != 0 {
				goto childerror
			}
			fd[i] = nextfd
			nextfd++
		}
	}

	// Pass 2: dup fd[i] down onto i.
	for i = 0; i < len(fd); i++ {
		if fd[i] == -1 {
			RawSyscall(SYS_CLOSE, uintptr(i), 0, 0)
			continue
		}
		if fd[i] == i {
			// dup2(i, i) won't clear close-on-exec flag on Linux,
			// probably not elsewhere either.
			_, _, err1 = RawSyscall(SYS_FCNTL, uintptr(fd[i]), F_SETFD, 0)
			if err1 != 0 {
				goto childerror
			}
			continue
		}
		// The new fd is created NOT close-on-exec,
		// which is exactly what we want.
		_, _, err1 = RawSyscall(SYS_DUP2, uintptr(fd[i]), uintptr(i), 0)
		if err1 != 0 {
			goto childerror
		}
	}

	// By convention, we don't close-on-exec the fds we are
	// started with, so if len(fd) < 3, close 0, 1, 2 as needed.
	// Programs that know they inherit fds >= 3 will need
	// to set them close-on-exec.
	for i = len(fd); i < 3; i++ {
		RawSyscall(SYS_CLOSE, uintptr(i), 0, 0)
	}

	// Detach fd 0 from tty
	if sys.Noctty {
		_, _, err1 = RawSyscall(SYS_IOCTL, 0, uintptr(TIOCNOTTY), 0)
		if err1 != 0 {
			goto childerror
		}
	}

	// Set the controlling TTY to Ctty
	if sys.Setctty {
		_, _, err1 = RawSyscall(SYS_IOCTL, uintptr(sys.Ctty), uintptr(TIOCSCTTY), 0)
		if err1 != 0 {
			goto childerror
		}
	}

	// Restore original rlimit.
	if rlim != nil {
		RawSyscall(SYS_SETRLIMIT, uintptr(RLIMIT_NOFILE), uintptr(unsafe.Pointer(rlim)), 0)
	}

	// Time to exec.
	_, _, err1 = RawSyscall(SYS_EXECVE,
		uintptr(unsafe.Pointer(argv0)),
		uintptr(unsafe.Pointer(&argv[0])),
		uintptr(unsafe.Pointer(&envv[0])))

childerror:
	// send error code on pipe
	RawSyscall(SYS_WRITE, uintptr(pipe), uintptr(unsafe.Pointer(&err1)), unsafe.Sizeof(err1))
	for {
		RawSyscall(SYS_EXIT, 253, 0, 0)
	}
}

// forkAndExecFailureCleanup cleans up after an exec failure.
func forkAndExecFailureCleanup(attr *ProcAttr, sys *SysProcAttr) {
	// Nothing to do.
}
