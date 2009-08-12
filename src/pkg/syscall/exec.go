// Copyright 2009 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// Fork, exec, wait, etc.

package syscall

import (
	"sync";
	"unsafe";
)

// Lock synchronizing creation of new file descriptors with fork.
//
// We want the child in a fork/exec sequence to inherit only the
// file descriptors we intend.  To do that, we mark all file
// descriptors close-on-exec and then, in the child, explicitly
// unmark the ones we want the exec'ed program to keep.
// Unix doesn't make this easy: there is, in general, no way to
// allocate a new file descriptor close-on-exec.  Instead you
// have to allocate the descriptor and then mark it close-on-exec.
// If a fork happens between those two events, the child's exec
// will inherit an unwanted file descriptor.
//
// This lock solves that race: the create new fd/mark close-on-exec
// operation is done holding ForkLock for reading, and the fork itself
// is done holding ForkLock for writing.  At least, that's the idea.
// There are some complications.
//
// Some system calls that create new file descriptors can block
// for arbitrarily long times: open on a hung NFS server or named
// pipe, accept on a socket, and so on.  We can't reasonably grab
// the lock across those operations.
//
// It is worse to inherit some file descriptors than others.
// If a non-malicious child accidentally inherits an open ordinary file,
// that's not a big deal.  On the other hand, if a long-lived child
// accidentally inherits the write end of a pipe, then the reader
// of that pipe will not see EOF until that child exits, potentially
// causing the parent program to hang.  This is a common problem
// in threaded C programs that use popen.
//
// Luckily, the file descriptors that are most important not to
// inherit are not the ones that can take an arbitrarily long time
// to create: pipe returns instantly, and the net package uses
// non-blocking I/O to accept on a listening socket.
// The rules for which file descriptor-creating operations use the
// ForkLock are as follows:
//
// 1) Pipe.    Does not block.  Use the ForkLock.
// 2) Socket.  Does not block.  Use the ForkLock.
// 3) Accept.  If using non-blocking mode, use the ForkLock.
//             Otherwise, live with the race.
// 4) Open.    Can block.  Use O_CLOEXEC if available (Linux).
//             Otherwise, live with the race.
// 5) Dup.     Does not block.  Use the ForkLock.
//             On Linux, could use fcntl F_DUPFD_CLOEXEC
//             instead of the ForkLock, but only for dup(fd, -1).

var ForkLock sync.RWMutex

// Convert array of string to array
// of NUL-terminated byte pointer.
func StringArrayPtr(ss []string) []*byte {
	bb := make([]*byte, len(ss)+1);
	for i := 0; i < len(ss); i++ {
		bb[i] = StringBytePtr(ss[i]);
	}
	bb[len(ss)] = nil;
	return bb;
}

func CloseOnExec(fd int) {
	fcntl(fd, F_SETFD, FD_CLOEXEC);
}

func SetNonblock(fd int, nonblocking bool) (errno int) {
	flag, err := fcntl(fd, F_GETFL, 0);
	if err != 0 {
		return err;
	}
	if nonblocking {
		flag |= O_NONBLOCK;
	} else {
		flag &= ^O_NONBLOCK;
	}
	flag, err = fcntl(fd, F_SETFL, flag);
	return err;
}


// Fork, dup fd onto 0..len(fd), and exec(argv0, argvv, envv) in child.
// If a dup or exec fails, write the errno int to pipe.
// (Pipe is close-on-exec so if exec succeeds, it will be closed.)
// In the child, this function must not acquire any locks, because
// they might have been locked at the time of the fork.  This means
// no rescheduling, no malloc calls, and no new stack segments.
// The calls to RawSyscall are okay because they are assembly
// functions that do not grow the stack.
func forkAndExecInChild(argv0 *byte, argv []*byte, envv []*byte, traceme bool, dir *byte, fd []int, pipe int)
	(pid int, err int)
{
	// Declare all variables at top in case any
	// declarations require heap allocation (e.g., err1).
	var r1, r2, err1 uintptr;
	var nextfd int;
	var i int;

	darwin := OS == "darwin";

	// About to call fork.
	// No more allocation or calls of non-assembly functions.
	r1, r2, err1 = RawSyscall(SYS_FORK, 0, 0, 0);
	if err1 != 0 {
		return 0, int(err1)
	}

	// On Darwin:
	//	r1 = child pid in both parent and child.
	//	r2 = 0 in parent, 1 in child.
	// Convert to normal Unix r1 = 0 in child.
	if darwin && r2 == 1 {
		r1 = 0;
	}

	if r1 != 0 {
		// parent; return PID
		return int(r1), 0
	}

	// Fork succeeded, now in child.

	// Enable tracing if requested.
	if traceme {
		r1, r2, err1 = RawSyscall(SYS_PTRACE, uintptr(_PTRACE_TRACEME), 0, 0);
		if err1 != 0 {
			goto childerror;
		}
	}

	// Chdir
	if dir != nil {
		r1, r2, err1 = RawSyscall(SYS_CHDIR, uintptr(unsafe.Pointer(dir)), 0, 0);
		if err1 != 0 {
			goto childerror;
		}
	}

	// Pass 1: look for fd[i] < i and move those up above len(fd)
	// so that pass 2 won't stomp on an fd it needs later.
	nextfd = int(len(fd));
	if pipe < nextfd {
		r1, r2, err1 = RawSyscall(SYS_DUP2, uintptr(pipe), uintptr(nextfd), 0);
		if err1 != 0 {
			goto childerror;
		}
		RawSyscall(SYS_FCNTL, uintptr(nextfd), F_SETFD, FD_CLOEXEC);
		pipe = nextfd;
		nextfd++;
	}
	for i = 0; i < len(fd); i++ {
		if fd[i] >= 0 && fd[i] < int(i) {
			r1, r2, err1 = RawSyscall(SYS_DUP2, uintptr(fd[i]), uintptr(nextfd), 0);
			if err1 != 0 {
				goto childerror;
			}
			RawSyscall(SYS_FCNTL, uintptr(nextfd), F_SETFD, FD_CLOEXEC);
			fd[i] = nextfd;
			nextfd++;
			if nextfd == pipe {	// don't stomp on pipe
				nextfd++;
			}
		}
	}

	// Pass 2: dup fd[i] down onto i.
	for i = 0; i < len(fd); i++ {
		if fd[i] == -1 {
			RawSyscall(SYS_CLOSE, uintptr(i), 0, 0);
			continue;
		}
		if fd[i] == int(i) {
			// dup2(i, i) won't clear close-on-exec flag on Linux,
			// probably not elsewhere either.
			r1, r2, err1 = RawSyscall(SYS_FCNTL, uintptr(fd[i]), F_SETFD, 0);
			if err1 != 0 {
				goto childerror;
			}
			continue;
		}
		// The new fd is created NOT close-on-exec,
		// which is exactly what we want.
		r1, r2, err1 = RawSyscall(SYS_DUP2, uintptr(fd[i]), uintptr(i), 0);
		if err1 != 0 {
			goto childerror;
		}
	}

	// By convention, we don't close-on-exec the fds we are
	// started with, so if len(fd) < 3, close 0, 1, 2 as needed.
	// Programs that know they inherit fds >= 3 will need
	// to set them close-on-exec.
	for i = len(fd); i < 3; i++ {
		RawSyscall(SYS_CLOSE, uintptr(i), 0, 0);
	}

	// Time to exec.
	r1, r2, err1 = RawSyscall(SYS_EXECVE,
		uintptr(unsafe.Pointer(argv0)),
		uintptr(unsafe.Pointer(&argv[0])),
		uintptr(unsafe.Pointer(&envv[0])));

childerror:
	// send error code on pipe
	RawSyscall(SYS_WRITE, uintptr(pipe), uintptr(unsafe.Pointer(&err1)), uintptr(unsafe.Sizeof(err1)));
	for {
		RawSyscall(SYS_EXIT, 253, 0, 0);
	}

	// Calling panic is not actually safe,
	// but the for loop above won't break
	// and this shuts up the compiler.
	panic("unreached");
}

func forkExec(argv0 string, argv []string, envv []string, traceme bool, dir string, fd []int)
	(pid int, err int)
{
	var p [2]int;
	var r1 int;
	var n int;
	var err1 uintptr;
	var wstatus WaitStatus;

	p[0] = -1;
	p[1] = -1;

	// Convert args to C form.
	argv0p := StringBytePtr(argv0);
	argvp := StringArrayPtr(argv);
	envvp := StringArrayPtr(envv);
	var dirp *byte;
	if len(dir) > 0 {
		dirp = StringBytePtr(dir);
	}

	// Acquire the fork lock so that no other threads
	// create new fds that are not yet close-on-exec
	// before we fork.
	ForkLock.Lock();

	// Allocate child status pipe close on exec.
	if err = Pipe(&p); err != 0 {
		goto error;
	}
	var val int;
	if val, err = fcntl(p[0], F_SETFD, FD_CLOEXEC); err != 0 {
		goto error;
	}
	if val, err = fcntl(p[1], F_SETFD, FD_CLOEXEC); err != 0 {
		goto error;
	}

	// Kick off child.
	pid, err = forkAndExecInChild(argv0p, argvp, envvp, traceme, dirp, fd, p[1]);
	if err != 0 {
	error:
		if p[0] >= 0 {
			Close(p[0]);
			Close(p[1]);
		}
		ForkLock.Unlock();
		return 0, err
	}
	ForkLock.Unlock();

	// Read child error status from pipe.
	Close(p[1]);
	n, err = read(p[0], (*byte)(unsafe.Pointer(&err1)), unsafe.Sizeof(err1));
	Close(p[0]);
	if err != 0 || n != 0 {
		if n == unsafe.Sizeof(err1) {
			err = int(err1);
		}
		if err == 0 {
			err = EPIPE;
		}

		// Child failed; wait for it to exit, to make sure
		// the zombies don't accumulate.
		pid1, err1 := Wait4(pid, &wstatus, 0, nil);
		for err1 == EINTR {
			pid1, err1 = Wait4(pid, &wstatus, 0, nil);
		}
		return 0, err
	}

	// Read got EOF, so pipe closed on exec, so exec succeeded.
	return pid, 0
}

// Combination of fork and exec, careful to be thread safe.
func ForkExec(argv0 string, argv []string, envv []string, dir string, fd []int)
	(pid int, err int)
{
	return forkExec(argv0, argv, envv, false, dir, fd);
}

// PtraceForkExec is like ForkExec, but starts the child in a traced state.
func PtraceForkExec(argv0 string, argv []string, envv []string, dir string, fd []int)
	(pid int, err int)
{
	return forkExec(argv0, argv, envv, true, dir, fd);
}

// Ordinary exec.
func Exec(argv0 string, argv []string, envv []string) (err int) {
	r1, r2, err1 := RawSyscall(SYS_EXECVE,
		uintptr(unsafe.Pointer(StringBytePtr(argv0))),
		uintptr(unsafe.Pointer(&StringArrayPtr(argv)[0])),
		uintptr(unsafe.Pointer(&StringArrayPtr(envv)[0])));
	return int(err1);
}

