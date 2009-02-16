// Copyright 2009 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// Fork, exec, wait, etc.

package syscall

import (
	"sync";
	"syscall";
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

func CloseOnExec(fd int64) {
	Fcntl(fd, F_SETFD, FD_CLOEXEC);
}

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

func Wait4(pid int64, wstatus *WaitStatus, options int64, rusage *Rusage)
	(wpid, err int64)
{
	var s WaitStatus;
	r1, r2, err1 := Syscall6(SYS_WAIT4,
		pid,
		int64(uintptr(unsafe.Pointer(&s))),
		options,
		int64(uintptr(unsafe.Pointer(rusage))), 0, 0);
	if wstatus != nil {
		*wstatus = s;
	}
	return r1, err1;
}

// Fork, dup fd onto 0..len(fd), and exec(argv0, argvv, envv) in child.
// If a dup or exec fails, write the errno int64 to pipe.
// (Pipe is close-on-exec so if exec succeeds, it will be closed.)
// In the child, this function must not acquire any locks, because
// they might have been locked at the time of the fork.  This means
// no rescheduling, no malloc calls, and no new stack segments.
// The calls to RawSyscall are okay because they are assembly
// functions that do not grow the stack.
func forkAndExecInChild(argv0 *byte, argv []*byte, envv []*byte, fd []int64, pipe int64)
	(pid int64, err int64)
{
	// Declare all variables at top in case any
	// declarations require heap allocation (e.g., err1).
	var r1, r2, err1 int64;
	var nextfd int64;
	var i int;

	darwin := OS == "darwin";

	// About to call fork.
	// No more allocation or calls of non-assembly functions.
	r1, r2, err1 = RawSyscall(SYS_FORK, 0, 0, 0);
	if err1 != 0 {
		return 0, err1
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
		return r1, 0
	}

	// Fork succeeded, now in child.

	// Pass 1: look for fd[i] < i and move those up above len(fd)
	// so that pass 2 won't stomp on an fd it needs later.
	nextfd = int64(len(fd));
	if pipe < nextfd {
		r1, r2, err = RawSyscall(SYS_DUP2, pipe, nextfd, 0);
		if err != 0 {
			goto childerror;
		}
		RawSyscall(SYS_FCNTL, nextfd, F_SETFD, FD_CLOEXEC);
		pipe = nextfd;
		nextfd++;
	}
	for i = 0; i < len(fd); i++ {
		if fd[i] >= 0 && fd[i] < int64(i) {
			r1, r2, err = RawSyscall(SYS_DUP2, fd[i], nextfd, 0);
			if err != 0 {
				goto childerror;
			}
			RawSyscall(SYS_FCNTL, nextfd, F_SETFD, FD_CLOEXEC);
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
			RawSyscall(SYS_CLOSE, int64(i), 0, 0);
			continue;
		}
		if fd[i] == int64(i) {
			// dup2(i, i) won't clear close-on-exec flag on Linux,
			// probably not elsewhere either.
			r1, r2, err = RawSyscall(SYS_FCNTL, fd[i], F_SETFD, 0);
			if err != 0 {
				goto childerror;
			}
			continue;
		}
		// The new fd is created NOT close-on-exec,
		// which is exactly what we want.
		r1, r2, err = RawSyscall(SYS_DUP2, fd[i], int64(i), 0);
		if err != 0 {
			goto childerror;
		}
	}

	// By convention, we don't close-on-exec the fds we are
	// started with, so if len(fd) < 3, close 0, 1, 2 as needed.
	// Programs that know they inherit fds >= 3 will need
	// to set them close-on-exec.
	for i = len(fd); i < 3; i++ {
		RawSyscall(SYS_CLOSE, int64(i), 0, 0);
	}

	// Time to exec.
	r1, r2, err1 = RawSyscall(SYS_EXECVE,
		int64(uintptr(unsafe.Pointer(argv0))),
		int64(uintptr(unsafe.Pointer(&argv[0]))),
		int64(uintptr(unsafe.Pointer(&envv[0]))));

childerror:
	// send error code on pipe
	RawSyscall(SYS_WRITE, pipe, int64(uintptr(unsafe.Pointer(&err1))), 8);
	for {
		RawSyscall(SYS_EXIT, 253, 0, 0);
	}

	// Calling panic is not actually safe,
	// but the for loop above won't break
	// and this shuts up the compiler.
	panic("unreached");
}

// Combination of fork and exec, careful to be thread safe.
func ForkExec(argv0 string, argv []string, envv []string, fd []int64)
	(pid int64, err int64)
{
	var p [2]int64;
	var r1 int64;
	var n, err1 int64;
	var wstatus WaitStatus;

	p[0] = -1;
	p[1] = -1;

	// Convert args to C form.
	argv0p := StringBytePtr(argv0);
	argvp := StringArrayPtr(argv);
	envvp := StringArrayPtr(envv);

	// Acquire the fork lock so that no other threads
	// create new fds that are not yet close-on-exec
	// before we fork.
	ForkLock.Lock();

	// Allocate child status pipe close on exec.
	if r1, err = Pipe(&p); err != 0 {
		goto error;
	}
	if r1, err = Fcntl(p[0], F_SETFD, FD_CLOEXEC); err != 0 {
		goto error;
	}
	if r1, err = Fcntl(p[1], F_SETFD, FD_CLOEXEC); err != 0 {
		goto error;
	}

	// Kick off child.
	pid, err = forkAndExecInChild(argv0p, argvp, envvp, fd, p[1]);
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
	n, r1, err = Syscall(SYS_READ, p[0], int64(uintptr(unsafe.Pointer(&err1))), 8);
	Close(p[0]);
	if err != 0 || n != 0 {
		if n == 8 {
			err = err1;
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

// Ordinary exec.
func Exec(argv0 string, argv []string, envv []string) (err int64) {
	r1, r2, err1 := RawSyscall(SYS_EXECVE,
		int64(uintptr(unsafe.Pointer(StringBytePtr(argv0)))),
		int64(uintptr(unsafe.Pointer(&StringArrayPtr(argv)[0]))),
		int64(uintptr(unsafe.Pointer(&StringArrayPtr(envv)[0]))));
	return err1;
}

