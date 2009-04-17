// Copyright 2009 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package os

import (
	"os";
	"syscall";
)

// ForkExec forks the current process and invokes Exec with the file, arguments,
// and environment specified by argv0, argv, and envv.  It returns the process
// id of the forked process and an Error, if any.  The fd array specifies the
// file descriptors to be set up in the new process: fd[0] will be Unix file
// descriptor 0 (standard input), fd[1] descriptor 1, and so on.  A nil entry
// will cause the child to have no open file descriptor with that index.
func ForkExec(argv0 string, argv []string, envv []string, fd []*File)
	(pid int, err Error)
{
	// Create array of integer (system) fds.
	intfd := make([]int64, len(fd));
	for i, f := range(fd) {
		if f == nil {
			intfd[i] = -1;
		} else {
			intfd[i] = f.Fd();
		}
	}

	p, e := syscall.ForkExec(argv0, argv, envv, intfd);
	return int(p), ErrnoToError(e);
}

// Exec replaces the current process with an execution of the program
// named by argv0, with arguments argv and environment envv.
// If successful, Exec never returns.  If it fails, it returns an Error.
// ForkExec is almost always a better way to execute a program.
func Exec(argv0 string, argv []string, envv []string) Error {
	if envv == nil {
		envv = Environ();
	}
	e := syscall.Exec(argv0, argv, envv);
	return ErrnoToError(e);
}

// TODO(rsc): Should os implement its own syscall.WaitStatus
// wrapper with the methods, or is exposing the underlying one enough?
//
// TODO(rsc): Certainly need to have os.Rusage struct,
// since syscall one might have different field types across
// different OS.

// Waitmsg stores the information about an exited process as reported by Wait.
type Waitmsg struct {
	Pid int;	// The process's id.
	syscall.WaitStatus;	// System-dependent status info.
	Rusage *syscall.Rusage;	// System-dependent resource usage info.
}

// Options for Wait.
const (
	WNOHANG = syscall.WNOHANG;	// Don't wait if no process has exited.
	WSTOPPED = syscall.WSTOPPED;	// If set, status of stopped subprocesses is also reported.
	WUNTRACED = WSTOPPED;
	WRUSAGE = 1<<60;	// Record resource usage.
)

// Wait waits for process pid to exit or stop, and then returns a
// Waitmsg describing its status and an Error, if any. The options
// (WNOHANG etc.) affect the behavior of the Wait call.
func Wait(pid int, options uint64) (w *Waitmsg, err Error) {
	var status syscall.WaitStatus;
	var rusage *syscall.Rusage;
	if options & WRUSAGE != 0 {
		rusage = new(syscall.Rusage);
		options ^= WRUSAGE;
	}
	pid1, e := syscall.Wait4(int64(pid), &status, int64(options), rusage);
	if e != 0 {
		return nil, ErrnoToError(e);
	}
	w = new(Waitmsg);
	w.Pid = pid;
	w.WaitStatus = status;
	w.Rusage = rusage;
	return w, nil;
}

