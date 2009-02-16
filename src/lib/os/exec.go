// Copyright 2009 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package os

import (
	"os";
	"syscall";
)

func ForkExec(argv0 string, argv []string, envv []string, fd []*FD)
	(pid int, err *Error)
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

func Exec(argv0 string, argv []string, envv []string) *Error {
	e := syscall.Exec(argv0, argv, envv);
	return ErrnoToError(e);
}

// TODO(rsc): Should os implement its own syscall.WaitStatus
// wrapper with the methods, or is exposing the underlying one enough?
//
// TODO(rsc): Certainly need to have os.Rusage struct,
// since syscall one might have different field types across
// different OS.

type Waitmsg struct {
	Pid int;
	syscall.WaitStatus;
	Rusage *syscall.Rusage;
}

const (
	WNOHANG = syscall.WNOHANG;
	WSTOPPED = syscall.WSTOPPED;
	WRUSAGE = 1<<60;
)

func Wait(pid int, options uint64) (w *Waitmsg, err *Error) {
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

