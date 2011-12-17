// Copyright 2009 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// +build darwin freebsd linux netbsd openbsd

package os

import (
	"errors"
	"runtime"
	"syscall"
)

// Options for Wait.
const (
	WNOHANG   = syscall.WNOHANG   // Don't wait if no process has exited.
	WSTOPPED  = syscall.WSTOPPED  // If set, status of stopped subprocesses is also reported.
	WUNTRACED = syscall.WUNTRACED // Usually an alias for WSTOPPED.
	WRUSAGE   = 1 << 20           // Record resource usage.
)

// WRUSAGE must not be too high a bit, to avoid clashing with Linux's
// WCLONE, WALL, and WNOTHREAD flags, which sit in the top few bits of
// the options

// Wait waits for the Process to exit or stop, and then returns a
// Waitmsg describing its status and an error, if any. The options
// (WNOHANG etc.) affect the behavior of the Wait call.
func (p *Process) Wait(options int) (w *Waitmsg, err error) {
	if p.Pid == -1 {
		return nil, EINVAL
	}
	var status syscall.WaitStatus
	var rusage *syscall.Rusage
	if options&WRUSAGE != 0 {
		rusage = new(syscall.Rusage)
		options ^= WRUSAGE
	}
	pid1, e := syscall.Wait4(p.Pid, &status, options, rusage)
	if e != nil {
		return nil, NewSyscallError("wait", e)
	}
	// With WNOHANG pid is 0 if child has not exited.
	if pid1 != 0 && options&WSTOPPED == 0 {
		p.done = true
	}
	w = new(Waitmsg)
	w.Pid = pid1
	w.WaitStatus = status
	w.Rusage = rusage
	return w, nil
}

// Signal sends a signal to the Process.
func (p *Process) Signal(sig Signal) error {
	if p.done {
		return errors.New("os: process already finished")
	}
	if e := syscall.Kill(p.Pid, int(sig.(UnixSignal))); e != nil {
		return e
	}
	return nil
}

// Release releases any resources associated with the Process.
func (p *Process) Release() error {
	// NOOP for unix.
	p.Pid = -1
	// no need for a finalizer anymore
	runtime.SetFinalizer(p, nil)
	return nil
}

// FindProcess looks for a running process by its pid.
// The Process it returns can be used to obtain information
// about the underlying operating system process.
func FindProcess(pid int) (p *Process, err error) {
	// NOOP for unix.
	return newProcess(pid, 0), nil
}
