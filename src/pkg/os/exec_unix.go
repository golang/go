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

// Wait waits for the Process to exit or stop, and then returns a
// Waitmsg describing its status and an error, if any.
func (p *Process) Wait() (w *Waitmsg, err error) {
	if p.Pid == -1 {
		return nil, syscall.EINVAL
	}
	var status syscall.WaitStatus
	pid1, e := syscall.Wait4(p.Pid, &status, 0, nil)
	if e != nil {
		return nil, NewSyscallError("wait", e)
	}
	if pid1 != 0 {
		p.done = true
	}
	w = new(Waitmsg)
	w.Pid = pid1
	w.WaitStatus = status
	return w, nil
}

// Signal sends a signal to the Process.
func (p *Process) Signal(sig Signal) error {
	if p.done {
		return errors.New("os: process already finished")
	}
	s, ok := sig.(syscall.Signal)
	if !ok {
		return errors.New("os: unsupported signal type")
	}
	if e := syscall.Kill(p.Pid, s); e != nil {
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

func findProcess(pid int) (p *Process, err error) {
	// NOOP for unix.
	return newProcess(pid, 0), nil
}
