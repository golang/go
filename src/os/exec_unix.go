// Copyright 2009 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

//go:build unix || (js && wasm) || wasip1

package os

import (
	"errors"
	"runtime"
	"syscall"
	"time"
)

func (p *Process) wait() (ps *ProcessState, err error) {
	if p.handle == nil {
		return nil, syscall.EINVAL
	}
	var (
		status syscall.WaitStatus
		rusage syscall.Rusage
	)
	if e := p.handle.Wait4(&status, &rusage); e != nil {
		return nil, NewSyscallError("wait", e)
	}
	ps = &ProcessState{
		pid:    p.Pid,
		status: status,
		rusage: &rusage,
	}
	return ps, nil
}

func (p *Process) signal(sig Signal) error {
	if p.Pid == -1 {
		return errors.New("os: process already released")
	}
	if p.Pid == 0 {
		return errors.New("os: process not initialized")
	}
	if p.handle != nil {
		if !p.handle.Hold() {
			return ErrProcessDone
		}
		defer p.handle.Unhold()
	}
	s, ok := sig.(syscall.Signal)
	if !ok {
		return errors.New("os: unsupported signal type")
	}
	if e := syscall.Kill(p.Pid, s); e != nil {
		if e == syscall.ESRCH {
			return ErrProcessDone
		}
		return e
	}
	return nil
}

func (p *Process) release() error {
	// NOOP for unix.
	p.Pid = -1
	p.handle = nil
	// no need for a finalizer anymore
	runtime.SetFinalizer(p, nil)
	return nil
}

func findProcess(pid int) (p *Process, err error) {
	// NOOP for unix.
	return newProcess(pid, nil), nil
}

func (p *ProcessState) userTime() time.Duration {
	return time.Duration(p.rusage.Utime.Nano()) * time.Nanosecond
}

func (p *ProcessState) systemTime() time.Duration {
	return time.Duration(p.rusage.Stime.Nano()) * time.Nanosecond
}
