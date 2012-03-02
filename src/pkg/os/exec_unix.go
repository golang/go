// Copyright 2009 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// +build darwin freebsd linux netbsd openbsd

package os

import (
	"errors"
	"runtime"
	"syscall"
	"time"
)

func (p *Process) wait() (ps *ProcessState, err error) {
	if p.Pid == -1 {
		return nil, syscall.EINVAL
	}
	var status syscall.WaitStatus
	var rusage syscall.Rusage
	pid1, e := syscall.Wait4(p.Pid, &status, 0, &rusage)
	if e != nil {
		return nil, NewSyscallError("wait", e)
	}
	if pid1 != 0 {
		p.done = true
	}
	ps = &ProcessState{
		pid:    pid1,
		status: status,
		rusage: &rusage,
	}
	return ps, nil
}

func (p *Process) signal(sig Signal) error {
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

func (p *Process) release() error {
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

func (p *ProcessState) userTime() time.Duration {
	return time.Duration(p.rusage.Utime.Nano()) * time.Nanosecond
}

func (p *ProcessState) systemTime() time.Duration {
	return time.Duration(p.rusage.Stime.Nano()) * time.Nanosecond
}
