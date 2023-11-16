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
	if p.Pid == -1 {
		return nil, syscall.EINVAL
	}
	// Use pidfd if possible; fallback on ENOSYS or EPERM (the latter can be
	// returned if syscall is prohibited by seccomp or a similar mechanism).
	//
	// When pidfd is used, there is no wait/kill race (described in CL 23967)
	// because PID recycle issue doesn't exist (IOW, pidfd, unlike PID, is
	// guaranteed to refer to one particular process). Thus, there is no
	// need for the workaround (blockUntilWaitable + sigMu) below.
	if ps, e := p.pidfdWait(); e != syscall.ENOSYS && e != syscall.EPERM {
		return ps, NewSyscallError("waitid", e)
	}

	// If we can block until Wait4 will succeed immediately, do so.
	ready, err := p.blockUntilWaitable()
	if err != nil {
		return nil, err
	}
	if ready {
		// Mark the process done now, before the call to Wait4,
		// so that Process.signal will not send a signal.
		p.setDone()
		// Acquire a write lock on sigMu to wait for any
		// active call to the signal method to complete.
		p.sigMu.Lock()
		p.sigMu.Unlock()
	}

	var (
		status syscall.WaitStatus
		rusage syscall.Rusage
		pid1   int
		e      error
	)
	for {
		pid1, e = syscall.Wait4(p.Pid, &status, 0, &rusage)
		if e != syscall.EINTR {
			break
		}
	}
	if e != nil {
		return nil, NewSyscallError("wait", e)
	}
	p.setDone()
	ps = &ProcessState{
		pid:    pid1,
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
	s, ok := sig.(syscall.Signal)
	if !ok {
		return errors.New("os: unsupported signal type")
	}
	// Use pidfd if possible; fallback on ENOSYS.
	if err := p.pidfdSendSignal(s); err != syscall.ENOSYS {
		return err
	}
	p.sigMu.RLock()
	defer p.sigMu.RUnlock()
	if p.done() {
		return ErrProcessDone
	}
	return convertESRCH(syscall.Kill(p.Pid, s))
}

func convertESRCH(err error) error {
	if err == syscall.ESRCH {
		return ErrProcessDone
	}
	return err
}

func (p *Process) release() error {
	p.pidfdRelease()
	p.Pid = -1
	// no need for a finalizer anymore
	runtime.SetFinalizer(p, nil)
	return nil
}

func findProcess(pid int) (p *Process, err error) {
	h, err := pidfdFind(pid)
	if err == ErrProcessDone {
		return nil, err
	}
	// Ignore all other errors from pidfdFind,
	// as the callers do not expect them, and
	// we can use pid anyway.
	return newProcess(pid, h), nil
}

func (p *ProcessState) userTime() time.Duration {
	return time.Duration(p.rusage.Utime.Nano()) * time.Nanosecond
}

func (p *ProcessState) systemTime() time.Duration {
	return time.Duration(p.rusage.Stime.Nano()) * time.Nanosecond
}
