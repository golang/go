// Copyright 2009 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

//go:build unix || (js && wasm) || wasip1

package os

import (
	"errors"
	"syscall"
	"time"
)

const (
	// Special values for Process.Pid.
	pidUnset    = 0
	pidReleased = -1
)

func (p *Process) wait() (ps *ProcessState, err error) {
	// Which type of Process do we have?
	if p.handle != nil {
		// pidfd
		return p.pidfdWait()
	} else {
		// Regular PID
		return p.pidWait()
	}
}

func (p *Process) pidWait() (*ProcessState, error) {
	// TODO(go.dev/issue/67642): When there are concurrent Wait calls, one
	// may wait on the wrong process if the PID is reused after the
	// completes its wait.
	//
	// Checking for statusDone here would not be a complete fix, as the PID
	// could still be waited on and reused prior to blockUntilWaitable.
	switch p.pidStatus() {
	case statusReleased:
		return nil, syscall.EINVAL
	}

	// If we can block until Wait4 will succeed immediately, do so.
	ready, err := p.blockUntilWaitable()
	if err != nil {
		return nil, err
	}
	if ready {
		// Mark the process done now, before the call to Wait4,
		// so that Process.pidSignal will not send a signal.
		p.doRelease(statusDone)
		// Acquire a write lock on sigMu to wait for any
		// active call to the signal method to complete.
		p.sigMu.Lock()
		p.sigMu.Unlock()
	}

	var (
		status syscall.WaitStatus
		rusage syscall.Rusage
	)
	pid1, err := ignoringEINTR2(func() (int, error) {
		return syscall.Wait4(p.Pid, &status, 0, &rusage)
	})
	if err != nil {
		return nil, NewSyscallError("wait", err)
	}
	p.doRelease(statusDone)
	return &ProcessState{
		pid:    pid1,
		status: status,
		rusage: &rusage,
	}, nil
}

func (p *Process) signal(sig Signal) error {
	s, ok := sig.(syscall.Signal)
	if !ok {
		return errors.New("os: unsupported signal type")
	}

	// Which type of Process do we have?
	if p.handle != nil {
		// pidfd
		return p.pidfdSendSignal(s)
	} else {
		// Regular PID
		return p.pidSignal(s)
	}
}

func (p *Process) pidSignal(s syscall.Signal) error {
	if p.Pid == pidReleased {
		return errors.New("os: process already released")
	}
	if p.Pid == pidUnset {
		return errors.New("os: process not initialized")
	}

	p.sigMu.RLock()
	defer p.sigMu.RUnlock()

	switch p.pidStatus() {
	case statusDone:
		return ErrProcessDone
	case statusReleased:
		return errors.New("os: process already released")
	}

	return convertESRCH(syscall.Kill(p.Pid, s))
}

func convertESRCH(err error) error {
	if err == syscall.ESRCH {
		return ErrProcessDone
	}
	return err
}

func findProcess(pid int) (p *Process, err error) {
	h, err := pidfdFind(pid)
	if err == ErrProcessDone {
		// We can't return an error here since users are not expecting
		// it. Instead, return a process with a "done" state already
		// and let a subsequent Signal or Wait call catch that.
		return newDoneProcess(pid), nil
	} else if err != nil {
		// Ignore other errors from pidfdFind, as the callers
		// do not expect them. Fall back to using the PID.
		return newPIDProcess(pid), nil
	}
	// Use the handle.
	return newHandleProcess(pid, h), nil
}

func (p *ProcessState) userTime() time.Duration {
	return time.Duration(p.rusage.Utime.Nano()) * time.Nanosecond
}

func (p *ProcessState) systemTime() time.Duration {
	return time.Duration(p.rusage.Stime.Nano()) * time.Nanosecond
}
