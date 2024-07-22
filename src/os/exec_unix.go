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

const (
	// Special values for Process.Pid.
	pidUnset    = 0
	pidReleased = -1
)

func (p *Process) wait() (ps *ProcessState, err error) {
	// Which type of Process do we have?
	switch p.mode {
	case modeHandle:
		// pidfd
		return p.pidfdWait()
	case modePID:
		// Regular PID
		return p.pidWait()
	default:
		panic("unreachable")
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
		p.pidDeactivate(statusDone)
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
	p.pidDeactivate(statusDone)
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
	switch p.mode {
	case modeHandle:
		// pidfd
		return p.pidfdSendSignal(s)
	case modePID:
		// Regular PID
		return p.pidSignal(s)
	default:
		panic("unreachable")
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

func (p *Process) release() error {
	// We clear the Pid field only for API compatibility. On Unix, Release
	// has always set Pid to -1. Internally, the implementation relies
	// solely on statusReleased to determine that the Process is released.
	p.Pid = pidReleased

	switch p.mode {
	case modeHandle:
		// Drop the Process' reference and mark handle unusable for
		// future calls.
		//
		// Ignore the return value: we don't care if this was a no-op
		// racing with Wait, or a double Release.
		p.handlePersistentRelease(statusReleased)
	case modePID:
		// Just mark the PID unusable.
		p.pidDeactivate(statusReleased)
	}
	// no need for a finalizer anymore
	runtime.SetFinalizer(p, nil)
	return nil
}

func findProcess(pid int) (p *Process, err error) {
	h, err := pidfdFind(pid)
	if errors.Is(err, ErrProcessDone) {
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
