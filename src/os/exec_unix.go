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
	"unsafe"
)

const CLD_EXITED = 1
const CLD_KILLED = 2
const CLD_DUMPED = 3
const CLD_TRAPPED = 4
const CLD_STOPPED = 5
const CLD_CONTINUED = 6

type Siginfo struct {
	Signo int32
	Errno int32
	Code  int32
	_     int32
	_     [112]byte
}

func (s Siginfo) Exited() bool { return s.Code == CLD_EXITED }

func (s Siginfo) Signaled() bool { return s.Code == CLD_KILLED }

func (s Siginfo) Stopped() bool { return s.Code == CLD_STOPPED }

func (s Siginfo) Trapped() bool { return s.Code == CLD_TRAPPED }

func (s Siginfo) Continued() bool { return s.Code == CLD_CONTINUED }

func (s Siginfo) CoreDump() bool { return s.Code == CLD_DUMPED }

func (s Siginfo) ExitStatus() int {
	if !s.Exited() {
		return -1
	}
	return int(s.Errno)
}

func (s Siginfo) Signal() syscall.Signal {
	if !s.Signaled() {
		return -1
	}
	return syscall.Signal(s.Errno)
}

func (s Siginfo) StopSignal() syscall.Signal {
	if !s.Stopped() {
		return -1
	}
	return syscall.Signal(s.Errno)
}

func (s Siginfo) TrapCause() int {
	if !s.Trapped() {
		return -1
	}
	return int(s.Errno)
}

func (p *Process) wait() (ps *ProcessState, err error) {
	if p.Pid == -1 {
		return nil, syscall.EINVAL
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
		siginfo Siginfo
		rusage  syscall.Rusage
		e       syscall.Errno
	)
	for {
		_, _, e = syscall.Syscall6(syscall.SYS_WAITID, _P_PIDFD, p.handle, uintptr(unsafe.Pointer(&siginfo)), syscall.WEXITED, uintptr(unsafe.Pointer(&rusage)), 0)
		if e == syscall.EINTR {
			continue
		} else if e == syscall.ENOSYS {
			// waitid has been available since Linux 2.6.9, but
			// reportedly is not available in Ubuntu on Windows.
			// See issue 16610.
			panic("TODO: Implement fallback")
		} else if e != 0 {
			break
		}
		// During ptrace the wait might return also for non-exit reasons. In that case we retry.
		// See: https://lwn.net/Articles/688624/
		if siginfo.Exited() || siginfo.Signaled() || siginfo.CoreDump() {
			break
		}
	}
	runtime.KeepAlive(p)
	if e != 0 {
		return nil, NewSyscallError("waitid", e)
	}
	p.setDone()
	ps = &ProcessState{
		pid:     p.Pid,
		siginfo: siginfo,
		rusage:  &rusage,
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
	p.sigMu.RLock()
	defer p.sigMu.RUnlock()
	if p.done() {
		return ErrProcessDone
	}
	s, ok := sig.(syscall.Signal)
	if !ok {
		return errors.New("os: unsupported signal type")
	}
	if _, _, e := syscall.RawSyscall6(syscall.SYS_PIDFD_SEND_SIGNAL, p.handle, uintptr(s), 0, 0, 0, 0); e != 0 {
		if e == syscall.ESRCH {
			return ErrProcessDone
		}
		return NewSyscallError("pidfd_send_signal", e)
	}
	runtime.KeepAlive(p)
	return nil
}

func (p *Process) release() error {
	e := syscall.Close(int(p.handle))
	if e != nil {
		return NewSyscallError("close", e)

	}
	p.Pid = -1
	// no need for a finalizer anymore
	runtime.SetFinalizer(p, nil)
	return nil
}

func findProcess(pid int) (p *Process, err error) {
	fd, _, e := syscall.Syscall(syscall.SYS_PIDFD_OPEN, uintptr(pid), 0, 0)
	runtime.KeepAlive(p)
	if e != 0 {
		return nil, NewSyscallError("pidfd_open", e)
	}
	return newProcess(pid, fd), nil
}

func (p *ProcessState) userTime() time.Duration {
	return time.Duration(p.rusage.Utime.Nano()) * time.Nanosecond
}

func (p *ProcessState) systemTime() time.Duration {
	return time.Duration(p.rusage.Stime.Nano()) * time.Nanosecond
}
