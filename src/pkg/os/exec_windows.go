// Copyright 2009 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package os

import (
	"runtime"
	"syscall"
)

func (p *Process) Wait(options int) (w *Waitmsg, err Error) {
	s, e := syscall.WaitForSingleObject(syscall.Handle(p.handle), syscall.INFINITE)
	switch s {
	case syscall.WAIT_OBJECT_0:
		break
	case syscall.WAIT_FAILED:
		return nil, NewSyscallError("WaitForSingleObject", e)
	default:
		return nil, NewError("os: unexpected result from WaitForSingleObject")
	}
	var ec uint32
	e = syscall.GetExitCodeProcess(syscall.Handle(p.handle), &ec)
	if e != 0 {
		return nil, NewSyscallError("GetExitCodeProcess", e)
	}
	p.done = true
	return &Waitmsg{p.Pid, syscall.WaitStatus{s, ec}, new(syscall.Rusage)}, nil
}

// Signal sends a signal to the Process.
func (p *Process) Signal(sig Signal) Error {
	if p.done {
		return NewError("os: process already finished")
	}
	switch sig.(UnixSignal) {
	case SIGKILL:
		e := syscall.TerminateProcess(syscall.Handle(p.handle), 1)
		return NewSyscallError("TerminateProcess", e)
	}
	return Errno(syscall.EWINDOWS)
}

func (p *Process) Release() Error {
	if p.handle == -1 {
		return EINVAL
	}
	e := syscall.CloseHandle(syscall.Handle(p.handle))
	if e != 0 {
		return NewSyscallError("CloseHandle", e)
	}
	p.handle = -1
	// no need for a finalizer anymore
	runtime.SetFinalizer(p, nil)
	return nil
}

func FindProcess(pid int) (p *Process, err Error) {
	const da = syscall.STANDARD_RIGHTS_READ |
		syscall.PROCESS_QUERY_INFORMATION | syscall.SYNCHRONIZE
	h, e := syscall.OpenProcess(da, false, uint32(pid))
	if e != 0 {
		return nil, NewSyscallError("OpenProcess", e)
	}
	return newProcess(pid, int(h)), nil
}
