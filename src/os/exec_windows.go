// Copyright 2009 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package os

import (
	"errors"
	"runtime"
	"sync/atomic"
	"syscall"
	"time"
	"unsafe"
)

func (p *Process) wait() (ps *ProcessState, err error) {
	handle := atomic.LoadUintptr(&p.handle)
	s, e := syscall.WaitForSingleObject(syscall.Handle(handle), syscall.INFINITE)
	switch s {
	case syscall.WAIT_OBJECT_0:
		break
	case syscall.WAIT_FAILED:
		return nil, NewSyscallError("WaitForSingleObject", e)
	default:
		return nil, errors.New("os: unexpected result from WaitForSingleObject")
	}
	var ec uint32
	e = syscall.GetExitCodeProcess(syscall.Handle(handle), &ec)
	if e != nil {
		return nil, NewSyscallError("GetExitCodeProcess", e)
	}
	var u syscall.Rusage
	e = syscall.GetProcessTimes(syscall.Handle(handle), &u.CreationTime, &u.ExitTime, &u.KernelTime, &u.UserTime)
	if e != nil {
		return nil, NewSyscallError("GetProcessTimes", e)
	}
	p.setDone()
	// NOTE(brainman): It seems that sometimes process is not dead
	// when WaitForSingleObject returns. But we do not know any
	// other way to wait for it. Sleeping for a while seems to do
	// the trick sometimes. So we will sleep and smell the roses.
	defer time.Sleep(5 * time.Millisecond)
	defer p.Release()
	return &ProcessState{p.Pid, syscall.WaitStatus{ExitCode: ec}, &u}, nil
}

func terminateProcess(pid, exitcode int) error {
	h, e := syscall.OpenProcess(syscall.PROCESS_TERMINATE, false, uint32(pid))
	if e != nil {
		return NewSyscallError("OpenProcess", e)
	}
	defer syscall.CloseHandle(h)
	e = syscall.TerminateProcess(h, uint32(exitcode))
	return NewSyscallError("TerminateProcess", e)
}

func (p *Process) signal(sig Signal) error {
	handle := atomic.LoadUintptr(&p.handle)
	if handle == uintptr(syscall.InvalidHandle) {
		return syscall.EINVAL
	}
	if p.done() {
		return errors.New("os: process already finished")
	}
	if sig == Kill {
		return terminateProcess(p.Pid, 1)
	}
	// TODO(rsc): Handle Interrupt too?
	return syscall.Errno(syscall.EWINDOWS)
}

func (p *Process) release() error {
	handle := atomic.LoadUintptr(&p.handle)
	if handle == uintptr(syscall.InvalidHandle) {
		return syscall.EINVAL
	}
	e := syscall.CloseHandle(syscall.Handle(handle))
	if e != nil {
		return NewSyscallError("CloseHandle", e)
	}
	atomic.StoreUintptr(&p.handle, uintptr(syscall.InvalidHandle))
	// no need for a finalizer anymore
	runtime.SetFinalizer(p, nil)
	return nil
}

func findProcess(pid int) (p *Process, err error) {
	const da = syscall.STANDARD_RIGHTS_READ |
		syscall.PROCESS_QUERY_INFORMATION | syscall.SYNCHRONIZE
	h, e := syscall.OpenProcess(da, false, uint32(pid))
	if e != nil {
		return nil, NewSyscallError("OpenProcess", e)
	}
	return newProcess(pid, uintptr(h)), nil
}

func init() {
	var argc int32
	cmd := syscall.GetCommandLine()
	argv, e := syscall.CommandLineToArgv(cmd, &argc)
	if e != nil {
		return
	}
	defer syscall.LocalFree(syscall.Handle(uintptr(unsafe.Pointer(argv))))
	Args = make([]string, argc)
	for i, v := range (*argv)[:argc] {
		Args[i] = syscall.UTF16ToString((*v)[:])
	}
}

func ftToDuration(ft *syscall.Filetime) time.Duration {
	n := int64(ft.HighDateTime)<<32 + int64(ft.LowDateTime) // in 100-nanosecond intervals
	return time.Duration(n*100) * time.Nanosecond
}

func (p *ProcessState) userTime() time.Duration {
	return ftToDuration(&p.rusage.UserTime)
}

func (p *ProcessState) systemTime() time.Duration {
	return ftToDuration(&p.rusage.KernelTime)
}
