// Copyright 2009 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package os

import (
	"errors"
	"runtime"
	"syscall"
	"unsafe"
)

// Wait waits for the Process to exit or stop, and then returns a
// Waitmsg describing its status and an error, if any.
func (p *Process) Wait(options int) (w *Waitmsg, err error) {
	s, e := syscall.WaitForSingleObject(syscall.Handle(p.handle), syscall.INFINITE)
	switch s {
	case syscall.WAIT_OBJECT_0:
		break
	case syscall.WAIT_FAILED:
		return nil, NewSyscallError("WaitForSingleObject", e)
	default:
		return nil, errors.New("os: unexpected result from WaitForSingleObject")
	}
	var ec uint32
	e = syscall.GetExitCodeProcess(syscall.Handle(p.handle), &ec)
	if e != nil {
		return nil, NewSyscallError("GetExitCodeProcess", e)
	}
	p.done = true
	return &Waitmsg{p.Pid, syscall.WaitStatus{Status: s, ExitCode: ec}, new(syscall.Rusage)}, nil
}

// Signal sends a signal to the Process.
func (p *Process) Signal(sig Signal) error {
	if p.done {
		return errors.New("os: process already finished")
	}
	if us, ok := sig.(UnixSignal); ok && us == syscall.SIGKILL {
		e := syscall.TerminateProcess(syscall.Handle(p.handle), 1)
		return NewSyscallError("TerminateProcess", e)
	}
	return syscall.Errno(syscall.EWINDOWS)
}

// Release releases any resources associated with the Process.
func (p *Process) Release() error {
	if p.handle == uintptr(syscall.InvalidHandle) {
		return EINVAL
	}
	e := syscall.CloseHandle(syscall.Handle(p.handle))
	if e != nil {
		return NewSyscallError("CloseHandle", e)
	}
	p.handle = uintptr(syscall.InvalidHandle)
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
		Args[i] = string(syscall.UTF16ToString((*v)[:]))
	}
}
