// Copyright 2009 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package os

import (
	"errors"
	"runtime"
	"syscall"
)

// StartProcess starts a new process with the program, arguments and attributes
// specified by name, argv and attr.
// If there is an error, it will be of type *PathError.
func StartProcess(name string, argv []string, attr *ProcAttr) (p *Process, err error) {
	sysattr := &syscall.ProcAttr{
		Dir: attr.Dir,
		Env: attr.Env,
		Sys: attr.Sys,
	}

	// Create array of integer (system) fds.
	intfd := make([]int, len(attr.Files))
	for i, f := range attr.Files {
		if f == nil {
			intfd[i] = -1
		} else {
			intfd[i] = f.Fd()
		}
	}

	sysattr.Files = intfd

	pid, h, e := syscall.StartProcess(name, argv, sysattr)
	if e != nil {
		return nil, &PathError{"fork/exec", name, e}
	}

	return newProcess(pid, h), nil
}

// Plan9Note implements the Signal interface on Plan 9.
type Plan9Note string

func (note Plan9Note) String() string {
	return string(note)
}

func (p *Process) Signal(sig Signal) error {
	if p.done {
		return errors.New("os: process already finished")
	}

	f, e := OpenFile("/proc/"+itoa(p.Pid)+"/note", O_WRONLY, 0)
	if e != nil {
		return NewSyscallError("signal", e)
	}
	defer f.Close()
	_, e = f.Write([]byte(sig.String()))
	return e
}

// Kill causes the Process to exit immediately.
func (p *Process) Kill() error {
	f, e := OpenFile("/proc/"+itoa(p.Pid)+"/ctl", O_WRONLY, 0)
	if e != nil {
		return NewSyscallError("kill", e)
	}
	defer f.Close()
	_, e = f.Write([]byte("kill"))
	return e
}

// Waitmsg stores the information about an exited process as reported by Wait.
type Waitmsg struct {
	syscall.Waitmsg
}

// Wait waits for the Process to exit or stop, and then returns a
// Waitmsg describing its status and an error, if any. The options
// (WNOHANG etc.) affect the behavior of the Wait call.
func (p *Process) Wait(options int) (w *Waitmsg, err error) {
	var waitmsg syscall.Waitmsg

	if p.Pid == -1 {
		return nil, EINVAL
	}

	for true {
		err = syscall.Await(&waitmsg)

		if err != nil {
			return nil, NewSyscallError("wait", err)
		}

		if waitmsg.Pid == p.Pid {
			p.done = true
			break
		}
	}

	return &Waitmsg{waitmsg}, nil
}

// Wait waits for process pid to exit or stop, and then returns a
// Waitmsg describing its status and an error, if any. The options
// (WNOHANG etc.) affect the behavior of the Wait call.
// Wait is equivalent to calling FindProcess and then Wait
// and Release on the result.
func Wait(pid int, options int) (w *Waitmsg, err error) {
	p, e := FindProcess(pid)
	if e != nil {
		return nil, e
	}
	defer p.Release()
	return p.Wait(options)
}

// Release releases any resources associated with the Process.
func (p *Process) Release() error {
	// NOOP for Plan 9.
	p.Pid = -1
	// no need for a finalizer anymore
	runtime.SetFinalizer(p, nil)
	return nil
}

func findProcess(pid int) (p *Process, err error) {
	// NOOP for Plan 9.
	return newProcess(pid, 0), nil
}

func (w *Waitmsg) String() string {
	if w == nil {
		return "<nil>"
	}
	return "exit status: " + w.Msg
}
