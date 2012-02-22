// Copyright 2009 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package os

import (
	"errors"
	"runtime"
	"syscall"
	"time"
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

	for _, f := range attr.Files {
		sysattr.Files = append(sysattr.Files, f.Fd())
	}

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

// Wait waits for the Process to exit or stop, and then returns a
// ProcessState describing its status and an error, if any.
func (p *Process) Wait() (ps *ProcessState, err error) {
	var waitmsg syscall.Waitmsg

	if p.Pid == -1 {
		return nil, ErrInvalid
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

	ps = &ProcessState{
		pid:    waitmsg.Pid,
		status: &waitmsg,
	}
	return ps, nil
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

// ProcessState stores information about process as reported by Wait.
type ProcessState struct {
	pid    int              // The process's id.
	status *syscall.Waitmsg // System-dependent status info.
}

// Pid returns the process id of the exited process.
func (p *ProcessState) Pid() int {
	return p.pid
}

// Exited returns whether the program has exited.
func (p *ProcessState) Exited() bool {
	return p.status.Exited()
}

// Success reports whether the program exited successfully,
// such as with exit status 0 on Unix.
func (p *ProcessState) Success() bool {
	return p.status.ExitStatus() == 0
}

// Sys returns system-dependent exit information about
// the process.  Convert it to the appropriate underlying
// type, such as *syscall.Waitmsg on Plan 9, to access its contents.
func (p *ProcessState) Sys() interface{} {
	return p.status
}

// SysUsage returns system-dependent resource usage information about
// the exited process.  Convert it to the appropriate underlying
// type, such as *syscall.Waitmsg on Plan 9, to access its contents.
func (p *ProcessState) SysUsage() interface{} {
	return p.status
}

// UserTime returns the user CPU time of the exited process and its children.
// It is always reported as 0 on Windows.
func (p *ProcessState) UserTime() time.Duration {
	return time.Duration(p.status.Time[0]) * time.Millisecond
}

// SystemTime returns the system CPU time of the exited process and its children.
// It is always reported as 0 on Windows.
func (p *ProcessState) SystemTime() time.Duration {
	return time.Duration(p.status.Time[1]) * time.Millisecond
}

func (p *ProcessState) String() string {
	if p == nil {
		return "<nil>"
	}
	return "exit status: " + p.status.Msg
}
