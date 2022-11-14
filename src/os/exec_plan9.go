// Copyright 2009 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package os

import (
	"internal/itoa"
	"runtime"
	"syscall"
	"time"
)

// The only signal values guaranteed to be present in the os package
// on all systems are Interrupt (send the process an interrupt) and
// Kill (force the process to exit). Interrupt is not implemented on
// Windows; using it with os.Process.Signal will return an error.
var (
	Interrupt Signal = syscall.Note("interrupt")
	Kill      Signal = syscall.Note("kill")
)

func startProcess(name string, argv []string, attr *ProcAttr) (p *Process, err error) {
	sysattr := &syscall.ProcAttr{
		Dir: attr.Dir,
		Env: attr.Env,
		Sys: attr.Sys,
	}

	sysattr.Files = make([]uintptr, 0, len(attr.Files))
	for _, f := range attr.Files {
		sysattr.Files = append(sysattr.Files, f.Fd())
	}

	pid, h, e := syscall.StartProcess(name, argv, sysattr)
	if e != nil {
		return nil, &PathError{Op: "fork/exec", Path: name, Err: e}
	}

	return newProcess(pid, h), nil
}

func (p *Process) writeProcFile(file string, data string) error {
	f, e := OpenFile("/proc/"+itoa.Itoa(p.Pid)+"/"+file, O_WRONLY, 0)
	if e != nil {
		return e
	}
	defer f.Close()
	_, e = f.Write([]byte(data))
	return e
}

func (p *Process) signal(sig Signal) error {
	if p.done() {
		return ErrProcessDone
	}
	if e := p.writeProcFile("note", sig.String()); e != nil {
		return NewSyscallError("signal", e)
	}
	return nil
}

func (p *Process) kill() error {
	return p.signal(Kill)
}

func (p *Process) wait() (ps *ProcessState, err error) {
	var waitmsg syscall.Waitmsg

	if p.Pid == -1 {
		return nil, ErrInvalid
	}
	err = syscall.WaitProcess(p.Pid, &waitmsg)
	if err != nil {
		return nil, NewSyscallError("wait", err)
	}

	p.setDone()
	ps = &ProcessState{
		pid:    waitmsg.Pid,
		status: &waitmsg,
	}
	return ps, nil
}

func (p *Process) release() error {
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

// ProcessState stores information about a process, as reported by Wait.
type ProcessState struct {
	pid    int              // The process's id.
	status *syscall.Waitmsg // System-dependent status info.
}

// Pid returns the process id of the exited process.
func (p *ProcessState) Pid() int {
	return p.pid
}

func (p *ProcessState) exited() bool {
	return p.status.Exited()
}

func (p *ProcessState) success() bool {
	return p.status.ExitStatus() == 0
}

func (p *ProcessState) sys() any {
	return p.status
}

func (p *ProcessState) sysUsage() any {
	return p.status
}

func (p *ProcessState) userTime() time.Duration {
	return time.Duration(p.status.Time[0]) * time.Millisecond
}

func (p *ProcessState) systemTime() time.Duration {
	return time.Duration(p.status.Time[1]) * time.Millisecond
}

func (p *ProcessState) String() string {
	if p == nil {
		return "<nil>"
	}
	return "exit status: " + p.status.Msg
}

// ExitCode returns the exit code of the exited process, or -1
// if the process hasn't exited or was terminated by a signal.
func (p *ProcessState) ExitCode() int {
	// return -1 if the process hasn't started.
	if p == nil {
		return -1
	}
	return p.status.ExitStatus()
}
