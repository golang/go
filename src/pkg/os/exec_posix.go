// Copyright 2009 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// +build darwin freebsd linux openbsd windows

package os

import (
	"runtime"
	"syscall"
)

type UnixSignal int32

func (sig UnixSignal) String() string {
	s := runtime.Signame(int32(sig))
	if len(s) > 0 {
		return s
	}
	return "UnixSignal"
}

// StartProcess starts a new process with the program, arguments and attributes
// specified by name, argv and attr.
//
// StartProcess is a low-level interface. The exec package provides
// higher-level interfaces.
func StartProcess(name string, argv []string, attr *ProcAttr) (p *Process, err Error) {
	sysattr := &syscall.ProcAttr{
		Dir: attr.Dir,
		Env: attr.Env,
		Sys: attr.Sys,
	}
	if sysattr.Env == nil {
		sysattr.Env = Environ()
	}
	for _, f := range attr.Files {
		sysattr.Files = append(sysattr.Files, f.Fd())
	}

	pid, h, e := syscall.StartProcess(name, argv, sysattr)
	if iserror(e) {
		return nil, &PathError{"fork/exec", name, Errno(e)}
	}
	return newProcess(pid, h), nil
}

// Kill causes the Process to exit immediately.
func (p *Process) Kill() Error {
	return p.Signal(SIGKILL)
}

// Exec replaces the current process with an execution of the
// named binary, with arguments argv and environment envv.
// If successful, Exec never returns.  If it fails, it returns an Error.
//
// To run a child process, see StartProcess (for a low-level interface)
// or the exec package (for higher-level interfaces).
func Exec(name string, argv []string, envv []string) Error {
	if envv == nil {
		envv = Environ()
	}
	e := syscall.Exec(name, argv, envv)
	if iserror(e) {
		return &PathError{"exec", name, Errno(e)}
	}
	return nil
}

// TODO(rsc): Should os implement its own syscall.WaitStatus
// wrapper with the methods, or is exposing the underlying one enough?
//
// TODO(rsc): Certainly need to have Rusage struct,
// since syscall one might have different field types across
// different OS.

// Waitmsg stores the information about an exited process as reported by Wait.
type Waitmsg struct {
	Pid                int             // The process's id.
	syscall.WaitStatus                 // System-dependent status info.
	Rusage             *syscall.Rusage // System-dependent resource usage info.
}

// Wait waits for process pid to exit or stop, and then returns a
// Waitmsg describing its status and an Error, if any. The options
// (WNOHANG etc.) affect the behavior of the Wait call.
// Wait is equivalent to calling FindProcess and then Wait
// and Release on the result.
func Wait(pid int, options int) (w *Waitmsg, err Error) {
	p, e := FindProcess(pid)
	if e != nil {
		return nil, e
	}
	defer p.Release()
	return p.Wait(options)
}

// Convert i to decimal string.
func itod(i int) string {
	if i == 0 {
		return "0"
	}

	u := uint64(i)
	if i < 0 {
		u = -u
	}

	// Assemble decimal in reverse order.
	var b [32]byte
	bp := len(b)
	for ; u > 0; u /= 10 {
		bp--
		b[bp] = byte(u%10) + '0'
	}

	if i < 0 {
		bp--
		b[bp] = '-'
	}

	return string(b[bp:])
}

func (w *Waitmsg) String() string {
	if w == nil {
		return "<nil>"
	}
	// TODO(austin) Use signal names when possible?
	res := ""
	switch {
	case w.Exited():
		res = "exit status " + itod(w.ExitStatus())
	case w.Signaled():
		res = "signal " + itod(w.Signal())
	case w.Stopped():
		res = "stop signal " + itod(w.StopSignal())
		if w.StopSignal() == syscall.SIGTRAP && w.TrapCause() != 0 {
			res += " (trap " + itod(w.TrapCause()) + ")"
		}
	case w.Continued():
		res = "continued"
	}
	if w.CoreDump() {
		res += " (core dumped)"
	}
	return res
}
