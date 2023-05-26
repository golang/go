// Copyright 2009 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

//go:build unix || (js && wasm) || wasip1 || windows

package os

import (
	"internal/itoa"
	"internal/syscall/execenv"
	"runtime"
	"syscall"
)

// The only signal values guaranteed to be present in the os package on all
// systems are os.Interrupt (send the process an interrupt) and os.Kill (force
// the process to exit). On Windows, sending os.Interrupt to a process with
// os.Process.Signal is not implemented; it will return an error instead of
// sending a signal.
var (
	Interrupt Signal = syscall.SIGINT
	Kill      Signal = syscall.SIGKILL
)

func startProcess(name string, argv []string, attr *ProcAttr) (p *Process, err error) {
	// If there is no SysProcAttr (ie. no Chroot or changed
	// UID/GID), double-check existence of the directory we want
	// to chdir into. We can make the error clearer this way.
	if attr != nil && attr.Sys == nil && attr.Dir != "" {
		if _, err := Stat(attr.Dir); err != nil {
			pe := err.(*PathError)
			pe.Op = "chdir"
			return nil, pe
		}
	}

	sysattr := &syscall.ProcAttr{
		Dir: attr.Dir,
		Env: attr.Env,
		Sys: attr.Sys,
	}
	if sysattr.Env == nil {
		sysattr.Env, err = execenv.Default(sysattr.Sys)
		if err != nil {
			return nil, err
		}
	}
	sysattr.Files = make([]uintptr, 0, len(attr.Files))
	for _, f := range attr.Files {
		sysattr.Files = append(sysattr.Files, f.Fd())
	}

	pid, h, e := syscall.StartProcess(name, argv, sysattr)

	// Make sure we don't run the finalizers of attr.Files.
	runtime.KeepAlive(attr)

	if e != nil {
		return nil, &PathError{Op: "fork/exec", Path: name, Err: e}
	}

	return newProcess(pid, h), nil
}

func (p *Process) kill() error {
	return p.Signal(Kill)
}

// ProcessState stores information about a process, as reported by Wait.
type ProcessState struct {
	pid     int     // The process's id.
	siginfo Siginfo // System-dependent status info.
	rusage  *syscall.Rusage
}

// Pid returns the process id of the exited process.
func (p *ProcessState) Pid() int {
	return p.pid
}

func (p *ProcessState) exited() bool {
	return p.siginfo.Exited()
}

func (p *ProcessState) success() bool {
	return p.siginfo.ExitStatus() == 0
}

func (p *ProcessState) sys() any {
	return p.siginfo
}

func (p *ProcessState) sysUsage() any {
	return p.rusage
}

func (p *ProcessState) String() string {
	if p == nil {
		return "<nil>"
	}
	siginfo := p.Sys().(Siginfo)
	res := ""
	switch {
	case siginfo.Exited():
		code := siginfo.ExitStatus()
		if runtime.GOOS == "windows" && uint(code) >= 1<<16 { // windows uses large hex numbers
			res = "exit status " + uitox(uint(code))
		} else { // unix systems use small decimal integers
			res = "exit status " + itoa.Itoa(code) // unix
		}
	case siginfo.Signaled():
		res = "signal: " + siginfo.Signal().String()
	case siginfo.Stopped():
		res = "stop signal: " + siginfo.StopSignal().String()
		if siginfo.Trapped() && siginfo.TrapCause() != 0 {
			res += " (trap " + itoa.Itoa(siginfo.TrapCause()) + ")"
		}
	case siginfo.Continued():
		res = "continued"
	}
	if siginfo.CoreDump() {
		res += " (core dumped)"
	}
	return res
}

// ExitCode returns the exit code of the exited process, or -1
// if the process hasn't exited or was terminated by a signal.
func (p *ProcessState) ExitCode() int {
	// return -1 if the process hasn't started.
	if p == nil {
		return -1
	}
	return p.siginfo.ExitStatus()
}
