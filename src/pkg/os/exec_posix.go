// Copyright 2009 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// +build darwin freebsd linux netbsd openbsd windows

package os

import (
	"syscall"
)

// The only signal values guaranteed to be present on all systems
// are Interrupt (send the process an interrupt) and Kill (force
// the process to exit).
var (
	Interrupt Signal = syscall.SIGINT
	Kill      Signal = syscall.SIGKILL
)

func startProcess(name string, argv []string, attr *ProcAttr) (p *Process, err error) {
	// If there is no SysProcAttr (ie. no Chroot or changed
	// UID/GID), double-check existence of the directory we want
	// to chdir into.  We can make the error clearer this way.
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
		sysattr.Env = Environ()
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

func (p *Process) kill() error {
	return p.Signal(Kill)
}

// ProcessState stores information about a process, as reported by Wait.
type ProcessState struct {
	pid    int                // The process's id.
	status syscall.WaitStatus // System-dependent status info.
	rusage *syscall.Rusage
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

func (p *ProcessState) sys() interface{} {
	return p.status
}

func (p *ProcessState) sysUsage() interface{} {
	return p.rusage
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

func (p *ProcessState) String() string {
	if p == nil {
		return "<nil>"
	}
	status := p.Sys().(syscall.WaitStatus)
	res := ""
	switch {
	case status.Exited():
		res = "exit status " + itod(status.ExitStatus())
	case status.Signaled():
		res = "signal " + itod(int(status.Signal()))
	case status.Stopped():
		res = "stop signal " + itod(int(status.StopSignal()))
		if status.StopSignal() == syscall.SIGTRAP && status.TrapCause() != 0 {
			res += " (trap " + itod(status.TrapCause()) + ")"
		}
	case status.Continued():
		res = "continued"
	}
	if status.CoreDump() {
		res += " (core dumped)"
	}
	return res
}
