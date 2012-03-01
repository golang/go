// Copyright 2009 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package os

import (
	"runtime"
	"syscall"
)

// Process stores the information about a process created by StartProcess.
type Process struct {
	Pid    int
	handle uintptr
	done   bool // process has been successfully waited on
}

func newProcess(pid int, handle uintptr) *Process {
	p := &Process{Pid: pid, handle: handle}
	runtime.SetFinalizer(p, (*Process).Release)
	return p
}

// ProcAttr holds the attributes that will be applied to a new process
// started by StartProcess.
type ProcAttr struct {
	// If Dir is non-empty, the child changes into the directory before
	// creating the process.
	Dir string
	// If Env is non-nil, it gives the environment variables for the
	// new process in the form returned by Environ.
	// If it is nil, the result of Environ will be used.
	Env []string
	// Files specifies the open files inherited by the new process.  The
	// first three entries correspond to standard input, standard output, and
	// standard error.  An implementation may support additional entries,
	// depending on the underlying operating system.  A nil entry corresponds
	// to that file being closed when the process starts.
	Files []*File

	// Operating system-specific process creation attributes.
	// Note that setting this field means that your program
	// may not execute properly or even compile on some
	// operating systems.
	Sys *syscall.SysProcAttr
}

// A Signal represents an operating system signal.
// The usual underlying implementation is operating system-dependent:
// on Unix it is syscall.Signal.
type Signal interface {
	String() string
	Signal() // to distinguish from other Stringers
}

// The only signal values guaranteed to be present on all systems
// are Interrupt (send the process an interrupt) and
// Kill (force the process to exit).
var (
	Interrupt Signal = syscall.SIGINT
	Kill      Signal = syscall.SIGKILL
)

// Getpid returns the process id of the caller.
func Getpid() int { return syscall.Getpid() }

// Getppid returns the process id of the caller's parent.
func Getppid() int { return syscall.Getppid() }
