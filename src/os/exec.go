// Copyright 2009 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package os

import (
	"errors"
	"internal/testlog"
	"runtime"
	"sync"
	"sync/atomic"
	"syscall"
	"time"
)

// ErrProcessDone indicates a [Process] has finished.
var ErrProcessDone = errors.New("os: process already finished")

type processStatus uint32

const (
	// statusOK means that the Process is ready to use.
	statusOK processStatus = iota

	// statusDone indicates that the PID/handle should not be used because
	// the process is done (has been successfully Wait'd on).
	statusDone

	// statusReleased indicates that the PID/handle should not be used
	// because the process is released.
	statusReleased
)

// Process stores the information about a process created by [StartProcess].
type Process struct {
	Pid int

	// state contains the atomic process state.
	//
	// This consists of the processStatus fields,
	// which indicate if the process is done/released.
	state atomic.Uint32

	// Used only when handle is nil
	sigMu sync.RWMutex // avoid race between wait and signal

	// handle, if not nil, is a pointer to a struct
	// that holds the OS-specific process handle.
	// This pointer is set when Process is created,
	// and never changed afterward.
	// This is a pointer to a separate memory allocation
	// so that we can use runtime.AddCleanup.
	handle *processHandle

	// cleanup is used to clean up the process handle.
	cleanup runtime.Cleanup
}

// processHandle holds an operating system handle to a process.
// This is only used on systems that support that concept,
// currently Linux and Windows.
// This maintains a reference count to the handle,
// and closes the handle when the reference drops to zero.
type processHandle struct {
	// The actual handle. This field should not be used directly.
	// Instead, use the acquire and release methods.
	//
	// On Windows this is a handle returned by OpenProcess.
	// On Linux this is a pidfd.
	handle uintptr

	// Number of active references. When this drops to zero
	// the handle is closed.
	refs atomic.Int32
}

// acquire adds a reference and returns the handle.
// The bool result reports whether acquire succeeded;
// it fails if the handle is already closed.
// Every successful call to acquire should be paired with a call to release.
func (ph *processHandle) acquire() (uintptr, bool) {
	for {
		refs := ph.refs.Load()
		if refs < 0 {
			panic("internal error: negative process handle reference count")
		}
		if refs == 0 {
			return 0, false
		}
		if ph.refs.CompareAndSwap(refs, refs+1) {
			return ph.handle, true
		}
	}
}

// release releases a reference to the handle.
func (ph *processHandle) release() {
	for {
		refs := ph.refs.Load()
		if refs <= 0 {
			panic("internal error: too many releases of process handle")
		}
		if ph.refs.CompareAndSwap(refs, refs-1) {
			if refs == 1 {
				ph.closeHandle()
			}
			return
		}
	}
}

func newPIDProcess(pid int) *Process {
	p := &Process{
		Pid: pid,
	}
	return p
}

func newHandleProcess(pid int, handle uintptr) *Process {
	ph := &processHandle{
		handle: handle,
	}

	// Start the reference count as 1,
	// meaning the reference from the returned Process.
	ph.refs.Store(1)

	p := &Process{
		Pid:    pid,
		handle: ph,
	}

	p.cleanup = runtime.AddCleanup(p, (*processHandle).release, ph)

	return p
}

func newDoneProcess(pid int) *Process {
	p := &Process{
		Pid: pid,
	}
	p.state.Store(uint32(statusDone)) // No persistent reference, as there is no handle.
	return p
}

func (p *Process) handleTransientAcquire() (uintptr, processStatus) {
	if p.handle == nil {
		panic("handleTransientAcquire called in invalid mode")
	}

	status := processStatus(p.state.Load())
	if status != statusOK {
		return 0, status
	}
	h, ok := p.handle.acquire()
	if ok {
		return h, statusOK
	}

	// This case means that the handle has been closed.
	// We always set the status to non-zero before closing the handle.
	// If we get here the status must have been set non-zero after
	// we just checked it above.
	status = processStatus(p.state.Load())
	if status == statusOK {
		panic("inconsistent process status")
	}
	return 0, status
}

func (p *Process) handleTransientRelease() {
	if p.handle == nil {
		panic("handleTransientRelease called in invalid mode")
	}
	p.handle.release()
}

func (p *Process) pidStatus() processStatus {
	if p.handle != nil {
		panic("pidStatus called in invalid mode")
	}

	return processStatus(p.state.Load())
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
	// Files specifies the open files inherited by the new process. The
	// first three entries correspond to standard input, standard output, and
	// standard error. An implementation may support additional entries,
	// depending on the underlying operating system. A nil entry corresponds
	// to that file being closed when the process starts.
	// On Unix systems, StartProcess will change these File values
	// to blocking mode, which means that SetDeadline will stop working
	// and calling Close will not interrupt a Read or Write.
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

// Getpid returns the process id of the caller.
func Getpid() int { return syscall.Getpid() }

// Getppid returns the process id of the caller's parent.
func Getppid() int { return syscall.Getppid() }

// FindProcess looks for a running process by its pid.
//
// The [Process] it returns can be used to obtain information
// about the underlying operating system process.
//
// On Unix systems, FindProcess always succeeds and returns a Process
// for the given pid, regardless of whether the process exists. To test whether
// the process actually exists, see whether p.Signal(syscall.Signal(0)) reports
// an error.
func FindProcess(pid int) (*Process, error) {
	return findProcess(pid)
}

// StartProcess starts a new process with the program, arguments and attributes
// specified by name, argv and attr. The argv slice will become [os.Args] in the
// new process, so it normally starts with the program name.
//
// If the calling goroutine has locked the operating system thread
// with [runtime.LockOSThread] and modified any inheritable OS-level
// thread state (for example, Linux or Plan 9 name spaces), the new
// process will inherit the caller's thread state.
//
// StartProcess is a low-level interface. The [os/exec] package provides
// higher-level interfaces.
//
// If there is an error, it will be of type [*PathError].
func StartProcess(name string, argv []string, attr *ProcAttr) (*Process, error) {
	testlog.Open(name)
	return startProcess(name, argv, attr)
}

// Release releases any resources associated with the [Process] p,
// rendering it unusable in the future.
// Release only needs to be called if [Process.Wait] is not.
func (p *Process) Release() error {
	// Unfortunately, for historical reasons, on systems other
	// than Windows, Release sets the Pid field to -1.
	// This causes the race detector to report a problem
	// on concurrent calls to Release, but we can't change it now.
	if runtime.GOOS != "windows" {
		p.Pid = -1
	}

	oldStatus := p.doRelease(statusReleased)

	// For backward compatibility, on Windows only,
	// we return EINVAL on a second call to Release.
	if runtime.GOOS == "windows" {
		if oldStatus == statusReleased {
			return syscall.EINVAL
		}
	}

	return nil
}

// doRelease releases a [Process], setting the status to newStatus.
// If the previous status is not statusOK, this does nothing.
// It returns the previous status.
func (p *Process) doRelease(newStatus processStatus) processStatus {
	for {
		state := p.state.Load()
		oldStatus := processStatus(state)
		if oldStatus != statusOK {
			return oldStatus
		}

		if !p.state.CompareAndSwap(state, uint32(newStatus)) {
			continue
		}

		// We have successfully released the Process.
		// If it has a handle, release the reference we
		// created in newHandleProcess.
		if p.handle != nil {
			// No need for more cleanup.
			p.cleanup.Stop()

			p.handle.release()
		}

		return statusOK
	}
}

// Kill causes the [Process] to exit immediately. Kill does not wait until
// the Process has actually exited. This only kills the Process itself,
// not any other processes it may have started.
func (p *Process) Kill() error {
	return p.kill()
}

// Wait waits for the [Process] to exit, and then returns a
// ProcessState describing its status and an error, if any.
// Wait releases any resources associated with the Process.
// On most operating systems, the Process must be a child
// of the current process or an error will be returned.
func (p *Process) Wait() (*ProcessState, error) {
	return p.wait()
}

// Signal sends a signal to the [Process].
// Sending [Interrupt] on Windows is not implemented.
func (p *Process) Signal(sig Signal) error {
	return p.signal(sig)
}

// UserTime returns the user CPU time of the exited process and its children.
func (p *ProcessState) UserTime() time.Duration {
	return p.userTime()
}

// SystemTime returns the system CPU time of the exited process and its children.
func (p *ProcessState) SystemTime() time.Duration {
	return p.systemTime()
}

// Exited reports whether the program has exited.
// On Unix systems this reports true if the program exited due to calling exit,
// but false if the program terminated due to a signal.
func (p *ProcessState) Exited() bool {
	return p.exited()
}

// Success reports whether the program exited successfully,
// such as with exit status 0 on Unix.
func (p *ProcessState) Success() bool {
	return p.success()
}

// Sys returns system-dependent exit information about
// the process. Convert it to the appropriate underlying
// type, such as [syscall.WaitStatus] on Unix, to access its contents.
func (p *ProcessState) Sys() any {
	return p.sys()
}

// SysUsage returns system-dependent resource usage information about
// the exited process. Convert it to the appropriate underlying
// type, such as [*syscall.Rusage] on Unix, to access its contents.
// (On Unix, *syscall.Rusage matches struct rusage as defined in the
// getrusage(2) manual page.)
func (p *ProcessState) SysUsage() any {
	return p.sysUsage()
}
