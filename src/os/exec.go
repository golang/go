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

type processMode uint8

const (
	// modePID means that Process operations such use the raw PID from the
	// Pid field. handle is not used.
	//
	// This may be due to the host not supporting handles, or because
	// Process was created as a literal, leaving handle unset.
	//
	// This must be the zero value so Process literals get modePID.
	modePID processMode = iota

	// modeHandle means that Process operations use handle, which is
	// initialized with an OS process handle.
	//
	// Note that Release and Wait will deactivate and eventually close the
	// handle, so acquire may fail, indicating the reason.
	modeHandle
)

type processStatus uint64

const (
	// PID/handle OK to use.
	statusOK processStatus = 0

	// statusDone indicates that the PID/handle should not be used because
	// the process is done (has been successfully Wait'd on).
	statusDone processStatus = 1 << 62

	// statusReleased indicates that the PID/handle should not be used
	// because the process is released.
	statusReleased processStatus = 1 << 63

	processStatusMask = 0x3 << 62
)

// Process stores the information about a process created by [StartProcess].
type Process struct {
	Pid int

	mode processMode

	// State contains the atomic process state.
	//
	// In modePID, this consists only of the processStatus fields, which
	// indicate if the process is done/released.
	//
	// In modeHandle, the lower bits also contain a reference count for the
	// handle field.
	//
	// The Process itself initially holds 1 persistent reference. Any
	// operation that uses the handle with a system call temporarily holds
	// an additional transient reference. This prevents the handle from
	// being closed prematurely, which could result in the OS allocating a
	// different handle with the same value, leading to Process' methods
	// operating on the wrong process.
	//
	// Release and Wait both drop the Process' persistent reference, but
	// other concurrent references may delay actually closing the handle
	// because they hold a transient reference.
	//
	// Regardless, we want new method calls to immediately treat the handle
	// as unavailable after Release or Wait to avoid extending this delay.
	// This is achieved by setting either processStatus flag when the
	// Process' persistent reference is dropped. The only difference in the
	// flags is the reason the handle is unavailable, which affects the
	// errors returned by concurrent calls.
	state atomic.Uint64

	// Used only in modePID.
	sigMu sync.RWMutex // avoid race between wait and signal

	// handle, if not nil, is a pointer to a struct
	// that holds the OS-specific process handle.
	// This pointer is set when Process is created,
	// and never changed afterward.
	// This is a pointer to a separate memory allocation
	// so that we can use runtime.AddCleanup.
	handle *processHandle
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
		Pid:  pid,
		mode: modePID,
	}
	runtime.SetFinalizer(p, (*Process).Release)
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
		mode:   modeHandle,
		handle: ph,
	}
	p.state.Store(1) // 1 persistent reference
	runtime.SetFinalizer(p, (*Process).Release)
	return p
}

func newDoneProcess(pid int) *Process {
	p := &Process{
		Pid:  pid,
		mode: modePID,
	}
	p.state.Store(uint64(statusDone)) // No persistent reference, as there is no handle.
	runtime.SetFinalizer(p, (*Process).Release)
	return p
}

func (p *Process) handleTransientAcquire() (uintptr, processStatus) {
	if p.mode != modeHandle {
		panic("handleTransientAcquire called in invalid mode")
	}

	for {
		refs := p.state.Load()
		if refs&processStatusMask != 0 {
			return 0, processStatus(refs & processStatusMask)
		}
		new := refs + 1
		if !p.state.CompareAndSwap(refs, new) {
			continue
		}
		h, ok := p.handle.acquire()
		if !ok {
			panic("inconsistent reference counts")
		}
		return h, statusOK
	}
}

func (p *Process) handleTransientRelease() {
	if p.mode != modeHandle {
		panic("handleTransientRelease called in invalid mode")
	}

	for {
		state := p.state.Load()
		refs := state &^ processStatusMask
		status := processStatus(state & processStatusMask)
		if refs == 0 {
			// This should never happen because
			// handleTransientRelease is always paired with
			// handleTransientAcquire.
			panic("release of handle with refcount 0")
		}
		if refs == 1 && status == statusOK {
			// Process holds a persistent reference and always sets
			// a status when releasing that reference
			// (handlePersistentRelease). Thus something has gone
			// wrong if this is the last release but a status has
			// not always been set.
			panic("final release of handle without processStatus")
		}
		new := state - 1
		if !p.state.CompareAndSwap(state, new) {
			continue
		}
		p.handle.release()
		return
	}
}

// Drop the Process' persistent reference on the handle, deactivating future
// Wait/Signal calls with the passed reason.
//
// Returns the status prior to this call. If this is not statusOK, then the
// reference was not dropped or status changed.
func (p *Process) handlePersistentRelease(reason processStatus) processStatus {
	if p.mode != modeHandle {
		panic("handlePersistentRelease called in invalid mode")
	}

	for {
		refs := p.state.Load()
		status := processStatus(refs & processStatusMask)
		if status != statusOK {
			// Both Release and successful Wait will drop the
			// Process' persistent reference on the handle. We
			// can't allow concurrent calls to drop the reference
			// twice, so we use the status as a guard to ensure the
			// reference is dropped exactly once.
			return status
		}
		if refs == 0 {
			// This should never happen because dropping the
			// persistent reference always sets a status.
			panic("release of handle with refcount 0")
		}
		new := (refs - 1) | uint64(reason)
		if !p.state.CompareAndSwap(refs, new) {
			continue
		}
		p.handle.release()
		return status
	}
}

func (p *Process) pidStatus() processStatus {
	if p.mode != modePID {
		panic("pidStatus called in invalid mode")
	}

	return processStatus(p.state.Load())
}

func (p *Process) pidDeactivate(reason processStatus) {
	if p.mode != modePID {
		panic("pidDeactivate called in invalid mode")
	}

	// Both Release and successful Wait will deactivate the PID. Only one
	// of those should win, so nothing left to do here if the compare
	// fails.
	//
	// N.B. This means that results can be inconsistent. e.g., with a
	// racing Release and Wait, Wait may successfully wait on the process,
	// returning the wait status, while future calls error with "process
	// released" rather than "process done".
	p.state.CompareAndSwap(0, uint64(reason))
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
	// Note to future authors: the Release API is cursed.
	//
	// On Unix and Plan 9, Release sets p.Pid = -1. This is the only part of the
	// Process API that is not thread-safe, but it can't be changed now.
	//
	// On Windows, Release does _not_ modify p.Pid.
	//
	// On Windows, Wait calls Release after successfully waiting to
	// proactively clean up resources.
	//
	// On Unix and Plan 9, Wait also proactively cleans up resources, but
	// can not call Release, as Wait does not set p.Pid = -1.
	//
	// On Unix and Plan 9, calling Release a second time has no effect.
	//
	// On Windows, calling Release a second time returns EINVAL.
	return p.release()
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
