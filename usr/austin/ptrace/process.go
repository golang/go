// Copyright 2009 The Go Authors.  All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// Package ptrace provides a platform-independent interface for
// tracing and controlling running processes.  It supports
// multi-threaded processes and provides typical low-level debugging
// controls such as breakpoints, single stepping, and manipulating
// memory and registers.
package ptrace

import (
	"os";
	"strconv";
)

type Word uint64

// A Cause explains why a thread is stopped.
type Cause interface {
	String() string;
}

// Regs is a set of named machine registers, including a program
// counter, link register, and stack pointer.
type Regs interface {
	// PC returns the value of the program counter.
	PC() Word;

	// Link returns the link register, if any.
	Link() Word;

	// SP returns the value of the stack pointer.
	SP() Word;

	// Names returns the names of all of the registers.
	Names() []string;

	// Get returns the value of a register, where i corresponds to
	// the index of the register's name in the array returned by
	// Names.
	Get(i int) Word;

	// Set sets the value of a register.
	Set(i int, val Word);
}

// Thread is a thread in the process being traced.
type Thread interface {
	// Step steps this thread by a single instruction.  The thread
	// must be stopped.  If the thread is currently stopped on a
	// breakpoint, this will step over the breakpoint.
	//
	// XXX What if it's stopped because of a signal?
	Step() os.Error;

	// Stopped returns the reason that this thread is stopped.  It
	// is an error is the thread not stopped.
	Stopped() (Cause, os.Error);

	// Regs retrieves the current register values from this
	// thread.  The thread must be stopped.
	Regs() (Regs, os.Error);

	// Peek reads len(out) bytes from the address addr in this
	// thread into out.  The thread must be stopped.  It returns
	// the number of bytes successfully read.  If an error occurs,
	// such as attempting to read unmapped memory, this count
	// could be short and an error will be returned.  If this does
	// encounter unmapped memory, it will read up to the byte
	// preceding the unmapped area.
	Peek(addr Word, out []byte) (int, os.Error);

	// Poke writes b to the address addr in this thread.  The
	// thread must be stopped.  It returns the number of bytes
	// successfully written.  If an error occurs, such as
	// attempting to write to unmapped memory, this count could be
	// short and an error will be returned.  If this does
	// encounter unmapped memory, it will write up to the byte
	// preceding the unmapped area.
	Poke(addr Word, b []byte) (int, os.Error);
}

// Process is a process being traced.  It consists of a set of
// threads.  A process can be running, stopped, or terminated.  The
// process's state extends to all of its threads.
type Process interface {
	// Threads returns an array of all threads in this process.
	Threads() []*Thread;

	// AddBreakpoint creates a new breakpoint at program counter
	// pc.  Breakpoints can only be created when the process is
	// stopped.  It is an error if a breakpoint already exists at
	// pc.
	AddBreakpoint(pc Word) os.Error;

	// RemoveBreakpoint removes the breakpoint at the program
	// counter pc.  It is an error if no breakpoint exists at pc.
	RemoveBreakpoint(pc Word) os.Error;

	// Stop stops all running threads in this process before
	// returning.
	Stop() os.Error;

	// Continue resumes execution of all threads in this process.
	// Any thread that is stopped on a breakpoint will be stepped
	// over that breakpoint.  Any thread that is stopped because
	// of a signal will receive the pending signal.
	Continue() os.Error;

	// WaitStop waits until all threads in process p are stopped
	// as a result of some thread hitting a breakpoint, receiving
	// a signal, creating a new thread, or exiting.
	WaitStop() os.Error;

	// Detach detaches from this process.  All stopped threads
	// will be resumed.
	Detach() os.Error;
}

// Paused is a stop cause used for threads that are stopped either by
// user request (e.g., from the Stop method or after single stepping),
// or that are stopped because some other thread caused the program to
// stop.
type Paused struct {}

func (c Paused) String() string {
	return "paused";
}

// Breakpoint is a stop cause resulting from a thread reaching a set
// breakpoint.
type Breakpoint	Word

// PC returns the program counter that the program is stopped at.
func (c Breakpoint) PC() Word {
	return Word(c);
}

func (c Breakpoint) String() string {
	return "breakpoint at 0x" + strconv.Uitob64(uint64(c.PC()), 16);
}

// Signal is a stop cause resulting from a thread receiving a signal.
// When the process is continued, the signal will be delivered.
type Signal string

// Signal returns the signal being delivered to the thread.
func (c Signal) Name() string {
	return string(c);
}

func (c Signal) String() string {
	return c.Name();
}

// ThreadCreate is a stop cause returned from an existing thread when
// it creates a new thread.  The new thread exists in a primordial
// form at this point and will begin executing in earnest when the
// process is continued.
type ThreadCreate struct {
	thread Thread;
}

func (c ThreadCreate) NewThread() Thread {
	return c.thread;
}

func (c ThreadCreate) String() string {
	return "thread create";
}

// ThreadExit is a stop cause resulting from a thread exiting.  When
// this cause first arises, the thread will still be in the list of
// process threads and its registers and memory will still be
// accessible.
type ThreadExit struct {
	exitStatus int;
	signal int;
}

// Exited returns true if the thread exited normally.
func (c ThreadExit) Exited() bool {
	return c.exitStatus != -1;
}

// ExitStatus returns the exit status of the thread if it exited
// normally or -1 otherwise.
func (c ThreadExit) ExitStatus() int {
	return c.exitStatus;
}

// Signaled returns true if the thread was terminated by a signal.
func (c ThreadExit) Signaled() bool {
	return c.signal != -1;
}

// StopSignal returns the signal that terminated the thread, or -1 if
// it was not terminated by a signal.
func (c ThreadExit) StopSignal() int {
	return c.signal;
}

func (c ThreadExit) String() string {
	res := "thread exited ";
	switch {
	case c.Exited():
		res += "with status " + strconv.Itoa(c.ExitStatus());
	case c.Signaled():
		res += "from signal " + strconv.Itoa(c.StopSignal());
	default:
		res += "from unknown cause";
	}
	return res;
}
