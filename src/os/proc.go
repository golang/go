// Copyright 2009 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// Process etc.

package os

import (
	"internal/testlog"
	"runtime"
	"syscall"
)

// Args hold the command-line arguments, starting with the program name.
var Args []string

func init() {
	if runtime.GOOS == "windows" {
		// Initialized in exec_windows.go.
		return
	}
	Args = runtime_args()
}

func runtime_args() []string // in package runtime

// Getuid returns the numeric user id of the caller.
//
// On Windows, it returns -1.
func Getuid() int { return syscall.Getuid() }

// Geteuid returns the numeric effective user id of the caller.
//
// On Windows, it returns -1.
func Geteuid() int { return syscall.Geteuid() }

// Getgid returns the numeric group id of the caller.
//
// On Windows, it returns -1.
func Getgid() int { return syscall.Getgid() }

// Getegid returns the numeric effective group id of the caller.
//
// On Windows, it returns -1.
func Getegid() int { return syscall.Getegid() }

// Getgroups returns a list of the numeric ids of groups that the caller belongs to.
//
// On Windows, it returns [syscall.EWINDOWS]. See the [os/user] package
// for a possible alternative.
func Getgroups() ([]int, error) {
	gids, e := syscall.Getgroups()
	return gids, NewSyscallError("getgroups", e)
}

// Exit causes the current program to exit with the given status code.
// Conventionally, code zero indicates success, non-zero an error.
// The program terminates immediately; deferred functions are not run.
//
// For portability, the status code should be in the range [0, 125].
func Exit(code int) {
	if code == 0 && testlog.PanicOnExit0() {
		// We were told to panic on calls to os.Exit(0).
		// This is used to fail tests that make an early
		// unexpected call to os.Exit(0).
		panic("unexpected call to os.Exit(0) during test")
	}

	// Inform the runtime that os.Exit is being called. If -race is
	// enabled, this will give race detector a chance to fail the
	// program (racy programs do not have the right to finish
	// successfully). If coverage is enabled, then this call will
	// enable us to write out a coverage data file.
	runtime_beforeExit(code)

	syscall.Exit(code)
}

func runtime_beforeExit(exitCode int) // implemented in runtime
