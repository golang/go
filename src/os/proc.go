// Copyright 2009 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// Process etc.

package os

import (
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
func Getuid() int { return syscall.Getuid() }

// Geteuid returns the numeric effective user id of the caller.
func Geteuid() int { return syscall.Geteuid() }

// Getgid returns the numeric group id of the caller.
func Getgid() int { return syscall.Getgid() }

// Getegid returns the numeric effective group id of the caller.
func Getegid() int { return syscall.Getegid() }

// Getgroups returns a list of the numeric ids of groups that the caller belongs to.
func Getgroups() ([]int, error) {
	gids, e := syscall.Getgroups()
	return gids, NewSyscallError("getgroups", e)
}

// Exit causes the current program to exit with the given status code.
// Conventionally, code zero indicates success, non-zero an error.
// The program terminates immediately; deferred functions are not run.
func Exit(code int) {
	if code == 0 {
		// Give race detector a chance to fail the program.
		// Racy programs do not have the right to finish successfully.
		runtime_beforeExit()
	}
	syscall.Exit(code)
}

func runtime_beforeExit() // implemented in runtime
