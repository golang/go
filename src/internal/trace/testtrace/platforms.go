// Copyright 2025 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package testtrace

import (
	"runtime"
	"testing"
)

// MustHaveSyscallEvents skips the current test if the current
// platform does not support true system call events.
func MustHaveSyscallEvents(t *testing.T) {
	if HasSyscallEvents() {
		return
	}
	t.Skip("current platform has no true syscall events")
}

// HasSyscallEvents returns true if the current platform
// has real syscall events available.
func HasSyscallEvents() bool {
	switch runtime.GOOS {
	case "js", "wasip1":
		// js and wasip1 emulate system calls by blocking on channels
		// while yielding back to the environment. They never actually
		// call entersyscall/exitsyscall.
		return false
	}
	return true
}
