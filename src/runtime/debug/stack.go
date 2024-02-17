// Copyright 2011 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// Package debug contains facilities for programs to debug themselves while
// they are running.
package debug

import (
	"internal/poll"
	"os"
	"runtime"
	_ "unsafe" // for linkname
)

// PrintStack prints to standard error the stack trace returned by runtime.Stack.
func PrintStack() {
	os.Stderr.Write(Stack())
}

// Stack returns a formatted stack trace of the goroutine that calls it.
// It calls [runtime.Stack] with a large enough buffer to capture the entire trace.
func Stack() []byte {
	buf := make([]byte, 1024)
	for {
		n := runtime.Stack(buf, false)
		if n < len(buf) {
			return buf[:n]
		}
		buf = make([]byte, 2*len(buf))
	}
}

// SetCrashOutput configures a single additional file where unhandled
// panics and other fatal errors are printed, in addition to standard error.
// There is only one additional file: calling SetCrashOutput again
// overrides any earlier call; it does not close the previous file.
// SetCrashOutput(nil) disables the use of any additional file.
func SetCrashOutput(f *os.File) error {
	fd := ^uintptr(0)
	if f != nil {
		// The runtime will write to this file descriptor from
		// low-level routines during a panic, possibly without
		// a G, so we must call f.Fd() eagerly. This creates a
		// danger that that the file descriptor is no longer
		// valid at the time of the write, because the caller
		// (incorrectly) called f.Close() and the kernel
		// reissued the fd in a later call to open(2), leading
		// to crashes being written to the wrong file.
		//
		// So, we duplicate the fd to obtain a private one
		// that cannot be closed by the user.
		// This also alleviates us from concerns about the
		// lifetime and finalization of f.
		// (DupCloseOnExec returns an fd, not a *File, so
		// there is no finalizer, and we are responsible for
		// closing it.)
		//
		// The new fd must be close-on-exec, otherwise if the
		// crash monitor is a child process, it may inherit
		// it, so it will never see EOF from the pipe even
		// when this process crashes.
		//
		// A side effect of Fd() is that it calls SetBlocking,
		// which is important so that writes of a crash report
		// to a full pipe buffer don't get lost.
		fd2, _, err := poll.DupCloseOnExec(int(f.Fd()))
		if err != nil {
			return err
		}
		runtime.KeepAlive(f) // prevent finalization before dup
		fd = uintptr(fd2)
	}
	if prev := runtime_setCrashFD(fd); prev != ^uintptr(0) {
		// We use NewFile+Close because it is portable
		// unlike syscall.Close, whose parameter type varies.
		os.NewFile(prev, "").Close() // ignore error
	}
	return nil
}

//go:linkname runtime_setCrashFD runtime.setCrashFD
func runtime_setCrashFD(uintptr) uintptr
