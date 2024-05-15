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

// CrashOptions provides options that control the formatting of the
// fatal crash message.
type CrashOptions struct {
	/* for future expansion */
}

// SetCrashOutput configures a single additional file where unhandled
// panics and other fatal errors are printed, in addition to standard error.
// There is only one additional file: calling SetCrashOutput again overrides
// any earlier call.
// SetCrashOutput duplicates f's file descriptor, so the caller may safely
// close f as soon as SetCrashOutput returns.
// To disable this additional crash output, call SetCrashOutput(nil).
// If called concurrently with a crash, some in-progress output may be written
// to the old file even after an overriding SetCrashOutput returns.
//
// TODO(adonovan): the variadic ... is a short-term measure to avoid
// breaking the call in x/telemetry; it will be removed before the
// go1.23 freeze.
func SetCrashOutput(f *os.File, opts ...CrashOptions) error {
	if len(opts) > 1 {
		panic("supply at most 1 CrashOptions")
	}
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
