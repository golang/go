// Copyright 2009 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package time

import (
	"os"
	"syscall"
)

// Sleep pauses the current goroutine for at least ns nanoseconds. Higher resolution
// sleeping may be provided by syscall.Nanosleep on some operating systems.
func Sleep(ns int64) os.Error {
	// TODO(cw): use monotonic-time once it's available
	t := Nanoseconds()
	end := t + ns
	for t < end {
		errno := syscall.Sleep(end - t)
		if errno != 0 && errno != syscall.EINTR {
			return os.NewSyscallError("sleep", errno)
		}
		t = Nanoseconds()
	}
	return nil
}
