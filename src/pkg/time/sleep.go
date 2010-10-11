// Copyright 2009 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package time

import (
	"os"
	"syscall"
)

// Sleep pauses the current goroutine for at least ns nanoseconds.
// Higher resolution sleeping may be provided by syscall.Nanosleep 
// on some operating systems.
func Sleep(ns int64) os.Error {
	_, err := sleep(Nanoseconds(), ns)
	return err
}

// After waits at least ns nanoseconds before sending the current time
// on the returned channel.
func After(ns int64) <-chan int64 {
	t := Nanoseconds()
	ch := make(chan int64, 1)
	go func() {
		t, _ = sleep(t, ns)
		ch <- t
	}()
	return ch
}

// sleep takes the current time and a duration,
// pauses for at least ns nanoseconds, and
// returns the current time and an error.
func sleep(t, ns int64) (int64, os.Error) {
	// TODO(cw): use monotonic-time once it's available
	end := t + ns
	for t < end {
		errno := syscall.Sleep(end - t)
		if errno != 0 && errno != syscall.EINTR {
			return 0, os.NewSyscallError("sleep", errno)
		}
		t = Nanoseconds()
	}
	return t, nil
}
