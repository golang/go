// Copyright 2017 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// Package poll supports non-blocking I/O on file descriptors with polling.
// This supports I/O operations that block only a goroutine, not a thread.
// This is used by the net and os packages.
// It uses a poller built into the runtime, with support from the
// runtime scheduler.
package poll

import (
	"errors"
	"internal/oserror"
)

// ErrNetClosing is returned when a network descriptor is used after
// it has been closed. Keep this string consistent because of issue
// #4373: since historically programs have not been able to detect
// this error, they look for the string.
var ErrNetClosing = errors.New("use of closed network connection")

// ErrFileClosing is returned when a file descriptor is used after it
// has been closed.
var ErrFileClosing = errors.New("use of closed file")

// ErrNoDeadline is returned when a request is made to set a deadline
// on a file type that does not use the poller.
var ErrNoDeadline = errors.New("file type does not support deadline")

// Return the appropriate closing error based on isFile.
func errClosing(isFile bool) error {
	if isFile {
		return ErrFileClosing
	}
	return ErrNetClosing
}

// ErrTimeout is returned for an expired deadline.
var ErrTimeout error = &TimeoutError{}

// TimeoutError is returned for an expired deadline.
type TimeoutError struct{}

// Implement the net.Error interface.
func (e *TimeoutError) Error() string   { return "i/o timeout" }
func (e *TimeoutError) Timeout() bool   { return true }
func (e *TimeoutError) Temporary() bool { return true }

func (e *TimeoutError) Is(target error) bool {
	return target == oserror.ErrTimeout || target == oserror.ErrTemporary
}

// ErrNotPollable is returned when the file or socket is not suitable
// for event notification.
var ErrNotPollable = errors.New("not pollable")

// consume removes data from a slice of byte slices, for writev.
func consume(v *[][]byte, n int64) {
	for len(*v) > 0 {
		ln0 := int64(len((*v)[0]))
		if ln0 > n {
			(*v)[0] = (*v)[0][n:]
			return
		}
		n -= ln0
		*v = (*v)[1:]
	}
}

// TestHookDidWritev is a hook for testing writev.
var TestHookDidWritev = func(wrote int) {}
