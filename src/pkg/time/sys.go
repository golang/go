// Copyright 2009 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package time

// Seconds reports the number of seconds since the Unix epoch,
// January 1, 1970 00:00:00 UTC.
func Seconds() int64 {
	return Nanoseconds() / 1e9
}

// Nanoseconds is implemented by package runtime.

// Nanoseconds reports the number of nanoseconds since the Unix epoch,
// January 1, 1970 00:00:00 UTC.
func Nanoseconds() int64

// Sleep pauses the current goroutine for at least ns nanoseconds.
func Sleep(ns int64)
