// Copyright 2009 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package os

import "syscall"

// Time returns the current time, in whole seconds and
// fractional nanoseconds, plus an error if any. The current
// time is thus 1e9*sec+nsec, in nanoseconds.  The zero of
// time is the Unix epoch.
func Time() (sec int64, nsec int64, err error) {
	var tv syscall.Timeval
	if e := syscall.Gettimeofday(&tv); e != nil {
		return 0, 0, NewSyscallError("gettimeofday", e)
	}
	return int64(tv.Sec), int64(tv.Usec) * 1000, err
}
