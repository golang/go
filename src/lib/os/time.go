// Copyright 2009 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package os

import (
	"os";
	"syscall"
)


// Time returns the current time, in whole seconds and
// fractional nanoseconds, plus an Error if any. The current
// time is thus 1e9*sec+nsec, in nanoseconds.  The zero of
// time is the Unix epoch.
func Time() (sec int64, nsec int64, err Error) {
	var errno int64;
	sec, nsec, errno = syscall.Gettimeofday();
	if errno != 0 {
		return 0, 0, ErrnoToError(errno)
	}
	return sec, nsec, nil
}

