// Copyright 2009 The Go Authors.  All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package syscall

import syscall "syscall"

export func gettimeofday() (sec, nsec, errno int64) {
	const GETTIMEOFDAY = 116;
	// The "1" in the call is the timeval pointer, which must be
	// non-zero but is otherwise unused.  The results
	// are returned in r1, r2.
	r1, r2, err := syscall.Syscall(GETTIMEOFDAY, 1, 0, 0);
	if err != 0 {
		return 0, 0, err
	}
	return r1, r2*1000, 0
}
