// Copyright 2009 The Go Authors.  All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package syscall

import syscall "syscall"

export func gettimeofday() (sec, nsec, errno int64) {
	// The "1" in the call is the timeval pointer, which must be
	// non-zero but is otherwise unused.  The results
	// are returned in r1, r2.
	r1, r2, err := Syscall(SYS_GETTIMEOFDAY, 1, 0, 0);
	if err != 0 {
		return 0, 0, err
	}
	return r1, r2*1000, 0
}

export func nstotimeval(ns int64, tv *Timeval) {
	ns += 999;	// round up
	tv.sec = int64(ns/1000000000);
	tv.usec = uint32(ns%1000000000 / 1000);
}
