// Copyright 2009 The Go Authors.  All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package syscall

import syscall "syscall"

export func gettimeofday() (sec, nsec, errno int64) {
	var tv Timeval;
	r1, r2, e := Syscall(SYS_GETTIMEOFDAY, TimevalPtr(&tv), 0, 0);
	if e != 0 {
		return 0, 0, e
	}
	return int64(tv.sec), int64(tv.usec*1000), 0
}
