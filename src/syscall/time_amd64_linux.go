// Copyright 2009 The Go Authors.  All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package syscall

import syscall "syscall"

func	Int64Ptr(s *int64) int64;

export func gettimeofday() (sec, nsec, errno int64) {
	const GETTIMEOFDAY = 96
	var tv [2]int64;	// struct timeval
	r1, r2, err := syscall.Syscall(GETTIMEOFDAY, Int64Ptr(&tv[0]), 0, 0);
	if err != 0 {
		return 0, 0, err
	}
	return tv[0], tv[1], 0
}
