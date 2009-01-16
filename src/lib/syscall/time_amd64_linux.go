// Copyright 2009 The Go Authors.  All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package syscall

import (
	"syscall";
	"unsafe";
)

export func Gettimeofday() (sec, nsec, errno int64) {
	var tv Timeval;
	r1, r2, e := Syscall(SYS_GETTIMEOFDAY, int64(uintptr(unsafe.pointer(&tv))), 0, 0);
	if e != 0 {
		return 0, 0, e
	}
	return int64(tv.Sec), int64(tv.Usec*1000), 0
}

export func Nstotimeval(ns int64, tv *Timeval) {
	ns += 999;	// round up
	tv.Sec = int64(ns/1000000000);
	tv.Usec = uint64(ns%1000000000 / 1000);
}
