// Copyright 2009 The Go Authors.  All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package syscall

func Getpagesize() int	{ return 4096 }

func NsecToTimeval(nsec int64) (tv Timeval) {
	tv.Sec = int32(nsec / 1e9);
	tv.Usec = int32(nsec % 1e9 / 1e3);
	return;
}

func NsecToTimespec(nsec int64) (ts Timespec) {
	ts.Sec = int32(nsec / 1e9);
	ts.Nsec = int32(nsec % 1e9);
	return;
}
