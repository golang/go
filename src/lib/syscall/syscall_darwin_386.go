// Copyright 2009 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package syscall

import "syscall"

func TimespecToNsec(ts Timespec) int64 {
	return int64(ts.Sec)*1e9 + int64(ts.Nsec);
}

func NsecToTimespec(nsec int64) (ts Timespec) {
	ts.Sec = int32(nsec / 1e9);
	ts.Nsec = int32(nsec % 1e9);
	return;
}

func TimevalToNsec(tv Timeval) int64 {
	return int64(tv.Sec)*1e9 + int64(tv.Usec)*1e3;
}

func NsecToTimeval(nsec int64) (tv Timeval) {
	nsec += 999;	// round up to microsecond
	tv.Usec = int32(nsec%1e9 / 1e3);
	tv.Sec = int32(nsec/1e9);
	return;
}

//sys	gettimeofday(tp *Timeval) (sec int64, usec int32, errno int)
func Gettimeofday(tv *Timeval) (errno int) {
	// The tv passed to gettimeofday must be non-nil
	// but is otherwise unused.  The answers come back
	// in the two registers.
	sec, usec, err := gettimeofday(tv);
	tv.Sec = int32(sec);
	tv.Usec = int32(usec);
	return err;
}

