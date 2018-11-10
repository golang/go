// Copyright 2016 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// +build darwin dragonfly freebsd linux nacl netbsd openbsd solaris

package syscall

// TimespecToNsec converts a Timespec value into a number of
// nanoseconds since the Unix epoch.
func TimespecToNsec(ts Timespec) int64 { return int64(ts.Sec)*1e9 + int64(ts.Nsec) }

// NsecToTimespec takes a number of nanoseconds since the Unix epoch
// and returns the corresponding Timespec value.
func NsecToTimespec(nsec int64) Timespec {
	sec := nsec / 1e9
	nsec = nsec % 1e9
	if nsec < 0 {
		nsec += 1e9
		sec--
	}
	return setTimespec(sec, nsec)
}

// TimevalToNsec converts a Timeval value into a number of nanoseconds
// since the Unix epoch.
func TimevalToNsec(tv Timeval) int64 { return int64(tv.Sec)*1e9 + int64(tv.Usec)*1e3 }

// NsecToTimeval takes a number of nanoseconds since the Unix epoch
// and returns the corresponding Timeval value.
func NsecToTimeval(nsec int64) Timeval {
	nsec += 999 // round up to microsecond
	usec := nsec % 1e9 / 1e3
	sec := nsec / 1e9
	if usec < 0 {
		usec += 1e6
		sec--
	}
	return setTimeval(sec, usec)
}
