// Copyright 2017 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// +build darwin dragonfly freebsd linux netbsd openbsd solaris

package unix

import "time"

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

// TimeToTimespec converts t into a Timespec.
// On some 32-bit systems the range of valid Timespec values are smaller
// than that of time.Time values.  So if t is out of the valid range of
// Timespec, it returns a zero Timespec and ERANGE.
func TimeToTimespec(t time.Time) (Timespec, error) {
	sec := t.Unix()
	nsec := int64(t.Nanosecond())
	ts := setTimespec(sec, nsec)

	// Currently all targets have either int32 or int64 for Timespec.Sec.
	// If there were a new target with floating point type for it, we have
	// to consider the rounding error.
	if int64(ts.Sec) != sec {
		return Timespec{}, ERANGE
	}
	return ts, nil
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

// Unix returns ts as the number of seconds and nanoseconds elapsed since the
// Unix epoch.
func (ts *Timespec) Unix() (sec int64, nsec int64) {
	return int64(ts.Sec), int64(ts.Nsec)
}

// Unix returns tv as the number of seconds and nanoseconds elapsed since the
// Unix epoch.
func (tv *Timeval) Unix() (sec int64, nsec int64) {
	return int64(tv.Sec), int64(tv.Usec) * 1000
}

// Nano returns ts as the number of nanoseconds elapsed since the Unix epoch.
func (ts *Timespec) Nano() int64 {
	return int64(ts.Sec)*1e9 + int64(ts.Nsec)
}

// Nano returns tv as the number of nanoseconds elapsed since the Unix epoch.
func (tv *Timeval) Nano() int64 {
	return int64(tv.Sec)*1e9 + int64(tv.Usec)*1000
}
