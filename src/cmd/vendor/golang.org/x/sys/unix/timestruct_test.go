// Copyright 2017 The Go Authors. All right reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// +build darwin dragonfly freebsd linux netbsd openbsd solaris

package unix_test

import (
	"testing"
	"time"
	"unsafe"

	"golang.org/x/sys/unix"
)

func TestTimeToTimespec(t *testing.T) {
	timeTests := []struct {
		time  time.Time
		valid bool
	}{
		{time.Unix(0, 0), true},
		{time.Date(2009, time.November, 10, 23, 0, 0, 0, time.UTC), true},
		{time.Date(2262, time.December, 31, 23, 0, 0, 0, time.UTC), false},
		{time.Unix(0x7FFFFFFF, 0), true},
		{time.Unix(0x80000000, 0), false},
		{time.Unix(0x7FFFFFFF, 1000000000), false},
		{time.Unix(0x7FFFFFFF, 999999999), true},
		{time.Unix(-0x80000000, 0), true},
		{time.Unix(-0x80000001, 0), false},
		{time.Date(2038, time.January, 19, 3, 14, 7, 0, time.UTC), true},
		{time.Date(2038, time.January, 19, 3, 14, 8, 0, time.UTC), false},
		{time.Date(1901, time.December, 13, 20, 45, 52, 0, time.UTC), true},
		{time.Date(1901, time.December, 13, 20, 45, 51, 0, time.UTC), false},
	}

	// Currently all targets have either int32 or int64 for Timespec.Sec.
	// If there were a new target with unsigned or floating point type for
	// it, this test must be adjusted.
	have64BitTime := (unsafe.Sizeof(unix.Timespec{}.Sec) == 8)
	for _, tt := range timeTests {
		ts, err := unix.TimeToTimespec(tt.time)
		tt.valid = tt.valid || have64BitTime
		if tt.valid && err != nil {
			t.Errorf("TimeToTimespec(%v): %v", tt.time, err)
		}
		if err == nil {
			tstime := time.Unix(int64(ts.Sec), int64(ts.Nsec))
			if !tstime.Equal(tt.time) {
				t.Errorf("TimeToTimespec(%v) is the time %v", tt.time, tstime)
			}
		}
	}
}
