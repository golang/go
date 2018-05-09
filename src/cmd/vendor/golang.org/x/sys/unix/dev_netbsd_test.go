// Copyright 2017 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// +build go1.7

package unix_test

import (
	"fmt"
	"testing"

	"golang.org/x/sys/unix"
)

func TestDevices(t *testing.T) {
	testCases := []struct {
		path  string
		major uint32
		minor uint32
	}{
		// well known major/minor numbers according to /dev/MAKEDEV on
		// NetBSD 8.0
		{"/dev/null", 2, 2},
		{"/dev/zero", 2, 12},
		{"/dev/random", 46, 0},
		{"/dev/urandom", 46, 1},
	}
	for _, tc := range testCases {
		t.Run(fmt.Sprintf("%s %v:%v", tc.path, tc.major, tc.minor), func(t *testing.T) {
			var stat unix.Stat_t
			err := unix.Stat(tc.path, &stat)
			if err != nil {
				t.Errorf("failed to stat device: %v", err)
				return
			}

			dev := uint64(stat.Rdev)
			if unix.Major(dev) != tc.major {
				t.Errorf("for %s Major(%#x) == %d, want %d", tc.path, dev, unix.Major(dev), tc.major)
			}
			if unix.Minor(dev) != tc.minor {
				t.Errorf("for %s Minor(%#x) == %d, want %d", tc.path, dev, unix.Minor(dev), tc.minor)
			}
			if unix.Mkdev(tc.major, tc.minor) != dev {
				t.Errorf("for %s Mkdev(%d, %d) == %#x, want %#x", tc.path, tc.major, tc.minor, unix.Mkdev(tc.major, tc.minor), dev)
			}
		})
	}
}
