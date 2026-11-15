// Copyright 2026 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package time_test

import (
	"testing"
	. "time"
)

func TestZoneBoundsAcrossYear(t *testing.T) {
	for _, name := range []string{"America/New_York", "Australia/Sydney"} {
		loc, err := LoadLocation(name)
		if err != nil {
			t.Fatal(err)
		}
		for _, year := range []int{2040, 2041} {
			for _, month := range []Month{January, December} {
				now := Date(year, month, 15, 12, 0, 0, 0, loc)
				start, end := now.ZoneBounds()
				_, offset := now.Zone()
				_, before := start.Add(-Nanosecond).Zone()
				_, after := end.Zone()
				if before == offset {
					t.Errorf("%v: zone does not change at start bound %v", now, start)
				}
				if after == offset {
					t.Errorf("%v: zone does not change at end bound %v", now, end)
				}
			}
		}
	}
}
