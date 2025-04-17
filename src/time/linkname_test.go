// Copyright 2024 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package time_test

import (
	"testing"
	"time"
	_ "unsafe" // for linkname
)

//go:linkname timeAbs time.Time.abs
func timeAbs(time.Time) uint64

//go:linkname absClock time.absClock
func absClock(uint64) (hour, min, sec int)

//go:linkname absDate time.absDate
func absDate(uint64, bool) (year int, month time.Month, day int, yday int)

func TestLinkname(t *testing.T) {
	tm := time.Date(2006, time.January, 2, 15, 4, 5, 6, time.UTC)
	abs := timeAbs(tm)
	// wantAbs should be Jan 1 based, not Mar 1 based.
	// See absolute time description in time.go.
	const wantAbs = 9223372029851535845 // NOT 9223372029877973939
	if abs != wantAbs {
		t.Fatalf("timeAbs(2006-01-02 15:04:05 UTC) = %d, want %d", abs, uint64(wantAbs))
	}

	year, month, day, yday := absDate(abs, true)
	if year != 2006 || month != time.January || day != 2 || yday != 1 {
		t.Errorf("absDate() = %v, %v, %v, %v, want 2006, January, 2, 1", year, month, day, yday)
	}

	hour, min, sec := absClock(abs)
	if hour != 15 || min != 4 || sec != 5 {
		t.Errorf("absClock() = %v, %v, %v, 15, 4, 5", hour, min, sec)
	}
}
