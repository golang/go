// Copyright 2009 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// $G $F.go && $L $F.$A && ./$A.out

package time

import (
	"testing";
	"time";
)

type TimeTest struct {
	seconds int64;
	golden Time;
}

var utctests = []TimeTest {
	TimeTest{0, Time{1970, 1, 1, 0, 0, 0, Thursday, 0, "GMT"}},
	TimeTest{1221681866, Time{2008, 9, 17, 20, 4, 26, Wednesday, 0, "GMT"}},
	TimeTest{-1221681866, Time{1931, 4, 16, 3, 55, 34, Thursday, 0, "GMT"}},
	TimeTest{1e18, Time{31688740476, 10, 23, 1, 46, 40, Friday, 0, "GMT"}},
	TimeTest{-1e18, Time{-31688736537, 3, 10, 22, 13, 20, Tuesday, 0, "GMT"}},
	TimeTest{0x7fffffffffffffff, Time{292277026596, 12, 4, 15, 30, 7, Sunday, 0, "GMT"}},
	TimeTest{-0x8000000000000000, Time{-292277022657, 1, 27, 8, 29, 52, Sunday, 0, "GMT"}}
}

var localtests = []TimeTest {
	TimeTest{0, Time{1969, 12, 31, 16, 0, 0, Wednesday, -8*60*60, "PST"}},
	TimeTest{1221681866, Time{2008, 9, 17, 13, 4, 26, Wednesday, -7*60*60, "PDT"}}
}

func Same(t, u *Time) bool {
	return t.year == u.year
		&& t.month == u.month
		&& t.day == u.day
		&& t.hour == u.hour
		&& t.minute == u.minute
		&& t.second == u.second
		&& t.weekday == u.weekday
		&& t.zoneoffset == u.zoneoffset
		&& t.zone == u.zone
}

export func TestSecondsToUTC(t *testing.T) {
	for i := 0; i < len(utctests); i++ {
		sec := utctests[i].seconds;
		golden := &utctests[i].golden;
		tm := SecondsToUTC(sec);
		newsec := tm.Seconds();
		if newsec != sec {
			t.Errorf("SecondsToUTC(%d).Seconds() = %d", sec, newsec);
		}
		if !Same(tm, golden) {
			t.Errorf("SecondsToUTC(%d):", sec);
			t.Errorf("  want=%v", *golden);
			t.Errorf("  have=%v", *tm);
		}
	}
}

export func TestSecondsToLocalTime(t *testing.T) {
	for i := 0; i < len(localtests); i++ {
		sec := localtests[i].seconds;
		golden := &localtests[i].golden;
		tm := SecondsToLocalTime(sec);
		newsec := tm.Seconds();
		if newsec != sec {
			t.Errorf("SecondsToLocalTime(%d).Seconds() = %d", sec, newsec);
		}
		if !Same(tm, golden) {
			t.Errorf("SecondsToLocalTime(%d):", sec);
			t.Errorf("  want=%v", *golden);
			t.Errorf("  have=%v", *tm);
		}
	}
}

