// Copyright 2009 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package time

import (
	"os";
	"testing";
	"time";
)

func init() {
	// Force US Pacific time for daylight-savings
	// tests below (localtests).  Needs to be set
	// before the first call into the time library.
	os.Setenv("TZ", "US/Pacific");
}

type TimeTest struct {
	seconds int64;
	golden Time;
}

var utctests = []TimeTest (
	TimeTest(0, Time(1970, 1, 1, 0, 0, 0, Thursday, 0, "UTC")),
	TimeTest(1221681866, Time(2008, 9, 17, 20, 4, 26, Wednesday, 0, "UTC")),
	TimeTest(-1221681866, Time(1931, 4, 16, 3, 55, 34, Thursday, 0, "UTC")),
	TimeTest(1e18, Time(31688740476, 10, 23, 1, 46, 40, Friday, 0, "UTC")),
	TimeTest(-1e18, Time(-31688736537, 3, 10, 22, 13, 20, Tuesday, 0, "UTC")),
	TimeTest(0x7fffffffffffffff, Time(292277026596, 12, 4, 15, 30, 7, Sunday, 0, "UTC")),
	TimeTest(-0x8000000000000000, Time(-292277022657, 1, 27, 8, 29, 52, Sunday, 0, "UTC"))
)

var localtests = []TimeTest (
	TimeTest(0, Time(1969, 12, 31, 16, 0, 0, Wednesday, -8*60*60, "PST")),
	TimeTest(1221681866, Time(2008, 9, 17, 13, 4, 26, Wednesday, -7*60*60, "PDT"))
)

func same(t, u *Time) bool {
	return t.Year == u.Year
		&& t.Month == u.Month
		&& t.Day == u.Day
		&& t.Hour == u.Hour
		&& t.Minute == u.Minute
		&& t.Second == u.Second
		&& t.Weekday == u.Weekday
		&& t.ZoneOffset == u.ZoneOffset
		&& t.Zone == u.Zone
}

func TestSecondsToUTC(t *testing.T) {
	for i := 0; i < len(utctests); i++ {
		sec := utctests[i].seconds;
		golden := &utctests[i].golden;
		tm := SecondsToUTC(sec);
		newsec := tm.Seconds();
		if newsec != sec {
			t.Errorf("SecondsToUTC(%d).Seconds() = %d", sec, newsec);
		}
		if !same(tm, golden) {
			t.Errorf("SecondsToUTC(%d):", sec);
			t.Errorf("  want=%v", *golden);
			t.Errorf("  have=%v", *tm);
		}
	}
}

func TestSecondsToLocalTime(t *testing.T) {
	for i := 0; i < len(localtests); i++ {
		sec := localtests[i].seconds;
		golden := &localtests[i].golden;
		tm := SecondsToLocalTime(sec);
		newsec := tm.Seconds();
		if newsec != sec {
			t.Errorf("SecondsToLocalTime(%d).Seconds() = %d", sec, newsec);
		}
		if !same(tm, golden) {
			t.Errorf("SecondsToLocalTime(%d):", sec);
			t.Errorf("  want=%v", *golden);
			t.Errorf("  have=%v", *tm);
		}
	}
}

