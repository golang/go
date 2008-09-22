// Copyright 2009 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// $G $F.go && $L $F.$A && ./$A.out

package main

import "time"

type Test struct {
	seconds int64;
	golden time.Time;
}

var UTCTests = []Test {
	Test{0, time.Time{1970, 1, 1, 0, 0, 0, time.Thursday, 0, "GMT"}},
	Test{1221681866, time.Time{2008, 9, 17, 20, 4, 26, time.Wednesday, 0, "GMT"}},
	Test{-1221681866, time.Time{1931, 4, 16, 3, 55, 34, time.Thursday, 0, "GMT"}},
	Test{1e18, time.Time{31688740476, 10, 23, 1, 46, 40, time.Friday, 0, "GMT"}},
	Test{-1e18, time.Time{-31688736537, 3, 10, 22, 13, 20, time.Tuesday, 0, "GMT"}},
	Test{0x7fffffffffffffff, time.Time{292277026596, 12, 4, 15, 30, 7, time.Sunday, 0, "GMT"}},
	Test{-0x8000000000000000, time.Time{-292277022657, 1, 27, 8, 29, 52, time.Sunday, 0, "GMT"}}
}

var LocalTests = []Test {
	Test{0, time.Time{1969, 12, 31, 16, 0, 0, time.Wednesday, -8*60*60, "PST"}},
	Test{1221681866, time.Time{2008, 9, 17, 13, 4, 26, time.Wednesday, -7*60*60, "PDT"}}
}

func Same(t, u *time.Time) bool {
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

func Diff(t, u *time.Time) {
	if t.year != u.year { print("year: ", t.year, " ", u.year, "\n") }
	if t.month != u.month { print("month: ", t.month, " ", u.month, "\n") }
	if t.day != u.day { print("day: ", t.day, " ", u.day, "\n") }
	if t.hour != u.hour { print("hour: ", t.hour, " ", u.hour, "\n") }
	if t.minute != u.minute { print("minute: ", t.minute, " ", u.minute, "\n") }
	if t.second != u.second { print("second: ", t.second, " ", u.second, "\n") }
	if t.weekday != u.weekday { print("weekday: ", t.weekday, " ", u.weekday, "\n") }
	if t.zoneoffset != u.zoneoffset { print("zoneoffset: ", t.zoneoffset, " ", u.zoneoffset, "\n") }
	if t.zone != u.zone { print("zone: ", t.zone, " ", u.zone, "\n") }
}

func main() {
	for i := 0; i < len(UTCTests); i++ {
		sec := UTCTests[i].seconds;
		golden := &UTCTests[i].golden;
		t := time.SecondsToUTC(sec);
		newsec := t.Seconds()
		if newsec != sec {
			panic("SecondsToUTC and back ", sec, " ", newsec)
		}
		if !Same(t, golden) {
			Diff(t, golden);
			panic("SecondsToUTC ", sec, " ", t.String(), " ", t.year, " golden=", golden.String(), " ", golden.year)
		}
	//	print(t.String(), "\n")
	}

	for i := 0; i < len(LocalTests); i++ {
		sec := LocalTests[i].seconds;
		golden := &LocalTests[i].golden;
		t := time.SecondsToLocalTime(sec);
		newsec := t.Seconds()
		if newsec != sec {
			panic("SecondsToLocalTime and back ", sec, " ", newsec)
		}
		if !Same(t, golden) {
			Diff(t, golden);
			panic("SecondsToLocalTime ", sec, " ", t.String(), " ", len(t.zone), " golden=", golden.String(), " ", len(t.zone))
		}
	//	print(t.String(), "\n")
	}
}

