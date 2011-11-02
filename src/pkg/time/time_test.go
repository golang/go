// Copyright 2009 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package time_test

import (
	"strconv"
	"strings"
	"testing"
	"testing/quick"
	. "time"
)

// We should be in PST/PDT, but if the time zone files are missing we
// won't be. The purpose of this test is to at least explain why some of
// the subsequent tests fail.
func TestZoneData(t *testing.T) {
	lt := LocalTime()
	// PST is 8 hours west, PDT is 7 hours west.  We could use the name but it's not unique.
	if off := lt.ZoneOffset; off != -8*60*60 && off != -7*60*60 {
		t.Errorf("Unable to find US Pacific time zone data for testing; time zone is %q offset %d", lt.Zone, off)
		t.Error("Likely problem: the time zone files have not been installed.")
	}
}

type TimeTest struct {
	seconds int64
	golden  Time
}

var utctests = []TimeTest{
	{0, Time{1970, 1, 1, 0, 0, 0, 0, 0, "UTC"}},
	{1221681866, Time{2008, 9, 17, 20, 4, 26, 0, 0, "UTC"}},
	{-1221681866, Time{1931, 4, 16, 3, 55, 34, 0, 0, "UTC"}},
	{-11644473600, Time{1601, 1, 1, 0, 0, 0, 0, 0, "UTC"}},
	{599529660, Time{1988, 12, 31, 0, 1, 0, 0, 0, "UTC"}},
	{978220860, Time{2000, 12, 31, 0, 1, 0, 0, 0, "UTC"}},
	{1e18, Time{31688740476, 10, 23, 1, 46, 40, 0, 0, "UTC"}},
	{-1e18, Time{-31688736537, 3, 10, 22, 13, 20, 0, 0, "UTC"}},
	{0x7fffffffffffffff, Time{292277026596, 12, 4, 15, 30, 7, 0, 0, "UTC"}},
	{-0x8000000000000000, Time{-292277022657, 1, 27, 8, 29, 52, 0, 0, "UTC"}},
}

var nanoutctests = []TimeTest{
	{0, Time{1970, 1, 1, 0, 0, 0, 1e8, 0, "UTC"}},
	{1221681866, Time{2008, 9, 17, 20, 4, 26, 2e8, 0, "UTC"}},
}

var localtests = []TimeTest{
	{0, Time{1969, 12, 31, 16, 0, 0, 0, -8 * 60 * 60, "PST"}},
	{1221681866, Time{2008, 9, 17, 13, 4, 26, 0, -7 * 60 * 60, "PDT"}},
}

var nanolocaltests = []TimeTest{
	{0, Time{1969, 12, 31, 16, 0, 0, 1e8, -8 * 60 * 60, "PST"}},
	{1221681866, Time{2008, 9, 17, 13, 4, 26, 3e8, -7 * 60 * 60, "PDT"}},
}

func same(t, u *Time) bool {
	return t.Year == u.Year &&
		t.Month == u.Month &&
		t.Day == u.Day &&
		t.Hour == u.Hour &&
		t.Minute == u.Minute &&
		t.Second == u.Second &&
		t.Nanosecond == u.Nanosecond &&
		t.Weekday() == u.Weekday() &&
		t.ZoneOffset == u.ZoneOffset &&
		t.Zone == u.Zone
}

func TestSecondsToUTC(t *testing.T) {
	for _, test := range utctests {
		sec := test.seconds
		golden := &test.golden
		tm := SecondsToUTC(sec)
		newsec := tm.Seconds()
		if newsec != sec {
			t.Errorf("SecondsToUTC(%d).Seconds() = %d", sec, newsec)
		}
		if !same(tm, golden) {
			t.Errorf("SecondsToUTC(%d):", sec)
			t.Errorf("  want=%+v", *golden)
			t.Errorf("  have=%+v", *tm)
		}
	}
}

func TestNanosecondsToUTC(t *testing.T) {
	for _, test := range nanoutctests {
		golden := &test.golden
		nsec := test.seconds*1e9 + int64(golden.Nanosecond)
		tm := NanosecondsToUTC(nsec)
		newnsec := tm.Nanoseconds()
		if newnsec != nsec {
			t.Errorf("NanosecondsToUTC(%d).Nanoseconds() = %d", nsec, newnsec)
		}
		if !same(tm, golden) {
			t.Errorf("NanosecondsToUTC(%d):", nsec)
			t.Errorf("  want=%+v", *golden)
			t.Errorf("  have=%+v", *tm)
		}
	}
}

func TestSecondsToLocalTime(t *testing.T) {
	for _, test := range localtests {
		sec := test.seconds
		golden := &test.golden
		tm := SecondsToLocalTime(sec)
		newsec := tm.Seconds()
		if newsec != sec {
			t.Errorf("SecondsToLocalTime(%d).Seconds() = %d", sec, newsec)
		}
		if !same(tm, golden) {
			t.Errorf("SecondsToLocalTime(%d):", sec)
			t.Errorf("  want=%+v", *golden)
			t.Errorf("  have=%+v", *tm)
		}
	}
}

func TestNanoecondsToLocalTime(t *testing.T) {
	for _, test := range nanolocaltests {
		golden := &test.golden
		nsec := test.seconds*1e9 + int64(golden.Nanosecond)
		tm := NanosecondsToLocalTime(nsec)
		newnsec := tm.Nanoseconds()
		if newnsec != nsec {
			t.Errorf("NanosecondsToLocalTime(%d).Seconds() = %d", nsec, newnsec)
		}
		if !same(tm, golden) {
			t.Errorf("NanosecondsToLocalTime(%d):", nsec)
			t.Errorf("  want=%+v", *golden)
			t.Errorf("  have=%+v", *tm)
		}
	}
}

func TestSecondsToUTCAndBack(t *testing.T) {
	f := func(sec int64) bool { return SecondsToUTC(sec).Seconds() == sec }
	f32 := func(sec int32) bool { return f(int64(sec)) }
	cfg := &quick.Config{MaxCount: 10000}

	// Try a reasonable date first, then the huge ones.
	if err := quick.Check(f32, cfg); err != nil {
		t.Fatal(err)
	}
	if err := quick.Check(f, cfg); err != nil {
		t.Fatal(err)
	}
}

func TestNanosecondsToUTCAndBack(t *testing.T) {
	f := func(nsec int64) bool { return NanosecondsToUTC(nsec).Nanoseconds() == nsec }
	f32 := func(nsec int32) bool { return f(int64(nsec)) }
	cfg := &quick.Config{MaxCount: 10000}

	// Try a small date first, then the large ones. (The span is only a few hundred years
	// for nanoseconds in an int64.)
	if err := quick.Check(f32, cfg); err != nil {
		t.Fatal(err)
	}
	if err := quick.Check(f, cfg); err != nil {
		t.Fatal(err)
	}
}

type TimeFormatTest struct {
	time           Time
	formattedValue string
}

var rfc3339Formats = []TimeFormatTest{
	{Time{2008, 9, 17, 20, 4, 26, 0, 0, "UTC"}, "2008-09-17T20:04:26Z"},
	{Time{1994, 9, 17, 20, 4, 26, 0, -18000, "EST"}, "1994-09-17T20:04:26-05:00"},
	{Time{2000, 12, 26, 1, 15, 6, 0, 15600, "OTO"}, "2000-12-26T01:15:06+04:20"},
}

func TestRFC3339Conversion(t *testing.T) {
	for _, f := range rfc3339Formats {
		if f.time.Format(RFC3339) != f.formattedValue {
			t.Error("RFC3339:")
			t.Errorf("  want=%+v", f.formattedValue)
			t.Errorf("  have=%+v", f.time.Format(RFC3339))
		}
	}
}

type FormatTest struct {
	name   string
	format string
	result string
}

var formatTests = []FormatTest{
	{"ANSIC", ANSIC, "Wed Feb  4 21:00:57 2009"},
	{"UnixDate", UnixDate, "Wed Feb  4 21:00:57 PST 2009"},
	{"RubyDate", RubyDate, "Wed Feb 04 21:00:57 -0800 2009"},
	{"RFC822", RFC822, "04 Feb 09 2100 PST"},
	{"RFC850", RFC850, "Wednesday, 04-Feb-09 21:00:57 PST"},
	{"RFC1123", RFC1123, "Wed, 04 Feb 2009 21:00:57 PST"},
	{"RFC1123Z", RFC1123Z, "Wed, 04 Feb 2009 21:00:57 -0800"},
	{"RFC3339", RFC3339, "2009-02-04T21:00:57-08:00"},
	{"Kitchen", Kitchen, "9:00PM"},
	{"am/pm", "3pm", "9pm"},
	{"AM/PM", "3PM", "9PM"},
	{"two-digit year", "06 01 02", "09 02 04"},
	// Time stamps, Fractional seconds.
	{"Stamp", Stamp, "Feb  4 21:00:57"},
	{"StampMilli", StampMilli, "Feb  4 21:00:57.012"},
	{"StampMicro", StampMicro, "Feb  4 21:00:57.012345"},
	{"StampNano", StampNano, "Feb  4 21:00:57.012345678"},
}

func TestFormat(t *testing.T) {
	// The numeric time represents Thu Feb  4 21:00:57.012345678 PST 2010
	time := NanosecondsToLocalTime(1233810057012345678)
	for _, test := range formatTests {
		result := time.Format(test.format)
		if result != test.result {
			t.Errorf("%s expected %q got %q", test.name, test.result, result)
		}
	}
}

type ParseTest struct {
	name       string
	format     string
	value      string
	hasTZ      bool  // contains a time zone
	hasWD      bool  // contains a weekday
	yearSign   int64 // sign of year
	fracDigits int   // number of digits of fractional second
}

var parseTests = []ParseTest{
	{"ANSIC", ANSIC, "Thu Feb  4 21:00:57 2010", false, true, 1, 0},
	{"UnixDate", UnixDate, "Thu Feb  4 21:00:57 PST 2010", true, true, 1, 0},
	{"RubyDate", RubyDate, "Thu Feb 04 21:00:57 -0800 2010", true, true, 1, 0},
	{"RFC850", RFC850, "Thursday, 04-Feb-10 21:00:57 PST", true, true, 1, 0},
	{"RFC1123", RFC1123, "Thu, 04 Feb 2010 21:00:57 PST", true, true, 1, 0},
	{"RFC1123Z", RFC1123Z, "Thu, 04 Feb 2010 21:00:57 -0800", true, true, 1, 0},
	{"RFC3339", RFC3339, "2010-02-04T21:00:57-08:00", true, false, 1, 0},
	{"custom: \"2006-01-02 15:04:05-07\"", "2006-01-02 15:04:05-07", "2010-02-04 21:00:57-08", true, false, 1, 0},
	// Optional fractional seconds.
	{"ANSIC", ANSIC, "Thu Feb  4 21:00:57.0 2010", false, true, 1, 1},
	{"UnixDate", UnixDate, "Thu Feb  4 21:00:57.01 PST 2010", true, true, 1, 2},
	{"RubyDate", RubyDate, "Thu Feb 04 21:00:57.012 -0800 2010", true, true, 1, 3},
	{"RFC850", RFC850, "Thursday, 04-Feb-10 21:00:57.0123 PST", true, true, 1, 4},
	{"RFC1123", RFC1123, "Thu, 04 Feb 2010 21:00:57.01234 PST", true, true, 1, 5},
	{"RFC1123Z", RFC1123Z, "Thu, 04 Feb 2010 21:00:57.01234 -0800", true, true, 1, 5},
	{"RFC3339", RFC3339, "2010-02-04T21:00:57.012345678-08:00", true, false, 1, 9},
	// Amount of white space should not matter.
	{"ANSIC", ANSIC, "Thu Feb 4 21:00:57 2010", false, true, 1, 0},
	{"ANSIC", ANSIC, "Thu      Feb     4     21:00:57     2010", false, true, 1, 0},
	// Case should not matter
	{"ANSIC", ANSIC, "THU FEB 4 21:00:57 2010", false, true, 1, 0},
	{"ANSIC", ANSIC, "thu feb 4 21:00:57 2010", false, true, 1, 0},
	// Fractional seconds.
	{"millisecond", "Mon Jan _2 15:04:05.000 2006", "Thu Feb  4 21:00:57.012 2010", false, true, 1, 3},
	{"microsecond", "Mon Jan _2 15:04:05.000000 2006", "Thu Feb  4 21:00:57.012345 2010", false, true, 1, 6},
	{"nanosecond", "Mon Jan _2 15:04:05.000000000 2006", "Thu Feb  4 21:00:57.012345678 2010", false, true, 1, 9},
	// Leading zeros in other places should not be taken as fractional seconds.
	{"zero1", "2006.01.02.15.04.05.0", "2010.02.04.21.00.57.0", false, false, 1, 1},
	{"zero2", "2006.01.02.15.04.05.00", "2010.02.04.21.00.57.01", false, false, 1, 2},
}

func TestParse(t *testing.T) {
	for _, test := range parseTests {
		time, err := Parse(test.format, test.value)
		if err != nil {
			t.Errorf("%s error: %v", test.name, err)
		} else {
			checkTime(time, &test, t)
		}
	}
}

var rubyTests = []ParseTest{
	{"RubyDate", RubyDate, "Thu Feb 04 21:00:57 -0800 2010", true, true, 1, 0},
	// Ignore the time zone in the test. If it parses, it'll be OK.
	{"RubyDate", RubyDate, "Thu Feb 04 21:00:57 -0000 2010", false, true, 1, 0},
	{"RubyDate", RubyDate, "Thu Feb 04 21:00:57 +0000 2010", false, true, 1, 0},
	{"RubyDate", RubyDate, "Thu Feb 04 21:00:57 +1130 2010", false, true, 1, 0},
}

// Problematic time zone format needs special tests.
func TestRubyParse(t *testing.T) {
	for _, test := range rubyTests {
		time, err := Parse(test.format, test.value)
		if err != nil {
			t.Errorf("%s error: %v", test.name, err)
		} else {
			checkTime(time, &test, t)
		}
	}
}

func checkTime(time *Time, test *ParseTest, t *testing.T) {
	// The time should be Thu Feb  4 21:00:57 PST 2010
	if test.yearSign*time.Year != 2010 {
		t.Errorf("%s: bad year: %d not %d", test.name, time.Year, 2010)
	}
	if time.Month != 2 {
		t.Errorf("%s: bad month: %d not %d", test.name, time.Month, 2)
	}
	if time.Day != 4 {
		t.Errorf("%s: bad day: %d not %d", test.name, time.Day, 4)
	}
	if time.Hour != 21 {
		t.Errorf("%s: bad hour: %d not %d", test.name, time.Hour, 21)
	}
	if time.Minute != 0 {
		t.Errorf("%s: bad minute: %d not %d", test.name, time.Minute, 0)
	}
	if time.Second != 57 {
		t.Errorf("%s: bad second: %d not %d", test.name, time.Second, 57)
	}
	// Nanoseconds must be checked against the precision of the input.
	nanosec, err := strconv.Atoui("012345678"[:test.fracDigits] + "000000000"[:9-test.fracDigits])
	if err != nil {
		panic(err)
	}
	if time.Nanosecond != int(nanosec) {
		t.Errorf("%s: bad nanosecond: %d not %d", test.name, time.Nanosecond, nanosec)
	}
	if test.hasTZ && time.ZoneOffset != -28800 {
		t.Errorf("%s: bad tz offset: %d not %d", test.name, time.ZoneOffset, -28800)
	}
	if test.hasWD && time.Weekday() != 4 {
		t.Errorf("%s: bad weekday: %d not %d", test.name, time.Weekday(), 4)
	}
}

func TestFormatAndParse(t *testing.T) {
	const fmt = "Mon MST " + RFC3339 // all fields
	f := func(sec int64) bool {
		t1 := SecondsToLocalTime(sec)
		if t1.Year < 1000 || t1.Year > 9999 {
			// not required to work
			return true
		}
		t2, err := Parse(fmt, t1.Format(fmt))
		if err != nil {
			t.Errorf("error: %s", err)
			return false
		}
		if !same(t1, t2) {
			t.Errorf("different: %q %q", t1, t2)
			return false
		}
		return true
	}
	f32 := func(sec int32) bool { return f(int64(sec)) }
	cfg := &quick.Config{MaxCount: 10000}

	// Try a reasonable date first, then the huge ones.
	if err := quick.Check(f32, cfg); err != nil {
		t.Fatal(err)
	}
	if err := quick.Check(f, cfg); err != nil {
		t.Fatal(err)
	}
}

type ParseErrorTest struct {
	format string
	value  string
	expect string // must appear within the error
}

var parseErrorTests = []ParseErrorTest{
	{ANSIC, "Feb  4 21:00:60 2010", "cannot parse"}, // cannot parse Feb as Mon
	{ANSIC, "Thu Feb  4 21:00:57 @2010", "cannot parse"},
	{ANSIC, "Thu Feb  4 21:00:60 2010", "second out of range"},
	{ANSIC, "Thu Feb  4 21:61:57 2010", "minute out of range"},
	{ANSIC, "Thu Feb  4 24:00:60 2010", "hour out of range"},
	{"Mon Jan _2 15:04:05.000 2006", "Thu Feb  4 23:00:59x01 2010", "cannot parse"},
	{"Mon Jan _2 15:04:05.000 2006", "Thu Feb  4 23:00:59.xxx 2010", "cannot parse"},
	{"Mon Jan _2 15:04:05.000 2006", "Thu Feb  4 23:00:59.-123 2010", "fractional second out of range"},
}

func TestParseErrors(t *testing.T) {
	for _, test := range parseErrorTests {
		_, err := Parse(test.format, test.value)
		if err == nil {
			t.Errorf("expected error for %q %q", test.format, test.value)
		} else if strings.Index(err.Error(), test.expect) < 0 {
			t.Errorf("expected error with %q for %q %q; got %s", test.expect, test.format, test.value, err)
		}
	}
}

func TestNoonIs12PM(t *testing.T) {
	noon := Time{Hour: 12}
	const expect = "12:00PM"
	got := noon.Format("3:04PM")
	if got != expect {
		t.Errorf("got %q; expect %q", got, expect)
	}
	got = noon.Format("03:04PM")
	if got != expect {
		t.Errorf("got %q; expect %q", got, expect)
	}
}

func TestMidnightIs12AM(t *testing.T) {
	midnight := Time{Hour: 0}
	expect := "12:00AM"
	got := midnight.Format("3:04PM")
	if got != expect {
		t.Errorf("got %q; expect %q", got, expect)
	}
	got = midnight.Format("03:04PM")
	if got != expect {
		t.Errorf("got %q; expect %q", got, expect)
	}
}

func Test12PMIsNoon(t *testing.T) {
	noon, err := Parse("3:04PM", "12:00PM")
	if err != nil {
		t.Fatal("error parsing date:", err)
	}
	if noon.Hour != 12 {
		t.Errorf("got %d; expect 12", noon.Hour)
	}
	noon, err = Parse("03:04PM", "12:00PM")
	if err != nil {
		t.Fatal("error parsing date:", err)
	}
	if noon.Hour != 12 {
		t.Errorf("got %d; expect 12", noon.Hour)
	}
}

func Test12AMIsMidnight(t *testing.T) {
	midnight, err := Parse("3:04PM", "12:00AM")
	if err != nil {
		t.Fatal("error parsing date:", err)
	}
	if midnight.Hour != 0 {
		t.Errorf("got %d; expect 0", midnight.Hour)
	}
	midnight, err = Parse("03:04PM", "12:00AM")
	if err != nil {
		t.Fatal("error parsing date:", err)
	}
	if midnight.Hour != 0 {
		t.Errorf("got %d; expect 0", midnight.Hour)
	}
}

// Check that a time without a Zone still produces a (numeric) time zone
// when formatted with MST as a requested zone.
func TestMissingZone(t *testing.T) {
	time, err := Parse(RubyDate, "Thu Feb 02 16:10:03 -0500 2006")
	if err != nil {
		t.Fatal("error parsing date:", err)
	}
	expect := "Thu Feb  2 16:10:03 -0500 2006" // -0500 not EST
	str := time.Format(UnixDate)               // uses MST as its time zone
	if str != expect {
		t.Errorf("expected %q got %q", expect, str)
	}
}

func TestMinutesInTimeZone(t *testing.T) {
	time, err := Parse(RubyDate, "Mon Jan 02 15:04:05 +0123 2006")
	if err != nil {
		t.Fatal("error parsing date:", err)
	}
	expected := (1*60 + 23) * 60
	if time.ZoneOffset != expected {
		t.Errorf("ZoneOffset incorrect, expected %d got %d", expected, time.ZoneOffset)
	}
}

func BenchmarkSeconds(b *testing.B) {
	for i := 0; i < b.N; i++ {
		Seconds()
	}
}

func BenchmarkNanoseconds(b *testing.B) {
	for i := 0; i < b.N; i++ {
		Nanoseconds()
	}
}

func BenchmarkFormat(b *testing.B) {
	time := SecondsToLocalTime(1265346057)
	for i := 0; i < b.N; i++ {
		time.Format("Mon Jan  2 15:04:05 2006")
	}
}

func BenchmarkParse(b *testing.B) {
	for i := 0; i < b.N; i++ {
		Parse(ANSIC, "Mon Jan  2 15:04:05 2006")
	}
}
