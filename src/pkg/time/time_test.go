// Copyright 2009 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package time_test

import (
	"os"
	"strings"
	"testing"
	"testing/quick"
	. "time"
)

func init() {
	// Force US Pacific time for daylight-savings
	// tests below (localtests).  Needs to be set
	// before the first call into the time library.
	os.Setenv("TZ", "America/Los_Angeles")
}

type TimeTest struct {
	seconds int64
	golden  Time
}

var utctests = []TimeTest{
	{0, Time{1970, 1, 1, 0, 0, 0, Thursday, 0, "UTC"}},
	{1221681866, Time{2008, 9, 17, 20, 4, 26, Wednesday, 0, "UTC"}},
	{-1221681866, Time{1931, 4, 16, 3, 55, 34, Thursday, 0, "UTC"}},
	{-11644473600, Time{1601, 1, 1, 0, 0, 0, Monday, 0, "UTC"}},
	{599529660, Time{1988, 12, 31, 0, 1, 0, Saturday, 0, "UTC"}},
	{978220860, Time{2000, 12, 31, 0, 1, 0, Sunday, 0, "UTC"}},
	{1e18, Time{31688740476, 10, 23, 1, 46, 40, Friday, 0, "UTC"}},
	{-1e18, Time{-31688736537, 3, 10, 22, 13, 20, Tuesday, 0, "UTC"}},
	{0x7fffffffffffffff, Time{292277026596, 12, 4, 15, 30, 7, Sunday, 0, "UTC"}},
	{-0x8000000000000000, Time{-292277022657, 1, 27, 8, 29, 52, Sunday, 0, "UTC"}},
}

var localtests = []TimeTest{
	{0, Time{1969, 12, 31, 16, 0, 0, Wednesday, -8 * 60 * 60, "PST"}},
	{1221681866, Time{2008, 9, 17, 13, 4, 26, Wednesday, -7 * 60 * 60, "PDT"}},
}

func same(t, u *Time) bool {
	return t.Year == u.Year &&
		t.Month == u.Month &&
		t.Day == u.Day &&
		t.Hour == u.Hour &&
		t.Minute == u.Minute &&
		t.Second == u.Second &&
		t.Weekday == u.Weekday &&
		t.ZoneOffset == u.ZoneOffset &&
		t.Zone == u.Zone
}

func TestSecondsToUTC(t *testing.T) {
	for i := 0; i < len(utctests); i++ {
		sec := utctests[i].seconds
		golden := &utctests[i].golden
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

func TestSecondsToLocalTime(t *testing.T) {
	for i := 0; i < len(localtests); i++ {
		sec := localtests[i].seconds
		golden := &localtests[i].golden
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

type TimeFormatTest struct {
	time           Time
	formattedValue string
}

var rfc3339Formats = []TimeFormatTest{
	{Time{2008, 9, 17, 20, 4, 26, Wednesday, 0, "UTC"}, "2008-09-17T20:04:26Z"},
	{Time{1994, 9, 17, 20, 4, 26, Wednesday, -18000, "EST"}, "1994-09-17T20:04:26-05:00"},
	{Time{2000, 12, 26, 1, 15, 6, Wednesday, 15600, "OTO"}, "2000-12-26T01:15:06+04:20"},
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
	{"ANSIC", ANSIC, "Thu Feb  4 21:00:57 2010"},
	{"UnixDate", UnixDate, "Thu Feb  4 21:00:57 PST 2010"},
	{"RubyDate", RubyDate, "Thu Feb 04 21:00:57 -0800 2010"},
	{"RFC822", RFC822, "04 Feb 10 2100 PST"},
	{"RFC850", RFC850, "Thursday, 04-Feb-10 21:00:57 PST"},
	{"RFC1123", RFC1123, "Thu, 04 Feb 2010 21:00:57 PST"},
	{"RFC3339", RFC3339, "2010-02-04T21:00:57-08:00"},
	{"Kitchen", Kitchen, "9:00PM"},
	{"am/pm", "3pm", "9pm"},
	{"AM/PM", "3PM", "9PM"},
}

func TestFormat(t *testing.T) {
	// The numeric time represents Thu Feb  4 21:00:57 PST 2010
	time := SecondsToLocalTime(1265346057)
	for _, test := range formatTests {
		result := time.Format(test.format)
		if result != test.result {
			t.Errorf("%s expected %q got %q", test.name, test.result, result)
		}
	}
}

type ParseTest struct {
	name     string
	format   string
	value    string
	hasTZ    bool  // contains a time zone
	hasWD    bool  // contains a weekday
	yearSign int64 // sign of year
}

var parseTests = []ParseTest{
	{"ANSIC", ANSIC, "Thu Feb  4 21:00:57 2010", false, true, 1},
	{"UnixDate", UnixDate, "Thu Feb  4 21:00:57 PST 2010", true, true, 1},
	{"RubyDate", RubyDate, "Thu Feb 04 21:00:57 -0800 2010", true, true, 1},
	{"RFC850", RFC850, "Thursday, 04-Feb-10 21:00:57 PST", true, true, 1},
	{"RFC1123", RFC1123, "Thu, 04 Feb 2010 21:00:57 PST", true, true, 1},
	{"RFC3339", RFC3339, "2010-02-04T21:00:57-08:00", true, false, 1},
	{"custom: \"2006-01-02 15:04:05-07\"", "2006-01-02 15:04:05-07", "2010-02-04 21:00:57-08", true, false, 1},
	// Amount of white space should not matter.
	{"ANSIC", ANSIC, "Thu Feb 4 21:00:57 2010", false, true, 1},
	{"ANSIC", ANSIC, "Thu      Feb     4     21:00:57     2010", false, true, 1},
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
	{"RubyDate", RubyDate, "Thu Feb 04 21:00:57 -0800 2010", true, true, 1},
	// Ignore the time zone in the test. If it parses, it'll be OK.
	{"RubyDate", RubyDate, "Thu Feb 04 21:00:57 -0000 2010", false, true, 1},
	{"RubyDate", RubyDate, "Thu Feb 04 21:00:57 +0000 2010", false, true, 1},
	{"RubyDate", RubyDate, "Thu Feb 04 21:00:57 +1130 2010", false, true, 1},
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
	if test.hasTZ && time.ZoneOffset != -28800 {
		t.Errorf("%s: bad tz offset: %d not %d", test.name, time.ZoneOffset, -28800)
	}
	if test.hasWD && time.Weekday != 4 {
		t.Errorf("%s: bad weekday: %d not %d", test.name, time.Weekday, 4)
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
	{ANSIC, "Feb  4 21:00:60 2010", "parse"}, // cannot parse Feb as Mon
	{ANSIC, "Thu Feb  4 21:00:57 @2010", "parse"},
	{ANSIC, "Thu Feb  4 21:00:60 2010", "second out of range"},
	{ANSIC, "Thu Feb  4 21:61:57 2010", "minute out of range"},
	{ANSIC, "Thu Feb  4 24:00:60 2010", "hour out of range"},
}

func TestParseErrors(t *testing.T) {
	for _, test := range parseErrorTests {
		_, err := Parse(test.format, test.value)
		if err == nil {
			t.Errorf("expected error for %q %q", test.format, test.value)
		} else if strings.Index(err.String(), test.expect) < 0 {
			t.Errorf("expected error with %q for %q %q; got %s", test.expect, test.format, test.value, err)
		}
	}
}

// Check that a time without a Zone still produces a (numeric) time zone
// when formatted with MST as a requested zone.
func TestMissingZone(t *testing.T) {
	time, err := Parse(RubyDate, "Tue Feb 02 16:10:03 -0500 2006")
	if err != nil {
		t.Fatal("error parsing date:", err)
	}
	expect := "Tue Feb  2 16:10:03 -0500 2006" // -0500 not EST
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
