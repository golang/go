// Copyright 2009 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package time_test

import (
	"fmt"
	"strconv"
	"strings"
	"testing"
	"testing/quick"
	. "time"
)

var nextStdChunkTests = []string{
	"(2006)-(01)-(02)T(15):(04):(05)(Z07:00)",
	"(2006)-(01)-(02) (002) (15):(04):(05)",
	"(2006)-(01) (002) (15):(04):(05)",
	"(2006)-(002) (15):(04):(05)",
	"(2006)(002)(01) (15):(04):(05)",
	"(2006)(002)(04) (15):(04):(05)",
}

func TestNextStdChunk(t *testing.T) {
	// Most bugs in Parse or Format boil down to problems with
	// the exact detection of format chunk boundaries in the
	// helper function nextStdChunk (here called as NextStdChunk).
	// This test checks nextStdChunk's behavior directly,
	// instead of needing to test it only indirectly through Parse/Format.

	// markChunks returns format with each detected
	// 'format chunk' parenthesized.
	// For example showChunks("2006-01-02") == "(2006)-(01)-(02)".
	markChunks := func(format string) string {
		// Note that NextStdChunk and StdChunkNames
		// are not part of time's public API.
		// They are exported in export_test for this test.
		out := ""
		for s := format; s != ""; {
			prefix, std, suffix := NextStdChunk(s)
			out += prefix
			if std > 0 {
				out += "(" + StdChunkNames[std] + ")"
			}
			s = suffix
		}
		return out
	}

	noParens := func(r rune) rune {
		if r == '(' || r == ')' {
			return -1
		}
		return r
	}

	for _, marked := range nextStdChunkTests {
		// marked is an expected output from markChunks.
		// If we delete the parens and pass it through markChunks,
		// we should get the original back.
		format := strings.Map(noParens, marked)
		out := markChunks(format)
		if out != marked {
			t.Errorf("nextStdChunk parses %q as %q, want %q", format, out, marked)
		}
	}
}

type TimeFormatTest struct {
	time           Time
	formattedValue string
}

var rfc3339Formats = []TimeFormatTest{
	{Date(2008, 9, 17, 20, 4, 26, 0, UTC), "2008-09-17T20:04:26Z"},
	{Date(1994, 9, 17, 20, 4, 26, 0, FixedZone("EST", -18000)), "1994-09-17T20:04:26-05:00"},
	{Date(2000, 12, 26, 1, 15, 6, 0, FixedZone("OTO", 15600)), "2000-12-26T01:15:06+04:20"},
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
	{"RFC822", RFC822, "04 Feb 09 21:00 PST"},
	{"RFC850", RFC850, "Wednesday, 04-Feb-09 21:00:57 PST"},
	{"RFC1123", RFC1123, "Wed, 04 Feb 2009 21:00:57 PST"},
	{"RFC1123Z", RFC1123Z, "Wed, 04 Feb 2009 21:00:57 -0800"},
	{"RFC3339", RFC3339, "2009-02-04T21:00:57-08:00"},
	{"RFC3339Nano", RFC3339Nano, "2009-02-04T21:00:57.0123456-08:00"},
	{"Kitchen", Kitchen, "9:00PM"},
	{"am/pm", "3pm", "9pm"},
	{"AM/PM", "3PM", "9PM"},
	{"two-digit year", "06 01 02", "09 02 04"},
	// Three-letter months and days must not be followed by lower-case letter.
	{"Janet", "Hi Janet, the Month is January", "Hi Janet, the Month is February"},
	// Time stamps, Fractional seconds.
	{"Stamp", Stamp, "Feb  4 21:00:57"},
	{"StampMilli", StampMilli, "Feb  4 21:00:57.012"},
	{"StampMicro", StampMicro, "Feb  4 21:00:57.012345"},
	{"StampNano", StampNano, "Feb  4 21:00:57.012345600"},
	{"DateTime", DateTime, "2009-02-04 21:00:57"},
	{"DateOnly", DateOnly, "2009-02-04"},
	{"TimeOnly", TimeOnly, "21:00:57"},
	{"YearDay", "Jan  2 002 __2 2", "Feb  4 035  35 4"},
	{"Year", "2006 6 06 _6 __6 ___6", "2009 6 09 _6 __6 ___6"},
	{"Month", "Jan January 1 01 _1", "Feb February 2 02 _2"},
	{"DayOfMonth", "2 02 _2 __2", "4 04  4  35"},
	{"DayOfWeek", "Mon Monday", "Wed Wednesday"},
	{"Hour", "15 3 03 _3", "21 9 09 _9"},
	{"Minute", "4 04 _4", "0 00 _0"},
	{"Second", "5 05 _5", "57 57 _57"},
}

func TestFormat(t *testing.T) {
	// The numeric time represents Thu Feb  4 21:00:57.012345600 PST 2009
	time := Unix(0, 1233810057012345600)
	for _, test := range formatTests {
		result := time.Format(test.format)
		if result != test.result {
			t.Errorf("%s expected %q got %q", test.name, test.result, result)
		}
	}
}

var goStringTests = []struct {
	in   Time
	want string
}{
	{Date(2009, February, 5, 5, 0, 57, 12345600, UTC),
		"time.Date(2009, time.February, 5, 5, 0, 57, 12345600, time.UTC)"},
	{Date(2009, February, 5, 5, 0, 57, 12345600, Local),
		"time.Date(2009, time.February, 5, 5, 0, 57, 12345600, time.Local)"},
	{Date(2009, February, 5, 5, 0, 57, 12345600, FixedZone("Europe/Berlin", 3*60*60)),
		`time.Date(2009, time.February, 5, 5, 0, 57, 12345600, time.Location("Europe/Berlin"))`,
	},
	{Date(2009, February, 5, 5, 0, 57, 12345600, FixedZone("Non-ASCII character ⏰", 3*60*60)),
		`time.Date(2009, time.February, 5, 5, 0, 57, 12345600, time.Location("Non-ASCII character \xe2\x8f\xb0"))`,
	},
}

func TestGoString(t *testing.T) {
	// The numeric time represents Thu Feb  4 21:00:57.012345600 PST 2009
	for _, tt := range goStringTests {
		if tt.in.GoString() != tt.want {
			t.Errorf("GoString (%q): got %q want %q", tt.in, tt.in.GoString(), tt.want)
		}
	}
}

// issue 12440.
func TestFormatSingleDigits(t *testing.T) {
	time := Date(2001, 2, 3, 4, 5, 6, 700000000, UTC)
	test := FormatTest{"single digit format", "3:4:5", "4:5:6"}
	result := time.Format(test.format)
	if result != test.result {
		t.Errorf("%s expected %q got %q", test.name, test.result, result)
	}
}

func TestFormatShortYear(t *testing.T) {
	years := []int{
		-100001, -100000, -99999,
		-10001, -10000, -9999,
		-1001, -1000, -999,
		-101, -100, -99,
		-11, -10, -9,
		-1, 0, 1,
		9, 10, 11,
		99, 100, 101,
		999, 1000, 1001,
		9999, 10000, 10001,
		99999, 100000, 100001,
	}

	for _, y := range years {
		time := Date(y, January, 1, 0, 0, 0, 0, UTC)
		result := time.Format("2006.01.02")
		var want string
		if y < 0 {
			// The 4 in %04d counts the - sign, so print -y instead
			// and introduce our own - sign.
			want = fmt.Sprintf("-%04d.%02d.%02d", -y, 1, 1)
		} else {
			want = fmt.Sprintf("%04d.%02d.%02d", y, 1, 1)
		}
		if result != want {
			t.Errorf("(jan 1 %d).Format(\"2006.01.02\") = %q, want %q", y, result, want)
		}
	}
}

type ParseTest struct {
	name       string
	format     string
	value      string
	hasTZ      bool // contains a time zone
	hasWD      bool // contains a weekday
	yearSign   int  // sign of year, -1 indicates the year is not present in the format
	fracDigits int  // number of digits of fractional second
}

var parseTests = []ParseTest{
	{"ANSIC", ANSIC, "Thu Feb  4 21:00:57 2010", false, true, 1, 0},
	{"UnixDate", UnixDate, "Thu Feb  4 21:00:57 PST 2010", true, true, 1, 0},
	{"RubyDate", RubyDate, "Thu Feb 04 21:00:57 -0800 2010", true, true, 1, 0},
	{"RFC850", RFC850, "Thursday, 04-Feb-10 21:00:57 PST", true, true, 1, 0},
	{"RFC1123", RFC1123, "Thu, 04 Feb 2010 21:00:57 PST", true, true, 1, 0},
	{"RFC1123", RFC1123, "Thu, 04 Feb 2010 22:00:57 PDT", true, true, 1, 0},
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
	{"custom: \"2006-01-02 15:04:05\"", "2006-01-02 15:04:05", "2010-02-04 21:00:57.0", false, false, 1, 0},
	// Amount of white space should not matter.
	{"ANSIC", ANSIC, "Thu Feb 4 21:00:57 2010", false, true, 1, 0},
	{"ANSIC", ANSIC, "Thu      Feb     4     21:00:57     2010", false, true, 1, 0},
	// Case should not matter
	{"ANSIC", ANSIC, "THU FEB 4 21:00:57 2010", false, true, 1, 0},
	{"ANSIC", ANSIC, "thu feb 4 21:00:57 2010", false, true, 1, 0},
	// Fractional seconds.
	{"millisecond:: dot separator", "Mon Jan _2 15:04:05.000 2006", "Thu Feb  4 21:00:57.012 2010", false, true, 1, 3},
	{"microsecond:: dot separator", "Mon Jan _2 15:04:05.000000 2006", "Thu Feb  4 21:00:57.012345 2010", false, true, 1, 6},
	{"nanosecond:: dot separator", "Mon Jan _2 15:04:05.000000000 2006", "Thu Feb  4 21:00:57.012345678 2010", false, true, 1, 9},
	{"millisecond:: comma separator", "Mon Jan _2 15:04:05,000 2006", "Thu Feb  4 21:00:57.012 2010", false, true, 1, 3},
	{"microsecond:: comma separator", "Mon Jan _2 15:04:05,000000 2006", "Thu Feb  4 21:00:57.012345 2010", false, true, 1, 6},
	{"nanosecond:: comma separator", "Mon Jan _2 15:04:05,000000000 2006", "Thu Feb  4 21:00:57.012345678 2010", false, true, 1, 9},

	// Leading zeros in other places should not be taken as fractional seconds.
	{"zero1", "2006.01.02.15.04.05.0", "2010.02.04.21.00.57.0", false, false, 1, 1},
	{"zero2", "2006.01.02.15.04.05.00", "2010.02.04.21.00.57.01", false, false, 1, 2},
	// Month and day names only match when not followed by a lower-case letter.
	{"Janet", "Hi Janet, the Month is January: Jan _2 15:04:05 2006", "Hi Janet, the Month is February: Feb  4 21:00:57 2010", false, true, 1, 0},

	// GMT with offset.
	{"GMT-8", UnixDate, "Fri Feb  5 05:00:57 GMT-8 2010", true, true, 1, 0},

	// Accept any number of fractional second digits (including none) for .999...
	// In Go 1, .999... was completely ignored in the format, meaning the first two
	// cases would succeed, but the next four would not. Go 1.1 accepts all six.
	// decimal "." separator.
	{"", "2006-01-02 15:04:05.9999 -0700 MST", "2010-02-04 21:00:57 -0800 PST", true, false, 1, 0},
	{"", "2006-01-02 15:04:05.999999999 -0700 MST", "2010-02-04 21:00:57 -0800 PST", true, false, 1, 0},
	{"", "2006-01-02 15:04:05.9999 -0700 MST", "2010-02-04 21:00:57.0123 -0800 PST", true, false, 1, 4},
	{"", "2006-01-02 15:04:05.999999999 -0700 MST", "2010-02-04 21:00:57.0123 -0800 PST", true, false, 1, 4},
	{"", "2006-01-02 15:04:05.9999 -0700 MST", "2010-02-04 21:00:57.012345678 -0800 PST", true, false, 1, 9},
	{"", "2006-01-02 15:04:05.999999999 -0700 MST", "2010-02-04 21:00:57.012345678 -0800 PST", true, false, 1, 9},
	// comma "," separator.
	{"", "2006-01-02 15:04:05,9999 -0700 MST", "2010-02-04 21:00:57 -0800 PST", true, false, 1, 0},
	{"", "2006-01-02 15:04:05,999999999 -0700 MST", "2010-02-04 21:00:57 -0800 PST", true, false, 1, 0},
	{"", "2006-01-02 15:04:05,9999 -0700 MST", "2010-02-04 21:00:57.0123 -0800 PST", true, false, 1, 4},
	{"", "2006-01-02 15:04:05,999999999 -0700 MST", "2010-02-04 21:00:57.0123 -0800 PST", true, false, 1, 4},
	{"", "2006-01-02 15:04:05,9999 -0700 MST", "2010-02-04 21:00:57.012345678 -0800 PST", true, false, 1, 9},
	{"", "2006-01-02 15:04:05,999999999 -0700 MST", "2010-02-04 21:00:57.012345678 -0800 PST", true, false, 1, 9},

	// issue 4502.
	{"", StampNano, "Feb  4 21:00:57.012345678", false, false, -1, 9},
	{"", "Jan _2 15:04:05.999", "Feb  4 21:00:57.012300000", false, false, -1, 4},
	{"", "Jan _2 15:04:05.999", "Feb  4 21:00:57.012345678", false, false, -1, 9},
	{"", "Jan _2 15:04:05.999999999", "Feb  4 21:00:57.0123", false, false, -1, 4},
	{"", "Jan _2 15:04:05.999999999", "Feb  4 21:00:57.012345678", false, false, -1, 9},

	// Day of year.
	{"", "2006-01-02 002 15:04:05", "2010-02-04 035 21:00:57", false, false, 1, 0},
	{"", "2006-01 002 15:04:05", "2010-02 035 21:00:57", false, false, 1, 0},
	{"", "2006-002 15:04:05", "2010-035 21:00:57", false, false, 1, 0},
	{"", "200600201 15:04:05", "201003502 21:00:57", false, false, 1, 0},
	{"", "200600204 15:04:05", "201003504 21:00:57", false, false, 1, 0},
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

// All parsed with ANSIC.
var dayOutOfRangeTests = []struct {
	date string
	ok   bool
}{
	{"Thu Jan 99 21:00:57 2010", false},
	{"Thu Jan 31 21:00:57 2010", true},
	{"Thu Jan 32 21:00:57 2010", false},
	{"Thu Feb 28 21:00:57 2012", true},
	{"Thu Feb 29 21:00:57 2012", true},
	{"Thu Feb 29 21:00:57 2010", false},
	{"Thu Mar 31 21:00:57 2010", true},
	{"Thu Mar 32 21:00:57 2010", false},
	{"Thu Apr 30 21:00:57 2010", true},
	{"Thu Apr 31 21:00:57 2010", false},
	{"Thu May 31 21:00:57 2010", true},
	{"Thu May 32 21:00:57 2010", false},
	{"Thu Jun 30 21:00:57 2010", true},
	{"Thu Jun 31 21:00:57 2010", false},
	{"Thu Jul 31 21:00:57 2010", true},
	{"Thu Jul 32 21:00:57 2010", false},
	{"Thu Aug 31 21:00:57 2010", true},
	{"Thu Aug 32 21:00:57 2010", false},
	{"Thu Sep 30 21:00:57 2010", true},
	{"Thu Sep 31 21:00:57 2010", false},
	{"Thu Oct 31 21:00:57 2010", true},
	{"Thu Oct 32 21:00:57 2010", false},
	{"Thu Nov 30 21:00:57 2010", true},
	{"Thu Nov 31 21:00:57 2010", false},
	{"Thu Dec 31 21:00:57 2010", true},
	{"Thu Dec 32 21:00:57 2010", false},
	{"Thu Dec 00 21:00:57 2010", false},
}

func TestParseDayOutOfRange(t *testing.T) {
	for _, test := range dayOutOfRangeTests {
		_, err := Parse(ANSIC, test.date)
		switch {
		case test.ok && err == nil:
			// OK
		case !test.ok && err != nil:
			if !strings.Contains(err.Error(), "day out of range") {
				t.Errorf("%q: expected 'day' error, got %v", test.date, err)
			}
		case test.ok && err != nil:
			t.Errorf("%q: unexpected error: %v", test.date, err)
		case !test.ok && err == nil:
			t.Errorf("%q: expected 'day' error, got none", test.date)
		}
	}
}

// TestParseInLocation checks that the Parse and ParseInLocation
// functions do not get confused by the fact that AST (Arabia Standard
// Time) and AST (Atlantic Standard Time) are different time zones,
// even though they have the same abbreviation.
//
// ICANN has been slowly phasing out invented abbreviation in favor of
// numeric time zones (for example, the Asia/Baghdad time zone
// abbreviation got changed from AST to +03 in the 2017a tzdata
// release); but we still want to make sure that the time package does
// not get confused on systems with slightly older tzdata packages.
func TestParseInLocation(t *testing.T) {

	baghdad, err := LoadLocation("Asia/Baghdad")
	if err != nil {
		t.Fatal(err)
	}

	var t1, t2 Time

	t1, err = ParseInLocation("Jan 02 2006 MST", "Feb 01 2013 AST", baghdad)
	if err != nil {
		t.Fatal(err)
	}

	_, offset := t1.Zone()

	// A zero offset means that ParseInLocation did not recognize the
	// 'AST' abbreviation as matching the current location (Baghdad,
	// where we'd expect a +03 hrs offset); likely because we're using
	// a recent tzdata release (2017a or newer).
	// If it happens, skip the Baghdad test.
	if offset != 0 {
		t2 = Date(2013, February, 1, 00, 00, 00, 0, baghdad)
		if t1 != t2 {
			t.Fatalf("ParseInLocation(Feb 01 2013 AST, Baghdad) = %v, want %v", t1, t2)
		}
		if offset != 3*60*60 {
			t.Fatalf("ParseInLocation(Feb 01 2013 AST, Baghdad).Zone = _, %d, want _, %d", offset, 3*60*60)
		}
	}

	blancSablon, err := LoadLocation("America/Blanc-Sablon")
	if err != nil {
		t.Fatal(err)
	}

	// In this case 'AST' means 'Atlantic Standard Time', and we
	// expect the abbreviation to correctly match the american
	// location.
	t1, err = ParseInLocation("Jan 02 2006 MST", "Feb 01 2013 AST", blancSablon)
	if err != nil {
		t.Fatal(err)
	}
	t2 = Date(2013, February, 1, 00, 00, 00, 0, blancSablon)
	if t1 != t2 {
		t.Fatalf("ParseInLocation(Feb 01 2013 AST, Blanc-Sablon) = %v, want %v", t1, t2)
	}
	_, offset = t1.Zone()
	if offset != -4*60*60 {
		t.Fatalf("ParseInLocation(Feb 01 2013 AST, Blanc-Sablon).Zone = _, %d, want _, %d", offset, -4*60*60)
	}
}

func TestLoadLocationZipFile(t *testing.T) {
	undo := DisablePlatformSources()
	defer undo()

	_, err := LoadLocation("Australia/Sydney")
	if err != nil {
		t.Fatal(err)
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

func checkTime(time Time, test *ParseTest, t *testing.T) {
	// The time should be Thu Feb  4 21:00:57 PST 2010
	if test.yearSign >= 0 && test.yearSign*time.Year() != 2010 {
		t.Errorf("%s: bad year: %d not %d", test.name, time.Year(), 2010)
	}
	if time.Month() != February {
		t.Errorf("%s: bad month: %s not %s", test.name, time.Month(), February)
	}
	if time.Day() != 4 {
		t.Errorf("%s: bad day: %d not %d", test.name, time.Day(), 4)
	}
	if time.Hour() != 21 {
		t.Errorf("%s: bad hour: %d not %d", test.name, time.Hour(), 21)
	}
	if time.Minute() != 0 {
		t.Errorf("%s: bad minute: %d not %d", test.name, time.Minute(), 0)
	}
	if time.Second() != 57 {
		t.Errorf("%s: bad second: %d not %d", test.name, time.Second(), 57)
	}
	// Nanoseconds must be checked against the precision of the input.
	nanosec, err := strconv.ParseUint("012345678"[:test.fracDigits]+"000000000"[:9-test.fracDigits], 10, 0)
	if err != nil {
		panic(err)
	}
	if time.Nanosecond() != int(nanosec) {
		t.Errorf("%s: bad nanosecond: %d not %d", test.name, time.Nanosecond(), nanosec)
	}
	name, offset := time.Zone()
	if test.hasTZ && offset != -28800 {
		t.Errorf("%s: bad tz offset: %s %d not %d", test.name, name, offset, -28800)
	}
	if test.hasWD && time.Weekday() != Thursday {
		t.Errorf("%s: bad weekday: %s not %s", test.name, time.Weekday(), Thursday)
	}
}

func TestFormatAndParse(t *testing.T) {
	const fmt = "Mon MST " + RFC3339 // all fields
	f := func(sec int64) bool {
		t1 := Unix(sec/2, 0)
		if t1.Year() < 1000 || t1.Year() > 9999 || t1.Unix() != sec {
			// not required to work
			return true
		}
		t2, err := Parse(fmt, t1.Format(fmt))
		if err != nil {
			t.Errorf("error: %s", err)
			return false
		}
		if t1.Unix() != t2.Unix() || t1.Nanosecond() != t2.Nanosecond() {
			t.Errorf("FormatAndParse %d: %q(%d) %q(%d)", sec, t1, t1.Unix(), t2, t2.Unix())
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

type ParseTimeZoneTest struct {
	value  string
	length int
	ok     bool
}

var parseTimeZoneTests = []ParseTimeZoneTest{
	{"gmt hi there", 0, false},
	{"GMT hi there", 3, true},
	{"GMT+12 hi there", 6, true},
	{"GMT+00 hi there", 6, true},
	{"GMT+", 3, true},
	{"GMT+3", 5, true},
	{"GMT+a", 3, true},
	{"GMT+3a", 5, true},
	{"GMT-5 hi there", 5, true},
	{"GMT-51 hi there", 3, true},
	{"ChST hi there", 4, true},
	{"MeST hi there", 4, true},
	{"MSDx", 3, true},
	{"MSDY", 0, false}, // four letters must end in T.
	{"ESAST hi", 5, true},
	{"ESASTT hi", 0, false}, // run of upper-case letters too long.
	{"ESATY hi", 0, false},  // five letters must end in T.
	{"WITA hi", 4, true},    // Issue #18251
	// Issue #24071
	{"+03 hi", 3, true},
	{"-04 hi", 3, true},
	// Issue #26032
	{"+00", 3, true},
	{"-11", 3, true},
	{"-12", 3, true},
	{"-23", 3, true},
	{"-24", 0, false},
	{"+13", 3, true},
	{"+14", 3, true},
	{"+23", 3, true},
	{"+24", 0, false},
}

func TestParseTimeZone(t *testing.T) {
	for _, test := range parseTimeZoneTests {
		length, ok := ParseTimeZone(test.value)
		if ok != test.ok {
			t.Errorf("expected %t for %q got %t", test.ok, test.value, ok)
		} else if length != test.length {
			t.Errorf("expected %d for %q got %d", test.length, test.value, length)
		}
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
	// issue 4502. StampNano requires exactly 9 digits of precision.
	{StampNano, "Dec  7 11:22:01.000000", `cannot parse ".000000" as ".000000000"`},
	{StampNano, "Dec  7 11:22:01.0000000000", `extra text: "0"`},
	// issue 4493. Helpful errors.
	{RFC3339, "2006-01-02T15:04:05Z07:00", `parsing time "2006-01-02T15:04:05Z07:00": extra text: "07:00"`},
	{RFC3339, "2006-01-02T15:04_abc", `parsing time "2006-01-02T15:04_abc" as "2006-01-02T15:04:05Z07:00": cannot parse "_abc" as ":"`},
	{RFC3339, "2006-01-02T15:04:05_abc", `parsing time "2006-01-02T15:04:05_abc" as "2006-01-02T15:04:05Z07:00": cannot parse "_abc" as "Z07:00"`},
	{RFC3339, "2006-01-02T15:04:05Z_abc", `parsing time "2006-01-02T15:04:05Z_abc": extra text: "_abc"`},
	// invalid second followed by optional fractional seconds
	{RFC3339, "2010-02-04T21:00:67.012345678-08:00", "second out of range"},
	// issue 21113
	{"_2 Jan 06 15:04 MST", "4 --- 00 00:00 GMT", "cannot parse"},
	{"_2 January 06 15:04 MST", "4 --- 00 00:00 GMT", "cannot parse"},

	// invalid or mismatched day-of-year
	{"Jan _2 002 2006", "Feb  4 034 2006", "day-of-year does not match day"},
	{"Jan _2 002 2006", "Feb  4 004 2006", "day-of-year does not match month"},

	// issue 45391.
	{`"2006-01-02T15:04:05Z07:00"`, "0", `parsing time "0" as "\"2006-01-02T15:04:05Z07:00\"": cannot parse "0" as "\""`},
	{RFC3339, "\"", `parsing time "\"" as "2006-01-02T15:04:05Z07:00": cannot parse "\"" as "2006"`},
}

func TestParseErrors(t *testing.T) {
	for _, test := range parseErrorTests {
		_, err := Parse(test.format, test.value)
		if err == nil {
			t.Errorf("expected error for %q %q", test.format, test.value)
		} else if !strings.Contains(err.Error(), test.expect) {
			t.Errorf("expected error with %q for %q %q; got %s", test.expect, test.format, test.value, err)
		}
	}
}

func TestNoonIs12PM(t *testing.T) {
	noon := Date(0, January, 1, 12, 0, 0, 0, UTC)
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
	midnight := Date(0, January, 1, 0, 0, 0, 0, UTC)
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
	if noon.Hour() != 12 {
		t.Errorf("got %d; expect 12", noon.Hour())
	}
	noon, err = Parse("03:04PM", "12:00PM")
	if err != nil {
		t.Fatal("error parsing date:", err)
	}
	if noon.Hour() != 12 {
		t.Errorf("got %d; expect 12", noon.Hour())
	}
}

func Test12AMIsMidnight(t *testing.T) {
	midnight, err := Parse("3:04PM", "12:00AM")
	if err != nil {
		t.Fatal("error parsing date:", err)
	}
	if midnight.Hour() != 0 {
		t.Errorf("got %d; expect 0", midnight.Hour())
	}
	midnight, err = Parse("03:04PM", "12:00AM")
	if err != nil {
		t.Fatal("error parsing date:", err)
	}
	if midnight.Hour() != 0 {
		t.Errorf("got %d; expect 0", midnight.Hour())
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
		t.Errorf("got %s; expect %s", str, expect)
	}
}

func TestMinutesInTimeZone(t *testing.T) {
	time, err := Parse(RubyDate, "Mon Jan 02 15:04:05 +0123 2006")
	if err != nil {
		t.Fatal("error parsing date:", err)
	}
	expected := (1*60 + 23) * 60
	_, offset := time.Zone()
	if offset != expected {
		t.Errorf("ZoneOffset = %d, want %d", offset, expected)
	}
}

type SecondsTimeZoneOffsetTest struct {
	format         string
	value          string
	expectedoffset int
}

var secondsTimeZoneOffsetTests = []SecondsTimeZoneOffsetTest{
	{"2006-01-02T15:04:05-070000", "1871-01-01T05:33:02-003408", -(34*60 + 8)},
	{"2006-01-02T15:04:05-07:00:00", "1871-01-01T05:33:02-00:34:08", -(34*60 + 8)},
	{"2006-01-02T15:04:05-070000", "1871-01-01T05:33:02+003408", 34*60 + 8},
	{"2006-01-02T15:04:05-07:00:00", "1871-01-01T05:33:02+00:34:08", 34*60 + 8},
	{"2006-01-02T15:04:05Z070000", "1871-01-01T05:33:02-003408", -(34*60 + 8)},
	{"2006-01-02T15:04:05Z07:00:00", "1871-01-01T05:33:02+00:34:08", 34*60 + 8},
	{"2006-01-02T15:04:05-07", "1871-01-01T05:33:02+01", 1 * 60 * 60},
	{"2006-01-02T15:04:05-07", "1871-01-01T05:33:02-02", -2 * 60 * 60},
	{"2006-01-02T15:04:05Z07", "1871-01-01T05:33:02-02", -2 * 60 * 60},
}

func TestParseSecondsInTimeZone(t *testing.T) {
	// should accept timezone offsets with seconds like: Zone America/New_York   -4:56:02 -      LMT     1883 Nov 18 12:03:58
	for _, test := range secondsTimeZoneOffsetTests {
		time, err := Parse(test.format, test.value)
		if err != nil {
			t.Fatal("error parsing date:", err)
		}
		_, offset := time.Zone()
		if offset != test.expectedoffset {
			t.Errorf("ZoneOffset = %d, want %d", offset, test.expectedoffset)
		}
	}
}

func TestFormatSecondsInTimeZone(t *testing.T) {
	for _, test := range secondsTimeZoneOffsetTests {
		d := Date(1871, 1, 1, 5, 33, 2, 0, FixedZone("LMT", test.expectedoffset))
		timestr := d.Format(test.format)
		if timestr != test.value {
			t.Errorf("Format = %s, want %s", timestr, test.value)
		}
	}
}

// Issue 11334.
func TestUnderscoreTwoThousand(t *testing.T) {
	format := "15:04_20060102"
	input := "14:38_20150618"
	time, err := Parse(format, input)
	if err != nil {
		t.Error(err)
	}
	if y, m, d := time.Date(); y != 2015 || m != 6 || d != 18 {
		t.Errorf("Incorrect y/m/d, got %d/%d/%d", y, m, d)
	}
	if h := time.Hour(); h != 14 {
		t.Errorf("Incorrect hour, got %d", h)
	}
	if m := time.Minute(); m != 38 {
		t.Errorf("Incorrect minute, got %d", m)
	}
}

// Issue 29918, 29916
func TestStd0xParseError(t *testing.T) {
	tests := []struct {
		format, value, valueElemPrefix string
	}{
		{"01 MST", "0 MST", "0"},
		{"01 MST", "1 MST", "1"},
		{RFC850, "Thursday, 04-Feb-1 21:00:57 PST", "1"},
	}
	for _, tt := range tests {
		_, err := Parse(tt.format, tt.value)
		if err == nil {
			t.Errorf("Parse(%q, %q) did not fail as expected", tt.format, tt.value)
		} else if perr, ok := err.(*ParseError); !ok {
			t.Errorf("Parse(%q, %q) returned error type %T, expected ParseError", tt.format, tt.value, perr)
		} else if !strings.Contains(perr.Error(), "cannot parse") || !strings.HasPrefix(perr.ValueElem, tt.valueElemPrefix) {
			t.Errorf("Parse(%q, %q) returned wrong parsing error message: %v", tt.format, tt.value, perr)
		}
	}
}

var monthOutOfRangeTests = []struct {
	value string
	ok    bool
}{
	{"00-01", false},
	{"13-01", false},
	{"01-01", true},
}

func TestParseMonthOutOfRange(t *testing.T) {
	for _, test := range monthOutOfRangeTests {
		_, err := Parse("01-02", test.value)
		switch {
		case !test.ok && err != nil:
			if !strings.Contains(err.Error(), "month out of range") {
				t.Errorf("%q: expected 'month' error, got %v", test.value, err)
			}
		case test.ok && err != nil:
			t.Errorf("%q: unexpected error: %v", test.value, err)
		case !test.ok && err == nil:
			t.Errorf("%q: expected 'month' error, got none", test.value)
		}
	}
}

// Issue 37387.
func TestParseYday(t *testing.T) {
	t.Parallel()
	for i := 1; i <= 365; i++ {
		d := fmt.Sprintf("2020-%03d", i)
		tm, err := Parse("2006-002", d)
		if err != nil {
			t.Errorf("unexpected error for %s: %v", d, err)
		} else if tm.Year() != 2020 || tm.YearDay() != i {
			t.Errorf("got year %d yearday %d, want %d %d", tm.Year(), tm.YearDay(), 2020, i)
		}
	}
}

// Issue 45391.
func TestQuote(t *testing.T) {
	tests := []struct {
		s, want string
	}{
		{`"`, `"\""`},
		{`abc"xyz"`, `"abc\"xyz\""`},
		{"", `""`},
		{"abc", `"abc"`},
		{`☺`, `"\xe2\x98\xba"`},
		{`☺ hello ☺ hello`, `"\xe2\x98\xba hello \xe2\x98\xba hello"`},
		{"\x04", `"\x04"`},
	}
	for _, tt := range tests {
		if q := Quote(tt.s); q != tt.want {
			t.Errorf("Quote(%q) = got %q, want %q", tt.s, q, tt.want)
		}
	}

}

// Issue 48037
func TestFormatFractionalSecondSeparators(t *testing.T) {
	tests := []struct {
		s, want string
	}{
		{`15:04:05.000`, `21:00:57.012`},
		{`15:04:05.999`, `21:00:57.012`},
		{`15:04:05,000`, `21:00:57,012`},
		{`15:04:05,999`, `21:00:57,012`},
	}

	// The numeric time represents Thu Feb  4 21:00:57.012345600 PST 2009
	time := Unix(0, 1233810057012345600)
	for _, tt := range tests {
		if q := time.Format(tt.s); q != tt.want {
			t.Errorf("Format(%q) = got %q, want %q", tt.s, q, tt.want)
		}
	}
}

// Issue 48685 and 54567.
func TestParseFractionalSecondsLongerThanNineDigits(t *testing.T) {
	tests := []struct {
		s    string
		want int
	}{
		// 9 digits
		{"2021-09-29T16:04:33.000000000Z", 0},
		{"2021-09-29T16:04:33.000000001Z", 1},
		{"2021-09-29T16:04:33.100000000Z", 100_000_000},
		{"2021-09-29T16:04:33.100000001Z", 100_000_001},
		{"2021-09-29T16:04:33.999999999Z", 999_999_999},
		{"2021-09-29T16:04:33.012345678Z", 12_345_678},
		// 10 digits, truncates
		{"2021-09-29T16:04:33.0000000000Z", 0},
		{"2021-09-29T16:04:33.0000000001Z", 0},
		{"2021-09-29T16:04:33.1000000000Z", 100_000_000},
		{"2021-09-29T16:04:33.1000000009Z", 100_000_000},
		{"2021-09-29T16:04:33.9999999999Z", 999_999_999},
		{"2021-09-29T16:04:33.0123456789Z", 12_345_678},
		// 11 digits, truncates
		{"2021-09-29T16:04:33.10000000000Z", 100_000_000},
		{"2021-09-29T16:04:33.00123456789Z", 1_234_567},
		// 12 digits, truncates
		{"2021-09-29T16:04:33.000123456789Z", 123_456},
		// 15 digits, truncates
		{"2021-09-29T16:04:33.9999999999999999Z", 999_999_999},
	}

	for _, tt := range tests {
		for _, format := range []string{RFC3339, RFC3339Nano} {
			tm, err := Parse(format, tt.s)
			if err != nil {
				t.Errorf("Parse(%q, %q) error: %v", format, tt.s, err)
				continue
			}
			if got := tm.Nanosecond(); got != tt.want {
				t.Errorf("Parse(%q, %q) = got %d, want %d", format, tt.s, got, tt.want)
			}
		}
	}
}
