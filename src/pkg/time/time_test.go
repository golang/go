// Copyright 2009 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package time_test

import (
	"bytes"
	"encoding/gob"
	"encoding/json"
	"fmt"
	"math/big"
	"math/rand"
	"runtime"
	"testing"
	"testing/quick"
	. "time"
)

// We should be in PST/PDT, but if the time zone files are missing we
// won't be. The purpose of this test is to at least explain why some of
// the subsequent tests fail.
func TestZoneData(t *testing.T) {
	lt := Now()
	// PST is 8 hours west, PDT is 7 hours west.  We could use the name but it's not unique.
	if name, off := lt.Zone(); off != -8*60*60 && off != -7*60*60 {
		t.Errorf("Unable to find US Pacific time zone data for testing; time zone is %q offset %d", name, off)
		t.Error("Likely problem: the time zone files have not been installed.")
	}
}

// parsedTime is the struct representing a parsed time value.
type parsedTime struct {
	Year                 int
	Month                Month
	Day                  int
	Hour, Minute, Second int // 15:04:05 is 15, 4, 5.
	Nanosecond           int // Fractional second.
	Weekday              Weekday
	ZoneOffset           int    // seconds east of UTC, e.g. -7*60*60 for -0700
	Zone                 string // e.g., "MST"
}

type TimeTest struct {
	seconds int64
	golden  parsedTime
}

var utctests = []TimeTest{
	{0, parsedTime{1970, January, 1, 0, 0, 0, 0, Thursday, 0, "UTC"}},
	{1221681866, parsedTime{2008, September, 17, 20, 4, 26, 0, Wednesday, 0, "UTC"}},
	{-1221681866, parsedTime{1931, April, 16, 3, 55, 34, 0, Thursday, 0, "UTC"}},
	{-11644473600, parsedTime{1601, January, 1, 0, 0, 0, 0, Monday, 0, "UTC"}},
	{599529660, parsedTime{1988, December, 31, 0, 1, 0, 0, Saturday, 0, "UTC"}},
	{978220860, parsedTime{2000, December, 31, 0, 1, 0, 0, Sunday, 0, "UTC"}},
}

var nanoutctests = []TimeTest{
	{0, parsedTime{1970, January, 1, 0, 0, 0, 1e8, Thursday, 0, "UTC"}},
	{1221681866, parsedTime{2008, September, 17, 20, 4, 26, 2e8, Wednesday, 0, "UTC"}},
}

var localtests = []TimeTest{
	{0, parsedTime{1969, December, 31, 16, 0, 0, 0, Wednesday, -8 * 60 * 60, "PST"}},
	{1221681866, parsedTime{2008, September, 17, 13, 4, 26, 0, Wednesday, -7 * 60 * 60, "PDT"}},
}

var nanolocaltests = []TimeTest{
	{0, parsedTime{1969, December, 31, 16, 0, 0, 1e8, Wednesday, -8 * 60 * 60, "PST"}},
	{1221681866, parsedTime{2008, September, 17, 13, 4, 26, 3e8, Wednesday, -7 * 60 * 60, "PDT"}},
}

func same(t Time, u *parsedTime) bool {
	// Check aggregates.
	year, month, day := t.Date()
	hour, min, sec := t.Clock()
	name, offset := t.Zone()
	if year != u.Year || month != u.Month || day != u.Day ||
		hour != u.Hour || min != u.Minute || sec != u.Second ||
		name != u.Zone || offset != u.ZoneOffset {
		return false
	}
	// Check individual entries.
	return t.Year() == u.Year &&
		t.Month() == u.Month &&
		t.Day() == u.Day &&
		t.Hour() == u.Hour &&
		t.Minute() == u.Minute &&
		t.Second() == u.Second &&
		t.Nanosecond() == u.Nanosecond &&
		t.Weekday() == u.Weekday
}

func TestSecondsToUTC(t *testing.T) {
	for _, test := range utctests {
		sec := test.seconds
		golden := &test.golden
		tm := Unix(sec, 0).UTC()
		newsec := tm.Unix()
		if newsec != sec {
			t.Errorf("SecondsToUTC(%d).Seconds() = %d", sec, newsec)
		}
		if !same(tm, golden) {
			t.Errorf("SecondsToUTC(%d):  // %#v", sec, tm)
			t.Errorf("  want=%+v", *golden)
			t.Errorf("  have=%v", tm.Format(RFC3339+" MST"))
		}
	}
}

func TestNanosecondsToUTC(t *testing.T) {
	for _, test := range nanoutctests {
		golden := &test.golden
		nsec := test.seconds*1e9 + int64(golden.Nanosecond)
		tm := Unix(0, nsec).UTC()
		newnsec := tm.Unix()*1e9 + int64(tm.Nanosecond())
		if newnsec != nsec {
			t.Errorf("NanosecondsToUTC(%d).Nanoseconds() = %d", nsec, newnsec)
		}
		if !same(tm, golden) {
			t.Errorf("NanosecondsToUTC(%d):", nsec)
			t.Errorf("  want=%+v", *golden)
			t.Errorf("  have=%+v", tm.Format(RFC3339+" MST"))
		}
	}
}

func TestSecondsToLocalTime(t *testing.T) {
	for _, test := range localtests {
		sec := test.seconds
		golden := &test.golden
		tm := Unix(sec, 0)
		newsec := tm.Unix()
		if newsec != sec {
			t.Errorf("SecondsToLocalTime(%d).Seconds() = %d", sec, newsec)
		}
		if !same(tm, golden) {
			t.Errorf("SecondsToLocalTime(%d):", sec)
			t.Errorf("  want=%+v", *golden)
			t.Errorf("  have=%+v", tm.Format(RFC3339+" MST"))
		}
	}
}

func TestNanosecondsToLocalTime(t *testing.T) {
	for _, test := range nanolocaltests {
		golden := &test.golden
		nsec := test.seconds*1e9 + int64(golden.Nanosecond)
		tm := Unix(0, nsec)
		newnsec := tm.Unix()*1e9 + int64(tm.Nanosecond())
		if newnsec != nsec {
			t.Errorf("NanosecondsToLocalTime(%d).Seconds() = %d", nsec, newnsec)
		}
		if !same(tm, golden) {
			t.Errorf("NanosecondsToLocalTime(%d):", nsec)
			t.Errorf("  want=%+v", *golden)
			t.Errorf("  have=%+v", tm.Format(RFC3339+" MST"))
		}
	}
}

func TestSecondsToUTCAndBack(t *testing.T) {
	f := func(sec int64) bool { return Unix(sec, 0).UTC().Unix() == sec }
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
	f := func(nsec int64) bool {
		t := Unix(0, nsec).UTC()
		ns := t.Unix()*1e9 + int64(t.Nanosecond())
		return ns == nsec
	}
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

// The time routines provide no way to get absolute time
// (seconds since zero), but we need it to compute the right
// answer for bizarre roundings like "to the nearest 3 ns".
// Compute as t - year1 = (t - 1970) + (1970 - 2001) + (2001 - 1).
// t - 1970 is returned by Unix and Nanosecond.
// 1970 - 2001 is -(31*365+8)*86400 = -978307200 seconds.
// 2001 - 1 is 2000*365.2425*86400 = 63113904000 seconds.
const unixToZero = -978307200 + 63113904000

// abs returns the absolute time stored in t, as seconds and nanoseconds.
func abs(t Time) (sec, nsec int64) {
	unix := t.Unix()
	nano := t.Nanosecond()
	return unix + unixToZero, int64(nano)
}

// absString returns abs as a decimal string.
func absString(t Time) string {
	sec, nsec := abs(t)
	if sec < 0 {
		sec = -sec
		nsec = -nsec
		if nsec < 0 {
			nsec += 1e9
			sec--
		}
		return fmt.Sprintf("-%d%09d", sec, nsec)
	}
	return fmt.Sprintf("%d%09d", sec, nsec)
}

var truncateRoundTests = []struct {
	t Time
	d Duration
}{
	{Date(-1, January, 1, 12, 15, 30, 5e8, UTC), 3},
	{Date(-1, January, 1, 12, 15, 31, 5e8, UTC), 3},
	{Date(2012, January, 1, 12, 15, 30, 5e8, UTC), Second},
	{Date(2012, January, 1, 12, 15, 31, 5e8, UTC), Second},
}

func TestTruncateRound(t *testing.T) {
	var (
		bsec  = new(big.Int)
		bnsec = new(big.Int)
		bd    = new(big.Int)
		bt    = new(big.Int)
		br    = new(big.Int)
		bq    = new(big.Int)
		b1e9  = new(big.Int)
	)

	b1e9.SetInt64(1e9)

	testOne := func(ti, tns, di int64) bool {
		t0 := Unix(ti, int64(tns)).UTC()
		d := Duration(di)
		if d < 0 {
			d = -d
		}
		if d <= 0 {
			d = 1
		}

		// Compute bt = absolute nanoseconds.
		sec, nsec := abs(t0)
		bsec.SetInt64(sec)
		bnsec.SetInt64(nsec)
		bt.Mul(bsec, b1e9)
		bt.Add(bt, bnsec)

		// Compute quotient and remainder mod d.
		bd.SetInt64(int64(d))
		bq.DivMod(bt, bd, br)

		// To truncate, subtract remainder.
		// br is < d, so it fits in an int64.
		r := br.Int64()
		t1 := t0.Add(-Duration(r))

		// Check that time.Truncate works.
		if trunc := t0.Truncate(d); trunc != t1 {
			t.Errorf("Time.Truncate(%s, %s) = %s, want %s\n"+
				"%v trunc %v =\n%v want\n%v",
				t0.Format(RFC3339Nano), d, trunc, t1.Format(RFC3339Nano),
				absString(t0), int64(d), absString(trunc), absString(t1))
			return false
		}

		// To round, add d back if remainder r > d/2 or r == exactly d/2.
		// The commented out code would round half to even instead of up,
		// but that makes it time-zone dependent, which is a bit strange.
		if r > int64(d)/2 || r+r == int64(d) /*&& bq.Bit(0) == 1*/ {
			t1 = t1.Add(Duration(d))
		}

		// Check that time.Round works.
		if rnd := t0.Round(d); rnd != t1 {
			t.Errorf("Time.Round(%s, %s) = %s, want %s\n"+
				"%v round %v =\n%v want\n%v",
				t0.Format(RFC3339Nano), d, rnd, t1.Format(RFC3339Nano),
				absString(t0), int64(d), absString(rnd), absString(t1))
			return false
		}
		return true
	}

	// manual test cases
	for _, tt := range truncateRoundTests {
		testOne(tt.t.Unix(), int64(tt.t.Nanosecond()), int64(tt.d))
	}

	// exhaustive near 0
	for i := 0; i < 100; i++ {
		for j := 1; j < 100; j++ {
			testOne(unixToZero, int64(i), int64(j))
			testOne(unixToZero, -int64(i), int64(j))
			if t.Failed() {
				return
			}
		}
	}

	if t.Failed() {
		return
	}

	// randomly generated test cases
	cfg := &quick.Config{MaxCount: 100000}
	if testing.Short() {
		cfg.MaxCount = 1000
	}

	// divisors of Second
	f1 := func(ti int64, tns int32, logdi int32) bool {
		d := Duration(1)
		a, b := uint(logdi%9), (logdi>>16)%9
		d <<= a
		for i := 0; i < int(b); i++ {
			d *= 5
		}
		return testOne(ti, int64(tns), int64(d))
	}
	quick.Check(f1, cfg)

	// multiples of Second
	f2 := func(ti int64, tns int32, di int32) bool {
		d := Duration(di) * Second
		if d < 0 {
			d = -d
		}
		return testOne(ti, int64(tns), int64(d))
	}
	quick.Check(f2, cfg)

	// halfway cases
	f3 := func(tns, di int64) bool {
		di &= 0xfffffffe
		if di == 0 {
			di = 2
		}
		tns -= tns % di
		if tns < 0 {
			tns += di / 2
		} else {
			tns -= di / 2
		}
		return testOne(0, tns, di)
	}
	quick.Check(f3, cfg)

	// full generality
	f4 := func(ti int64, tns int32, di int64) bool {
		return testOne(ti, int64(tns), di)
	}
	quick.Check(f4, cfg)
}

type ISOWeekTest struct {
	year       int // year
	month, day int // month and day
	yex        int // expected year
	wex        int // expected week
}

var isoWeekTests = []ISOWeekTest{
	{1981, 1, 1, 1981, 1}, {1982, 1, 1, 1981, 53}, {1983, 1, 1, 1982, 52},
	{1984, 1, 1, 1983, 52}, {1985, 1, 1, 1985, 1}, {1986, 1, 1, 1986, 1},
	{1987, 1, 1, 1987, 1}, {1988, 1, 1, 1987, 53}, {1989, 1, 1, 1988, 52},
	{1990, 1, 1, 1990, 1}, {1991, 1, 1, 1991, 1}, {1992, 1, 1, 1992, 1},
	{1993, 1, 1, 1992, 53}, {1994, 1, 1, 1993, 52}, {1995, 1, 2, 1995, 1},
	{1996, 1, 1, 1996, 1}, {1996, 1, 7, 1996, 1}, {1996, 1, 8, 1996, 2},
	{1997, 1, 1, 1997, 1}, {1998, 1, 1, 1998, 1}, {1999, 1, 1, 1998, 53},
	{2000, 1, 1, 1999, 52}, {2001, 1, 1, 2001, 1}, {2002, 1, 1, 2002, 1},
	{2003, 1, 1, 2003, 1}, {2004, 1, 1, 2004, 1}, {2005, 1, 1, 2004, 53},
	{2006, 1, 1, 2005, 52}, {2007, 1, 1, 2007, 1}, {2008, 1, 1, 2008, 1},
	{2009, 1, 1, 2009, 1}, {2010, 1, 1, 2009, 53}, {2010, 1, 1, 2009, 53},
	{2011, 1, 1, 2010, 52}, {2011, 1, 2, 2010, 52}, {2011, 1, 3, 2011, 1},
	{2011, 1, 4, 2011, 1}, {2011, 1, 5, 2011, 1}, {2011, 1, 6, 2011, 1},
	{2011, 1, 7, 2011, 1}, {2011, 1, 8, 2011, 1}, {2011, 1, 9, 2011, 1},
	{2011, 1, 10, 2011, 2}, {2011, 1, 11, 2011, 2}, {2011, 6, 12, 2011, 23},
	{2011, 6, 13, 2011, 24}, {2011, 12, 25, 2011, 51}, {2011, 12, 26, 2011, 52},
	{2011, 12, 27, 2011, 52}, {2011, 12, 28, 2011, 52}, {2011, 12, 29, 2011, 52},
	{2011, 12, 30, 2011, 52}, {2011, 12, 31, 2011, 52}, {1995, 1, 1, 1994, 52},
	{2012, 1, 1, 2011, 52}, {2012, 1, 2, 2012, 1}, {2012, 1, 8, 2012, 1},
	{2012, 1, 9, 2012, 2}, {2012, 12, 23, 2012, 51}, {2012, 12, 24, 2012, 52},
	{2012, 12, 30, 2012, 52}, {2012, 12, 31, 2013, 1}, {2013, 1, 1, 2013, 1},
	{2013, 1, 6, 2013, 1}, {2013, 1, 7, 2013, 2}, {2013, 12, 22, 2013, 51},
	{2013, 12, 23, 2013, 52}, {2013, 12, 29, 2013, 52}, {2013, 12, 30, 2014, 1},
	{2014, 1, 1, 2014, 1}, {2014, 1, 5, 2014, 1}, {2014, 1, 6, 2014, 2},
	{2015, 1, 1, 2015, 1}, {2016, 1, 1, 2015, 53}, {2017, 1, 1, 2016, 52},
	{2018, 1, 1, 2018, 1}, {2019, 1, 1, 2019, 1}, {2020, 1, 1, 2020, 1},
	{2021, 1, 1, 2020, 53}, {2022, 1, 1, 2021, 52}, {2023, 1, 1, 2022, 52},
	{2024, 1, 1, 2024, 1}, {2025, 1, 1, 2025, 1}, {2026, 1, 1, 2026, 1},
	{2027, 1, 1, 2026, 53}, {2028, 1, 1, 2027, 52}, {2029, 1, 1, 2029, 1},
	{2030, 1, 1, 2030, 1}, {2031, 1, 1, 2031, 1}, {2032, 1, 1, 2032, 1},
	{2033, 1, 1, 2032, 53}, {2034, 1, 1, 2033, 52}, {2035, 1, 1, 2035, 1},
	{2036, 1, 1, 2036, 1}, {2037, 1, 1, 2037, 1}, {2038, 1, 1, 2037, 53},
	{2039, 1, 1, 2038, 52}, {2040, 1, 1, 2039, 52},
}

func TestISOWeek(t *testing.T) {
	// Selected dates and corner cases
	for _, wt := range isoWeekTests {
		dt := Date(wt.year, Month(wt.month), wt.day, 0, 0, 0, 0, UTC)
		y, w := dt.ISOWeek()
		if w != wt.wex || y != wt.yex {
			t.Errorf("got %d/%d; expected %d/%d for %d-%02d-%02d",
				y, w, wt.yex, wt.wex, wt.year, wt.month, wt.day)
		}
	}

	// The only real invariant: Jan 04 is in week 1
	for year := 1950; year < 2100; year++ {
		if y, w := Date(year, January, 4, 0, 0, 0, 0, UTC).ISOWeek(); y != year || w != 1 {
			t.Errorf("got %d/%d; expected %d/1 for Jan 04", y, w, year)
		}
	}
}

type YearDayTest struct {
	year, month, day int
	yday             int
}

// Test YearDay in several different scenarios
// and corner cases
var yearDayTests = []YearDayTest{
	// Non-leap-year tests
	{2007, 1, 1, 1},
	{2007, 1, 15, 15},
	{2007, 2, 1, 32},
	{2007, 2, 15, 46},
	{2007, 3, 1, 60},
	{2007, 3, 15, 74},
	{2007, 4, 1, 91},
	{2007, 12, 31, 365},

	// Leap-year tests
	{2008, 1, 1, 1},
	{2008, 1, 15, 15},
	{2008, 2, 1, 32},
	{2008, 2, 15, 46},
	{2008, 3, 1, 61},
	{2008, 3, 15, 75},
	{2008, 4, 1, 92},
	{2008, 12, 31, 366},

	// Looks like leap-year (but isn't) tests
	{1900, 1, 1, 1},
	{1900, 1, 15, 15},
	{1900, 2, 1, 32},
	{1900, 2, 15, 46},
	{1900, 3, 1, 60},
	{1900, 3, 15, 74},
	{1900, 4, 1, 91},
	{1900, 12, 31, 365},

	// Year one tests (non-leap)
	{1, 1, 1, 1},
	{1, 1, 15, 15},
	{1, 2, 1, 32},
	{1, 2, 15, 46},
	{1, 3, 1, 60},
	{1, 3, 15, 74},
	{1, 4, 1, 91},
	{1, 12, 31, 365},

	// Year minus one tests (non-leap)
	{-1, 1, 1, 1},
	{-1, 1, 15, 15},
	{-1, 2, 1, 32},
	{-1, 2, 15, 46},
	{-1, 3, 1, 60},
	{-1, 3, 15, 74},
	{-1, 4, 1, 91},
	{-1, 12, 31, 365},

	// 400 BC tests (leap-year)
	{-400, 1, 1, 1},
	{-400, 1, 15, 15},
	{-400, 2, 1, 32},
	{-400, 2, 15, 46},
	{-400, 3, 1, 61},
	{-400, 3, 15, 75},
	{-400, 4, 1, 92},
	{-400, 12, 31, 366},

	// Special Cases

	// Gregorian calendar change (no effect)
	{1582, 10, 4, 277},
	{1582, 10, 15, 288},
}

// Check to see if YearDay is location sensitive
var yearDayLocations = []*Location{
	FixedZone("UTC-8", -8*60*60),
	FixedZone("UTC-4", -4*60*60),
	UTC,
	FixedZone("UTC+4", 4*60*60),
	FixedZone("UTC+8", 8*60*60),
}

func TestYearDay(t *testing.T) {
	for _, loc := range yearDayLocations {
		for _, ydt := range yearDayTests {
			dt := Date(ydt.year, Month(ydt.month), ydt.day, 0, 0, 0, 0, loc)
			yday := dt.YearDay()
			if yday != ydt.yday {
				t.Errorf("got %d, expected %d for %d-%02d-%02d in %v",
					yday, ydt.yday, ydt.year, ydt.month, ydt.day, loc)
			}
		}
	}
}

var durationTests = []struct {
	str string
	d   Duration
}{
	{"0", 0},
	{"1ns", 1 * Nanosecond},
	{"1.1µs", 1100 * Nanosecond},
	{"2.2ms", 2200 * Microsecond},
	{"3.3s", 3300 * Millisecond},
	{"4m5s", 4*Minute + 5*Second},
	{"4m5.001s", 4*Minute + 5001*Millisecond},
	{"5h6m7.001s", 5*Hour + 6*Minute + 7001*Millisecond},
	{"8m0.000000001s", 8*Minute + 1*Nanosecond},
	{"2562047h47m16.854775807s", 1<<63 - 1},
	{"-2562047h47m16.854775808s", -1 << 63},
}

func TestDurationString(t *testing.T) {
	for _, tt := range durationTests {
		if str := tt.d.String(); str != tt.str {
			t.Errorf("Duration(%d).String() = %s, want %s", int64(tt.d), str, tt.str)
		}
		if tt.d > 0 {
			if str := (-tt.d).String(); str != "-"+tt.str {
				t.Errorf("Duration(%d).String() = %s, want %s", int64(-tt.d), str, "-"+tt.str)
			}
		}
	}
}

var dateTests = []struct {
	year, month, day, hour, min, sec, nsec int
	z                                      *Location
	unix                                   int64
}{
	{2011, 11, 6, 1, 0, 0, 0, Local, 1320566400},   // 1:00:00 PDT
	{2011, 11, 6, 1, 59, 59, 0, Local, 1320569999}, // 1:59:59 PDT
	{2011, 11, 6, 2, 0, 0, 0, Local, 1320573600},   // 2:00:00 PST

	{2011, 3, 13, 1, 0, 0, 0, Local, 1300006800},   // 1:00:00 PST
	{2011, 3, 13, 1, 59, 59, 0, Local, 1300010399}, // 1:59:59 PST
	{2011, 3, 13, 3, 0, 0, 0, Local, 1300010400},   // 3:00:00 PDT
	{2011, 3, 13, 2, 30, 0, 0, Local, 1300008600},  // 2:30:00 PDT ≡ 1:30 PST

	// Many names for Fri Nov 18 7:56:35 PST 2011
	{2011, 11, 18, 7, 56, 35, 0, Local, 1321631795},                 // Nov 18 7:56:35
	{2011, 11, 19, -17, 56, 35, 0, Local, 1321631795},               // Nov 19 -17:56:35
	{2011, 11, 17, 31, 56, 35, 0, Local, 1321631795},                // Nov 17 31:56:35
	{2011, 11, 18, 6, 116, 35, 0, Local, 1321631795},                // Nov 18 6:116:35
	{2011, 10, 49, 7, 56, 35, 0, Local, 1321631795},                 // Oct 49 7:56:35
	{2011, 11, 18, 7, 55, 95, 0, Local, 1321631795},                 // Nov 18 7:55:95
	{2011, 11, 18, 7, 56, 34, 1e9, Local, 1321631795},               // Nov 18 7:56:34 + 10⁹ns
	{2011, 12, -12, 7, 56, 35, 0, Local, 1321631795},                // Dec -21 7:56:35
	{2012, 1, -43, 7, 56, 35, 0, Local, 1321631795},                 // Jan -52 7:56:35 2012
	{2012, int(January - 2), 18, 7, 56, 35, 0, Local, 1321631795},   // (Jan-2) 18 7:56:35 2012
	{2010, int(December + 11), 18, 7, 56, 35, 0, Local, 1321631795}, // (Dec+11) 18 7:56:35 2010
}

func TestDate(t *testing.T) {
	for _, tt := range dateTests {
		time := Date(tt.year, Month(tt.month), tt.day, tt.hour, tt.min, tt.sec, tt.nsec, tt.z)
		want := Unix(tt.unix, 0)
		if !time.Equal(want) {
			t.Errorf("Date(%d, %d, %d, %d, %d, %d, %d, %s) = %v, want %v",
				tt.year, tt.month, tt.day, tt.hour, tt.min, tt.sec, tt.nsec, tt.z,
				time, want)
		}
	}
}

// Several ways of getting from
// Fri Nov 18 7:56:35 PST 2011
// to
// Thu Mar 19 7:56:35 PST 2016
var addDateTests = []struct {
	years, months, days int
}{
	{4, 4, 1},
	{3, 16, 1},
	{3, 15, 30},
	{5, -6, -18 - 30 - 12},
}

func TestAddDate(t *testing.T) {
	t0 := Date(2011, 11, 18, 7, 56, 35, 0, UTC)
	t1 := Date(2016, 3, 19, 7, 56, 35, 0, UTC)
	for _, at := range addDateTests {
		time := t0.AddDate(at.years, at.months, at.days)
		if !time.Equal(t1) {
			t.Errorf("AddDate(%d, %d, %d) = %v, want %v",
				at.years, at.months, at.days,
				time, t1)
		}
	}
}

var daysInTests = []struct {
	year, month, di int
}{
	{2011, 1, 31},  // January, first month, 31 days
	{2011, 2, 28},  // February, non-leap year, 28 days
	{2012, 2, 29},  // February, leap year, 29 days
	{2011, 6, 30},  // June, 30 days
	{2011, 12, 31}, // December, last month, 31 days
}

func TestDaysIn(t *testing.T) {
	// The daysIn function is not exported.
	// Test the daysIn function via the `var DaysIn = daysIn`
	// statement in the internal_test.go file.
	for _, tt := range daysInTests {
		di := DaysIn(Month(tt.month), tt.year)
		if di != tt.di {
			t.Errorf("got %d; expected %d for %d-%02d",
				di, tt.di, tt.year, tt.month)
		}
	}
}

func TestAddToExactSecond(t *testing.T) {
	// Add an amount to the current time to round it up to the next exact second.
	// This test checks that the nsec field still lies within the range [0, 999999999].
	t1 := Now()
	t2 := t1.Add(Second - Duration(t1.Nanosecond()))
	sec := (t1.Second() + 1) % 60
	if t2.Second() != sec || t2.Nanosecond() != 0 {
		t.Errorf("sec = %d, nsec = %d, want sec = %d, nsec = 0", t2.Second(), t2.Nanosecond(), sec)
	}
}

func equalTimeAndZone(a, b Time) bool {
	aname, aoffset := a.Zone()
	bname, boffset := b.Zone()
	return a.Equal(b) && aoffset == boffset && aname == bname
}

var gobTests = []Time{
	Date(0, 1, 2, 3, 4, 5, 6, UTC),
	Date(7, 8, 9, 10, 11, 12, 13, FixedZone("", 0)),
	Unix(81985467080890095, 0x76543210), // Time.sec: 0x0123456789ABCDEF
	{}, // nil location
	Date(1, 2, 3, 4, 5, 6, 7, FixedZone("", 32767*60)),
	Date(1, 2, 3, 4, 5, 6, 7, FixedZone("", -32768*60)),
}

func TestTimeGob(t *testing.T) {
	var b bytes.Buffer
	enc := gob.NewEncoder(&b)
	dec := gob.NewDecoder(&b)
	for _, tt := range gobTests {
		var gobtt Time
		if err := enc.Encode(&tt); err != nil {
			t.Errorf("%v gob Encode error = %q, want nil", tt, err)
		} else if err := dec.Decode(&gobtt); err != nil {
			t.Errorf("%v gob Decode error = %q, want nil", tt, err)
		} else if !equalTimeAndZone(gobtt, tt) {
			t.Errorf("Decoded time = %v, want %v", gobtt, tt)
		}
		b.Reset()
	}
}

var invalidEncodingTests = []struct {
	bytes []byte
	want  string
}{
	{[]byte{}, "Time.UnmarshalBinary: no data"},
	{[]byte{0, 2, 3}, "Time.UnmarshalBinary: unsupported version"},
	{[]byte{1, 2, 3}, "Time.UnmarshalBinary: invalid length"},
}

func TestInvalidTimeGob(t *testing.T) {
	for _, tt := range invalidEncodingTests {
		var ignored Time
		err := ignored.GobDecode(tt.bytes)
		if err == nil || err.Error() != tt.want {
			t.Errorf("time.GobDecode(%#v) error = %v, want %v", tt.bytes, err, tt.want)
		}
		err = ignored.UnmarshalBinary(tt.bytes)
		if err == nil || err.Error() != tt.want {
			t.Errorf("time.UnmarshalBinary(%#v) error = %v, want %v", tt.bytes, err, tt.want)
		}
	}
}

var notEncodableTimes = []struct {
	time Time
	want string
}{
	{Date(0, 1, 2, 3, 4, 5, 6, FixedZone("", 1)), "Time.MarshalBinary: zone offset has fractional minute"},
	{Date(0, 1, 2, 3, 4, 5, 6, FixedZone("", -1*60)), "Time.MarshalBinary: unexpected zone offset"},
	{Date(0, 1, 2, 3, 4, 5, 6, FixedZone("", -32769*60)), "Time.MarshalBinary: unexpected zone offset"},
	{Date(0, 1, 2, 3, 4, 5, 6, FixedZone("", 32768*60)), "Time.MarshalBinary: unexpected zone offset"},
}

func TestNotGobEncodableTime(t *testing.T) {
	for _, tt := range notEncodableTimes {
		_, err := tt.time.GobEncode()
		if err == nil || err.Error() != tt.want {
			t.Errorf("%v GobEncode error = %v, want %v", tt.time, err, tt.want)
		}
		_, err = tt.time.MarshalBinary()
		if err == nil || err.Error() != tt.want {
			t.Errorf("%v MarshalBinary error = %v, want %v", tt.time, err, tt.want)
		}
	}
}

var jsonTests = []struct {
	time Time
	json string
}{
	{Date(9999, 4, 12, 23, 20, 50, 520*1e6, UTC), `"9999-04-12T23:20:50.52Z"`},
	{Date(1996, 12, 19, 16, 39, 57, 0, Local), `"1996-12-19T16:39:57-08:00"`},
	{Date(0, 1, 1, 0, 0, 0, 1, FixedZone("", 1*60)), `"0000-01-01T00:00:00.000000001+00:01"`},
}

func TestTimeJSON(t *testing.T) {
	for _, tt := range jsonTests {
		var jsonTime Time

		if jsonBytes, err := json.Marshal(tt.time); err != nil {
			t.Errorf("%v json.Marshal error = %v, want nil", tt.time, err)
		} else if string(jsonBytes) != tt.json {
			t.Errorf("%v JSON = %#q, want %#q", tt.time, string(jsonBytes), tt.json)
		} else if err = json.Unmarshal(jsonBytes, &jsonTime); err != nil {
			t.Errorf("%v json.Unmarshal error = %v, want nil", tt.time, err)
		} else if !equalTimeAndZone(jsonTime, tt.time) {
			t.Errorf("Unmarshaled time = %v, want %v", jsonTime, tt.time)
		}
	}
}

func TestInvalidTimeJSON(t *testing.T) {
	var tt Time
	err := json.Unmarshal([]byte(`{"now is the time":"buddy"}`), &tt)
	_, isParseErr := err.(*ParseError)
	if !isParseErr {
		t.Errorf("expected *time.ParseError unmarshaling JSON, got %v", err)
	}
}

var notJSONEncodableTimes = []struct {
	time Time
	want string
}{
	{Date(10000, 1, 1, 0, 0, 0, 0, UTC), "Time.MarshalJSON: year outside of range [0,9999]"},
	{Date(-1, 1, 1, 0, 0, 0, 0, UTC), "Time.MarshalJSON: year outside of range [0,9999]"},
}

func TestNotJSONEncodableTime(t *testing.T) {
	for _, tt := range notJSONEncodableTimes {
		_, err := tt.time.MarshalJSON()
		if err == nil || err.Error() != tt.want {
			t.Errorf("%v MarshalJSON error = %v, want %v", tt.time, err, tt.want)
		}
	}
}

var parseDurationTests = []struct {
	in   string
	ok   bool
	want Duration
}{
	// simple
	{"0", true, 0},
	{"5s", true, 5 * Second},
	{"30s", true, 30 * Second},
	{"1478s", true, 1478 * Second},
	// sign
	{"-5s", true, -5 * Second},
	{"+5s", true, 5 * Second},
	{"-0", true, 0},
	{"+0", true, 0},
	// decimal
	{"5.0s", true, 5 * Second},
	{"5.6s", true, 5*Second + 600*Millisecond},
	{"5.s", true, 5 * Second},
	{".5s", true, 500 * Millisecond},
	{"1.0s", true, 1 * Second},
	{"1.00s", true, 1 * Second},
	{"1.004s", true, 1*Second + 4*Millisecond},
	{"1.0040s", true, 1*Second + 4*Millisecond},
	{"100.00100s", true, 100*Second + 1*Millisecond},
	// different units
	{"10ns", true, 10 * Nanosecond},
	{"11us", true, 11 * Microsecond},
	{"12µs", true, 12 * Microsecond}, // U+00B5
	{"12μs", true, 12 * Microsecond}, // U+03BC
	{"13ms", true, 13 * Millisecond},
	{"14s", true, 14 * Second},
	{"15m", true, 15 * Minute},
	{"16h", true, 16 * Hour},
	// composite durations
	{"3h30m", true, 3*Hour + 30*Minute},
	{"10.5s4m", true, 4*Minute + 10*Second + 500*Millisecond},
	{"-2m3.4s", true, -(2*Minute + 3*Second + 400*Millisecond)},
	{"1h2m3s4ms5us6ns", true, 1*Hour + 2*Minute + 3*Second + 4*Millisecond + 5*Microsecond + 6*Nanosecond},
	{"39h9m14.425s", true, 39*Hour + 9*Minute + 14*Second + 425*Millisecond},
	// large value
	{"52763797000ns", true, 52763797000 * Nanosecond},
	// more than 9 digits after decimal point, see http://golang.org/issue/6617
	{"0.3333333333333333333h", true, 20 * Minute},

	// errors
	{"", false, 0},
	{"3", false, 0},
	{"-", false, 0},
	{"s", false, 0},
	{".", false, 0},
	{"-.", false, 0},
	{".s", false, 0},
	{"+.s", false, 0},
	{"3000000h", false, 0}, // overflow
}

func TestParseDuration(t *testing.T) {
	for _, tc := range parseDurationTests {
		d, err := ParseDuration(tc.in)
		if tc.ok && (err != nil || d != tc.want) {
			t.Errorf("ParseDuration(%q) = %v, %v, want %v, nil", tc.in, d, err, tc.want)
		} else if !tc.ok && err == nil {
			t.Errorf("ParseDuration(%q) = _, nil, want _, non-nil", tc.in)
		}
	}
}

func TestParseDurationRoundTrip(t *testing.T) {
	for i := 0; i < 100; i++ {
		// Resolutions finer than milliseconds will result in
		// imprecise round-trips.
		d0 := Duration(rand.Int31()) * Millisecond
		s := d0.String()
		d1, err := ParseDuration(s)
		if err != nil || d0 != d1 {
			t.Errorf("round-trip failed: %d => %q => %d, %v", d0, s, d1, err)
		}
	}
}

// golang.org/issue/4622
func TestLocationRace(t *testing.T) {
	ResetLocalOnceForTest() // reset the Once to trigger the race

	c := make(chan string, 1)
	go func() {
		c <- Now().String()
	}()
	Now().String()
	<-c
	Sleep(100 * Millisecond)

	// Back to Los Angeles for subsequent tests:
	ForceUSPacificForTesting()
}

var (
	t Time
	u int64
)

var mallocTest = []struct {
	count int
	desc  string
	fn    func()
}{
	{0, `time.Now()`, func() { t = Now() }},
	{0, `time.Now().UnixNano()`, func() { u = Now().UnixNano() }},
}

func TestCountMallocs(t *testing.T) {
	if testing.Short() {
		t.Skip("skipping malloc count in short mode")
	}
	if runtime.GOMAXPROCS(0) > 1 {
		t.Skip("skipping; GOMAXPROCS>1")
	}
	for _, mt := range mallocTest {
		allocs := int(testing.AllocsPerRun(100, mt.fn))
		if allocs > mt.count {
			t.Errorf("%s: %d allocs, want %d", mt.desc, allocs, mt.count)
		}
	}
}

func TestLoadFixed(t *testing.T) {
	// Issue 4064: handle locations without any zone transitions.
	loc, err := LoadLocation("Etc/GMT+1")
	if err != nil {
		t.Fatal(err)
	}

	// The tzdata name Etc/GMT+1 uses "east is negative",
	// but Go and most other systems use "east is positive".
	// So GMT+1 corresponds to -3600 in the Go zone, not +3600.
	name, offset := Now().In(loc).Zone()
	if name != "GMT+1" || offset != -1*60*60 {
		t.Errorf("Now().In(loc).Zone() = %q, %d, want %q, %d", name, offset, "GMT+1", -1*60*60)
	}
}

const (
	minDuration Duration = -1 << 63
	maxDuration Duration = 1<<63 - 1
)

var subTests = []struct {
	t Time
	u Time
	d Duration
}{
	{Time{}, Time{}, Duration(0)},
	{Date(2009, 11, 23, 0, 0, 0, 1, UTC), Date(2009, 11, 23, 0, 0, 0, 0, UTC), Duration(1)},
	{Date(2009, 11, 23, 0, 0, 0, 0, UTC), Date(2009, 11, 24, 0, 0, 0, 0, UTC), -24 * Hour},
	{Date(2009, 11, 24, 0, 0, 0, 0, UTC), Date(2009, 11, 23, 0, 0, 0, 0, UTC), 24 * Hour},
	{Date(-2009, 11, 24, 0, 0, 0, 0, UTC), Date(-2009, 11, 23, 0, 0, 0, 0, UTC), 24 * Hour},
	{Time{}, Date(2109, 11, 23, 0, 0, 0, 0, UTC), Duration(minDuration)},
	{Date(2109, 11, 23, 0, 0, 0, 0, UTC), Time{}, Duration(maxDuration)},
	{Time{}, Date(-2109, 11, 23, 0, 0, 0, 0, UTC), Duration(maxDuration)},
	{Date(-2109, 11, 23, 0, 0, 0, 0, UTC), Time{}, Duration(minDuration)},
	{Date(2290, 1, 1, 0, 0, 0, 0, UTC), Date(2000, 1, 1, 0, 0, 0, 0, UTC), 290*365*24*Hour + 71*24*Hour},
	{Date(2300, 1, 1, 0, 0, 0, 0, UTC), Date(2000, 1, 1, 0, 0, 0, 0, UTC), Duration(maxDuration)},
	{Date(2000, 1, 1, 0, 0, 0, 0, UTC), Date(2290, 1, 1, 0, 0, 0, 0, UTC), -290*365*24*Hour - 71*24*Hour},
	{Date(2000, 1, 1, 0, 0, 0, 0, UTC), Date(2300, 1, 1, 0, 0, 0, 0, UTC), Duration(minDuration)},
}

func TestSub(t *testing.T) {
	for i, st := range subTests {
		got := st.t.Sub(st.u)
		if got != st.d {
			t.Errorf("#%d: Sub(%v, %v): got %v; want %v", i, st.t, st.u, got, st.d)
		}
	}
}

var nsDurationTests = []struct {
	d    Duration
	want int64
}{
	{Duration(-1000), -1000},
	{Duration(-1), -1},
	{Duration(1), 1},
	{Duration(1000), 1000},
}

func TestDurationNanoseconds(t *testing.T) {
	for _, tt := range nsDurationTests {
		if got := tt.d.Nanoseconds(); got != tt.want {
			t.Errorf("d.Nanoseconds() = %d; want: %d", got, tt.want)
		}
	}
}

var minDurationTests = []struct {
	d    Duration
	want float64
}{
	{Duration(-60000000000), -1},
	{Duration(-1), -1 / 60e9},
	{Duration(1), 1 / 60e9},
	{Duration(60000000000), 1},
}

func TestDurationMinutes(t *testing.T) {
	for _, tt := range minDurationTests {
		if got := tt.d.Minutes(); got != tt.want {
			t.Errorf("d.Minutes() = %g; want: %g", got, tt.want)
		}
	}
}

var hourDurationTests = []struct {
	d    Duration
	want float64
}{
	{Duration(-3600000000000), -1},
	{Duration(-1), -1 / 3600e9},
	{Duration(1), 1 / 3600e9},
	{Duration(3600000000000), 1},
}

func TestDurationHours(t *testing.T) {
	for _, tt := range hourDurationTests {
		if got := tt.d.Hours(); got != tt.want {
			t.Errorf("d.Hours() = %g; want: %g", got, tt.want)
		}
	}
}

func BenchmarkNow(b *testing.B) {
	for i := 0; i < b.N; i++ {
		t = Now()
	}
}

func BenchmarkNowUnixNano(b *testing.B) {
	for i := 0; i < b.N; i++ {
		u = Now().UnixNano()
	}
}

func BenchmarkFormat(b *testing.B) {
	t := Unix(1265346057, 0)
	for i := 0; i < b.N; i++ {
		t.Format("Mon Jan  2 15:04:05 2006")
	}
}

func BenchmarkFormatNow(b *testing.B) {
	// Like BenchmarkFormat, but easier, because the time zone
	// lookup cache is optimized for the present.
	t := Now()
	for i := 0; i < b.N; i++ {
		t.Format("Mon Jan  2 15:04:05 2006")
	}
}

func BenchmarkParse(b *testing.B) {
	for i := 0; i < b.N; i++ {
		Parse(ANSIC, "Mon Jan  2 15:04:05 2006")
	}
}

func BenchmarkHour(b *testing.B) {
	t := Now()
	for i := 0; i < b.N; i++ {
		_ = t.Hour()
	}
}

func BenchmarkSecond(b *testing.B) {
	t := Now()
	for i := 0; i < b.N; i++ {
		_ = t.Second()
	}
}

func BenchmarkYear(b *testing.B) {
	t := Now()
	for i := 0; i < b.N; i++ {
		_ = t.Year()
	}
}

func BenchmarkDay(b *testing.B) {
	t := Now()
	for i := 0; i < b.N; i++ {
		_ = t.Day()
	}
}
