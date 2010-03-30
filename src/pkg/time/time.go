// Copyright 2009 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// The time package provides functionality for measuring and
// displaying time.
package time

import (
	"os"
)

// Seconds reports the number of seconds since the Unix epoch,
// January 1, 1970 00:00:00 UTC.
func Seconds() int64 {
	sec, _, err := os.Time()
	if err != nil {
		panic(err)
	}
	return sec
}

// Nanoseconds reports the number of nanoseconds since the Unix epoch,
// January 1, 1970 00:00:00 UTC.
func Nanoseconds() int64 {
	sec, nsec, err := os.Time()
	if err != nil {
		panic(err)
	}
	return sec*1e9 + nsec
}

// Days of the week.
const (
	Sunday = iota
	Monday
	Tuesday
	Wednesday
	Thursday
	Friday
	Saturday
)

// Time is the struct representing a parsed time value.
type Time struct {
	Year                 int64  // 2006 is 2006
	Month, Day           int    // Jan-2 is 1, 2
	Hour, Minute, Second int    // 15:04:05 is 15, 4, 5.
	Weekday              int    // Sunday, Monday, ...
	ZoneOffset           int    // seconds east of UTC, e.g. -7*60 for -0700
	Zone                 string // e.g., "MST"
}

var nonleapyear = []int{31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31}
var leapyear = []int{31, 29, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31}

func months(year int64) []int {
	if year%4 == 0 && (year%100 != 0 || year%400 == 0) {
		return leapyear
	}
	return nonleapyear
}

const (
	secondsPerDay   = 24 * 60 * 60
	daysPer400Years = 365*400 + 97
	daysPer100Years = 365*100 + 24
	daysPer4Years   = 365*4 + 1
	days1970To2001  = 31*365 + 8
)

// SecondsToUTC converts sec, in number of seconds since the Unix epoch,
// into a parsed Time value in the UTC time zone.
func SecondsToUTC(sec int64) *Time {
	t := new(Time)

	// Split into time and day.
	day := sec / secondsPerDay
	sec -= day * secondsPerDay
	if sec < 0 {
		day--
		sec += secondsPerDay
	}

	// Time
	t.Hour = int(sec / 3600)
	t.Minute = int((sec / 60) % 60)
	t.Second = int(sec % 60)

	// Day 0 = January 1, 1970 was a Thursday
	t.Weekday = int((day + Thursday) % 7)
	if t.Weekday < 0 {
		t.Weekday += 7
	}

	// Change day from 0 = 1970 to 0 = 2001,
	// to make leap year calculations easier
	// (2001 begins 4-, 100-, and 400-year cycles ending in a leap year.)
	day -= days1970To2001

	year := int64(2001)
	if day < 0 {
		// Go back enough 400 year cycles to make day positive.
		n := -day/daysPer400Years + 1
		year -= 400 * n
		day += daysPer400Years * n
	}

	// Cut off 400 year cycles.
	n := day / daysPer400Years
	year += 400 * n
	day -= daysPer400Years * n

	// Cut off 100-year cycles
	n = day / daysPer100Years
	if n > 3 { // happens on last day of 400th year
		n = 3
	}
	year += 100 * n
	day -= daysPer100Years * n

	// Cut off 4-year cycles
	n = day / daysPer4Years
	if n > 24 { // happens on last day of 100th year
		n = 24
	}
	year += 4 * n
	day -= daysPer4Years * n

	// Cut off non-leap years.
	n = day / 365
	if n > 3 { // happens on last day of 4th year
		n = 3
	}
	year += n
	day -= 365 * n

	t.Year = year

	// If someone ever needs yearday,
	// tyearday = day (+1?)

	months := months(year)
	var m int
	yday := int(day)
	for m = 0; m < 12 && yday >= months[m]; m++ {
		yday -= months[m]
	}
	t.Month = m + 1
	t.Day = yday + 1
	t.Zone = "UTC"

	return t
}

// UTC returns the current time as a parsed Time value in the UTC time zone.
func UTC() *Time { return SecondsToUTC(Seconds()) }

// SecondsToLocalTime converts sec, in number of seconds since the Unix epoch,
// into a parsed Time value in the local time zone.
func SecondsToLocalTime(sec int64) *Time {
	z, offset := lookupTimezone(sec)
	t := SecondsToUTC(sec + int64(offset))
	t.Zone = z
	t.ZoneOffset = offset
	return t
}

// LocalTime returns the current time as a parsed Time value in the local time zone.
func LocalTime() *Time { return SecondsToLocalTime(Seconds()) }

// Seconds returns the number of seconds since January 1, 1970 represented by the
// parsed Time value.
func (t *Time) Seconds() int64 {
	// First, accumulate days since January 1, 2001.
	// Using 2001 instead of 1970 makes the leap-year
	// handling easier (see SecondsToUTC), because
	// it is at the beginning of the 4-, 100-, and 400-year cycles.
	day := int64(0)

	// Rewrite year to be >= 2001.
	year := t.Year
	if year < 2001 {
		n := (2001-year)/400 + 1
		year += 400 * n
		day -= daysPer400Years * n
	}

	// Add in days from 400-year cycles.
	n := (year - 2001) / 400
	year -= 400 * n
	day += daysPer400Years * n

	// Add in 100-year cycles.
	n = (year - 2001) / 100
	year -= 100 * n
	day += daysPer100Years * n

	// Add in 4-year cycles.
	n = (year - 2001) / 4
	year -= 4 * n
	day += daysPer4Years * n

	// Add in non-leap years.
	n = year - 2001
	day += 365 * n

	// Add in days this year.
	months := months(t.Year)
	for m := 0; m < t.Month-1; m++ {
		day += int64(months[m])
	}
	day += int64(t.Day - 1)

	// Convert days to seconds since January 1, 2001.
	sec := day * secondsPerDay

	// Add in time elapsed today.
	sec += int64(t.Hour) * 3600
	sec += int64(t.Minute) * 60
	sec += int64(t.Second)

	// Convert from seconds since 2001 to seconds since 1970.
	sec += days1970To2001 * secondsPerDay

	// Account for local time zone.
	sec -= int64(t.ZoneOffset)
	return sec
}
