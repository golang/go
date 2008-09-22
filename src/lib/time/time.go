// Copyright 2009 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package time

import (
	"os";
	"time"
)

// Seconds since January 1, 1970 00:00:00 GMT
export func Seconds() (sec int64, err *os.Error) {
	var nsec int64;
	sec, nsec, err = os.Time()
	return sec, err
}

// Nanoseconds since January 1, 1970 00:00:00 GMT
export func Nanoseconds() (nsec int64, err *os.Error) {
	var sec int64;
	sec, nsec, err = os.Time()
	return sec*1e9 + nsec, err
}

export const (
	Sunday = iota;
	Monday;
	Tuesday;
	Wednesday;
	Thursday;
	Friday;
	Saturday;
)

export type Time struct {
	year int64;	// 2008 is 2008
	month, day int;	// Sep-17 is 9, 17
	hour, minute, second int;	// 10:43:12 is 10, 43, 12
	weekday int;		// Sunday = 0, Monday = 1, ...
	zoneoffset int;	// seconds west of UTC
	zone string;
}

var RegularMonths = []int{
	31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31
}
var LeapMonths = []int{
	31, 29, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31
}

func Months(year int64) *[]int {
	if year%4 == 0 && (year%100 != 0 || year%400 == 0) {
		return &LeapMonths
	} else {
		return &RegularMonths
	}
	return nil	// not reached
}

const (
	SecondsPerDay = 24*60*60;

	DaysPer400Years = 365*400+97;
	DaysPer100Years = 365*100+24;
	DaysPer4Years = 365*4+1;

	Days1970To2001 = 31*365+8;
)

export func SecondsToUTC(sec int64) *Time {
	t := new(Time);

	// Split into time and day.
	day := sec/SecondsPerDay;
	sec -= day*SecondsPerDay;
	if sec < 0 {
		day--
		sec += SecondsPerDay
	}

	// Time
	t.hour = int(sec/3600);
	t.minute = int((sec/60)%60);
	t.second = int(sec%60);

	// Day 0 = January 1, 1970 was a Thursday
	t.weekday = int((day + Thursday) % 7)
	if t.weekday < 0 {
		t.weekday += 7
	}

	// Change day from 0 = 1970 to 0 = 2001,
	// to make leap year calculations easier
	// (2001 begins 4-, 100-, and 400-year cycles ending in a leap year.)
	day -= Days1970To2001;

	year := int64(2001)
	if day < 0 {
		// Go back enough 400 year cycles to make day positive.
		n := -day/DaysPer400Years + 1;
		year -= 400*n;
		day += DaysPer400Years*n;
	} else {
		// Cut off 400 year cycles.
		n := day/DaysPer400Years;
		year += 400*n;
		day -= DaysPer400Years*n;
	}

	// Cut off 100-year cycles
	n := day/DaysPer100Years;
	year += 100*n;
	day -= DaysPer100Years*n;

	// Cut off 4-year cycles
	n = day/DaysPer4Years;
	year += 4*n;
	day -= DaysPer4Years*n;

	// Cut off non-leap years.
	n = day/365;
	year += n;
	day -= 365*n;

	t.year = year;

	// If someone ever needs yearday,
	// tyearday = day (+1?)

	months := Months(year);
	var m int;
	yday := int(day);
	for m = 0; m < 12 && yday >= months[m]; m++ {
		yday -= months[m]
	}
	t.month = m+1;
	t.day = yday+1;
	t.zone = "GMT";

	return t;
}

export func UTC() (t *Time, err *os.Error) {
	var sec int64;
	sec, err = Seconds()
	if err != nil {
		return nil, err
	}
	return SecondsToUTC(sec), nil
}

// TODO: Should this return an error?
export func SecondsToLocalTime(sec int64) *Time {
	zone, offset, err := time.LookupTimezone(sec)
	if err != nil {
		return SecondsToUTC(sec)
	}
	t := SecondsToUTC(sec+int64(offset));
	t.zone = zone;
	t.zoneoffset = offset;
	return t
}

export func LocalTime() (t *Time, err *os.Error) {
	var sec int64;
	sec, err = Seconds()
	if err != nil {
		return nil, err
	}
	return SecondsToLocalTime(sec), nil
}

// Compute number of seconds since January 1, 1970.
func (t *Time) Seconds() int64 {
	// First, accumulate days since January 1, 2001.
	// Using 2001 instead of 1970 makes the leap-year
	// handling easier (see SecondsToUTC), because 
	// it is at the beginning of the 4-, 100-, and 400-year cycles.
	day := int64(0);

	// Rewrite year to be >= 2001.
	year := t.year;
	if year < 2001 {
		n := (2001 - year)/400 + 1;
		year += 400*n;
		day -= DaysPer400Years*n;
	}

	// Add in days from 400-year cycles.
	n := (year - 2001) / 400;
	year -= 400*n;
	day += DaysPer400Years*n;

	// Add in 100-year cycles.
	n = (year - 2001) / 100;
	year -= 100*n;
	day += DaysPer100Years*n;

	// Add in 4-year cycles.
	n = (year - 2001) / 4;
	year -= 4*n;
	day += DaysPer4Years*n;

	// Add in non-leap years.
	n = year - 2001;
	day += 365*n;

	// Add in days this year.
	months := Months(t.year);
	for m := 0; m < t.month-1; m++ {
		day += int64(months[m])
	}
	day += int64(t.day - 1);

	// Convert days to seconds since January 1, 2001.
	sec := day * SecondsPerDay;

	// Add in time elapsed today.
	sec += int64(t.hour) * 3600;
	sec += int64(t.minute) * 60;
	sec += int64(t.second);

	// Convert from seconds since 2001 to seconds since 1970.
	sec += Days1970To2001 * SecondsPerDay;

	// Account for local time zone.
	sec -= int64(t.zoneoffset)
	return sec
}

var LongDayNames = []string{
	"Sunday",
	"Monday",
	"Tuesday",
	"Wednesday",
	"Thursday",
	"Friday",
	"Saturday"
}

var ShortDayNames = []string{
	"Sun",
	"Mon",
	"Tue",
	"Wed",
	"Thu",
	"Fri",
	"Sat"
}

var ShortMonthNames = []string{
	"Jan",
	"Feb",
	"Mar",
	"Apr",
	"May",
	"Jun",
	"Jul",
	"Aug",
	"Sep",
	"Oct",
	"Nov",
	"Dec"
}

func Copy(dst *[]byte, s string) {
	for i := 0; i < len(s); i++ {
		dst[i] = s[i]
	}
}

func Decimal(dst *[]byte, n int) {
	if n < 0 {
		n = 0
	}
	for i := len(dst)-1; i >= 0; i-- {
		dst[i] = byte(n%10 + '0');
		n /= 10
	}
}

func AddString(buf *[]byte, bp int, s string) int {
	n := len(s);
	Copy(buf[bp:bp+n], s)
	return bp+n
}

// Just enough of strftime to implement the date formats below.
// Not exported.
func Format(t *Time, fmt string) string {
	buf := new([]byte, 128);
	bp := 0

	for i := 0; i < len(fmt); i++ {
		if fmt[i] == '%' {
			i++
			switch fmt[i] {
			case 'A':	// %A full weekday name
				bp = AddString(buf, bp, LongDayNames[t.weekday])
			case 'a':	// %a abbreviated weekday name
				bp = AddString(buf, bp, ShortDayNames[t.weekday])
			case 'b':	// %b abbreviated month name
				bp = AddString(buf, bp, ShortMonthNames[t.month-1])
			case 'd':	// %d day of month (01-31)
				Decimal(buf[bp:bp+2], t.day);
				bp += 2
			case 'e':	// %e day of month ( 1-31)
				if t.day >= 10 {
					Decimal(buf[bp:bp+2], t.day)
				} else {
					buf[bp] = ' ';
					buf[bp+1] = byte(t.day + '0')
				}
				bp += 2
			case 'H':	// %H hour 00-23
				Decimal(buf[bp:bp+2], t.hour);
				bp += 2
			case 'M':	// %M minute 00-59
				Decimal(buf[bp:bp+2], t.minute);
				bp += 2
			case 'S':	// %S second 00-59
				Decimal(buf[bp:bp+2], t.second);
				bp += 2
			case 'Y':	// %Y year 2008
				Decimal(buf[bp:bp+4], int(t.year));
				bp += 4
			case 'y':	// %y year 08
				Decimal(buf[bp:bp+2], int(t.year%100));
				bp += 2
			case 'Z':
				bp = AddString(buf, bp, t.zone)
			default:
				buf[bp] = '%';
				buf[bp+1] = fmt[i];
				bp += 2
			}
		} else {
			buf[bp] = fmt[i];
			bp++
		}
	}
	return string(buf[0:bp])
}

// ANSI C asctime: Sun Nov  6 08:49:37 1994
func (t *Time) Asctime() string {
	return Format(t, "%a %b %e %H:%M:%S %Y")
}

// RFC 850: Sunday, 06-Nov-94 08:49:37 GMT
func (t *Time) RFC850() string {
	return Format(t, "%A, %d-%b-%y %H:%M:%S %Z")
}

// RFC 1123: Sun, 06 Nov 1994 08:49:37 GMT
func (t *Time) RFC1123() string {
	return Format(t, "%a, %d %b %Y %H:%M:%S %Z")
}

// date(1) - Sun Nov  6 08:49:37 GMT 1994
func (t *Time) String() string {
	return Format(t, "%a %b %e %H:%M:%S %Z %Y")
}

