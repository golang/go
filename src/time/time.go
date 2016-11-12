// Copyright 2009 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// Package time provides functionality for measuring and displaying time.
//
// The calendrical calculations always assume a Gregorian calendar, with
// no leap seconds.
package time

import "errors"

// A Time represents an instant in time with nanosecond precision.
//
// Programs using times should typically store and pass them as values,
// not pointers. That is, time variables and struct fields should be of
// type time.Time, not *time.Time. A Time value can be used by
// multiple goroutines simultaneously.
//
// Time instants can be compared using the Before, After, and Equal methods.
// The Sub method subtracts two instants, producing a Duration.
// The Add method adds a Time and a Duration, producing a Time.
//
// The zero value of type Time is January 1, year 1, 00:00:00.000000000 UTC.
// As this time is unlikely to come up in practice, the IsZero method gives
// a simple way of detecting a time that has not been initialized explicitly.
//
// Each Time has associated with it a Location, consulted when computing the
// presentation form of the time, such as in the Format, Hour, and Year methods.
// The methods Local, UTC, and In return a Time with a specific location.
// Changing the location in this way changes only the presentation; it does not
// change the instant in time being denoted and therefore does not affect the
// computations described in earlier paragraphs.
//
// Note that the Go == operator compares not just the time instant but also the
// Location. Therefore, Time values should not be used as map or database keys
// without first guaranteeing that the identical Location has been set for all
// values, which can be achieved through use of the UTC or Local method.
//
type Time struct {
	// sec gives the number of seconds elapsed since
	// January 1, year 1 00:00:00 UTC.
	sec int64

	// nsec specifies a non-negative nanosecond
	// offset within the second named by Seconds.
	// It must be in the range [0, 999999999].
	nsec int32

	// loc specifies the Location that should be used to
	// determine the minute, hour, month, day, and year
	// that correspond to this Time.
	// The nil location means UTC.
	// All UTC times are represented with loc==nil, never loc==&utcLoc.
	loc *Location
}

func (t *Time) setLoc(loc *Location) {
	if loc == &utcLoc {
		loc = nil
	}
	t.loc = loc
}

// After reports whether the time instant t is after u.
func (t Time) After(u Time) bool {
	return t.sec > u.sec || t.sec == u.sec && t.nsec > u.nsec
}

// Before reports whether the time instant t is before u.
func (t Time) Before(u Time) bool {
	return t.sec < u.sec || t.sec == u.sec && t.nsec < u.nsec
}

// Equal reports whether t and u represent the same time instant.
// Two times can be equal even if they are in different locations.
// For example, 6:00 +0200 CEST and 4:00 UTC are Equal.
// Do not use == with Time values.
func (t Time) Equal(u Time) bool {
	return t.sec == u.sec && t.nsec == u.nsec
}

// A Month specifies a month of the year (January = 1, ...).
type Month int

const (
	January Month = 1 + iota
	February
	March
	April
	May
	June
	July
	August
	September
	October
	November
	December
)

var months = [...]string{
	"January",
	"February",
	"March",
	"April",
	"May",
	"June",
	"July",
	"August",
	"September",
	"October",
	"November",
	"December",
}

// String returns the English name of the month ("January", "February", ...).
func (m Month) String() string {
	if January <= m && m <= December {
		return months[m-1]
	}
	buf := make([]byte, 20)
	n := fmtInt(buf, uint64(m))
	return "%!Month(" + string(buf[n:]) + ")"
}

// A Weekday specifies a day of the week (Sunday = 0, ...).
type Weekday int

const (
	Sunday Weekday = iota
	Monday
	Tuesday
	Wednesday
	Thursday
	Friday
	Saturday
)

var days = [...]string{
	"Sunday",
	"Monday",
	"Tuesday",
	"Wednesday",
	"Thursday",
	"Friday",
	"Saturday",
}

// String returns the English name of the day ("Sunday", "Monday", ...).
func (d Weekday) String() string { return days[d] }

// Computations on time.
//
// The zero value for a Time is defined to be
//	January 1, year 1, 00:00:00.000000000 UTC
// which (1) looks like a zero, or as close as you can get in a date
// (1-1-1 00:00:00 UTC), (2) is unlikely enough to arise in practice to
// be a suitable "not set" sentinel, unlike Jan 1 1970, and (3) has a
// non-negative year even in time zones west of UTC, unlike 1-1-0
// 00:00:00 UTC, which would be 12-31-(-1) 19:00:00 in New York.
//
// The zero Time value does not force a specific epoch for the time
// representation. For example, to use the Unix epoch internally, we
// could define that to distinguish a zero value from Jan 1 1970, that
// time would be represented by sec=-1, nsec=1e9.  However, it does
// suggest a representation, namely using 1-1-1 00:00:00 UTC as the
// epoch, and that's what we do.
//
// The Add and Sub computations are oblivious to the choice of epoch.
//
// The presentation computations - year, month, minute, and so on - all
// rely heavily on division and modulus by positive constants. For
// calendrical calculations we want these divisions to round down, even
// for negative values, so that the remainder is always positive, but
// Go's division (like most hardware division instructions) rounds to
// zero. We can still do those computations and then adjust the result
// for a negative numerator, but it's annoying to write the adjustment
// over and over. Instead, we can change to a different epoch so long
// ago that all the times we care about will be positive, and then round
// to zero and round down coincide. These presentation routines already
// have to add the zone offset, so adding the translation to the
// alternate epoch is cheap. For example, having a non-negative time t
// means that we can write
//
//	sec = t % 60
//
// instead of
//
//	sec = t % 60
//	if sec < 0 {
//		sec += 60
//	}
//
// everywhere.
//
// The calendar runs on an exact 400 year cycle: a 400-year calendar
// printed for 1970-2469 will apply as well to 2370-2769.  Even the days
// of the week match up. It simplifies the computations to choose the
// cycle boundaries so that the exceptional years are always delayed as
// long as possible. That means choosing a year equal to 1 mod 400, so
// that the first leap year is the 4th year, the first missed leap year
// is the 100th year, and the missed missed leap year is the 400th year.
// So we'd prefer instead to print a calendar for 2001-2400 and reuse it
// for 2401-2800.
//
// Finally, it's convenient if the delta between the Unix epoch and
// long-ago epoch is representable by an int64 constant.
//
// These three considerations—choose an epoch as early as possible, that
// uses a year equal to 1 mod 400, and that is no more than 2⁶³ seconds
// earlier than 1970—bring us to the year -292277022399.  We refer to
// this year as the absolute zero year, and to times measured as a uint64
// seconds since this year as absolute times.
//
// Times measured as an int64 seconds since the year 1—the representation
// used for Time's sec field—are called internal times.
//
// Times measured as an int64 seconds since the year 1970 are called Unix
// times.
//
// It is tempting to just use the year 1 as the absolute epoch, defining
// that the routines are only valid for years >= 1.  However, the
// routines would then be invalid when displaying the epoch in time zones
// west of UTC, since it is year 0.  It doesn't seem tenable to say that
// printing the zero time correctly isn't supported in half the time
// zones. By comparison, it's reasonable to mishandle some times in
// the year -292277022399.
//
// All this is opaque to clients of the API and can be changed if a
// better implementation presents itself.

const (
	// The unsigned zero year for internal calculations.
	// Must be 1 mod 400, and times before it will not compute correctly,
	// but otherwise can be changed at will.
	absoluteZeroYear = -292277022399

	// The year of the zero Time.
	// Assumed by the unixToInternal computation below.
	internalYear = 1

	// Offsets to convert between internal and absolute or Unix times.
	absoluteToInternal int64 = (absoluteZeroYear - internalYear) * 365.2425 * secondsPerDay
	internalToAbsolute       = -absoluteToInternal

	unixToInternal int64 = (1969*365 + 1969/4 - 1969/100 + 1969/400) * secondsPerDay
	internalToUnix int64 = -unixToInternal
)

// IsZero reports whether t represents the zero time instant,
// January 1, year 1, 00:00:00 UTC.
func (t Time) IsZero() bool {
	return t.sec == 0 && t.nsec == 0
}

// abs returns the time t as an absolute time, adjusted by the zone offset.
// It is called when computing a presentation property like Month or Hour.
func (t Time) abs() uint64 {
	l := t.loc
	// Avoid function calls when possible.
	if l == nil || l == &localLoc {
		l = l.get()
	}
	sec := t.sec + internalToUnix
	if l != &utcLoc {
		if l.cacheZone != nil && l.cacheStart <= sec && sec < l.cacheEnd {
			sec += int64(l.cacheZone.offset)
		} else {
			_, offset, _, _, _ := l.lookup(sec)
			sec += int64(offset)
		}
	}
	return uint64(sec + (unixToInternal + internalToAbsolute))
}

// locabs is a combination of the Zone and abs methods,
// extracting both return values from a single zone lookup.
func (t Time) locabs() (name string, offset int, abs uint64) {
	l := t.loc
	if l == nil || l == &localLoc {
		l = l.get()
	}
	// Avoid function call if we hit the local time cache.
	sec := t.sec + internalToUnix
	if l != &utcLoc {
		if l.cacheZone != nil && l.cacheStart <= sec && sec < l.cacheEnd {
			name = l.cacheZone.name
			offset = l.cacheZone.offset
		} else {
			name, offset, _, _, _ = l.lookup(sec)
		}
		sec += int64(offset)
	} else {
		name = "UTC"
	}
	abs = uint64(sec + (unixToInternal + internalToAbsolute))
	return
}

// Date returns the year, month, and day in which t occurs.
func (t Time) Date() (year int, month Month, day int) {
	year, month, day, _ = t.date(true)
	return
}

// Year returns the year in which t occurs.
func (t Time) Year() int {
	year, _, _, _ := t.date(false)
	return year
}

// Month returns the month of the year specified by t.
func (t Time) Month() Month {
	_, month, _, _ := t.date(true)
	return month
}

// Day returns the day of the month specified by t.
func (t Time) Day() int {
	_, _, day, _ := t.date(true)
	return day
}

// Weekday returns the day of the week specified by t.
func (t Time) Weekday() Weekday {
	return absWeekday(t.abs())
}

// absWeekday is like Weekday but operates on an absolute time.
func absWeekday(abs uint64) Weekday {
	// January 1 of the absolute year, like January 1 of 2001, was a Monday.
	sec := (abs + uint64(Monday)*secondsPerDay) % secondsPerWeek
	return Weekday(int(sec) / secondsPerDay)
}

// ISOWeek returns the ISO 8601 year and week number in which t occurs.
// Week ranges from 1 to 53. Jan 01 to Jan 03 of year n might belong to
// week 52 or 53 of year n-1, and Dec 29 to Dec 31 might belong to week 1
// of year n+1.
func (t Time) ISOWeek() (year, week int) {
	year, month, day, yday := t.date(true)
	wday := int(t.Weekday()+6) % 7 // weekday but Monday = 0.
	const (
		Mon int = iota
		Tue
		Wed
		Thu
		Fri
		Sat
		Sun
	)

	// Calculate week as number of Mondays in year up to
	// and including today, plus 1 because the first week is week 0.
	// Putting the + 1 inside the numerator as a + 7 keeps the
	// numerator from being negative, which would cause it to
	// round incorrectly.
	week = (yday - wday + 7) / 7

	// The week number is now correct under the assumption
	// that the first Monday of the year is in week 1.
	// If Jan 1 is a Tuesday, Wednesday, or Thursday, the first Monday
	// is actually in week 2.
	jan1wday := (wday - yday + 7*53) % 7
	if Tue <= jan1wday && jan1wday <= Thu {
		week++
	}

	// If the week number is still 0, we're in early January but in
	// the last week of last year.
	if week == 0 {
		year--
		week = 52
		// A year has 53 weeks when Jan 1 or Dec 31 is a Thursday,
		// meaning Jan 1 of the next year is a Friday
		// or it was a leap year and Jan 1 of the next year is a Saturday.
		if jan1wday == Fri || (jan1wday == Sat && isLeap(year)) {
			week++
		}
	}

	// December 29 to 31 are in week 1 of next year if
	// they are after the last Thursday of the year and
	// December 31 is a Monday, Tuesday, or Wednesday.
	if month == December && day >= 29 && wday < Thu {
		if dec31wday := (wday + 31 - day) % 7; Mon <= dec31wday && dec31wday <= Wed {
			year++
			week = 1
		}
	}

	return
}

// Clock returns the hour, minute, and second within the day specified by t.
func (t Time) Clock() (hour, min, sec int) {
	return absClock(t.abs())
}

// absClock is like clock but operates on an absolute time.
func absClock(abs uint64) (hour, min, sec int) {
	sec = int(abs % secondsPerDay)
	hour = sec / secondsPerHour
	sec -= hour * secondsPerHour
	min = sec / secondsPerMinute
	sec -= min * secondsPerMinute
	return
}

// Hour returns the hour within the day specified by t, in the range [0, 23].
func (t Time) Hour() int {
	return int(t.abs()%secondsPerDay) / secondsPerHour
}

// Minute returns the minute offset within the hour specified by t, in the range [0, 59].
func (t Time) Minute() int {
	return int(t.abs()%secondsPerHour) / secondsPerMinute
}

// Second returns the second offset within the minute specified by t, in the range [0, 59].
func (t Time) Second() int {
	return int(t.abs() % secondsPerMinute)
}

// Nanosecond returns the nanosecond offset within the second specified by t,
// in the range [0, 999999999].
func (t Time) Nanosecond() int {
	return int(t.nsec)
}

// YearDay returns the day of the year specified by t, in the range [1,365] for non-leap years,
// and [1,366] in leap years.
func (t Time) YearDay() int {
	_, _, _, yday := t.date(false)
	return yday + 1
}

// A Duration represents the elapsed time between two instants
// as an int64 nanosecond count. The representation limits the
// largest representable duration to approximately 290 years.
type Duration int64

const (
	minDuration Duration = -1 << 63
	maxDuration Duration = 1<<63 - 1
)

// Common durations. There is no definition for units of Day or larger
// to avoid confusion across daylight savings time zone transitions.
//
// To count the number of units in a Duration, divide:
//	second := time.Second
//	fmt.Print(int64(second/time.Millisecond)) // prints 1000
//
// To convert an integer number of units to a Duration, multiply:
//	seconds := 10
//	fmt.Print(time.Duration(seconds)*time.Second) // prints 10s
//
const (
	Nanosecond  Duration = 1
	Microsecond          = 1000 * Nanosecond
	Millisecond          = 1000 * Microsecond
	Second               = 1000 * Millisecond
	Minute               = 60 * Second
	Hour                 = 60 * Minute
)

// String returns a string representing the duration in the form "72h3m0.5s".
// Leading zero units are omitted. As a special case, durations less than one
// second format use a smaller unit (milli-, micro-, or nanoseconds) to ensure
// that the leading digit is non-zero. The zero duration formats as 0s.
func (d Duration) String() string {
	// Largest time is 2540400h10m10.000000000s
	var buf [32]byte
	w := len(buf)

	u := uint64(d)
	neg := d < 0
	if neg {
		u = -u
	}

	if u < uint64(Second) {
		// Special case: if duration is smaller than a second,
		// use smaller units, like 1.2ms
		var prec int
		w--
		buf[w] = 's'
		w--
		switch {
		case u == 0:
			return "0s"
		case u < uint64(Microsecond):
			// print nanoseconds
			prec = 0
			buf[w] = 'n'
		case u < uint64(Millisecond):
			// print microseconds
			prec = 3
			// U+00B5 'µ' micro sign == 0xC2 0xB5
			w-- // Need room for two bytes.
			copy(buf[w:], "µ")
		default:
			// print milliseconds
			prec = 6
			buf[w] = 'm'
		}
		w, u = fmtFrac(buf[:w], u, prec)
		w = fmtInt(buf[:w], u)
	} else {
		w--
		buf[w] = 's'

		w, u = fmtFrac(buf[:w], u, 9)

		// u is now integer seconds
		w = fmtInt(buf[:w], u%60)
		u /= 60

		// u is now integer minutes
		if u > 0 {
			w--
			buf[w] = 'm'
			w = fmtInt(buf[:w], u%60)
			u /= 60

			// u is now integer hours
			// Stop at hours because days can be different lengths.
			if u > 0 {
				w--
				buf[w] = 'h'
				w = fmtInt(buf[:w], u)
			}
		}
	}

	if neg {
		w--
		buf[w] = '-'
	}

	return string(buf[w:])
}

// fmtFrac formats the fraction of v/10**prec (e.g., ".12345") into the
// tail of buf, omitting trailing zeros.  it omits the decimal
// point too when the fraction is 0.  It returns the index where the
// output bytes begin and the value v/10**prec.
func fmtFrac(buf []byte, v uint64, prec int) (nw int, nv uint64) {
	// Omit trailing zeros up to and including decimal point.
	w := len(buf)
	print := false
	for i := 0; i < prec; i++ {
		digit := v % 10
		print = print || digit != 0
		if print {
			w--
			buf[w] = byte(digit) + '0'
		}
		v /= 10
	}
	if print {
		w--
		buf[w] = '.'
	}
	return w, v
}

// fmtInt formats v into the tail of buf.
// It returns the index where the output begins.
func fmtInt(buf []byte, v uint64) int {
	w := len(buf)
	if v == 0 {
		w--
		buf[w] = '0'
	} else {
		for v > 0 {
			w--
			buf[w] = byte(v%10) + '0'
			v /= 10
		}
	}
	return w
}

// Nanoseconds returns the duration as an integer nanosecond count.
func (d Duration) Nanoseconds() int64 { return int64(d) }

// These methods return float64 because the dominant
// use case is for printing a floating point number like 1.5s, and
// a truncation to integer would make them not useful in those cases.
// Splitting the integer and fraction ourselves guarantees that
// converting the returned float64 to an integer rounds the same
// way that a pure integer conversion would have, even in cases
// where, say, float64(d.Nanoseconds())/1e9 would have rounded
// differently.

// Seconds returns the duration as a floating point number of seconds.
func (d Duration) Seconds() float64 {
	sec := d / Second
	nsec := d % Second
	return float64(sec) + float64(nsec)/1e9
}

// Minutes returns the duration as a floating point number of minutes.
func (d Duration) Minutes() float64 {
	min := d / Minute
	nsec := d % Minute
	return float64(min) + float64(nsec)/(60*1e9)
}

// Hours returns the duration as a floating point number of hours.
func (d Duration) Hours() float64 {
	hour := d / Hour
	nsec := d % Hour
	return float64(hour) + float64(nsec)/(60*60*1e9)
}

// Add returns the time t+d.
func (t Time) Add(d Duration) Time {
	t.sec += int64(d / 1e9)
	nsec := t.nsec + int32(d%1e9)
	if nsec >= 1e9 {
		t.sec++
		nsec -= 1e9
	} else if nsec < 0 {
		t.sec--
		nsec += 1e9
	}
	t.nsec = nsec
	return t
}

// Sub returns the duration t-u. If the result exceeds the maximum (or minimum)
// value that can be stored in a Duration, the maximum (or minimum) duration
// will be returned.
// To compute t-d for a duration d, use t.Add(-d).
func (t Time) Sub(u Time) Duration {
	d := Duration(t.sec-u.sec)*Second + Duration(t.nsec-u.nsec)
	// Check for overflow or underflow.
	switch {
	case u.Add(d).Equal(t):
		return d // d is correct
	case t.Before(u):
		return minDuration // t - u is negative out of range
	default:
		return maxDuration // t - u is positive out of range
	}
}

// Since returns the time elapsed since t.
// It is shorthand for time.Now().Sub(t).
func Since(t Time) Duration {
	return Now().Sub(t)
}

// Until returns the duration until t.
// It is shorthand for t.Sub(time.Now()).
func Until(t Time) Duration {
	return t.Sub(Now())
}

// AddDate returns the time corresponding to adding the
// given number of years, months, and days to t.
// For example, AddDate(-1, 2, 3) applied to January 1, 2011
// returns March 4, 2010.
//
// AddDate normalizes its result in the same way that Date does,
// so, for example, adding one month to October 31 yields
// December 1, the normalized form for November 31.
func (t Time) AddDate(years int, months int, days int) Time {
	year, month, day := t.Date()
	hour, min, sec := t.Clock()
	return Date(year+years, month+Month(months), day+days, hour, min, sec, int(t.nsec), t.Location())
}

const (
	secondsPerMinute = 60
	secondsPerHour   = 60 * 60
	secondsPerDay    = 24 * secondsPerHour
	secondsPerWeek   = 7 * secondsPerDay
	daysPer400Years  = 365*400 + 97
	daysPer100Years  = 365*100 + 24
	daysPer4Years    = 365*4 + 1
)

// date computes the year, day of year, and when full=true,
// the month and day in which t occurs.
func (t Time) date(full bool) (year int, month Month, day int, yday int) {
	return absDate(t.abs(), full)
}

// absDate is like date but operates on an absolute time.
func absDate(abs uint64, full bool) (year int, month Month, day int, yday int) {
	// Split into time and day.
	d := abs / secondsPerDay

	// Account for 400 year cycles.
	n := d / daysPer400Years
	y := 400 * n
	d -= daysPer400Years * n

	// Cut off 100-year cycles.
	// The last cycle has one extra leap year, so on the last day
	// of that year, day / daysPer100Years will be 4 instead of 3.
	// Cut it back down to 3 by subtracting n>>2.
	n = d / daysPer100Years
	n -= n >> 2
	y += 100 * n
	d -= daysPer100Years * n

	// Cut off 4-year cycles.
	// The last cycle has a missing leap year, which does not
	// affect the computation.
	n = d / daysPer4Years
	y += 4 * n
	d -= daysPer4Years * n

	// Cut off years within a 4-year cycle.
	// The last year is a leap year, so on the last day of that year,
	// day / 365 will be 4 instead of 3.  Cut it back down to 3
	// by subtracting n>>2.
	n = d / 365
	n -= n >> 2
	y += n
	d -= 365 * n

	year = int(int64(y) + absoluteZeroYear)
	yday = int(d)

	if !full {
		return
	}

	day = yday
	if isLeap(year) {
		// Leap year
		switch {
		case day > 31+29-1:
			// After leap day; pretend it wasn't there.
			day--
		case day == 31+29-1:
			// Leap day.
			month = February
			day = 29
			return
		}
	}

	// Estimate month on assumption that every month has 31 days.
	// The estimate may be too low by at most one month, so adjust.
	month = Month(day / 31)
	end := int(daysBefore[month+1])
	var begin int
	if day >= end {
		month++
		begin = end
	} else {
		begin = int(daysBefore[month])
	}

	month++ // because January is 1
	day = day - begin + 1
	return
}

// daysBefore[m] counts the number of days in a non-leap year
// before month m begins. There is an entry for m=12, counting
// the number of days before January of next year (365).
var daysBefore = [...]int32{
	0,
	31,
	31 + 28,
	31 + 28 + 31,
	31 + 28 + 31 + 30,
	31 + 28 + 31 + 30 + 31,
	31 + 28 + 31 + 30 + 31 + 30,
	31 + 28 + 31 + 30 + 31 + 30 + 31,
	31 + 28 + 31 + 30 + 31 + 30 + 31 + 31,
	31 + 28 + 31 + 30 + 31 + 30 + 31 + 31 + 30,
	31 + 28 + 31 + 30 + 31 + 30 + 31 + 31 + 30 + 31,
	31 + 28 + 31 + 30 + 31 + 30 + 31 + 31 + 30 + 31 + 30,
	31 + 28 + 31 + 30 + 31 + 30 + 31 + 31 + 30 + 31 + 30 + 31,
}

func daysIn(m Month, year int) int {
	if m == February && isLeap(year) {
		return 29
	}
	return int(daysBefore[m] - daysBefore[m-1])
}

// Provided by package runtime.
func now() (sec int64, nsec int32)

// Now returns the current local time.
func Now() Time {
	sec, nsec := now()
	return Time{sec + unixToInternal, nsec, Local}
}

// UTC returns t with the location set to UTC.
func (t Time) UTC() Time {
	t.setLoc(&utcLoc)
	return t
}

// Local returns t with the location set to local time.
func (t Time) Local() Time {
	t.setLoc(Local)
	return t
}

// In returns t with the location information set to loc.
//
// In panics if loc is nil.
func (t Time) In(loc *Location) Time {
	if loc == nil {
		panic("time: missing Location in call to Time.In")
	}
	t.setLoc(loc)
	return t
}

// Location returns the time zone information associated with t.
func (t Time) Location() *Location {
	l := t.loc
	if l == nil {
		l = UTC
	}
	return l
}

// Zone computes the time zone in effect at time t, returning the abbreviated
// name of the zone (such as "CET") and its offset in seconds east of UTC.
func (t Time) Zone() (name string, offset int) {
	name, offset, _, _, _ = t.loc.lookup(t.sec + internalToUnix)
	return
}

// Unix returns t as a Unix time, the number of seconds elapsed
// since January 1, 1970 UTC.
func (t Time) Unix() int64 {
	return t.sec + internalToUnix
}

// UnixNano returns t as a Unix time, the number of nanoseconds elapsed
// since January 1, 1970 UTC. The result is undefined if the Unix time
// in nanoseconds cannot be represented by an int64 (a date before the year
// 1678 or after 2262). Note that this means the result of calling UnixNano
// on the zero Time is undefined.
func (t Time) UnixNano() int64 {
	return (t.sec+internalToUnix)*1e9 + int64(t.nsec)
}

const timeBinaryVersion byte = 1

// MarshalBinary implements the encoding.BinaryMarshaler interface.
func (t Time) MarshalBinary() ([]byte, error) {
	var offsetMin int16 // minutes east of UTC. -1 is UTC.

	if t.Location() == UTC {
		offsetMin = -1
	} else {
		_, offset := t.Zone()
		if offset%60 != 0 {
			return nil, errors.New("Time.MarshalBinary: zone offset has fractional minute")
		}
		offset /= 60
		if offset < -32768 || offset == -1 || offset > 32767 {
			return nil, errors.New("Time.MarshalBinary: unexpected zone offset")
		}
		offsetMin = int16(offset)
	}

	enc := []byte{
		timeBinaryVersion, // byte 0 : version
		byte(t.sec >> 56), // bytes 1-8: seconds
		byte(t.sec >> 48),
		byte(t.sec >> 40),
		byte(t.sec >> 32),
		byte(t.sec >> 24),
		byte(t.sec >> 16),
		byte(t.sec >> 8),
		byte(t.sec),
		byte(t.nsec >> 24), // bytes 9-12: nanoseconds
		byte(t.nsec >> 16),
		byte(t.nsec >> 8),
		byte(t.nsec),
		byte(offsetMin >> 8), // bytes 13-14: zone offset in minutes
		byte(offsetMin),
	}

	return enc, nil
}

// UnmarshalBinary implements the encoding.BinaryUnmarshaler interface.
func (t *Time) UnmarshalBinary(data []byte) error {
	buf := data
	if len(buf) == 0 {
		return errors.New("Time.UnmarshalBinary: no data")
	}

	if buf[0] != timeBinaryVersion {
		return errors.New("Time.UnmarshalBinary: unsupported version")
	}

	if len(buf) != /*version*/ 1+ /*sec*/ 8+ /*nsec*/ 4+ /*zone offset*/ 2 {
		return errors.New("Time.UnmarshalBinary: invalid length")
	}

	buf = buf[1:]
	t.sec = int64(buf[7]) | int64(buf[6])<<8 | int64(buf[5])<<16 | int64(buf[4])<<24 |
		int64(buf[3])<<32 | int64(buf[2])<<40 | int64(buf[1])<<48 | int64(buf[0])<<56

	buf = buf[8:]
	t.nsec = int32(buf[3]) | int32(buf[2])<<8 | int32(buf[1])<<16 | int32(buf[0])<<24

	buf = buf[4:]
	offset := int(int16(buf[1])|int16(buf[0])<<8) * 60

	if offset == -1*60 {
		t.setLoc(&utcLoc)
	} else if _, localoff, _, _, _ := Local.lookup(t.sec + internalToUnix); offset == localoff {
		t.setLoc(Local)
	} else {
		t.setLoc(FixedZone("", offset))
	}

	return nil
}

// TODO(rsc): Remove GobEncoder, GobDecoder, MarshalJSON, UnmarshalJSON in Go 2.
// The same semantics will be provided by the generic MarshalBinary, MarshalText,
// UnmarshalBinary, UnmarshalText.

// GobEncode implements the gob.GobEncoder interface.
func (t Time) GobEncode() ([]byte, error) {
	return t.MarshalBinary()
}

// GobDecode implements the gob.GobDecoder interface.
func (t *Time) GobDecode(data []byte) error {
	return t.UnmarshalBinary(data)
}

// MarshalJSON implements the json.Marshaler interface.
// The time is a quoted string in RFC 3339 format, with sub-second precision added if present.
func (t Time) MarshalJSON() ([]byte, error) {
	if y := t.Year(); y < 0 || y >= 10000 {
		// RFC 3339 is clear that years are 4 digits exactly.
		// See golang.org/issue/4556#c15 for more discussion.
		return nil, errors.New("Time.MarshalJSON: year outside of range [0,9999]")
	}

	b := make([]byte, 0, len(RFC3339Nano)+2)
	b = append(b, '"')
	b = t.AppendFormat(b, RFC3339Nano)
	b = append(b, '"')
	return b, nil
}

// UnmarshalJSON implements the json.Unmarshaler interface.
// The time is expected to be a quoted string in RFC 3339 format.
func (t *Time) UnmarshalJSON(data []byte) error {
	// Ignore null, like in the main JSON package.
	if string(data) == "null" {
		return nil
	}
	// Fractional seconds are handled implicitly by Parse.
	var err error
	*t, err = Parse(`"`+RFC3339+`"`, string(data))
	return err
}

// MarshalText implements the encoding.TextMarshaler interface.
// The time is formatted in RFC 3339 format, with sub-second precision added if present.
func (t Time) MarshalText() ([]byte, error) {
	if y := t.Year(); y < 0 || y >= 10000 {
		return nil, errors.New("Time.MarshalText: year outside of range [0,9999]")
	}

	b := make([]byte, 0, len(RFC3339Nano))
	return t.AppendFormat(b, RFC3339Nano), nil
}

// UnmarshalText implements the encoding.TextUnmarshaler interface.
// The time is expected to be in RFC 3339 format.
func (t *Time) UnmarshalText(data []byte) error {
	// Fractional seconds are handled implicitly by Parse.
	var err error
	*t, err = Parse(RFC3339, string(data))
	return err
}

// Unix returns the local Time corresponding to the given Unix time,
// sec seconds and nsec nanoseconds since January 1, 1970 UTC.
// It is valid to pass nsec outside the range [0, 999999999].
// Not all sec values have a corresponding time value. One such
// value is 1<<63-1 (the largest int64 value).
func Unix(sec int64, nsec int64) Time {
	if nsec < 0 || nsec >= 1e9 {
		n := nsec / 1e9
		sec += n
		nsec -= n * 1e9
		if nsec < 0 {
			nsec += 1e9
			sec--
		}
	}
	return Time{sec + unixToInternal, int32(nsec), Local}
}

func isLeap(year int) bool {
	return year%4 == 0 && (year%100 != 0 || year%400 == 0)
}

// norm returns nhi, nlo such that
//	hi * base + lo == nhi * base + nlo
//	0 <= nlo < base
func norm(hi, lo, base int) (nhi, nlo int) {
	if lo < 0 {
		n := (-lo-1)/base + 1
		hi -= n
		lo += n * base
	}
	if lo >= base {
		n := lo / base
		hi += n
		lo -= n * base
	}
	return hi, lo
}

// Date returns the Time corresponding to
//	yyyy-mm-dd hh:mm:ss + nsec nanoseconds
// in the appropriate zone for that time in the given location.
//
// The month, day, hour, min, sec, and nsec values may be outside
// their usual ranges and will be normalized during the conversion.
// For example, October 32 converts to November 1.
//
// A daylight savings time transition skips or repeats times.
// For example, in the United States, March 13, 2011 2:15am never occurred,
// while November 6, 2011 1:15am occurred twice. In such cases, the
// choice of time zone, and therefore the time, is not well-defined.
// Date returns a time that is correct in one of the two zones involved
// in the transition, but it does not guarantee which.
//
// Date panics if loc is nil.
func Date(year int, month Month, day, hour, min, sec, nsec int, loc *Location) Time {
	if loc == nil {
		panic("time: missing Location in call to Date")
	}

	// Normalize month, overflowing into year.
	m := int(month) - 1
	year, m = norm(year, m, 12)
	month = Month(m) + 1

	// Normalize nsec, sec, min, hour, overflowing into day.
	sec, nsec = norm(sec, nsec, 1e9)
	min, sec = norm(min, sec, 60)
	hour, min = norm(hour, min, 60)
	day, hour = norm(day, hour, 24)

	y := uint64(int64(year) - absoluteZeroYear)

	// Compute days since the absolute epoch.

	// Add in days from 400-year cycles.
	n := y / 400
	y -= 400 * n
	d := daysPer400Years * n

	// Add in 100-year cycles.
	n = y / 100
	y -= 100 * n
	d += daysPer100Years * n

	// Add in 4-year cycles.
	n = y / 4
	y -= 4 * n
	d += daysPer4Years * n

	// Add in non-leap years.
	n = y
	d += 365 * n

	// Add in days before this month.
	d += uint64(daysBefore[month-1])
	if isLeap(year) && month >= March {
		d++ // February 29
	}

	// Add in days before today.
	d += uint64(day - 1)

	// Add in time elapsed today.
	abs := d * secondsPerDay
	abs += uint64(hour*secondsPerHour + min*secondsPerMinute + sec)

	unix := int64(abs) + (absoluteToInternal + internalToUnix)

	// Look for zone offset for t, so we can adjust to UTC.
	// The lookup function expects UTC, so we pass t in the
	// hope that it will not be too close to a zone transition,
	// and then adjust if it is.
	_, offset, _, start, end := loc.lookup(unix)
	if offset != 0 {
		switch utc := unix - int64(offset); {
		case utc < start:
			_, offset, _, _, _ = loc.lookup(start - 1)
		case utc >= end:
			_, offset, _, _, _ = loc.lookup(end)
		}
		unix -= int64(offset)
	}

	t := Time{unix + unixToInternal, int32(nsec), nil}
	t.setLoc(loc)
	return t
}

// Truncate returns the result of rounding t down to a multiple of d (since the zero time).
// If d <= 0, Truncate returns t unchanged.
//
// Truncate operates on the time as an absolute duration since the
// zero time; it does not operate on the presentation form of the
// time. Thus, Truncate(Hour) may return a time with a non-zero
// minute, depending on the time's Location.
func (t Time) Truncate(d Duration) Time {
	if d <= 0 {
		return t
	}
	_, r := div(t, d)
	return t.Add(-r)
}

// Round returns the result of rounding t to the nearest multiple of d (since the zero time).
// The rounding behavior for halfway values is to round up.
// If d <= 0, Round returns t unchanged.
//
// Round operates on the time as an absolute duration since the
// zero time; it does not operate on the presentation form of the
// time. Thus, Round(Hour) may return a time with a non-zero
// minute, depending on the time's Location.
func (t Time) Round(d Duration) Time {
	if d <= 0 {
		return t
	}
	_, r := div(t, d)
	if r+r < d {
		return t.Add(-r)
	}
	return t.Add(d - r)
}

// div divides t by d and returns the quotient parity and remainder.
// We don't use the quotient parity anymore (round half up instead of round to even)
// but it's still here in case we change our minds.
func div(t Time, d Duration) (qmod2 int, r Duration) {
	neg := false
	nsec := t.nsec
	if t.sec < 0 {
		// Operate on absolute value.
		neg = true
		t.sec = -t.sec
		nsec = -nsec
		if nsec < 0 {
			nsec += 1e9
			t.sec-- // t.sec >= 1 before the -- so safe
		}
	}

	switch {
	// Special case: 2d divides 1 second.
	case d < Second && Second%(d+d) == 0:
		qmod2 = int(nsec/int32(d)) & 1
		r = Duration(nsec % int32(d))

	// Special case: d is a multiple of 1 second.
	case d%Second == 0:
		d1 := int64(d / Second)
		qmod2 = int(t.sec/d1) & 1
		r = Duration(t.sec%d1)*Second + Duration(nsec)

	// General case.
	// This could be faster if more cleverness were applied,
	// but it's really only here to avoid special case restrictions in the API.
	// No one will care about these cases.
	default:
		// Compute nanoseconds as 128-bit number.
		sec := uint64(t.sec)
		tmp := (sec >> 32) * 1e9
		u1 := tmp >> 32
		u0 := tmp << 32
		tmp = (sec & 0xFFFFFFFF) * 1e9
		u0x, u0 := u0, u0+tmp
		if u0 < u0x {
			u1++
		}
		u0x, u0 = u0, u0+uint64(nsec)
		if u0 < u0x {
			u1++
		}

		// Compute remainder by subtracting r<<k for decreasing k.
		// Quotient parity is whether we subtract on last round.
		d1 := uint64(d)
		for d1>>63 != 1 {
			d1 <<= 1
		}
		d0 := uint64(0)
		for {
			qmod2 = 0
			if u1 > d1 || u1 == d1 && u0 >= d0 {
				// subtract
				qmod2 = 1
				u0x, u0 = u0, u0-d0
				if u0 > u0x {
					u1--
				}
				u1 -= d1
			}
			if d1 == 0 && d0 == uint64(d) {
				break
			}
			d0 >>= 1
			d0 |= (d1 & 1) << 63
			d1 >>= 1
		}
		r = Duration(u0)
	}

	if neg && r != 0 {
		// If input was negative and not an exact multiple of d, we computed q, r such that
		//	q*d + r = -t
		// But the right answers are given by -(q-1), d-r:
		//	q*d + r = -t
		//	-q*d - r = t
		//	-(q-1)*d + (d - r) = t
		qmod2 ^= 1
		r = d - r
	}
	return
}
