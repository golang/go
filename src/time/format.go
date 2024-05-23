// Copyright 2010 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package time

import (
	"errors"
	"internal/stringslite"
)

// These are predefined layouts for use in [Time.Format] and [time.Parse].
// The reference time used in these layouts is the specific time stamp:
//
//	01/02 03:04:05PM '06 -0700
//
// (January 2, 15:04:05, 2006, in time zone seven hours west of GMT).
// That value is recorded as the constant named [Layout], listed below. As a Unix
// time, this is 1136239445. Since MST is GMT-0700, the reference would be
// printed by the Unix date command as:
//
//	Mon Jan 2 15:04:05 MST 2006
//
// It is a regrettable historic error that the date uses the American convention
// of putting the numerical month before the day.
//
// The example for Time.Format demonstrates the working of the layout string
// in detail and is a good reference.
//
// Note that the [RFC822], [RFC850], and [RFC1123] formats should be applied
// only to local times. Applying them to UTC times will use "UTC" as the
// time zone abbreviation, while strictly speaking those RFCs require the
// use of "GMT" in that case.
// In general [RFC1123Z] should be used instead of [RFC1123] for servers
// that insist on that format, and [RFC3339] should be preferred for new protocols.
// [RFC3339], [RFC822], [RFC822Z], [RFC1123], and [RFC1123Z] are useful for formatting;
// when used with time.Parse they do not accept all the time formats
// permitted by the RFCs and they do accept time formats not formally defined.
// The [RFC3339Nano] format removes trailing zeros from the seconds field
// and thus may not sort correctly once formatted.
//
// Most programs can use one of the defined constants as the layout passed to
// Format or Parse. The rest of this comment can be ignored unless you are
// creating a custom layout string.
//
// To define your own format, write down what the reference time would look like
// formatted your way; see the values of constants like [ANSIC], [StampMicro] or
// [Kitchen] for examples. The model is to demonstrate what the reference time
// looks like so that the Format and Parse methods can apply the same
// transformation to a general time value.
//
// Here is a summary of the components of a layout string. Each element shows by
// example the formatting of an element of the reference time. Only these values
// are recognized. Text in the layout string that is not recognized as part of
// the reference time is echoed verbatim during Format and expected to appear
// verbatim in the input to Parse.
//
//	Year: "2006" "06"
//	Month: "Jan" "January" "01" "1"
//	Day of the week: "Mon" "Monday"
//	Day of the month: "2" "_2" "02"
//	Day of the year: "__2" "002"
//	Hour: "15" "3" "03" (PM or AM)
//	Minute: "4" "04"
//	Second: "5" "05"
//	AM/PM mark: "PM"
//
// Numeric time zone offsets format as follows:
//
//	"-0700"     ±hhmm
//	"-07:00"    ±hh:mm
//	"-07"       ±hh
//	"-070000"   ±hhmmss
//	"-07:00:00" ±hh:mm:ss
//
// Replacing the sign in the format with a Z triggers
// the ISO 8601 behavior of printing Z instead of an
// offset for the UTC zone. Thus:
//
//	"Z0700"      Z or ±hhmm
//	"Z07:00"     Z or ±hh:mm
//	"Z07"        Z or ±hh
//	"Z070000"    Z or ±hhmmss
//	"Z07:00:00"  Z or ±hh:mm:ss
//
// Within the format string, the underscores in "_2" and "__2" represent spaces
// that may be replaced by digits if the following number has multiple digits,
// for compatibility with fixed-width Unix time formats. A leading zero represents
// a zero-padded value.
//
// The formats __2 and 002 are space-padded and zero-padded
// three-character day of year; there is no unpadded day of year format.
//
// A comma or decimal point followed by one or more zeros represents
// a fractional second, printed to the given number of decimal places.
// A comma or decimal point followed by one or more nines represents
// a fractional second, printed to the given number of decimal places, with
// trailing zeros removed.
// For example "15:04:05,000" or "15:04:05.000" formats or parses with
// millisecond precision.
//
// Some valid layouts are invalid time values for time.Parse, due to formats
// such as _ for space padding and Z for zone information.
const (
	Layout      = "01/02 03:04:05PM '06 -0700" // The reference time, in numerical order.
	ANSIC       = "Mon Jan _2 15:04:05 2006"
	UnixDate    = "Mon Jan _2 15:04:05 MST 2006"
	RubyDate    = "Mon Jan 02 15:04:05 -0700 2006"
	RFC822      = "02 Jan 06 15:04 MST"
	RFC822Z     = "02 Jan 06 15:04 -0700" // RFC822 with numeric zone
	RFC850      = "Monday, 02-Jan-06 15:04:05 MST"
	RFC1123     = "Mon, 02 Jan 2006 15:04:05 MST"
	RFC1123Z    = "Mon, 02 Jan 2006 15:04:05 -0700" // RFC1123 with numeric zone
	RFC3339     = "2006-01-02T15:04:05Z07:00"
	RFC3339Nano = "2006-01-02T15:04:05.999999999Z07:00"
	Kitchen     = "3:04PM"
	// Handy time stamps.
	Stamp      = "Jan _2 15:04:05"
	StampMilli = "Jan _2 15:04:05.000"
	StampMicro = "Jan _2 15:04:05.000000"
	StampNano  = "Jan _2 15:04:05.000000000"
	DateTime   = "2006-01-02 15:04:05"
	DateOnly   = "2006-01-02"
	TimeOnly   = "15:04:05"
)

const (
	_                        = iota
	stdLongMonth             = iota + stdNeedDate  // "January"
	stdMonth                                       // "Jan"
	stdNumMonth                                    // "1"
	stdZeroMonth                                   // "01"
	stdLongWeekDay                                 // "Monday"
	stdWeekDay                                     // "Mon"
	stdDay                                         // "2"
	stdUnderDay                                    // "_2"
	stdZeroDay                                     // "02"
	stdUnderYearDay                                // "__2"
	stdZeroYearDay                                 // "002"
	stdHour                  = iota + stdNeedClock // "15"
	stdHour12                                      // "3"
	stdZeroHour12                                  // "03"
	stdMinute                                      // "4"
	stdZeroMinute                                  // "04"
	stdSecond                                      // "5"
	stdZeroSecond                                  // "05"
	stdLongYear              = iota + stdNeedDate  // "2006"
	stdYear                                        // "06"
	stdPM                    = iota + stdNeedClock // "PM"
	stdpm                                          // "pm"
	stdTZ                    = iota                // "MST"
	stdISO8601TZ                                   // "Z0700"  // prints Z for UTC
	stdISO8601SecondsTZ                            // "Z070000"
	stdISO8601ShortTZ                              // "Z07"
	stdISO8601ColonTZ                              // "Z07:00" // prints Z for UTC
	stdISO8601ColonSecondsTZ                       // "Z07:00:00"
	stdNumTZ                                       // "-0700"  // always numeric
	stdNumSecondsTz                                // "-070000"
	stdNumShortTZ                                  // "-07"    // always numeric
	stdNumColonTZ                                  // "-07:00" // always numeric
	stdNumColonSecondsTZ                           // "-07:00:00"
	stdFracSecond0                                 // ".0", ".00", ... , trailing zeros included
	stdFracSecond9                                 // ".9", ".99", ..., trailing zeros omitted

	stdNeedDate       = 1 << 8             // need month, day, year
	stdNeedClock      = 2 << 8             // need hour, minute, second
	stdArgShift       = 16                 // extra argument in high bits, above low stdArgShift
	stdSeparatorShift = 28                 // extra argument in high 4 bits for fractional second separators
	stdMask           = 1<<stdArgShift - 1 // mask out argument
)

// std0x records the std values for "01", "02", ..., "06".
var std0x = [...]int{stdZeroMonth, stdZeroDay, stdZeroHour12, stdZeroMinute, stdZeroSecond, stdYear}

// startsWithLowerCase reports whether the string has a lower-case letter at the beginning.
// Its purpose is to prevent matching strings like "Month" when looking for "Mon".
func startsWithLowerCase(str string) bool {
	if len(str) == 0 {
		return false
	}
	c := str[0]
	return 'a' <= c && c <= 'z'
}

// nextStdChunk finds the first occurrence of a std string in
// layout and returns the text before, the std string, and the text after.
func nextStdChunk(layout string) (prefix string, std int, suffix string) {
	for i := 0; i < len(layout); i++ {
		switch c := int(layout[i]); c {
		case 'J': // January, Jan
			if len(layout) >= i+3 && layout[i:i+3] == "Jan" {
				if len(layout) >= i+7 && layout[i:i+7] == "January" {
					return layout[0:i], stdLongMonth, layout[i+7:]
				}
				if !startsWithLowerCase(layout[i+3:]) {
					return layout[0:i], stdMonth, layout[i+3:]
				}
			}

		case 'M': // Monday, Mon, MST
			if len(layout) >= i+3 {
				if layout[i:i+3] == "Mon" {
					if len(layout) >= i+6 && layout[i:i+6] == "Monday" {
						return layout[0:i], stdLongWeekDay, layout[i+6:]
					}
					if !startsWithLowerCase(layout[i+3:]) {
						return layout[0:i], stdWeekDay, layout[i+3:]
					}
				}
				if layout[i:i+3] == "MST" {
					return layout[0:i], stdTZ, layout[i+3:]
				}
			}

		case '0': // 01, 02, 03, 04, 05, 06, 002
			if len(layout) >= i+2 && '1' <= layout[i+1] && layout[i+1] <= '6' {
				return layout[0:i], std0x[layout[i+1]-'1'], layout[i+2:]
			}
			if len(layout) >= i+3 && layout[i+1] == '0' && layout[i+2] == '2' {
				return layout[0:i], stdZeroYearDay, layout[i+3:]
			}

		case '1': // 15, 1
			if len(layout) >= i+2 && layout[i+1] == '5' {
				return layout[0:i], stdHour, layout[i+2:]
			}
			return layout[0:i], stdNumMonth, layout[i+1:]

		case '2': // 2006, 2
			if len(layout) >= i+4 && layout[i:i+4] == "2006" {
				return layout[0:i], stdLongYear, layout[i+4:]
			}
			return layout[0:i], stdDay, layout[i+1:]

		case '_': // _2, _2006, __2
			if len(layout) >= i+2 && layout[i+1] == '2' {
				//_2006 is really a literal _, followed by stdLongYear
				if len(layout) >= i+5 && layout[i+1:i+5] == "2006" {
					return layout[0 : i+1], stdLongYear, layout[i+5:]
				}
				return layout[0:i], stdUnderDay, layout[i+2:]
			}
			if len(layout) >= i+3 && layout[i+1] == '_' && layout[i+2] == '2' {
				return layout[0:i], stdUnderYearDay, layout[i+3:]
			}

		case '3':
			return layout[0:i], stdHour12, layout[i+1:]

		case '4':
			return layout[0:i], stdMinute, layout[i+1:]

		case '5':
			return layout[0:i], stdSecond, layout[i+1:]

		case 'P': // PM
			if len(layout) >= i+2 && layout[i+1] == 'M' {
				return layout[0:i], stdPM, layout[i+2:]
			}

		case 'p': // pm
			if len(layout) >= i+2 && layout[i+1] == 'm' {
				return layout[0:i], stdpm, layout[i+2:]
			}

		case '-': // -070000, -07:00:00, -0700, -07:00, -07
			if len(layout) >= i+7 && layout[i:i+7] == "-070000" {
				return layout[0:i], stdNumSecondsTz, layout[i+7:]
			}
			if len(layout) >= i+9 && layout[i:i+9] == "-07:00:00" {
				return layout[0:i], stdNumColonSecondsTZ, layout[i+9:]
			}
			if len(layout) >= i+5 && layout[i:i+5] == "-0700" {
				return layout[0:i], stdNumTZ, layout[i+5:]
			}
			if len(layout) >= i+6 && layout[i:i+6] == "-07:00" {
				return layout[0:i], stdNumColonTZ, layout[i+6:]
			}
			if len(layout) >= i+3 && layout[i:i+3] == "-07" {
				return layout[0:i], stdNumShortTZ, layout[i+3:]
			}

		case 'Z': // Z070000, Z07:00:00, Z0700, Z07:00,
			if len(layout) >= i+7 && layout[i:i+7] == "Z070000" {
				return layout[0:i], stdISO8601SecondsTZ, layout[i+7:]
			}
			if len(layout) >= i+9 && layout[i:i+9] == "Z07:00:00" {
				return layout[0:i], stdISO8601ColonSecondsTZ, layout[i+9:]
			}
			if len(layout) >= i+5 && layout[i:i+5] == "Z0700" {
				return layout[0:i], stdISO8601TZ, layout[i+5:]
			}
			if len(layout) >= i+6 && layout[i:i+6] == "Z07:00" {
				return layout[0:i], stdISO8601ColonTZ, layout[i+6:]
			}
			if len(layout) >= i+3 && layout[i:i+3] == "Z07" {
				return layout[0:i], stdISO8601ShortTZ, layout[i+3:]
			}

		case '.', ',': // ,000, or .000, or ,999, or .999 - repeated digits for fractional seconds.
			if i+1 < len(layout) && (layout[i+1] == '0' || layout[i+1] == '9') {
				ch := layout[i+1]
				j := i + 1
				for j < len(layout) && layout[j] == ch {
					j++
				}
				// String of digits must end here - only fractional second is all digits.
				if !isDigit(layout, j) {
					code := stdFracSecond0
					if layout[i+1] == '9' {
						code = stdFracSecond9
					}
					std := stdFracSecond(code, j-(i+1), c)
					return layout[0:i], std, layout[j:]
				}
			}
		}
	}
	return layout, 0, ""
}

var longDayNames = []string{
	"Sunday",
	"Monday",
	"Tuesday",
	"Wednesday",
	"Thursday",
	"Friday",
	"Saturday",
}

var shortDayNames = []string{
	"Sun",
	"Mon",
	"Tue",
	"Wed",
	"Thu",
	"Fri",
	"Sat",
}

var shortMonthNames = []string{
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
	"Dec",
}

var longMonthNames = []string{
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

// match reports whether s1 and s2 match ignoring case.
// It is assumed s1 and s2 are the same length.
func match(s1, s2 string) bool {
	for i := 0; i < len(s1); i++ {
		c1 := s1[i]
		c2 := s2[i]
		if c1 != c2 {
			// Switch to lower-case; 'a'-'A' is known to be a single bit.
			c1 |= 'a' - 'A'
			c2 |= 'a' - 'A'
			if c1 != c2 || c1 < 'a' || c1 > 'z' {
				return false
			}
		}
	}
	return true
}

func lookup(tab []string, val string) (int, string, error) {
	for i, v := range tab {
		if len(val) >= len(v) && match(val[0:len(v)], v) {
			return i, val[len(v):], nil
		}
	}
	return -1, val, errBad
}

// appendInt appends the decimal form of x to b and returns the result.
// If the decimal form (excluding sign) is shorter than width, the result is padded with leading 0's.
// Duplicates functionality in strconv, but avoids dependency.
func appendInt(b []byte, x int, width int) []byte {
	u := uint(x)
	if x < 0 {
		b = append(b, '-')
		u = uint(-x)
	}

	// 2-digit and 4-digit fields are the most common in time formats.
	utod := func(u uint) byte { return '0' + byte(u) }
	switch {
	case width == 2 && u < 1e2:
		return append(b, utod(u/1e1), utod(u%1e1))
	case width == 4 && u < 1e4:
		return append(b, utod(u/1e3), utod(u/1e2%1e1), utod(u/1e1%1e1), utod(u%1e1))
	}

	// Compute the number of decimal digits.
	var n int
	if u == 0 {
		n = 1
	}
	for u2 := u; u2 > 0; u2 /= 10 {
		n++
	}

	// Add 0-padding.
	for pad := width - n; pad > 0; pad-- {
		b = append(b, '0')
	}

	// Ensure capacity.
	if len(b)+n <= cap(b) {
		b = b[:len(b)+n]
	} else {
		b = append(b, make([]byte, n)...)
	}

	// Assemble decimal in reverse order.
	i := len(b) - 1
	for u >= 10 && i > 0 {
		q := u / 10
		b[i] = utod(u - q*10)
		u = q
		i--
	}
	b[i] = utod(u)
	return b
}

// Never printed, just needs to be non-nil for return by atoi.
var errAtoi = errors.New("time: invalid number")

// Duplicates functionality in strconv, but avoids dependency.
func atoi[bytes []byte | string](s bytes) (x int, err error) {
	neg := false
	if len(s) > 0 && (s[0] == '-' || s[0] == '+') {
		neg = s[0] == '-'
		s = s[1:]
	}
	q, rem, err := leadingInt(s)
	x = int(q)
	if err != nil || len(rem) > 0 {
		return 0, errAtoi
	}
	if neg {
		x = -x
	}
	return x, nil
}

// The "std" value passed to appendNano contains two packed fields: the number of
// digits after the decimal and the separator character (period or comma).
// These functions pack and unpack that variable.
func stdFracSecond(code, n, c int) int {
	// Use 0xfff to make the failure case even more absurd.
	if c == '.' {
		return code | ((n & 0xfff) << stdArgShift)
	}
	return code | ((n & 0xfff) << stdArgShift) | 1<<stdSeparatorShift
}

func digitsLen(std int) int {
	return (std >> stdArgShift) & 0xfff
}

func separator(std int) byte {
	if (std >> stdSeparatorShift) == 0 {
		return '.'
	}
	return ','
}

// appendNano appends a fractional second, as nanoseconds, to b
// and returns the result. The nanosec must be within [0, 999999999].
func appendNano(b []byte, nanosec int, std int) []byte {
	trim := std&stdMask == stdFracSecond9
	n := digitsLen(std)
	if trim && (n == 0 || nanosec == 0) {
		return b
	}
	dot := separator(std)
	b = append(b, dot)
	b = appendInt(b, nanosec, 9)
	if n < 9 {
		b = b[:len(b)-9+n]
	}
	if trim {
		for len(b) > 0 && b[len(b)-1] == '0' {
			b = b[:len(b)-1]
		}
		if len(b) > 0 && b[len(b)-1] == dot {
			b = b[:len(b)-1]
		}
	}
	return b
}

// String returns the time formatted using the format string
//
//	"2006-01-02 15:04:05.999999999 -0700 MST"
//
// If the time has a monotonic clock reading, the returned string
// includes a final field "m=±<value>", where value is the monotonic
// clock reading formatted as a decimal number of seconds.
//
// The returned string is meant for debugging; for a stable serialized
// representation, use t.MarshalText, t.MarshalBinary, or t.Format
// with an explicit format string.
func (t Time) String() string {
	s := t.Format("2006-01-02 15:04:05.999999999 -0700 MST")

	// Format monotonic clock reading as m=±ddd.nnnnnnnnn.
	if t.wall&hasMonotonic != 0 {
		m2 := uint64(t.ext)
		sign := byte('+')
		if t.ext < 0 {
			sign = '-'
			m2 = -m2
		}
		m1, m2 := m2/1e9, m2%1e9
		m0, m1 := m1/1e9, m1%1e9
		buf := make([]byte, 0, 24)
		buf = append(buf, " m="...)
		buf = append(buf, sign)
		wid := 0
		if m0 != 0 {
			buf = appendInt(buf, int(m0), 0)
			wid = 9
		}
		buf = appendInt(buf, int(m1), wid)
		buf = append(buf, '.')
		buf = appendInt(buf, int(m2), 9)
		s += string(buf)
	}
	return s
}

// GoString implements [fmt.GoStringer] and formats t to be printed in Go source
// code.
func (t Time) GoString() string {
	abs := t.abs()
	year, month, day := absDate(abs)
	hour, minute, second := absClock(abs)

	buf := make([]byte, 0, len("time.Date(9999, time.September, 31, 23, 59, 59, 999999999, time.Local)"))
	buf = append(buf, "time.Date("...)
	buf = appendInt(buf, year, 0)
	if January <= month && month <= December {
		buf = append(buf, ", time."...)
		buf = append(buf, longMonthNames[month-1]...)
	} else {
		// It's difficult to construct a time.Time with a date outside the
		// standard range but we might as well try to handle the case.
		buf = appendInt(buf, int(month), 0)
	}
	buf = append(buf, ", "...)
	buf = appendInt(buf, day, 0)
	buf = append(buf, ", "...)
	buf = appendInt(buf, hour, 0)
	buf = append(buf, ", "...)
	buf = appendInt(buf, minute, 0)
	buf = append(buf, ", "...)
	buf = appendInt(buf, second, 0)
	buf = append(buf, ", "...)
	buf = appendInt(buf, t.Nanosecond(), 0)
	buf = append(buf, ", "...)
	switch loc := t.Location(); loc {
	case UTC, nil:
		buf = append(buf, "time.UTC"...)
	case Local:
		buf = append(buf, "time.Local"...)
	default:
		// there are several options for how we could display this, none of
		// which are great:
		//
		// - use Location(loc.name), which is not technically valid syntax
		// - use LoadLocation(loc.name), which will cause a syntax error when
		// embedded and also would require us to escape the string without
		// importing fmt or strconv
		// - try to use FixedZone, which would also require escaping the name
		// and would represent e.g. "America/Los_Angeles" daylight saving time
		// shifts inaccurately
		// - use the pointer format, which is no worse than you'd get with the
		// old fmt.Sprintf("%#v", t) format.
		//
		// Of these, Location(loc.name) is the least disruptive. This is an edge
		// case we hope not to hit too often.
		buf = append(buf, `time.Location(`...)
		buf = append(buf, quote(loc.name)...)
		buf = append(buf, ')')
	}
	buf = append(buf, ')')
	return string(buf)
}

// Format returns a textual representation of the time value formatted according
// to the layout defined by the argument. See the documentation for the
// constant called [Layout] to see how to represent the layout format.
//
// The executable example for [Time.Format] demonstrates the working
// of the layout string in detail and is a good reference.
func (t Time) Format(layout string) string {
	const bufSize = 64
	var b []byte
	max := len(layout) + 10
	if max < bufSize {
		var buf [bufSize]byte
		b = buf[:0]
	} else {
		b = make([]byte, 0, max)
	}
	b = t.AppendFormat(b, layout)
	return string(b)
}

// AppendFormat is like [Time.Format] but appends the textual
// representation to b and returns the extended buffer.
func (t Time) AppendFormat(b []byte, layout string) []byte {
	// Optimize for RFC3339 as it accounts for over half of all representations.
	switch layout {
	case RFC3339:
		return t.appendFormatRFC3339(b, false)
	case RFC3339Nano:
		return t.appendFormatRFC3339(b, true)
	default:
		return t.appendFormat(b, layout)
	}
}

func (t Time) appendFormat(b []byte, layout string) []byte {
	var (
		name, offset, abs = t.locabs()

		year  int = -1
		month Month
		day   int
		yday  int
		hour  int = -1
		min   int
		sec   int
	)

	// Each iteration generates one std value.
	for layout != "" {
		prefix, std, suffix := nextStdChunk(layout)
		if prefix != "" {
			b = append(b, prefix...)
		}
		if std == 0 {
			break
		}
		layout = suffix

		// Compute year, month, day if needed.
		if year < 0 && std&stdNeedDate != 0 {
			_, month, day = absDate(abs)
			year, yday = absYearDay(abs)
			yday++
		}

		// Compute hour, minute, second if needed.
		if hour < 0 && std&stdNeedClock != 0 {
			hour, min, sec = absClock(abs)
		}

		switch std & stdMask {
		case stdYear:
			y := year
			if y < 0 {
				y = -y
			}
			b = appendInt(b, y%100, 2)
		case stdLongYear:
			b = appendInt(b, year, 4)
		case stdMonth:
			b = append(b, month.String()[:3]...)
		case stdLongMonth:
			m := month.String()
			b = append(b, m...)
		case stdNumMonth:
			b = appendInt(b, int(month), 0)
		case stdZeroMonth:
			b = appendInt(b, int(month), 2)
		case stdWeekDay:
			b = append(b, absWeekday(abs).String()[:3]...)
		case stdLongWeekDay:
			s := absWeekday(abs).String()
			b = append(b, s...)
		case stdDay:
			b = appendInt(b, day, 0)
		case stdUnderDay:
			if day < 10 {
				b = append(b, ' ')
			}
			b = appendInt(b, day, 0)
		case stdZeroDay:
			b = appendInt(b, day, 2)
		case stdUnderYearDay:
			if yday < 100 {
				b = append(b, ' ')
				if yday < 10 {
					b = append(b, ' ')
				}
			}
			b = appendInt(b, yday, 0)
		case stdZeroYearDay:
			b = appendInt(b, yday, 3)
		case stdHour:
			b = appendInt(b, hour, 2)
		case stdHour12:
			// Noon is 12PM, midnight is 12AM.
			hr := hour % 12
			if hr == 0 {
				hr = 12
			}
			b = appendInt(b, hr, 0)
		case stdZeroHour12:
			// Noon is 12PM, midnight is 12AM.
			hr := hour % 12
			if hr == 0 {
				hr = 12
			}
			b = appendInt(b, hr, 2)
		case stdMinute:
			b = appendInt(b, min, 0)
		case stdZeroMinute:
			b = appendInt(b, min, 2)
		case stdSecond:
			b = appendInt(b, sec, 0)
		case stdZeroSecond:
			b = appendInt(b, sec, 2)
		case stdPM:
			if hour >= 12 {
				b = append(b, "PM"...)
			} else {
				b = append(b, "AM"...)
			}
		case stdpm:
			if hour >= 12 {
				b = append(b, "pm"...)
			} else {
				b = append(b, "am"...)
			}
		case stdISO8601TZ, stdISO8601ColonTZ, stdISO8601SecondsTZ, stdISO8601ShortTZ, stdISO8601ColonSecondsTZ, stdNumTZ, stdNumColonTZ, stdNumSecondsTz, stdNumShortTZ, stdNumColonSecondsTZ:
			// Ugly special case. We cheat and take the "Z" variants
			// to mean "the time zone as formatted for ISO 8601".
			if offset == 0 && (std == stdISO8601TZ || std == stdISO8601ColonTZ || std == stdISO8601SecondsTZ || std == stdISO8601ShortTZ || std == stdISO8601ColonSecondsTZ) {
				b = append(b, 'Z')
				break
			}
			zone := offset / 60 // convert to minutes
			absoffset := offset
			if zone < 0 {
				b = append(b, '-')
				zone = -zone
				absoffset = -absoffset
			} else {
				b = append(b, '+')
			}
			b = appendInt(b, zone/60, 2)
			if std == stdISO8601ColonTZ || std == stdNumColonTZ || std == stdISO8601ColonSecondsTZ || std == stdNumColonSecondsTZ {
				b = append(b, ':')
			}
			if std != stdNumShortTZ && std != stdISO8601ShortTZ {
				b = appendInt(b, zone%60, 2)
			}

			// append seconds if appropriate
			if std == stdISO8601SecondsTZ || std == stdNumSecondsTz || std == stdNumColonSecondsTZ || std == stdISO8601ColonSecondsTZ {
				if std == stdNumColonSecondsTZ || std == stdISO8601ColonSecondsTZ {
					b = append(b, ':')
				}
				b = appendInt(b, absoffset%60, 2)
			}

		case stdTZ:
			if name != "" {
				b = append(b, name...)
				break
			}
			// No time zone known for this time, but we must print one.
			// Use the -0700 format.
			zone := offset / 60 // convert to minutes
			if zone < 0 {
				b = append(b, '-')
				zone = -zone
			} else {
				b = append(b, '+')
			}
			b = appendInt(b, zone/60, 2)
			b = appendInt(b, zone%60, 2)
		case stdFracSecond0, stdFracSecond9:
			b = appendNano(b, t.Nanosecond(), std)
		}
	}
	return b
}

var errBad = errors.New("bad value for field") // placeholder not passed to user

// ParseError describes a problem parsing a time string.
type ParseError struct {
	Layout     string
	Value      string
	LayoutElem string
	ValueElem  string
	Message    string
}

// newParseError creates a new ParseError.
// The provided value and valueElem are cloned to avoid escaping their values.
func newParseError(layout, value, layoutElem, valueElem, message string) *ParseError {
	valueCopy := stringslite.Clone(value)
	valueElemCopy := stringslite.Clone(valueElem)
	return &ParseError{layout, valueCopy, layoutElem, valueElemCopy, message}
}

// These are borrowed from unicode/utf8 and strconv and replicate behavior in
// that package, since we can't take a dependency on either.
const (
	lowerhex  = "0123456789abcdef"
	runeSelf  = 0x80
	runeError = '\uFFFD'
)

func quote(s string) string {
	buf := make([]byte, 1, len(s)+2) // slice will be at least len(s) + quotes
	buf[0] = '"'
	for i, c := range s {
		if c >= runeSelf || c < ' ' {
			// This means you are asking us to parse a time.Duration or
			// time.Location with unprintable or non-ASCII characters in it.
			// We don't expect to hit this case very often. We could try to
			// reproduce strconv.Quote's behavior with full fidelity but
			// given how rarely we expect to hit these edge cases, speed and
			// conciseness are better.
			var width int
			if c == runeError {
				width = 1
				if i+2 < len(s) && s[i:i+3] == string(runeError) {
					width = 3
				}
			} else {
				width = len(string(c))
			}
			for j := 0; j < width; j++ {
				buf = append(buf, `\x`...)
				buf = append(buf, lowerhex[s[i+j]>>4])
				buf = append(buf, lowerhex[s[i+j]&0xF])
			}
		} else {
			if c == '"' || c == '\\' {
				buf = append(buf, '\\')
			}
			buf = append(buf, string(c)...)
		}
	}
	buf = append(buf, '"')
	return string(buf)
}

// Error returns the string representation of a ParseError.
func (e *ParseError) Error() string {
	if e.Message == "" {
		return "parsing time " +
			quote(e.Value) + " as " +
			quote(e.Layout) + ": cannot parse " +
			quote(e.ValueElem) + " as " +
			quote(e.LayoutElem)
	}
	return "parsing time " +
		quote(e.Value) + e.Message
}

// isDigit reports whether s[i] is in range and is a decimal digit.
func isDigit[bytes []byte | string](s bytes, i int) bool {
	if len(s) <= i {
		return false
	}
	c := s[i]
	return '0' <= c && c <= '9'
}

// getnum parses s[0:1] or s[0:2] (fixed forces s[0:2])
// as a decimal integer and returns the integer and the
// remainder of the string.
func getnum(s string, fixed bool) (int, string, error) {
	if !isDigit(s, 0) {
		return 0, s, errBad
	}
	if !isDigit(s, 1) {
		if fixed {
			return 0, s, errBad
		}
		return int(s[0] - '0'), s[1:], nil
	}
	return int(s[0]-'0')*10 + int(s[1]-'0'), s[2:], nil
}

// getnum3 parses s[0:1], s[0:2], or s[0:3] (fixed forces s[0:3])
// as a decimal integer and returns the integer and the remainder
// of the string.
func getnum3(s string, fixed bool) (int, string, error) {
	var n, i int
	for i = 0; i < 3 && isDigit(s, i); i++ {
		n = n*10 + int(s[i]-'0')
	}
	if i == 0 || fixed && i != 3 {
		return 0, s, errBad
	}
	return n, s[i:], nil
}

func cutspace(s string) string {
	for len(s) > 0 && s[0] == ' ' {
		s = s[1:]
	}
	return s
}

// skip removes the given prefix from value,
// treating runs of space characters as equivalent.
func skip(value, prefix string) (string, error) {
	for len(prefix) > 0 {
		if prefix[0] == ' ' {
			if len(value) > 0 && value[0] != ' ' {
				return value, errBad
			}
			prefix = cutspace(prefix)
			value = cutspace(value)
			continue
		}
		if len(value) == 0 || value[0] != prefix[0] {
			return value, errBad
		}
		prefix = prefix[1:]
		value = value[1:]
	}
	return value, nil
}

// Parse parses a formatted string and returns the time value it represents.
// See the documentation for the constant called [Layout] to see how to
// represent the format. The second argument must be parseable using
// the format string (layout) provided as the first argument.
//
// The example for [Time.Format] demonstrates the working of the layout string
// in detail and is a good reference.
//
// When parsing (only), the input may contain a fractional second
// field immediately after the seconds field, even if the layout does not
// signify its presence. In that case either a comma or a decimal point
// followed by a maximal series of digits is parsed as a fractional second.
// Fractional seconds are truncated to nanosecond precision.
//
// Elements omitted from the layout are assumed to be zero or, when
// zero is impossible, one, so parsing "3:04pm" returns the time
// corresponding to Jan 1, year 0, 15:04:00 UTC (note that because the year is
// 0, this time is before the zero Time).
// Years must be in the range 0000..9999. The day of the week is checked
// for syntax but it is otherwise ignored.
//
// For layouts specifying the two-digit year 06, a value NN >= 69 will be treated
// as 19NN and a value NN < 69 will be treated as 20NN.
//
// The remainder of this comment describes the handling of time zones.
//
// In the absence of a time zone indicator, Parse returns a time in UTC.
//
// When parsing a time with a zone offset like -0700, if the offset corresponds
// to a time zone used by the current location ([Local]), then Parse uses that
// location and zone in the returned time. Otherwise it records the time as
// being in a fabricated location with time fixed at the given zone offset.
//
// When parsing a time with a zone abbreviation like MST, if the zone abbreviation
// has a defined offset in the current location, then that offset is used.
// The zone abbreviation "UTC" is recognized as UTC regardless of location.
// If the zone abbreviation is unknown, Parse records the time as being
// in a fabricated location with the given zone abbreviation and a zero offset.
// This choice means that such a time can be parsed and reformatted with the
// same layout losslessly, but the exact instant used in the representation will
// differ by the actual zone offset. To avoid such problems, prefer time layouts
// that use a numeric zone offset, or use [ParseInLocation].
func Parse(layout, value string) (Time, error) {
	// Optimize for RFC3339 as it accounts for over half of all representations.
	if layout == RFC3339 || layout == RFC3339Nano {
		if t, ok := parseRFC3339(value, Local); ok {
			return t, nil
		}
	}
	return parse(layout, value, UTC, Local)
}

// ParseInLocation is like Parse but differs in two important ways.
// First, in the absence of time zone information, Parse interprets a time as UTC;
// ParseInLocation interprets the time as in the given location.
// Second, when given a zone offset or abbreviation, Parse tries to match it
// against the Local location; ParseInLocation uses the given location.
func ParseInLocation(layout, value string, loc *Location) (Time, error) {
	// Optimize for RFC3339 as it accounts for over half of all representations.
	if layout == RFC3339 || layout == RFC3339Nano {
		if t, ok := parseRFC3339(value, loc); ok {
			return t, nil
		}
	}
	return parse(layout, value, loc, loc)
}

func parse(layout, value string, defaultLocation, local *Location) (Time, error) {
	alayout, avalue := layout, value
	rangeErrString := "" // set if a value is out of range
	amSet := false       // do we need to subtract 12 from the hour for midnight?
	pmSet := false       // do we need to add 12 to the hour?

	// Time being constructed.
	var (
		year       int
		month      int = -1
		day        int = -1
		yday       int = -1
		hour       int
		min        int
		sec        int
		nsec       int
		z          *Location
		zoneOffset int = -1
		zoneName   string
	)

	// Each iteration processes one std value.
	for {
		var err error
		prefix, std, suffix := nextStdChunk(layout)
		stdstr := layout[len(prefix) : len(layout)-len(suffix)]
		value, err = skip(value, prefix)
		if err != nil {
			return Time{}, newParseError(alayout, avalue, prefix, value, "")
		}
		if std == 0 {
			if len(value) != 0 {
				return Time{}, newParseError(alayout, avalue, "", value, ": extra text: "+quote(value))
			}
			break
		}
		layout = suffix
		var p string
		hold := value
		switch std & stdMask {
		case stdYear:
			if len(value) < 2 {
				err = errBad
				break
			}
			p, value = value[0:2], value[2:]
			year, err = atoi(p)
			if err != nil {
				break
			}
			if year >= 69 { // Unix time starts Dec 31 1969 in some time zones
				year += 1900
			} else {
				year += 2000
			}
		case stdLongYear:
			if len(value) < 4 || !isDigit(value, 0) {
				err = errBad
				break
			}
			p, value = value[0:4], value[4:]
			year, err = atoi(p)
		case stdMonth:
			month, value, err = lookup(shortMonthNames, value)
			month++
		case stdLongMonth:
			month, value, err = lookup(longMonthNames, value)
			month++
		case stdNumMonth, stdZeroMonth:
			month, value, err = getnum(value, std == stdZeroMonth)
			if err == nil && (month <= 0 || 12 < month) {
				rangeErrString = "month"
			}
		case stdWeekDay:
			// Ignore weekday except for error checking.
			_, value, err = lookup(shortDayNames, value)
		case stdLongWeekDay:
			_, value, err = lookup(longDayNames, value)
		case stdDay, stdUnderDay, stdZeroDay:
			if std == stdUnderDay && len(value) > 0 && value[0] == ' ' {
				value = value[1:]
			}
			day, value, err = getnum(value, std == stdZeroDay)
			// Note that we allow any one- or two-digit day here.
			// The month, day, year combination is validated after we've completed parsing.
		case stdUnderYearDay, stdZeroYearDay:
			for i := 0; i < 2; i++ {
				if std == stdUnderYearDay && len(value) > 0 && value[0] == ' ' {
					value = value[1:]
				}
			}
			yday, value, err = getnum3(value, std == stdZeroYearDay)
			// Note that we allow any one-, two-, or three-digit year-day here.
			// The year-day, year combination is validated after we've completed parsing.
		case stdHour:
			hour, value, err = getnum(value, false)
			if hour < 0 || 24 <= hour {
				rangeErrString = "hour"
			}
		case stdHour12, stdZeroHour12:
			hour, value, err = getnum(value, std == stdZeroHour12)
			if hour < 0 || 12 < hour {
				rangeErrString = "hour"
			}
		case stdMinute, stdZeroMinute:
			min, value, err = getnum(value, std == stdZeroMinute)
			if min < 0 || 60 <= min {
				rangeErrString = "minute"
			}
		case stdSecond, stdZeroSecond:
			sec, value, err = getnum(value, std == stdZeroSecond)
			if err != nil {
				break
			}
			if sec < 0 || 60 <= sec {
				rangeErrString = "second"
				break
			}
			// Special case: do we have a fractional second but no
			// fractional second in the format?
			if len(value) >= 2 && commaOrPeriod(value[0]) && isDigit(value, 1) {
				_, std, _ = nextStdChunk(layout)
				std &= stdMask
				if std == stdFracSecond0 || std == stdFracSecond9 {
					// Fractional second in the layout; proceed normally
					break
				}
				// No fractional second in the layout but we have one in the input.
				n := 2
				for ; n < len(value) && isDigit(value, n); n++ {
				}
				nsec, rangeErrString, err = parseNanoseconds(value, n)
				value = value[n:]
			}
		case stdPM:
			if len(value) < 2 {
				err = errBad
				break
			}
			p, value = value[0:2], value[2:]
			switch p {
			case "PM":
				pmSet = true
			case "AM":
				amSet = true
			default:
				err = errBad
			}
		case stdpm:
			if len(value) < 2 {
				err = errBad
				break
			}
			p, value = value[0:2], value[2:]
			switch p {
			case "pm":
				pmSet = true
			case "am":
				amSet = true
			default:
				err = errBad
			}
		case stdISO8601TZ, stdISO8601ColonTZ, stdISO8601SecondsTZ, stdISO8601ShortTZ, stdISO8601ColonSecondsTZ, stdNumTZ, stdNumShortTZ, stdNumColonTZ, stdNumSecondsTz, stdNumColonSecondsTZ:
			if (std == stdISO8601TZ || std == stdISO8601ShortTZ || std == stdISO8601ColonTZ) && len(value) >= 1 && value[0] == 'Z' {
				value = value[1:]
				z = UTC
				break
			}
			var sign, hour, min, seconds string
			if std == stdISO8601ColonTZ || std == stdNumColonTZ {
				if len(value) < 6 {
					err = errBad
					break
				}
				if value[3] != ':' {
					err = errBad
					break
				}
				sign, hour, min, seconds, value = value[0:1], value[1:3], value[4:6], "00", value[6:]
			} else if std == stdNumShortTZ || std == stdISO8601ShortTZ {
				if len(value) < 3 {
					err = errBad
					break
				}
				sign, hour, min, seconds, value = value[0:1], value[1:3], "00", "00", value[3:]
			} else if std == stdISO8601ColonSecondsTZ || std == stdNumColonSecondsTZ {
				if len(value) < 9 {
					err = errBad
					break
				}
				if value[3] != ':' || value[6] != ':' {
					err = errBad
					break
				}
				sign, hour, min, seconds, value = value[0:1], value[1:3], value[4:6], value[7:9], value[9:]
			} else if std == stdISO8601SecondsTZ || std == stdNumSecondsTz {
				if len(value) < 7 {
					err = errBad
					break
				}
				sign, hour, min, seconds, value = value[0:1], value[1:3], value[3:5], value[5:7], value[7:]
			} else {
				if len(value) < 5 {
					err = errBad
					break
				}
				sign, hour, min, seconds, value = value[0:1], value[1:3], value[3:5], "00", value[5:]
			}
			var hr, mm, ss int
			hr, _, err = getnum(hour, true)
			if err == nil {
				mm, _, err = getnum(min, true)
			}
			if err == nil {
				ss, _, err = getnum(seconds, true)
			}

			// The range test use > rather than >=,
			// as some people do write offsets of 24 hours
			// or 60 minutes or 60 seconds.
			if hr > 24 {
				rangeErrString = "time zone offset hour"
			}
			if mm > 60 {
				rangeErrString = "time zone offset minute"
			}
			if ss > 60 {
				rangeErrString = "time zone offset second"
			}

			zoneOffset = (hr*60+mm)*60 + ss // offset is in seconds
			switch sign[0] {
			case '+':
			case '-':
				zoneOffset = -zoneOffset
			default:
				err = errBad
			}
		case stdTZ:
			// Does it look like a time zone?
			if len(value) >= 3 && value[0:3] == "UTC" {
				z = UTC
				value = value[3:]
				break
			}
			n, ok := parseTimeZone(value)
			if !ok {
				err = errBad
				break
			}
			zoneName, value = value[:n], value[n:]

		case stdFracSecond0:
			// stdFracSecond0 requires the exact number of digits as specified in
			// the layout.
			ndigit := 1 + digitsLen(std)
			if len(value) < ndigit {
				err = errBad
				break
			}
			nsec, rangeErrString, err = parseNanoseconds(value, ndigit)
			value = value[ndigit:]

		case stdFracSecond9:
			if len(value) < 2 || !commaOrPeriod(value[0]) || value[1] < '0' || '9' < value[1] {
				// Fractional second omitted.
				break
			}
			// Take any number of digits, even more than asked for,
			// because it is what the stdSecond case would do.
			i := 0
			for i+1 < len(value) && '0' <= value[i+1] && value[i+1] <= '9' {
				i++
			}
			nsec, rangeErrString, err = parseNanoseconds(value, 1+i)
			value = value[1+i:]
		}
		if rangeErrString != "" {
			return Time{}, newParseError(alayout, avalue, stdstr, value, ": "+rangeErrString+" out of range")
		}
		if err != nil {
			return Time{}, newParseError(alayout, avalue, stdstr, hold, "")
		}
	}
	if pmSet && hour < 12 {
		hour += 12
	} else if amSet && hour == 12 {
		hour = 0
	}

	// Convert yday to day, month.
	if yday >= 0 {
		var d int
		var m int
		if isLeap(year) {
			if yday == 31+29 {
				m = int(February)
				d = 29
			} else if yday > 31+29 {
				yday--
			}
		}
		if yday < 1 || yday > 365 {
			return Time{}, newParseError(alayout, avalue, "", value, ": day-of-year out of range")
		}
		if m == 0 {
			m = (yday-1)/31 + 1
			if int(daysBefore[m]) < yday {
				m++
			}
			d = yday - int(daysBefore[m-1])
		}
		// If month, day already seen, yday's m, d must match.
		// Otherwise, set them from m, d.
		if month >= 0 && month != m {
			return Time{}, newParseError(alayout, avalue, "", value, ": day-of-year does not match month")
		}
		month = m
		if day >= 0 && day != d {
			return Time{}, newParseError(alayout, avalue, "", value, ": day-of-year does not match day")
		}
		day = d
	} else {
		if month < 0 {
			month = int(January)
		}
		if day < 0 {
			day = 1
		}
	}

	// Validate the day of the month.
	if day < 1 || day > daysIn(Month(month), year) {
		return Time{}, newParseError(alayout, avalue, "", value, ": day out of range")
	}

	if z != nil {
		return Date(year, Month(month), day, hour, min, sec, nsec, z), nil
	}

	if zoneOffset != -1 {
		t := Date(year, Month(month), day, hour, min, sec, nsec, UTC)
		t.addSec(-int64(zoneOffset))

		// Look for local zone with the given offset.
		// If that zone was in effect at the given time, use it.
		name, offset, _, _, _ := local.lookup(t.unixSec())
		if offset == zoneOffset && (zoneName == "" || name == zoneName) {
			t.setLoc(local)
			return t, nil
		}

		// Otherwise create fake zone to record offset.
		zoneNameCopy := stringslite.Clone(zoneName) // avoid leaking the input value
		t.setLoc(FixedZone(zoneNameCopy, zoneOffset))
		return t, nil
	}

	if zoneName != "" {
		t := Date(year, Month(month), day, hour, min, sec, nsec, UTC)
		// Look for local zone with the given offset.
		// If that zone was in effect at the given time, use it.
		offset, ok := local.lookupName(zoneName, t.unixSec())
		if ok {
			t.addSec(-int64(offset))
			t.setLoc(local)
			return t, nil
		}

		// Otherwise, create fake zone with unknown offset.
		if len(zoneName) > 3 && zoneName[:3] == "GMT" {
			offset, _ = atoi(zoneName[3:]) // Guaranteed OK by parseGMT.
			offset *= 3600
		}
		zoneNameCopy := stringslite.Clone(zoneName) // avoid leaking the input value
		t.setLoc(FixedZone(zoneNameCopy, offset))
		return t, nil
	}

	// Otherwise, fall back to default.
	return Date(year, Month(month), day, hour, min, sec, nsec, defaultLocation), nil
}

// parseTimeZone parses a time zone string and returns its length. Time zones
// are human-generated and unpredictable. We can't do precise error checking.
// On the other hand, for a correct parse there must be a time zone at the
// beginning of the string, so it's almost always true that there's one
// there. We look at the beginning of the string for a run of upper-case letters.
// If there are more than 5, it's an error.
// If there are 4 or 5 and the last is a T, it's a time zone.
// If there are 3, it's a time zone.
// Otherwise, other than special cases, it's not a time zone.
// GMT is special because it can have an hour offset.
func parseTimeZone(value string) (length int, ok bool) {
	if len(value) < 3 {
		return 0, false
	}
	// Special case 1: ChST and MeST are the only zones with a lower-case letter.
	if len(value) >= 4 && (value[:4] == "ChST" || value[:4] == "MeST") {
		return 4, true
	}
	// Special case 2: GMT may have an hour offset; treat it specially.
	if value[:3] == "GMT" {
		length = parseGMT(value)
		return length, true
	}
	// Special Case 3: Some time zones are not named, but have +/-00 format
	if value[0] == '+' || value[0] == '-' {
		length = parseSignedOffset(value)
		ok := length > 0 // parseSignedOffset returns 0 in case of bad input
		return length, ok
	}
	// How many upper-case letters are there? Need at least three, at most five.
	var nUpper int
	for nUpper = 0; nUpper < 6; nUpper++ {
		if nUpper >= len(value) {
			break
		}
		if c := value[nUpper]; c < 'A' || 'Z' < c {
			break
		}
	}
	switch nUpper {
	case 0, 1, 2, 6:
		return 0, false
	case 5: // Must end in T to match.
		if value[4] == 'T' {
			return 5, true
		}
	case 4:
		// Must end in T, except one special case.
		if value[3] == 'T' || value[:4] == "WITA" {
			return 4, true
		}
	case 3:
		return 3, true
	}
	return 0, false
}

// parseGMT parses a GMT time zone. The input string is known to start "GMT".
// The function checks whether that is followed by a sign and a number in the
// range -23 through +23 excluding zero.
func parseGMT(value string) int {
	value = value[3:]
	if len(value) == 0 {
		return 3
	}

	return 3 + parseSignedOffset(value)
}

// parseSignedOffset parses a signed timezone offset (e.g. "+03" or "-04").
// The function checks for a signed number in the range -23 through +23 excluding zero.
// Returns length of the found offset string or 0 otherwise.
func parseSignedOffset(value string) int {
	sign := value[0]
	if sign != '-' && sign != '+' {
		return 0
	}
	x, rem, err := leadingInt(value[1:])

	// fail if nothing consumed by leadingInt
	if err != nil || value[1:] == rem {
		return 0
	}
	if x > 23 {
		return 0
	}
	return len(value) - len(rem)
}

func commaOrPeriod(b byte) bool {
	return b == '.' || b == ','
}

func parseNanoseconds[bytes []byte | string](value bytes, nbytes int) (ns int, rangeErrString string, err error) {
	if !commaOrPeriod(value[0]) {
		err = errBad
		return
	}
	if nbytes > 10 {
		value = value[:10]
		nbytes = 10
	}
	if ns, err = atoi(value[1:nbytes]); err != nil {
		return
	}
	if ns < 0 {
		rangeErrString = "fractional second"
		return
	}
	// We need nanoseconds, which means scaling by the number
	// of missing digits in the format, maximum length 10.
	scaleDigits := 10 - nbytes
	for i := 0; i < scaleDigits; i++ {
		ns *= 10
	}
	return
}

var errLeadingInt = errors.New("time: bad [0-9]*") // never printed

// leadingInt consumes the leading [0-9]* from s.
func leadingInt[bytes []byte | string](s bytes) (x uint64, rem bytes, err error) {
	i := 0
	for ; i < len(s); i++ {
		c := s[i]
		if c < '0' || c > '9' {
			break
		}
		if x > 1<<63/10 {
			// overflow
			return 0, rem, errLeadingInt
		}
		x = x*10 + uint64(c) - '0'
		if x > 1<<63 {
			// overflow
			return 0, rem, errLeadingInt
		}
	}
	return x, s[i:], nil
}

// leadingFraction consumes the leading [0-9]* from s.
// It is used only for fractions, so does not return an error on overflow,
// it just stops accumulating precision.
func leadingFraction(s string) (x uint64, scale float64, rem string) {
	i := 0
	scale = 1
	overflow := false
	for ; i < len(s); i++ {
		c := s[i]
		if c < '0' || c > '9' {
			break
		}
		if overflow {
			continue
		}
		if x > (1<<63-1)/10 {
			// It's possible for overflow to give a positive number, so take care.
			overflow = true
			continue
		}
		y := x*10 + uint64(c) - '0'
		if y > 1<<63 {
			overflow = true
			continue
		}
		x = y
		scale *= 10
	}
	return x, scale, s[i:]
}

var unitMap = map[string]uint64{
	"ns": uint64(Nanosecond),
	"us": uint64(Microsecond),
	"µs": uint64(Microsecond), // U+00B5 = micro symbol
	"μs": uint64(Microsecond), // U+03BC = Greek letter mu
	"ms": uint64(Millisecond),
	"s":  uint64(Second),
	"m":  uint64(Minute),
	"h":  uint64(Hour),
}

// ParseDuration parses a duration string.
// A duration string is a possibly signed sequence of
// decimal numbers, each with optional fraction and a unit suffix,
// such as "300ms", "-1.5h" or "2h45m".
// Valid time units are "ns", "us" (or "µs"), "ms", "s", "m", "h".
func ParseDuration(s string) (Duration, error) {
	// [-+]?([0-9]*(\.[0-9]*)?[a-z]+)+
	orig := s
	var d uint64
	neg := false

	// Consume [-+]?
	if s != "" {
		c := s[0]
		if c == '-' || c == '+' {
			neg = c == '-'
			s = s[1:]
		}
	}
	// Special case: if all that is left is "0", this is zero.
	if s == "0" {
		return 0, nil
	}
	if s == "" {
		return 0, errors.New("time: invalid duration " + quote(orig))
	}
	for s != "" {
		var (
			v, f  uint64      // integers before, after decimal point
			scale float64 = 1 // value = v + f/scale
		)

		var err error

		// The next character must be [0-9.]
		if !(s[0] == '.' || '0' <= s[0] && s[0] <= '9') {
			return 0, errors.New("time: invalid duration " + quote(orig))
		}
		// Consume [0-9]*
		pl := len(s)
		v, s, err = leadingInt(s)
		if err != nil {
			return 0, errors.New("time: invalid duration " + quote(orig))
		}
		pre := pl != len(s) // whether we consumed anything before a period

		// Consume (\.[0-9]*)?
		post := false
		if s != "" && s[0] == '.' {
			s = s[1:]
			pl := len(s)
			f, scale, s = leadingFraction(s)
			post = pl != len(s)
		}
		if !pre && !post {
			// no digits (e.g. ".s" or "-.s")
			return 0, errors.New("time: invalid duration " + quote(orig))
		}

		// Consume unit.
		i := 0
		for ; i < len(s); i++ {
			c := s[i]
			if c == '.' || '0' <= c && c <= '9' {
				break
			}
		}
		if i == 0 {
			return 0, errors.New("time: missing unit in duration " + quote(orig))
		}
		u := s[:i]
		s = s[i:]
		unit, ok := unitMap[u]
		if !ok {
			return 0, errors.New("time: unknown unit " + quote(u) + " in duration " + quote(orig))
		}
		if v > 1<<63/unit {
			// overflow
			return 0, errors.New("time: invalid duration " + quote(orig))
		}
		v *= unit
		if f > 0 {
			// float64 is needed to be nanosecond accurate for fractions of hours.
			// v >= 0 && (f*unit/scale) <= 3.6e+12 (ns/h, h is the largest unit)
			v += uint64(float64(f) * (float64(unit) / scale))
			if v > 1<<63 {
				// overflow
				return 0, errors.New("time: invalid duration " + quote(orig))
			}
		}
		d += v
		if d > 1<<63 {
			return 0, errors.New("time: invalid duration " + quote(orig))
		}
	}
	if neg {
		return -Duration(d), nil
	}
	if d > 1<<63-1 {
		return 0, errors.New("time: invalid duration " + quote(orig))
	}
	return Duration(d), nil
}
