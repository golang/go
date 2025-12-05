// Copyright 2022 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package time

import "errors"

// RFC 3339 is the most commonly used format.
//
// It is implicitly used by the Time.(Marshal|Unmarshal)(Text|JSON) methods.
// Also, according to analysis on https://go.dev/issue/52746,
// RFC 3339 accounts for 57% of all explicitly specified time formats,
// with the second most popular format only being used 8% of the time.
// The overwhelming use of RFC 3339 compared to all other formats justifies
// the addition of logic to optimize formatting and parsing.

func (t Time) appendFormatRFC3339(b []byte, nanos bool) []byte {
	_, offset, abs := t.locabs()

	// Format date and time.
	year, month, day := abs.days().date()
	hour, min, sec := abs.clock()

	b = appendIntWidth4(b, year)
	b = append(b, '-',
		tensDigit[month], onesDigit[month],
		'-',
		tensDigit[day], onesDigit[day],
		'T',
		tensDigit[hour], onesDigit[hour],
		':',
		tensDigit[min], onesDigit[min],
		':',
		tensDigit[sec], onesDigit[sec],
	)

	if nanos {
		std := stdFracSecond(stdFracSecond9, 9, '.')
		b = appendNano(b, t.Nanosecond(), std)
	}

	if offset == 0 {
		return append(b, 'Z')
	}

	// Format zone.
	zone := offset / 60 // convert to minutes
	if zone < 0 {
		b = append(b, '-')
		zone = -zone
	} else {
		b = append(b, '+')
	}

	zone %= 3600
	b = append(b,
		tensDigit[zone/60], onesDigit[zone/60],
		':',
		tensDigit[zone%60], onesDigit[zone%60],
	)
	return b
}

func (t Time) appendStrictRFC3339(b []byte) ([]byte, error) {
	n0 := len(b)
	b = t.appendFormatRFC3339(b, true)

	// Not all valid Go timestamps can be serialized as valid RFC 3339.
	// Explicitly check for these edge cases.
	// See https://go.dev/issue/4556 and https://go.dev/issue/54580.
	num2 := func(b []byte) byte { return 10*(b[0]-'0') + (b[1] - '0') }
	switch {
	case b[n0+len("9999")] != '-': // year must be exactly 4 digits wide
		return b, errors.New("year outside of range [0,9999]")
	case b[len(b)-1] != 'Z':
		c := b[len(b)-len("Z07:00")]
		if ('0' <= c && c <= '9') || num2(b[len(b)-len("07:00"):]) >= 24 {
			return b, errors.New("timezone hour outside of range [0,23]")
		}
	}
	return b, nil
}

func parseRFC3339[bytes []byte | string](s bytes, local *Location) (Time, bool) {
	// parseUint parses s as an unsigned decimal integer and
	// verifies that it is within some range.
	// If it is invalid or out-of-range,
	// it sets ok to false and returns the min value.
	ok := true
	parseUint := func(s bytes, min, max int) (x int) {
		for _, c := range []byte(s) {
			if c < '0' || '9' < c {
				ok = false
				return min
			}
			x = x*10 + int(c) - '0'
		}
		if x < min || max < x {
			ok = false
			return min
		}
		return x
	}

	// Parse the date and time.
	if len(s) < len("2006-01-02T15:04:05") {
		return Time{}, false
	}
	year := parseUint(s[0:4], 0, 9999)                       // e.g., 2006
	month := parseUint(s[5:7], 1, 12)                        // e.g., 01
	day := parseUint(s[8:10], 1, daysIn(Month(month), year)) // e.g., 02
	hour := parseUint(s[11:13], 0, 23)                       // e.g., 15
	min := parseUint(s[14:16], 0, 59)                        // e.g., 04
	sec := parseUint(s[17:19], 0, 59)                        // e.g., 05
	if !ok || !(s[4] == '-' && s[7] == '-' && s[10] == 'T' && s[13] == ':' && s[16] == ':') {
		return Time{}, false
	}
	s = s[19:]

	// Parse the fractional second.
	var nsec int
	if len(s) >= 2 && s[0] == '.' && isDigit(s, 1) {
		n := 2
		for ; n < len(s) && isDigit(s, n); n++ {
		}
		nsec, _, _ = parseNanoseconds(s, n)
		s = s[n:]
	}

	// Parse the time zone.
	t := Date(year, Month(month), day, hour, min, sec, nsec, UTC)
	if len(s) != 1 || s[0] != 'Z' {
		if len(s) != len("-07:00") {
			return Time{}, false
		}
		hr := parseUint(s[1:3], 0, 23) // e.g., 07
		mm := parseUint(s[4:6], 0, 59) // e.g., 00
		if !ok || !((s[0] == '-' || s[0] == '+') && s[3] == ':') {
			return Time{}, false
		}
		zoneOffset := (hr*60 + mm) * 60
		if s[0] == '-' {
			zoneOffset *= -1
		}
		t.addSec(-int64(zoneOffset))

		// Use local zone with the given offset if possible.
		if _, offset, _, _, _ := local.lookup(t.unixSec()); offset == zoneOffset {
			t.setLoc(local)
		} else {
			t.setLoc(FixedZone("", zoneOffset))
		}
	}
	return t, true
}

func parseStrictRFC3339(b []byte) (Time, error) {
	t, ok := parseRFC3339(b, Local)
	if !ok {
		t, err := Parse(RFC3339, string(b))
		if err != nil {
			return Time{}, err
		}

		// The parse template syntax cannot correctly validate RFC 3339.
		// Explicitly check for cases that Parse is unable to validate for.
		// See https://go.dev/issue/54580.
		num2 := func(b []byte) byte { return 10*(b[0]-'0') + (b[1] - '0') }
		switch {
		// TODO(https://go.dev/issue/54580): Strict parsing is disabled for now.
		// Enable this again with a GODEBUG opt-out.
		case true:
			return t, nil
		case b[len("2006-01-02T")+1] == ':': // hour must be two digits
			return Time{}, &ParseError{RFC3339, string(b), "15", string(b[len("2006-01-02T"):][:1]), ""}
		case b[len("2006-01-02T15:04:05")] == ',': // sub-second separator must be a period
			return Time{}, &ParseError{RFC3339, string(b), ".", ",", ""}
		case b[len(b)-1] != 'Z':
			switch {
			case num2(b[len(b)-len("07:00"):]) >= 24: // timezone hour must be in range
				return Time{}, &ParseError{RFC3339, string(b), "Z07:00", string(b[len(b)-len("Z07:00"):]), ": timezone hour out of range"}
			case num2(b[len(b)-len("00"):]) >= 60: // timezone minute must be in range
				return Time{}, &ParseError{RFC3339, string(b), "Z07:00", string(b[len(b)-len("Z07:00"):]), ": timezone minute out of range"}
			}
		default: // unknown error; should not occur
			return Time{}, &ParseError{RFC3339, string(b), RFC3339, string(b), ""}
		}
	}
	return t, nil
}
