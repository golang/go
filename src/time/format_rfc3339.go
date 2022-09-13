// Copyright 2022 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package time

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

	// Format date.
	year, month, day, _ := absDate(abs, true)
	b = appendInt(b, year, 4)
	b = append(b, '-')
	b = appendInt(b, int(month), 2)
	b = append(b, '-')
	b = appendInt(b, day, 2)

	b = append(b, 'T')

	// Format time.
	hour, min, sec := absClock(abs)
	b = appendInt(b, hour, 2)
	b = append(b, ':')
	b = appendInt(b, min, 2)
	b = append(b, ':')
	b = appendInt(b, sec, 2)

	if nanos {
		std := stdFracSecond(stdFracSecond9, 9, '.')
		b = formatNano(b, uint(t.Nanosecond()), std)
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
	b = appendInt(b, zone/60, 2)
	b = append(b, ':')
	b = appendInt(b, zone%60, 2)
	return b
}

func parseRFC3339(s string, local *Location) (Time, bool) {
	// parseUint parses s as an unsigned decimal integer and
	// verifies that it is within some range.
	// If it is invalid or out-of-range,
	// it sets ok to false and returns the min value.
	ok := true
	parseUint := func(s string, min, max int) (x int) {
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
	if s != "Z" {
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
