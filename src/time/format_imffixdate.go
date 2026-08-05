// Copyright 2026 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package time

// imfFixdate is the IMF-fixdate layout: the format HTTP senders are required
// to generate for header fields carrying a date, defined in RFC 9110, section
// 5.6.7 as a fixed-length, single-zone subset of the Internet Message Format
// (RFC 5322). It is the value of net/http.TimeFormat, and after RFC 3339 it is
// the most widely used machine-readable layout, which is why it is
// specialized here.
//
// "GMT" is a literal in this layout, not a zone abbreviation, so formatting
// does not convert. An HTTP-date represents an instant in UTC; supplying one
// is the caller's responsibility, as net/http does.
const imfFixdate = "Mon, 02 Jan 2006 15:04:05 GMT"

// appendFormatIMFFixdate is a specialization of appendFormat for imfFixdate.
// It formats t in its own location, exactly as the general path does.
func (t Time) appendFormatIMFFixdate(b []byte) []byte {
	_, _, abs := t.locabs()
	days := abs.days()
	year, month, day := days.date()
	hour, min, sec := abs.clock()

	b = append(b, days.weekday().String()[:3]...)
	b = append(b, ',', ' ')
	b = appendInt(b, day, 2)
	b = append(b, ' ')
	b = append(b, month.String()[:3]...)
	b = append(b, ' ')
	b = appendInt(b, year, 4)
	b = append(b, ' ')
	b = appendInt(b, hour, 2)
	b = append(b, ':')
	b = appendInt(b, min, 2)
	b = append(b, ':')
	b = appendInt(b, sec, 2)
	return append(b, " GMT"...)
}
