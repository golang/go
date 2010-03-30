package time

import (
	"bytes"
	"once"
	"os"
	"strconv"
)

const (
	numeric = iota
	alphabetic
	separator
	plus
	minus
)

// These are predefined layouts for use in Time.Format.
// The standard time used in the layouts is:
//	Mon Jan 2 15:04:05 MST 2006  (MST is GMT-0700)
// which is Unix time 1136243045.
// (Think of it as 01/02 03:04:05PM '06 -0700.)
// An underscore _ represents a space that
// may be replaced by a digit if the following number
// (a day) has two digits; for compatibility with
// fixed-width Unix time formats.
const (
	ANSIC    = "Mon Jan _2 15:04:05 2006"
	UnixDate = "Mon Jan _2 15:04:05 MST 2006"
	RubyDate = "Mon Jan 02 15:04:05 -0700 2006"
	RFC822   = "02 Jan 06 1504 MST"
	// RFC822 with Zulu time.
	RFC822Z = "02 Jan 06 1504 -0700"
	RFC850  = "Monday, 02-Jan-06 15:04:05 MST"
	RFC1123 = "Mon, 02 Jan 2006 15:04:05 MST"
	Kitchen = "3:04PM"
	// Special case: use Z to get the time zone formatted according to ISO 8601,
	// which is -0700 or Z for UTC
	ISO8601 = "2006-01-02T15:04:05Z"
)

const (
	stdLongMonth   = "January"
	stdMonth       = "Jan"
	stdNumMonth    = "1"
	stdZeroMonth   = "01"
	stdLongWeekDay = "Monday"
	stdWeekDay     = "Mon"
	stdDay         = "2"
	stdUnderDay    = "_2"
	stdZeroDay     = "02"
	stdHour        = "15"
	stdHour12      = "3"
	stdZeroHour12  = "03"
	stdMinute      = "4"
	stdZeroMinute  = "04"
	stdSecond      = "5"
	stdZeroSecond  = "05"
	stdLongYear    = "2006"
	stdYear        = "06"
	stdPM          = "PM"
	stdpm          = "pm"
	stdTZ          = "MST"
	stdISO8601TZ   = "Z"     // prints Z for UTC
	stdNumTZ       = "-0700" // always numeric
)

// nextStdChunk finds the first occurrence of a std string in
// layout and returns the text before, the std string, and the text after.
func nextStdChunk(layout string) (prefix, std, suffix string) {
	for i := 0; i < len(layout); i++ {
		switch layout[i] {
		case 'J': // January, Jan
			if len(layout) >= i+7 && layout[i:i+7] == stdLongMonth {
				return layout[0:i], stdLongMonth, layout[i+7:]
			}
			if len(layout) >= i+3 && layout[i:i+3] == stdMonth {
				return layout[0:i], stdMonth, layout[i+3:]
			}

		case 'M': // Monday, Mon, MST
			if len(layout) >= i+6 && layout[i:i+6] == stdLongWeekDay {
				return layout[0:i], stdLongWeekDay, layout[i+6:]
			}
			if len(layout) >= i+3 {
				if layout[i:i+3] == stdWeekDay {
					return layout[0:i], stdWeekDay, layout[i+3:]
				}
				if layout[i:i+3] == stdTZ {
					return layout[0:i], stdTZ, layout[i+3:]
				}
			}

		case '0': // 01, 02, 03, 04, 05, 06
			if len(layout) >= i+2 && '1' <= layout[i+1] && layout[i+1] <= '6' {
				return layout[0:i], layout[i : i+2], layout[i+2:]
			}

		case '1': // 15, 1
			if len(layout) >= i+2 && layout[i+1] == '5' {
				return layout[0:i], stdHour, layout[i+2:]
			}
			return layout[0:i], stdNumMonth, layout[i+1:]

		case '2': // 2006, 2
			if len(layout) >= i+4 && layout[i:i+4] == stdLongYear {
				return layout[0:i], stdLongYear, layout[i+4:]
			}
			return layout[0:i], stdDay, layout[i+1:]

		case '_': // _2
			if len(layout) >= i+2 && layout[i+1] == '2' {
				return layout[0:i], stdUnderDay, layout[i+2:]
			}

		case '3', '4', '5', 'Z': // 3, 4, 5, Z
			return layout[0:i], layout[i : i+1], layout[i+1:]

		case 'P': // PM
			if len(layout) >= i+2 && layout[i+1] == 'M' {
				return layout[0:i], layout[i : i+2], layout[i+2:]
			}

		case 'p': // pm
			if len(layout) >= i+2 && layout[i+1] == 'm' {
				return layout[0:i], layout[i : i+2], layout[i+2:]
			}

		case '-': // -0700
			if len(layout) >= i+5 && layout[i:i+5] == stdNumTZ {
				return layout[0:i], layout[i : i+5], layout[i+5:]
			}
		}
	}
	return layout, "", ""
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
	"---",
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
	"---",
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

func lookup(tab []string, val string) (int, string, os.Error) {
	for i, v := range tab {
		if len(val) >= len(v) && val[0:len(v)] == v {
			return i, val[len(v):], nil
		}
	}
	return -1, val, errBad
}

func pad(i int, padding string) string {
	s := strconv.Itoa(i)
	if i < 10 {
		s = padding + s
	}
	return s
}

func zeroPad(i int) string { return pad(i, "0") }

// Format returns a textual representation of the time value formatted
// according to layout.  The layout defines the format by showing the
// representation of a standard time, which is then used to describe
// the time to be formatted.  Predefined layouts ANSIC, UnixDate,
// ISO8601 and others describe standard representations.
func (t *Time) Format(layout string) string {
	b := new(bytes.Buffer)
	// Each iteration generates one std value.
	for {
		prefix, std, suffix := nextStdChunk(layout)
		b.WriteString(prefix)
		if std == "" {
			break
		}
		var p string
		switch std {
		case stdYear:
			p = strconv.Itoa64(t.Year % 100)
		case stdLongYear:
			p = strconv.Itoa64(t.Year)
		case stdMonth:
			p = shortMonthNames[t.Month]
		case stdLongMonth:
			p = longMonthNames[t.Month]
		case stdNumMonth:
			p = strconv.Itoa(t.Month)
		case stdZeroMonth:
			p = zeroPad(t.Month)
		case stdWeekDay:
			p = shortDayNames[t.Weekday]
		case stdLongWeekDay:
			p = longDayNames[t.Weekday]
		case stdDay:
			p = strconv.Itoa(t.Day)
		case stdUnderDay:
			p = pad(t.Day, " ")
		case stdZeroDay:
			p = zeroPad(t.Day)
		case stdHour:
			p = zeroPad(t.Hour)
		case stdHour12:
			p = strconv.Itoa(t.Hour % 12)
		case stdZeroHour12:
			p = zeroPad(t.Hour % 12)
		case stdMinute:
			p = strconv.Itoa(t.Minute)
		case stdZeroMinute:
			p = zeroPad(t.Minute)
		case stdSecond:
			p = strconv.Itoa(t.Second)
		case stdZeroSecond:
			p = zeroPad(t.Second)
		case stdISO8601TZ, stdNumTZ:
			// Ugly special case.  We cheat and take "Z" to mean "the time
			// zone as formatted for ISO 8601".
			if std == stdISO8601TZ && t.ZoneOffset == 0 {
				p = "Z"
				break
			}
			zone := t.ZoneOffset / 60 // convert to minutes
			if zone < 0 {
				p = "-"
				zone = -zone
			} else {
				p = "+"
			}
			p += zeroPad(zone / 60)
			p += zeroPad(zone % 60)
		case stdPM:
			if t.Hour >= 12 {
				p = "PM"
			} else {
				p = "AM"
			}
		case stdpm:
			if t.Hour >= 12 {
				p = "pm"
			} else {
				p = "am"
			}
		case stdTZ:
			if t.Zone != "" {
				p = t.Zone
			} else {
				// No time zone known for this time, but we must print one.
				// Use the -0700 format.
				zone := t.ZoneOffset / 60 // convert to minutes
				if zone < 0 {
					p = "-"
					zone = -zone
				} else {
					p = "+"
				}
				p += zeroPad(zone / 60)
				p += zeroPad(zone % 60)
			}
		}
		b.WriteString(p)
		layout = suffix
	}
	return b.String()
}

// String returns a Unix-style representation of the time value.
func (t *Time) String() string { return t.Format(UnixDate) }

var errBad = os.ErrorString("bad") // just a marker; not returned to user

// ParseError describes a problem parsing a time string.
type ParseError struct {
	Layout     string
	Value      string
	LayoutElem string
	ValueElem  string
	Message    string
}

// String is the string representation of a ParseError.
func (e *ParseError) String() string {
	if e.Message == "" {
		return "parsing time " +
			strconv.Quote(e.Value) + " as " +
			strconv.Quote(e.Layout) + ": cannot parse " +
			strconv.Quote(e.ValueElem) + " as " +
			strconv.Quote(e.LayoutElem)
	}
	return "parsing time " +
		strconv.Quote(e.Value) + e.Message
}

// getnum parses s[0:1] or s[0:2] (fixed forces the latter)
// as a decimal integer and returns the integer and the
// remainder of the string.
func getnum(s string, fixed bool) (int, string, os.Error) {
	if len(s) == 0 || s[0] < '0' || s[0] > '9' {
		return 0, s, errBad
	}
	if len(s) == 1 || s[1] < '0' || s[1] > '9' {
		if fixed {
			return 0, s, errBad
		}
		return int(s[0] - '0'), s[1:], nil
	}
	return int(s[0]-'0')*10 + int(s[1]-'0'), s[2:], nil
}

func cutspace(s string) string {
	for len(s) > 0 && s[0] == ' ' {
		s = s[1:]
	}
	return s
}

// skip removes the given prefix from value,
// treating runs of space characters as equivalent.
func skip(value, prefix string) (string, os.Error) {
	for len(prefix) > 0 {
		if prefix[0] == ' ' {
			if len(value) > 0 && value[0] != ' ' {
				return "", errBad
			}
			prefix = cutspace(prefix)
			value = cutspace(value)
			continue
		}
		if len(value) == 0 || value[0] != prefix[0] {
			return "", errBad
		}
		prefix = prefix[1:]
		value = value[1:]
	}
	return value, nil
}

// Parse parses a formatted string and returns the time value it represents.
// The layout defines the format by showing the representation of a standard
// time, which is then used to describe the string to be parsed.  Predefined
// layouts ANSIC, UnixDate, ISO8601 and others describe standard
// representations.
//
// Only those elements present in the value will be set in the returned time
// structure.  Also, if the input string represents an inconsistent time
// (such as having the wrong day of the week), the returned value will also
// be inconsistent.  In any case, the elements of the returned time will be
// sane: hours in 0..23, minutes in 0..59, day of month in 0..31, etc.
// Years must be in the range 0000..9999.
func Parse(alayout, avalue string) (*Time, os.Error) {
	var t Time
	rangeErrString := "" // set if a value is out of range
	pmSet := false       // do we need to add 12 to the hour?
	layout, value := alayout, avalue
	// Each iteration processes one std value.
	for {
		var err os.Error
		prefix, std, suffix := nextStdChunk(layout)
		value, err = skip(value, prefix)
		if err != nil {
			return nil, &ParseError{alayout, avalue, prefix, value, ""}
		}
		if len(std) == 0 {
			if len(value) != 0 {
				return nil, &ParseError{alayout, avalue, "", value, ": extra text: " + value}
			}
			break
		}
		layout = suffix
		var p string
		switch std {
		case stdYear:
			if len(value) < 2 {
				err = errBad
				break
			}
			p, value = value[0:2], value[2:]
			t.Year, err = strconv.Atoi64(p)
			if t.Year >= 69 { // Unix time starts Dec 31 1969 in some time zones
				t.Year += 1900
			} else {
				t.Year += 2000
			}
		case stdLongYear:
			if len(value) < 4 || value[0] < '0' || value[0] > '9' {
				err = errBad
				break
			}
			p, value = value[0:4], value[4:]
			t.Year, err = strconv.Atoi64(p)
		case stdMonth:
			t.Month, value, err = lookup(shortMonthNames, value)
		case stdLongMonth:
			t.Month, value, err = lookup(longMonthNames, value)
		case stdNumMonth, stdZeroMonth:
			t.Month, value, err = getnum(value, std == stdZeroMonth)
			if t.Month <= 0 || 12 < t.Month {
				rangeErrString = "month"
			}
		case stdWeekDay:
			t.Weekday, value, err = lookup(shortDayNames, value)
		case stdLongWeekDay:
			t.Weekday, value, err = lookup(longDayNames, value)
		case stdDay, stdUnderDay, stdZeroDay:
			if std == stdUnderDay && len(value) > 0 && value[0] == ' ' {
				value = value[1:]
			}
			t.Day, value, err = getnum(value, std == stdZeroDay)
			if t.Day < 0 || 31 < t.Day {
				// TODO: be more thorough in date check?
				rangeErrString = "day"
			}
		case stdHour:
			t.Hour, value, err = getnum(value, false)
			if t.Hour < 0 || 24 <= t.Hour {
				rangeErrString = "hour"
			}
		case stdHour12, stdZeroHour12:
			t.Hour, value, err = getnum(value, std == stdZeroHour12)
			if t.Hour < 0 || 12 < t.Hour {
				rangeErrString = "hour"
			}
		case stdMinute, stdZeroMinute:
			t.Minute, value, err = getnum(value, std == stdZeroMinute)
			if t.Minute < 0 || 60 <= t.Minute {
				rangeErrString = "minute"
			}
		case stdSecond, stdZeroSecond:
			t.Second, value, err = getnum(value, std == stdZeroSecond)
			if t.Second < 0 || 60 <= t.Second {
				rangeErrString = "second"
			}
		case stdISO8601TZ, stdNumTZ:
			if std == stdISO8601TZ && len(value) >= 1 && value[0] == 'Z' {
				value = value[1:]
				t.Zone = "UTC"
				break
			}
			if len(value) < 5 {
				err = errBad
				break
			}
			var sign string
			sign, p, value = value[0:1], value[1:5], value[5:]
			var hr, min int
			hr, err = strconv.Atoi(p[0:2])
			if err != nil {
				min, err = strconv.Atoi(p[2:4])
			}
			t.ZoneOffset = (hr*60 + min) * 60 // offset is in seconds
			switch sign[0] {
			case '+':
			case '-':
				t.ZoneOffset = -t.ZoneOffset
			default:
				err = errBad
			}
		case stdPM:
			if len(value) < 2 {
				err = errBad
				break
			}
			p, value = value[0:2], value[2:]
			if p == "PM" {
				pmSet = true
			} else if p != "AM" {
				err = errBad
			}
		case stdpm:
			if len(value) < 2 {
				err = errBad
				break
			}
			p, value = value[0:2], value[2:]
			if p == "pm" {
				pmSet = true
			} else if p != "am" {
				err = errBad
			}
		case stdTZ:
			// Does it look like a time zone?
			if len(value) >= 3 && value[0:3] == "UTC" {
				t.Zone, value = value[0:3], value[3:]
				break
			}

			if len(value) >= 3 && value[2] == 'T' {
				p, value = value[0:3], value[3:]
			} else if len(value) >= 4 && value[3] == 'T' {
				p, value = value[0:4], value[4:]
			} else {
				err = errBad
				break
			}
			for i := 0; i < len(p); i++ {
				if p[i] < 'A' || 'Z' < p[i] {
					err = errBad
				}
			}
			if err != nil {
				break
			}
			// It's a valid format.
			t.Zone = p
			// Can we find it in the table?
			once.Do(setupZone)
			for _, z := range zones {
				if p == z.zone.name {
					t.ZoneOffset = z.zone.utcoff
					break
				}
			}
		}
		if rangeErrString != "" {
			return nil, &ParseError{alayout, avalue, std, value, ": " + rangeErrString + " out of range"}
		}
		if err != nil {
			return nil, &ParseError{alayout, avalue, std, value, ""}
		}
	}
	if pmSet && t.Hour < 12 {
		t.Hour += 12
	}
	return &t, nil
}
