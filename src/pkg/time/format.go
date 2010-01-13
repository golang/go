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
	RFC850   = "Monday, 02-Jan-06 15:04:05 MST"
	RFC1123  = "Mon, 02 Jan 2006 15:04:05 MST"
	Kitchen  = "3:04PM"
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
	stdZulu        = "1504"
	stdPM          = "PM"
	stdpm          = "pm"
	stdTZ          = "MST"
	stdISO8601TZ   = "Z"
)

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

func lookup(tab []string, val string) (int, os.Error) {
	for i, v := range tab {
		if v == val {
			return i, nil
		}
	}
	return -1, errBad
}

func charType(c uint8) int {
	switch {
	case '0' <= c && c <= '9':
		return numeric
	case c == '_': // underscore; treated like a number when printing
		return numeric
	case 'a' <= c && c < 'z', 'A' <= c && c <= 'Z':
		return alphabetic
	}
	return separator
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
	// Each iteration generates one piece
	for len(layout) > 0 {
		c := layout[0]
		pieceType := charType(c)
		i := 0
		for i < len(layout) && charType(layout[i]) == pieceType {
			i++
		}
		p := layout[0:i]
		layout = layout[i:]
		switch p {
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
		case stdZulu:
			p = zeroPad(t.Hour) + zeroPad(t.Minute)
		case stdISO8601TZ:
			// Rather ugly special case, required because the time zone is too broken down
			// in this format to recognize easily.  We cheat and take "Z" to mean "the time
			// zone as formatted for ISO 8601".
			if t.ZoneOffset == 0 {
				p = "Z"
			} else {
				zone := t.ZoneOffset / 60 // minutes
				if zone < 0 {
					p = "-"
					zone = -zone
				} else {
					p = "+"
				}
				p += zeroPad(zone / 60)
				p += zeroPad(zone % 60)
			}
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
			p = t.Zone
		}
		b.WriteString(p)
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

// To simplify comparison, collapse an initial run of spaces into a single space.
func collapseSpaces(s string) string {
	if len(s) <= 1 || s[0] != ' ' {
		return s
	}
	var i int
	for i = 1; i < len(s); i++ {
		if s[i] != ' ' {
			return s[i-1:]
		}
	}
	return " "
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
func Parse(alayout, avalue string) (*Time, os.Error) {
	var t Time
	const formatErr = ": different format from "
	rangeErrString := "" // set if a value is out of range
	pmSet := false       // do we need to add 12 to the hour?
	// Each iteration steps along one piece
	nextIsYear := false // whether next item is a Year; means we saw a minus sign.
	layout, value := alayout, avalue
	for len(layout) > 0 && len(value) > 0 {
		c := layout[0]
		pieceType := charType(c)
		var i int
		for i = 0; i < len(layout) && charType(layout[i]) == pieceType; i++ {
		}
		reference := layout[0:i]
		layout = layout[i:]
		if reference == "Z" {
			// Special case for ISO8601 time zone: "Z" or "-0800"
			if value[0] == 'Z' {
				i = 1
			} else if len(value) >= 5 {
				i = 5
			} else {
				return nil, &ParseError{Layout: alayout, Value: avalue, Message: formatErr + alayout}
			}
		} else {
			c = value[0]
			if charType(c) != pieceType {
				return nil, &ParseError{Layout: alayout, Value: avalue, Message: formatErr + alayout}
			}
			for i = 0; i < len(value) && charType(value[i]) == pieceType; i++ {
			}
		}
		p := value[0:i]
		value = value[i:]
		// Separators must match but:
		// - initial run of spaces is treated as a single space
		// - there could be a following minus sign for negative years
		if pieceType == separator {
			if len(p) != len(reference) {
				// must be exactly a following minus sign
				pp := collapseSpaces(p)
				rr := collapseSpaces(reference)
				if pp != rr {
					if len(pp) != len(rr)+1 || p[len(pp)-1] != '-' {
						return nil, &ParseError{Layout: alayout, Value: avalue, Message: formatErr + alayout}
					}
					nextIsYear = true
					continue
				}
			}
		}
		var err os.Error
		switch reference {
		case stdYear:
			t.Year, err = strconv.Atoi64(p)
			if t.Year >= 69 { // Unix time starts Dec 31 1969 in some time zones
				t.Year += 1900
			} else {
				t.Year += 2000
			}
		case stdLongYear:
			t.Year, err = strconv.Atoi64(p)
			if nextIsYear {
				t.Year = -t.Year
				nextIsYear = false
			}
		case stdMonth:
			t.Month, err = lookup(shortMonthNames, p)
		case stdLongMonth:
			t.Month, err = lookup(longMonthNames, p)
		case stdNumMonth, stdZeroMonth:
			t.Month, err = strconv.Atoi(p)
			if t.Month <= 0 || 12 < t.Month {
				rangeErrString = "month"
			}
		case stdWeekDay:
			t.Weekday, err = lookup(shortDayNames, p)
		case stdLongWeekDay:
			t.Weekday, err = lookup(longDayNames, p)
		case stdDay, stdUnderDay, stdZeroDay:
			t.Day, err = strconv.Atoi(p)
			if t.Day < 0 || 31 < t.Day {
				// TODO: be more thorough in date check?
				rangeErrString = "day"
			}
		case stdHour:
			t.Hour, err = strconv.Atoi(p)
			if t.Hour < 0 || 24 <= t.Hour {
				rangeErrString = "hour"
			}
		case stdHour12, stdZeroHour12:
			t.Hour, err = strconv.Atoi(p)
			if t.Hour < 0 || 12 < t.Hour {
				rangeErrString = "hour"
			}
		case stdMinute, stdZeroMinute:
			t.Minute, err = strconv.Atoi(p)
			if t.Minute < 0 || 60 <= t.Minute {
				rangeErrString = "minute"
			}
		case stdSecond, stdZeroSecond:
			t.Second, err = strconv.Atoi(p)
			if t.Second < 0 || 60 <= t.Second {
				rangeErrString = "second"
			}
		case stdZulu:
			if len(p) != 4 {
				err = os.ErrorString("HHMM value must be 4 digits")
				break
			}
			t.Hour, err = strconv.Atoi(p[0:2])
			if err != nil {
				t.Minute, err = strconv.Atoi(p[2:4])
			}
		case stdISO8601TZ:
			if p == "Z" {
				t.Zone = "UTC"
				break
			}
			// len(p) known to be 5: "-0800"
			var hr, min int
			hr, err = strconv.Atoi(p[1:3])
			if err != nil {
				min, err = strconv.Atoi(p[3:5])
			}
			t.ZoneOffset = (hr*60 + min) * 60 // offset is in seconds
			switch p[0] {
			case '+':
			case '-':
				t.ZoneOffset = -t.ZoneOffset
			default:
				err = errBad
			}
		case stdPM:
			if p == "PM" {
				pmSet = true
			} else if p != "AM" {
				err = errBad
			}
		case stdpm:
			if p == "pm" {
				pmSet = true
			} else if p != "am" {
				err = errBad
			}
		case stdTZ:
			// Does it look like a time zone?
			if p == "UTC" {
				t.Zone = p
				break
			}
			// All other time zones look like XXT or XXXT.
			if len(p) != 3 && len(p) != 4 || p[len(p)-1] != 'T' {
				err = errBad
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
		if nextIsYear {
			// Means we didn't see a year when we were expecting one
			return nil, &ParseError{Layout: alayout, Value: value, Message: formatErr + alayout}
		}
		if rangeErrString != "" {
			return nil, &ParseError{alayout, avalue, reference, p, ": " + rangeErrString + " out of range"}
		}
		if err != nil {
			return nil, &ParseError{alayout, avalue, reference, p, ""}
		}
	}
	if pmSet && t.Hour < 12 {
		t.Hour += 12
	}
	return &t, nil
}
