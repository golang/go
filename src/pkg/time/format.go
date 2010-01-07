package time

import (
	"bytes"
	"strconv"
)

const (
	numeric = iota
	alphabetic
	separator
)

// These are predefined layouts for use in Time.Format.
// The standard time used in the layouts is:
//	Mon Jan  2 15:04:05 MST 2006  (MST is GMT-0700)
// which is Unix time 1136243045.
// (Think of it as 01/02 03:04:05PM '06 -0700.)
const (
	ANSIC    = "Mon Jan  2 15:04:05 2006"
	UnixDate = "Mon Jan  2 15:04:05 MST 2006"
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

func charType(c uint8) int {
	switch {
	case '0' <= c && c <= '9':
		return numeric
	case 'a' <= c && c < 'z', 'A' <= c && c <= 'Z':
		return alphabetic
	}
	return separator
}

func zeroPad(i int) string {
	s := strconv.Itoa(i)
	if i < 10 {
		s = "0" + s
	}
	return s
}

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
