// Copyright 2009 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package time

import (
	"errors"
	"runtime"
	"syscall"
)

// TODO(rsc): Fall back to copy of zoneinfo files.

// BUG(brainman,rsc): On Windows, the operating system does not provide complete
// time zone information.
// The implementation assumes that this year's rules for daylight savings
// time apply to all previous and future years as well. 
// Also, time zone abbreviations are unavailable.  The implementation constructs
// them using the capital letters from a longer time zone description.	

// abbrev returns the abbreviation to use for the given zone name.
func abbrev(name []uint16) string {
	// name is 'Pacific Standard Time' but we want 'PST'.
	// Extract just capital letters.  It's not perfect but the
	// information we need is not available from the kernel.
	// Because time zone abbreviations are not unique,
	// Windows refuses to expose them.
	//
	// http://social.msdn.microsoft.com/Forums/eu/vclanguage/thread/a87e1d25-fb71-4fe0-ae9c-a9578c9753eb
	// http://stackoverflow.com/questions/4195948/windows-time-zone-abbreviations-in-asp-net
	var short []rune
	for _, c := range name {
		if 'A' <= c && c <= 'Z' {
			short = append(short, rune(c))
		}
	}
	return string(short)
}

// pseudoUnix returns the pseudo-Unix time (seconds since Jan 1 1970 *LOCAL TIME*)
// denoted by the system date+time d in the given year.
// It is up to the caller to convert this local time into a UTC-based time.
func pseudoUnix(year int, d *syscall.Systemtime) int64 {
	// Windows specifies daylight savings information in "day in month" format:
	// d.Month is month number (1-12)
	// d.DayOfWeek is appropriate weekday (Sunday=0 to Saturday=6)
	// d.Day is week within the month (1 to 5, where 5 is last week of the month)
	// d.Hour, d.Minute and d.Second are absolute time
	day := 1
	t := Date(year, Month(d.Month), day, int(d.Hour), int(d.Minute), int(d.Second), 0, UTC)
	i := int(d.DayOfWeek) - int(t.Weekday())
	if i < 0 {
		i += 7
	}
	day += i
	if week := int(d.Day) - 1; week < 4 {
		day += week * 7
	} else {
		// "Last" instance of the day.
		day += 4 * 7
		if day > daysIn(Month(d.Month), year) {
			day -= 7
		}
	}
	return t.sec + int64(day-1)*secondsPerDay + internalToUnix
}

func initLocalFromTZI(i *syscall.Timezoneinformation) {
	l := &localLoc

	nzone := 1
	if i.StandardDate.Month > 0 {
		nzone++
	}
	l.zone = make([]zone, nzone)

	std := &l.zone[0]
	std.name = abbrev(i.StandardName[0:])
	if nzone == 1 {
		// No daylight savings.
		std.offset = -int(i.Bias) * 60
		l.cacheStart = -1 << 63
		l.cacheEnd = 1<<63 - 1
		l.cacheZone = std
		l.tx = make([]zoneTrans, 1)
		l.tx[0].when = l.cacheStart
		l.tx[0].index = 0
		return
	}

	// StandardBias must be ignored if StandardDate is not set,
	// so this computation is delayed until after the nzone==1
	// return above.
	std.offset = -int(i.Bias+i.StandardBias) * 60

	dst := &l.zone[1]
	dst.name = abbrev(i.DaylightName[0:])
	dst.offset = -int(i.Bias+i.DaylightBias) * 60
	dst.isDST = true

	// Arrange so that d0 is first transition date, d1 second,
	// i0 is index of zone after first transition, i1 second.
	d0 := &i.StandardDate
	d1 := &i.DaylightDate
	i0 := 0
	i1 := 1
	if d0.Month > d1.Month {
		d0, d1 = d1, d0
		i0, i1 = i1, i0
	}

	// 2 tx per year, 100 years on each side of this year
	l.tx = make([]zoneTrans, 400)

	t := Now().UTC()
	year := t.Year()
	txi := 0
	for y := year - 100; y < year+100; y++ {
		tx := &l.tx[txi]
		tx.when = pseudoUnix(y, d0) - int64(l.zone[i1].offset)
		tx.index = uint8(i0)
		txi++

		tx = &l.tx[txi]
		tx.when = pseudoUnix(y, d1) - int64(l.zone[i0].offset)
		tx.index = uint8(i1)
		txi++
	}
}

var usPacific = syscall.Timezoneinformation{
	Bias: 8 * 60,
	StandardName: [32]uint16{
		'P', 'a', 'c', 'i', 'f', 'i', 'c', ' ', 'S', 't', 'a', 'n', 'd', 'a', 'r', 'd', ' ', 'T', 'i', 'm', 'e',
	},
	StandardDate: syscall.Systemtime{Month: 11, Day: 1, Hour: 2},
	DaylightName: [32]uint16{
		'P', 'a', 'c', 'i', 'f', 'i', 'c', ' ', 'D', 'a', 'y', 'l', 'i', 'g', 'h', 't', ' ', 'T', 'i', 'm', 'e',
	},
	DaylightDate: syscall.Systemtime{Month: 3, Day: 2, Hour: 2},
	DaylightBias: -60,
}

func initTestingZone() {
	initLocalFromTZI(&usPacific)
}

func initLocal() {
	var i syscall.Timezoneinformation
	if _, err := syscall.GetTimeZoneInformation(&i); err != nil {
		localLoc.name = "UTC"
		return
	}
	initLocalFromTZI(&i)
}

func loadLocation(name string) (*Location, error) {
	if z, err := loadZoneFile(runtime.GOROOT()+`\lib\time\zoneinfo.zip`, name); err == nil {
		z.name = name
		return z, nil
	}
	return nil, errors.New("unknown time zone " + name)
}
