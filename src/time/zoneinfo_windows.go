// Copyright 2009 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package time

import (
	"errors"
	"internal/syscall/windows/registry"
	"syscall"
)

var platformZoneSources []string // none: Windows uses system calls instead

// TODO(rsc): Fall back to copy of zoneinfo files.

// BUG(brainman,rsc): On Windows, the operating system does not provide complete
// time zone information.
// The implementation assumes that this year's rules for daylight savings
// time apply to all previous and future years as well.

// matchZoneKey checks if stdname and dstname match the corresponding key
// values "MUI_Std" and MUI_Dlt" or "Std" and "Dlt" (the latter down-level
// from Vista) in the kname key stored under the open registry key zones.
func matchZoneKey(zones registry.Key, kname string, stdname, dstname string) (matched bool, err2 error) {
	k, err := registry.OpenKey(zones, kname, registry.READ)
	if err != nil {
		return false, err
	}
	defer k.Close()

	var std, dlt string
	if err = registry.LoadRegLoadMUIString(); err == nil {
		// Try MUI_Std and MUI_Dlt first, fallback to Std and Dlt if *any* error occurs
		std, err = k.GetMUIStringValue("MUI_Std")
		if err == nil {
			dlt, err = k.GetMUIStringValue("MUI_Dlt")
		}
	}
	if err != nil { // Fallback to Std and Dlt
		if std, _, err = k.GetStringValue("Std"); err != nil {
			return false, err
		}
		if dlt, _, err = k.GetStringValue("Dlt"); err != nil {
			return false, err
		}
	}

	if std != stdname {
		return false, nil
	}
	if dlt != dstname && dstname != stdname {
		return false, nil
	}
	return true, nil
}

// toEnglishName searches the registry for an English name of a time zone
// whose zone names are stdname and dstname and returns the English name.
func toEnglishName(stdname, dstname string) (string, error) {
	k, err := registry.OpenKey(registry.LOCAL_MACHINE, `SOFTWARE\Microsoft\Windows NT\CurrentVersion\Time Zones`, registry.ENUMERATE_SUB_KEYS|registry.QUERY_VALUE)
	if err != nil {
		return "", err
	}
	defer k.Close()

	names, err := k.ReadSubKeyNames()
	if err != nil {
		return "", err
	}
	for _, name := range names {
		matched, err := matchZoneKey(k, name, stdname, dstname)
		if err == nil && matched {
			return name, nil
		}
	}
	return "", errors.New(`English name for time zone "` + stdname + `" not found in registry`)
}

// extractCAPS extracts capital letters from description desc.
func extractCAPS(desc string) string {
	var short []rune
	for _, c := range desc {
		if 'A' <= c && c <= 'Z' {
			short = append(short, c)
		}
	}
	return string(short)
}

// abbrev returns the abbreviations to use for the given zone z.
func abbrev(z *syscall.Timezoneinformation) (std, dst string) {
	stdName := syscall.UTF16ToString(z.StandardName[:])
	a, ok := abbrs[stdName]
	if !ok {
		dstName := syscall.UTF16ToString(z.DaylightName[:])
		// Perhaps stdName is not English. Try to convert it.
		englishName, err := toEnglishName(stdName, dstName)
		if err == nil {
			a, ok = abbrs[englishName]
			if ok {
				return a.std, a.dst
			}
		}
		// fallback to using capital letters
		return extractCAPS(stdName), extractCAPS(dstName)
	}
	return a.std, a.dst
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
	return t.sec() + int64(day-1)*secondsPerDay + internalToUnix
}

func initLocalFromTZI(i *syscall.Timezoneinformation) {
	l := &localLoc

	l.name = "Local"

	nzone := 1
	if i.StandardDate.Month > 0 {
		nzone++
	}
	l.zone = make([]zone, nzone)

	stdname, dstname := abbrev(i)

	std := &l.zone[0]
	std.name = stdname
	if nzone == 1 {
		// No daylight savings.
		std.offset = -int(i.Bias) * 60
		l.cacheStart = alpha
		l.cacheEnd = omega
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
	dst.name = dstname
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

var aus = syscall.Timezoneinformation{
	Bias: -10 * 60,
	StandardName: [32]uint16{
		'A', 'U', 'S', ' ', 'E', 'a', 's', 't', 'e', 'r', 'n', ' ', 'S', 't', 'a', 'n', 'd', 'a', 'r', 'd', ' ', 'T', 'i', 'm', 'e',
	},
	StandardDate: syscall.Systemtime{Month: 4, Day: 1, Hour: 3},
	DaylightName: [32]uint16{
		'A', 'U', 'S', ' ', 'E', 'a', 's', 't', 'e', 'r', 'n', ' ', 'D', 'a', 'y', 'l', 'i', 'g', 'h', 't', ' ', 'T', 'i', 'm', 'e',
	},
	DaylightDate: syscall.Systemtime{Month: 10, Day: 1, Hour: 2},
	DaylightBias: -60,
}

func initLocal() {
	var i syscall.Timezoneinformation
	if _, err := syscall.GetTimeZoneInformation(&i); err != nil {
		localLoc.name = "UTC"
		return
	}
	initLocalFromTZI(&i)
}
