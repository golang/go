// Copyright 2009 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package time

import (
	"errors"
	"runtime"
	"syscall"
	"unsafe"
)

//go:generate go run genzabbrs.go -output zoneinfo_abbrs_windows.go

// TODO(rsc): Fall back to copy of zoneinfo files.

// BUG(brainman,rsc): On Windows, the operating system does not provide complete
// time zone information.
// The implementation assumes that this year's rules for daylight savings
// time apply to all previous and future years as well.

// getKeyValue retrieves the string value kname associated with the open registry key kh.
func getKeyValue(kh syscall.Handle, kname string) (string, error) {
	var buf [50]uint16 // buf needs to be large enough to fit zone descriptions
	var typ uint32
	n := uint32(len(buf) * 2) // RegQueryValueEx's signature expects array of bytes, not uint16
	p, _ := syscall.UTF16PtrFromString(kname)
	if err := syscall.RegQueryValueEx(kh, p, nil, &typ, (*byte)(unsafe.Pointer(&buf[0])), &n); err != nil {
		return "", err
	}
	if typ != syscall.REG_SZ { // null terminated strings only
		return "", errors.New("Key is not string")
	}
	return syscall.UTF16ToString(buf[:]), nil
}

// matchZoneKey checks if stdname and dstname match the corresponding "Std"
// and "Dlt" key values in the kname key stored under the open registry key zones.
func matchZoneKey(zones syscall.Handle, kname string, stdname, dstname string) (matched bool, err2 error) {
	var h syscall.Handle
	p, _ := syscall.UTF16PtrFromString(kname)
	if err := syscall.RegOpenKeyEx(zones, p, 0, syscall.KEY_READ, &h); err != nil {
		return false, err
	}
	defer syscall.RegCloseKey(h)

	s, err := getKeyValue(h, "Std")
	if err != nil {
		return false, err
	}
	if s != stdname {
		return false, nil
	}
	s, err = getKeyValue(h, "Dlt")
	if err != nil {
		return false, err
	}
	if s != dstname && dstname != stdname {
		return false, nil
	}
	return true, nil
}

// toEnglishName searches the registry for an English name of a time zone
// whose zone names are stdname and dstname and returns the English name.
func toEnglishName(stdname, dstname string) (string, error) {
	var zones syscall.Handle
	p, _ := syscall.UTF16PtrFromString(`SOFTWARE\Microsoft\Windows NT\CurrentVersion\Time Zones`)
	if err := syscall.RegOpenKeyEx(syscall.HKEY_LOCAL_MACHINE, p, 0, syscall.KEY_READ, &zones); err != nil {
		return "", err
	}
	defer syscall.RegCloseKey(zones)

	var count uint32
	if err := syscall.RegQueryInfoKey(zones, nil, nil, nil, &count, nil, nil, nil, nil, nil, nil, nil); err != nil {
		return "", err
	}

	var buf [50]uint16 // buf needs to be large enough to fit zone descriptions
	for i := uint32(0); i < count; i++ {
		n := uint32(len(buf))
		if syscall.RegEnumKeyEx(zones, i, &buf[0], &n, nil, nil, nil, nil) != nil {
			continue
		}
		kname := syscall.UTF16ToString(buf[:])
		matched, err := matchZoneKey(zones, kname, stdname, dstname)
		if err == nil && matched {
			return kname, nil
		}
	}
	return "", errors.New(`English name for time zone "` + stdname + `" not found in registry`)
}

// extractCAPS extracts capital letters from description desc.
func extractCAPS(desc string) string {
	var short []rune
	for _, c := range desc {
		if 'A' <= c && c <= 'Z' {
			short = append(short, rune(c))
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
	return t.sec + int64(day-1)*secondsPerDay + internalToUnix
}

func initLocalFromTZI(i *syscall.Timezoneinformation) {
	l := &localLoc

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

func initTestingZone() {
	initLocalFromTZI(&usPacific)
}

func initAusTestingZone() {
	initLocalFromTZI(&aus)
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

func forceZipFileForTesting(zipOnly bool) {
	// We only use the zip file anyway.
}
