// Copyright 2009 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// +build darwin freebsd linux netbsd openbsd

// Parse "zoneinfo" time zone file.
// This is a fairly standard file format used on OS X, Linux, BSD, Sun, and others.
// See tzfile(5), http://en.wikipedia.org/wiki/Zoneinfo,
// and ftp://munnari.oz.au/pub/oldtz/

package time

import (
	"errors"
	"syscall"
)

const (
	headerSize = 4 + 16 + 4*7
)

// Simple I/O interface to binary blob of data.
type data struct {
	p     []byte
	error bool
}

func (d *data) read(n int) []byte {
	if len(d.p) < n {
		d.p = nil
		d.error = true
		return nil
	}
	p := d.p[0:n]
	d.p = d.p[n:]
	return p
}

func (d *data) big4() (n uint32, ok bool) {
	p := d.read(4)
	if len(p) < 4 {
		d.error = true
		return 0, false
	}
	return uint32(p[0])<<24 | uint32(p[1])<<16 | uint32(p[2])<<8 | uint32(p[3]), true
}

func (d *data) byte() (n byte, ok bool) {
	p := d.read(1)
	if len(p) < 1 {
		d.error = true
		return 0, false
	}
	return p[0], true
}

// Make a string by stopping at the first NUL
func byteString(p []byte) string {
	for i := 0; i < len(p); i++ {
		if p[i] == 0 {
			return string(p[0:i])
		}
	}
	return string(p)
}

var badData = errors.New("malformed time zone information")

func loadZoneData(bytes []byte) (l *Location, err error) {
	d := data{bytes, false}

	// 4-byte magic "TZif"
	if magic := d.read(4); string(magic) != "TZif" {
		return nil, badData
	}

	// 1-byte version, then 15 bytes of padding
	var p []byte
	if p = d.read(16); len(p) != 16 || p[0] != 0 && p[0] != '2' {
		return nil, badData
	}

	// six big-endian 32-bit integers:
	//	number of UTC/local indicators
	//	number of standard/wall indicators
	//	number of leap seconds
	//	number of transition times
	//	number of local time zones
	//	number of characters of time zone abbrev strings
	const (
		NUTCLocal = iota
		NStdWall
		NLeap
		NTime
		NZone
		NChar
	)
	var n [6]int
	for i := 0; i < 6; i++ {
		nn, ok := d.big4()
		if !ok {
			return nil, badData
		}
		n[i] = int(nn)
	}

	// Transition times.
	txtimes := data{d.read(n[NTime] * 4), false}

	// Time zone indices for transition times.
	txzones := d.read(n[NTime])

	// Zone info structures
	zonedata := data{d.read(n[NZone] * 6), false}

	// Time zone abbreviations.
	abbrev := d.read(n[NChar])

	// Leap-second time pairs
	d.read(n[NLeap] * 8)

	// Whether tx times associated with local time types
	// are specified as standard time or wall time.
	isstd := d.read(n[NStdWall])

	// Whether tx times associated with local time types
	// are specified as UTC or local time.
	isutc := d.read(n[NUTCLocal])

	if d.error { // ran out of data
		return nil, badData
	}

	// If version == 2, the entire file repeats, this time using
	// 8-byte ints for txtimes and leap seconds.
	// We won't need those until 2106.

	// Now we can build up a useful data structure.
	// First the zone information.
	//	utcoff[4] isdst[1] nameindex[1]
	zone := make([]zone, n[NZone])
	for i := range zone {
		var ok bool
		var n uint32
		if n, ok = zonedata.big4(); !ok {
			return nil, badData
		}
		zone[i].offset = int(n)
		var b byte
		if b, ok = zonedata.byte(); !ok {
			return nil, badData
		}
		zone[i].isDST = b != 0
		if b, ok = zonedata.byte(); !ok || int(b) >= len(abbrev) {
			return nil, badData
		}
		zone[i].name = byteString(abbrev[b:])
	}

	// Now the transition time info.
	tx := make([]zoneTrans, n[NTime])
	for i := range tx {
		var ok bool
		var n uint32
		if n, ok = txtimes.big4(); !ok {
			return nil, badData
		}
		tx[i].when = int64(int32(n))
		if int(txzones[i]) >= len(zone) {
			return nil, badData
		}
		tx[i].index = txzones[i]
		if i < len(isstd) {
			tx[i].isstd = isstd[i] != 0
		}
		if i < len(isutc) {
			tx[i].isutc = isutc[i] != 0
		}
	}

	// Commited to succeed.
	l = &Location{zone: zone, tx: tx}

	// Fill in the cache with information about right now,
	// since that will be the most common lookup.
	sec, _ := now()
	for i := range tx {
		if tx[i].when <= sec && (i+1 == len(tx) || sec < tx[i+1].when) {
			l.cacheStart = tx[i].when
			l.cacheEnd = 1<<63 - 1
			if i+1 < len(tx) {
				l.cacheEnd = tx[i+1].when
			}
			l.cacheZone = &l.zone[tx[i].index]
		}
	}

	return l, nil
}

func loadZoneFile(name string) (l *Location, err error) {
	buf, err := readFile(name)
	if err != nil {
		return
	}
	return loadZoneData(buf)
}

func initTestingZone() {
	syscall.Setenv("TZ", "America/Los_Angeles")
	initLocal()
}

// Many systems use /usr/share/zoneinfo, Solaris 2 has
// /usr/share/lib/zoneinfo, IRIX 6 has /usr/lib/locale/TZ.
var zoneDirs = []string{
	"/usr/share/zoneinfo/",
	"/usr/share/lib/zoneinfo/",
	"/usr/lib/locale/TZ/",
}

func initLocal() {
	// consult $TZ to find the time zone to use.
	// no $TZ means use the system default /etc/localtime.
	// $TZ="" means use UTC.
	// $TZ="foo" means use /usr/share/zoneinfo/foo.

	tz, ok := syscall.Getenv("TZ")
	switch {
	case !ok:
		z, err := loadZoneFile("/etc/localtime")
		if err == nil {
			localLoc = *z
			localLoc.name = "Local"
			return
		}
	case tz != "" && tz != "UTC":
		if z, err := loadLocation(tz); err == nil {
			localLoc = *z
			return
		}
	}

	// Fall back to UTC.
	localLoc.name = "UTC"
}

func loadLocation(name string) (*Location, error) {
	for _, zoneDir := range zoneDirs {
		if z, err := loadZoneFile(zoneDir + name); err == nil {
			z.name = name
			return z, nil
		}
	}
	return nil, errors.New("unknown time zone " + name)
}
