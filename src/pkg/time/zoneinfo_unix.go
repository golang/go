// Copyright 2009 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// +build darwin freebsd linux openbsd

// Parse "zoneinfo" time zone file.
// This is a fairly standard file format used on OS X, Linux, BSD, Sun, and others.
// See tzfile(5), http://en.wikipedia.org/wiki/Zoneinfo,
// and ftp://munnari.oz.au/pub/oldtz/

package time

import (
	"bytes"
	"os"
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

func parseinfo(bytes []byte) (zt []zonetime, ok bool) {
	d := data{bytes, false}

	// 4-byte magic "TZif"
	if magic := d.read(4); string(magic) != "TZif" {
		return nil, false
	}

	// 1-byte version, then 15 bytes of padding
	var p []byte
	if p = d.read(16); len(p) != 16 || p[0] != 0 && p[0] != '2' {
		return nil, false
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
			return nil, false
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
		return nil, false
	}

	// If version == 2, the entire file repeats, this time using
	// 8-byte ints for txtimes and leap seconds.
	// We won't need those until 2106.

	// Now we can build up a useful data structure.
	// First the zone information.
	//	utcoff[4] isdst[1] nameindex[1]
	z := make([]zone, n[NZone])
	for i := 0; i < len(z); i++ {
		var ok bool
		var n uint32
		if n, ok = zonedata.big4(); !ok {
			return nil, false
		}
		z[i].utcoff = int(n)
		var b byte
		if b, ok = zonedata.byte(); !ok {
			return nil, false
		}
		z[i].isdst = b != 0
		if b, ok = zonedata.byte(); !ok || int(b) >= len(abbrev) {
			return nil, false
		}
		z[i].name = byteString(abbrev[b:])
	}

	// Now the transition time info.
	zt = make([]zonetime, n[NTime])
	for i := 0; i < len(zt); i++ {
		var ok bool
		var n uint32
		if n, ok = txtimes.big4(); !ok {
			return nil, false
		}
		zt[i].time = int32(n)
		if int(txzones[i]) >= len(z) {
			return nil, false
		}
		zt[i].zone = &z[txzones[i]]
		if i < len(isstd) {
			zt[i].isstd = isstd[i] != 0
		}
		if i < len(isutc) {
			zt[i].isutc = isutc[i] != 0
		}
	}
	return zt, true
}

func readinfofile(name string) ([]zonetime, bool) {
	var b bytes.Buffer

	f, err := os.Open(name)
	if err != nil {
		return nil, false
	}
	defer f.Close()
	if _, err := b.ReadFrom(f); err != nil {
		return nil, false
	}
	return parseinfo(b.Bytes())
}

func setupTestingZone() {
	os.Setenv("TZ", "America/Los_Angeles")
	setupZone()
}

func setupZone() {
	// consult $TZ to find the time zone to use.
	// no $TZ means use the system default /etc/localtime.
	// $TZ="" means use UTC.
	// $TZ="foo" means use /usr/share/zoneinfo/foo.
	// Many systems use /usr/share/zoneinfo, Solaris 2 has
	// /usr/share/lib/zoneinfo, IRIX 6 has /usr/lib/locale/TZ.
	zoneDirs := []string{"/usr/share/zoneinfo/",
		"/usr/share/lib/zoneinfo/",
		"/usr/lib/locale/TZ/"}

	tz, err := os.Getenverror("TZ")
	switch {
	case err == os.ENOENV:
		zones, _ = readinfofile("/etc/localtime")
	case len(tz) > 0:
		for _, zoneDir := range zoneDirs {
			var ok bool
			if zones, ok = readinfofile(zoneDir + tz); ok {
				break
			}
		}
	case len(tz) == 0:
		// do nothing: use UTC
	}
}
