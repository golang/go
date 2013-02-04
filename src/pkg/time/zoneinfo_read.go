// Copyright 2009 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// Parse "zoneinfo" time zone file.
// This is a fairly standard file format used on OS X, Linux, BSD, Sun, and others.
// See tzfile(5), http://en.wikipedia.org/wiki/Zoneinfo,
// and ftp://munnari.oz.au/pub/oldtz/

package time

import "errors"

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
		zone[i].offset = int(int32(n))
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

	if len(tx) == 0 {
		// Build fake transition to cover all time.
		// This happens in fixed locations like "Etc/GMT0".
		tx = append(tx, zoneTrans{when: -1 << 63, index: 0})
	}

	// Committed to succeed.
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

func loadZoneFile(dir, name string) (l *Location, err error) {
	if len(dir) > 4 && dir[len(dir)-4:] == ".zip" {
		return loadZoneZip(dir, name)
	}
	if dir != "" {
		name = dir + "/" + name
	}
	buf, err := readFile(name)
	if err != nil {
		return
	}
	return loadZoneData(buf)
}

// There are 500+ zoneinfo files.  Rather than distribute them all
// individually, we ship them in an uncompressed zip file.
// Used this way, the zip file format serves as a commonly readable
// container for the individual small files.  We choose zip over tar
// because zip files have a contiguous table of contents, making
// individual file lookups faster, and because the per-file overhead
// in a zip file is considerably less than tar's 512 bytes.

// get4 returns the little-endian 32-bit value in b.
func get4(b []byte) int {
	if len(b) < 4 {
		return 0
	}
	return int(b[0]) | int(b[1])<<8 | int(b[2])<<16 | int(b[3])<<24
}

// get2 returns the little-endian 16-bit value in b.
func get2(b []byte) int {
	if len(b) < 2 {
		return 0
	}
	return int(b[0]) | int(b[1])<<8
}

func loadZoneZip(zipfile, name string) (l *Location, err error) {
	fd, err := open(zipfile)
	if err != nil {
		return nil, errors.New("open " + zipfile + ": " + err.Error())
	}
	defer closefd(fd)

	const (
		zecheader = 0x06054b50
		zcheader  = 0x02014b50
		ztailsize = 22

		zheadersize = 30
		zheader     = 0x04034b50
	)

	buf := make([]byte, ztailsize)
	if err := preadn(fd, buf, -ztailsize); err != nil || get4(buf) != zecheader {
		return nil, errors.New("corrupt zip file " + zipfile)
	}
	n := get2(buf[10:])
	size := get4(buf[12:])
	off := get4(buf[16:])

	buf = make([]byte, size)
	if err := preadn(fd, buf, off); err != nil {
		return nil, errors.New("corrupt zip file " + zipfile)
	}

	for i := 0; i < n; i++ {
		// zip entry layout:
		//	0	magic[4]
		//	4	madevers[1]
		//	5	madeos[1]
		//	6	extvers[1]
		//	7	extos[1]
		//	8	flags[2]
		//	10	meth[2]
		//	12	modtime[2]
		//	14	moddate[2]
		//	16	crc[4]
		//	20	csize[4]
		//	24	uncsize[4]
		//	28	namelen[2]
		//	30	xlen[2]
		//	32	fclen[2]
		//	34	disknum[2]
		//	36	iattr[2]
		//	38	eattr[4]
		//	42	off[4]
		//	46	name[namelen]
		//	46+namelen+xlen+fclen - next header
		//
		if get4(buf) != zcheader {
			break
		}
		meth := get2(buf[10:])
		size := get4(buf[24:])
		namelen := get2(buf[28:])
		xlen := get2(buf[30:])
		fclen := get2(buf[32:])
		off := get4(buf[42:])
		zname := buf[46 : 46+namelen]
		buf = buf[46+namelen+xlen+fclen:]
		if string(zname) != name {
			continue
		}
		if meth != 0 {
			return nil, errors.New("unsupported compression for " + name + " in " + zipfile)
		}

		// zip per-file header layout:
		//	0	magic[4]
		//	4	extvers[1]
		//	5	extos[1]
		//	6	flags[2]
		//	8	meth[2]
		//	10	modtime[2]
		//	12	moddate[2]
		//	14	crc[4]
		//	18	csize[4]
		//	22	uncsize[4]
		//	26	namelen[2]
		//	28	xlen[2]
		//	30	name[namelen]
		//	30+namelen+xlen - file data
		//
		buf = make([]byte, zheadersize+namelen)
		if err := preadn(fd, buf, off); err != nil ||
			get4(buf) != zheader ||
			get2(buf[8:]) != meth ||
			get2(buf[26:]) != namelen ||
			string(buf[30:30+namelen]) != name {
			return nil, errors.New("corrupt zip file " + zipfile)
		}
		xlen = get2(buf[28:])

		buf = make([]byte, size)
		if err := preadn(fd, buf, off+30+namelen+xlen); err != nil {
			return nil, errors.New("corrupt zip file " + zipfile)
		}

		return loadZoneData(buf)
	}

	return nil, errors.New("cannot find " + name + " in zip file " + zipfile)
}
