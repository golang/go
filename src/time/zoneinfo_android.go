// Copyright 2016 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// Parse the "tzdata" packed timezone file used on Android.
// The format is lifted from ZoneInfoDB.java and ZoneInfo.java in
// java/libcore/util in the AOSP.

package time

import (
	"errors"
	"syscall"
)

var platformZoneSources = []string{
	"/system/usr/share/zoneinfo/tzdata",
	"/data/misc/zoneinfo/current/tzdata",
}

func initLocal() {
	// TODO(elias.naur): getprop persist.sys.timezone
	localLoc = *UTC
}

func init() {
	loadTzinfoFromTzdata = androidLoadTzinfoFromTzdata
}

var allowGorootSource = true

func gorootZoneSource(goroot string) (string, bool) {
	if goroot == "" || !allowGorootSource {
		return "", false
	}
	return goroot + "/lib/time/zoneinfo.zip", true
}

func androidLoadTzinfoFromTzdata(file, name string) ([]byte, error) {
	const (
		headersize = 12 + 3*4
		namesize   = 40
		entrysize  = namesize + 3*4
	)
	if len(name) > namesize {
		return nil, errors.New(name + " is longer than the maximum zone name length (40 bytes)")
	}
	fd, err := open(file)
	if err != nil {
		return nil, err
	}
	defer closefd(fd)

	buf := make([]byte, headersize)
	if err := preadn(fd, buf, 0); err != nil {
		return nil, errors.New("corrupt tzdata file " + file)
	}
	d := dataIO{buf, false}
	if magic := d.read(6); string(magic) != "tzdata" {
		return nil, errors.New("corrupt tzdata file " + file)
	}
	d = dataIO{buf[12:], false}
	indexOff, _ := d.big4()
	dataOff, _ := d.big4()
	indexSize := dataOff - indexOff
	entrycount := indexSize / entrysize
	buf = make([]byte, indexSize)
	if err := preadn(fd, buf, int(indexOff)); err != nil {
		return nil, errors.New("corrupt tzdata file " + file)
	}
	for i := 0; i < int(entrycount); i++ {
		entry := buf[i*entrysize : (i+1)*entrysize]
		// len(name) <= namesize is checked at function entry
		if string(entry[:len(name)]) != name {
			continue
		}
		d := dataIO{entry[namesize:], false}
		off, _ := d.big4()
		size, _ := d.big4()
		buf := make([]byte, size)
		if err := preadn(fd, buf, int(off+dataOff)); err != nil {
			return nil, errors.New("corrupt tzdata file " + file)
		}
		return buf, nil
	}
	return nil, syscall.ENOENT
}
