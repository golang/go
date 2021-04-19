// Copyright 2016 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// Parse the "tzdata" packed timezone file used on Android.
// The format is lifted from ZoneInfoDB.java and ZoneInfo.java in
// java/libcore/util in the AOSP.

package time

import (
	"errors"
	"runtime"
)

var tzdataPaths = []string{
	"/system/usr/share/zoneinfo/tzdata",
	"/data/misc/zoneinfo/current/tzdata",
	runtime.GOROOT() + "/lib/time/zoneinfo.zip",
}

var origTzdataPaths = tzdataPaths

func forceZipFileForTesting(zipOnly bool) {
	tzdataPaths = make([]string, len(origTzdataPaths))
	copy(tzdataPaths, origTzdataPaths)
	if zipOnly {
		for i := 0; i < len(tzdataPaths)-1; i++ {
			tzdataPaths[i] = "/XXXNOEXIST"
		}
	}
}

func initTestingZone() {
	z, err := loadLocation("America/Los_Angeles")
	if err != nil {
		panic("cannot load America/Los_Angeles for testing: " + err.Error())
	}
	z.name = "Local"
	localLoc = *z
}

func initLocal() {
	// TODO(elias.naur): getprop persist.sys.timezone
	localLoc = *UTC
}

func loadLocation(name string) (*Location, error) {
	var firstErr error
	for _, path := range tzdataPaths {
		var z *Location
		var err error
		if len(path) > 4 && path[len(path)-4:] == ".zip" {
			z, err = loadZoneZip(path, name)
		} else {
			z, err = loadTzdataFile(path, name)
		}
		if err == nil {
			z.name = name
			return z, nil
		} else if firstErr == nil && !isNotExist(err) {
			firstErr = err
		}
	}
	if firstErr != nil {
		return nil, firstErr
	}
	return nil, errors.New("unknown time zone " + name)
}

func loadTzdataFile(file, name string) (*Location, error) {
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
	d := data{buf, false}
	if magic := d.read(6); string(magic) != "tzdata" {
		return nil, errors.New("corrupt tzdata file " + file)
	}
	d = data{buf[12:], false}
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
		d := data{entry[namesize:], false}
		off, _ := d.big4()
		size, _ := d.big4()
		buf := make([]byte, size)
		if err := preadn(fd, buf, int(off+dataOff)); err != nil {
			return nil, errors.New("corrupt tzdata file " + file)
		}
		return loadZoneData(buf)
	}
	return nil, errors.New("cannot find " + name + " in tzdata file " + file)
}
