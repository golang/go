// Copyright 2011 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package time

import (
	"sync"
	"syscall"
)

// A Location maps time instants to the zone in use at that time.
// Typically, the Location represents the collection of time offsets
// in use in a geographical area, such as CEST and CET for central Europe.
type Location struct {
	name string
	zone []zone
	tx   []zoneTrans

	// Most lookups will be for the current time.
	// To avoid the binary search through tx, keep a
	// static one-element cache that gives the correct
	// zone for the time when the Location was created.
	// if cacheStart <= t <= cacheEnd,
	// lookup can return cacheZone.
	// The units for cacheStart and cacheEnd are seconds
	// since January 1, 1970 UTC, to match the argument
	// to lookup.
	cacheStart int64
	cacheEnd   int64
	cacheZone  *zone
}

// A zone represents a single time zone such as CEST or CET.
type zone struct {
	name   string // abbreviated name, "CET"
	offset int    // seconds east of UTC
	isDST  bool   // is this zone Daylight Savings Time?
}

// A zoneTrans represents a single time zone transition.
type zoneTrans struct {
	when         int64 // transition time, in seconds since 1970 GMT
	index        uint8 // the index of the zone that goes into effect at that time
	isstd, isutc bool  // ignored - no idea what these mean
}

// UTC represents Universal Coordinated Time (UTC).
var UTC *Location = &utcLoc

// utcLoc is separate so that get can refer to &utcLoc
// and ensure that it never returns a nil *Location,
// even if a badly behaved client has changed UTC.
var utcLoc = Location{name: "UTC"}

// Local represents the system's local time zone.
var Local *Location = &localLoc

// localLoc is separate so that initLocal can initialize
// it even if a client has changed Local.
var localLoc Location
var localOnce sync.Once

func (l *Location) get() *Location {
	if l == nil {
		return &utcLoc
	}
	if l == &localLoc {
		localOnce.Do(initLocal)
	}
	return l
}

// String returns a descriptive name for the time zone information,
// corresponding to the argument to LoadLocation.
func (l *Location) String() string {
	return l.get().name
}

// FixedZone returns a Location that always uses
// the given zone name and offset (seconds east of UTC).
func FixedZone(name string, offset int) *Location {
	l := &Location{
		name:       name,
		zone:       []zone{{name, offset, false}},
		tx:         []zoneTrans{{-1 << 63, 0, false, false}},
		cacheStart: -1 << 63,
		cacheEnd:   1<<63 - 1,
	}
	l.cacheZone = &l.zone[0]
	return l
}

// lookup returns information about the time zone in use at an
// instant in time expressed as seconds since January 1, 1970 00:00:00 UTC.
//
// The returned information gives the name of the zone (such as "CET"),
// the start and end times bracketing sec when that zone is in effect,
// the offset in seconds east of UTC (such as -5*60*60), and whether
// the daylight savings is being observed at that time.
func (l *Location) lookup(sec int64) (name string, offset int, isDST bool, start, end int64) {
	l = l.get()

	if len(l.tx) == 0 {
		name = "UTC"
		offset = 0
		isDST = false
		start = -1 << 63
		end = 1<<63 - 1
		return
	}

	if zone := l.cacheZone; zone != nil && l.cacheStart <= sec && sec < l.cacheEnd {
		name = zone.name
		offset = zone.offset
		isDST = zone.isDST
		start = l.cacheStart
		end = l.cacheEnd
		return
	}

	// Binary search for entry with largest time <= sec.
	// Not using sort.Search to avoid dependencies.
	tx := l.tx
	end = 1<<63 - 1
	for len(tx) > 1 {
		m := len(tx) / 2
		lim := tx[m].when
		if sec < lim {
			end = lim
			tx = tx[0:m]
		} else {
			tx = tx[m:]
		}
	}
	zone := &l.zone[tx[0].index]
	name = zone.name
	offset = zone.offset
	isDST = zone.isDST
	start = tx[0].when
	// end = maintained during the search
	return
}

// lookupName returns information about the time zone with
// the given name (such as "EST").
func (l *Location) lookupName(name string) (offset int, isDST bool, ok bool) {
	l = l.get()
	for i := range l.zone {
		zone := &l.zone[i]
		if zone.name == name {
			return zone.offset, zone.isDST, true
		}
	}
	return
}

// lookupOffset returns information about the time zone with
// the given offset (such as -5*60*60).
func (l *Location) lookupOffset(offset int) (name string, isDST bool, ok bool) {
	l = l.get()
	for i := range l.zone {
		zone := &l.zone[i]
		if zone.offset == offset {
			return zone.name, zone.isDST, true
		}
	}
	return
}

// NOTE(rsc): Eventually we will need to accept the POSIX TZ environment
// syntax too, but I don't feel like implementing it today.

var zoneinfo, _ = syscall.Getenv("ZONEINFO")

// LoadLocation returns the Location with the given name.
//
// If the name is "" or "UTC", LoadLocation returns UTC.
// If the name is "Local", LoadLocation returns Local.
//
// Otherwise, the name is taken to be a location name corresponding to a file
// in the IANA Time Zone database, such as "America/New_York".
//
// The time zone database needed by LoadLocation may not be
// present on all systems, especially non-Unix systems.
// LoadLocation looks in the directory or uncompressed zip file
// named by the ZONEINFO environment variable, if any, then looks in
// known installation locations on Unix systems,
// and finally looks in $GOROOT/lib/time/zoneinfo.zip.
func LoadLocation(name string) (*Location, error) {
	if name == "" || name == "UTC" {
		return UTC, nil
	}
	if name == "Local" {
		return Local, nil
	}
	if zoneinfo != "" {
		if z, err := loadZoneFile(zoneinfo, name); err == nil {
			z.name = name
			return z, nil
		}
	}
	return loadLocation(name)
}
