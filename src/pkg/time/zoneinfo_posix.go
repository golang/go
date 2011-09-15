// Copyright 2011 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// +build darwin freebsd linux openbsd plan9

package time

import "sync"

// Parsed representation
type zone struct {
	utcoff int
	isdst  bool
	name   string
}

type zonetime struct {
	time         int32 // transition time, in seconds since 1970 GMT
	zone         *zone // the zone that goes into effect at that time
	isstd, isutc bool  // ignored - no idea what these mean
}

var zones []zonetime
var onceSetupZone sync.Once

// Look up the correct time zone (daylight savings or not) for the given unix time, in the current location.
func lookupTimezone(sec int64) (zone string, offset int) {
	onceSetupZone.Do(setupZone)
	if len(zones) == 0 {
		return "UTC", 0
	}

	// Binary search for entry with largest time <= sec
	tz := zones
	for len(tz) > 1 {
		m := len(tz) / 2
		if sec < int64(tz[m].time) {
			tz = tz[0:m]
		} else {
			tz = tz[m:]
		}
	}
	z := tz[0].zone
	return z.name, z.utcoff
}

// lookupByName returns the time offset for the
// time zone with the given abbreviation. It only considers
// time zones that apply to the current system.
// For example, for a system configured as being in New York,
// it only recognizes "EST" and "EDT".
// For a system in San Francisco, "PST" and "PDT".
// For a system in Sydney, "EST" and "EDT", though they have
// different meanings than they do in New York.
func lookupByName(name string) (off int, found bool) {
	onceSetupZone.Do(setupZone)
	for _, z := range zones {
		if name == z.zone.name {
			return z.zone.utcoff, true
		}
	}
	return 0, false
}
