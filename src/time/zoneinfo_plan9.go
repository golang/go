// Copyright 2011 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// Parse Plan 9 timezone(2) files.

package time

import (
	"syscall"
)

var platformZoneSources []string // none on Plan 9

func isSpace(r rune) bool {
	return r == ' ' || r == '\t' || r == '\n'
}

// Copied from strings to avoid a dependency.
func fields(s string) []string {
	// First count the fields.
	n := 0
	inField := false
	for _, rune := range s {
		wasInField := inField
		inField = !isSpace(rune)
		if inField && !wasInField {
			n++
		}
	}

	// Now create them.
	a := make([]string, n)
	na := 0
	fieldStart := -1 // Set to -1 when looking for start of field.
	for i, rune := range s {
		if isSpace(rune) {
			if fieldStart >= 0 {
				a[na] = s[fieldStart:i]
				na++
				fieldStart = -1
			}
		} else if fieldStart == -1 {
			fieldStart = i
		}
	}
	if fieldStart >= 0 { // Last field might end at EOF.
		a[na] = s[fieldStart:]
	}
	return a
}

func loadZoneDataPlan9(s string) (l *Location, err error) {
	f := fields(s)
	if len(f) < 4 {
		if len(f) == 2 && f[0] == "GMT" {
			return UTC, nil
		}
		return nil, badData
	}

	var zones [2]zone

	// standard timezone offset
	o, err := atoi(f[1])
	if err != nil {
		return nil, badData
	}
	zones[0] = zone{name: f[0], offset: o, isDST: false}

	// alternate timezone offset
	o, err = atoi(f[3])
	if err != nil {
		return nil, badData
	}
	zones[1] = zone{name: f[2], offset: o, isDST: true}

	// transition time pairs
	var tx []zoneTrans
	f = f[4:]
	for i := 0; i < len(f); i++ {
		zi := 0
		if i%2 == 0 {
			zi = 1
		}
		t, err := atoi(f[i])
		if err != nil {
			return nil, badData
		}
		t -= zones[0].offset
		tx = append(tx, zoneTrans{when: int64(t), index: uint8(zi)})
	}

	// Committed to succeed.
	l = &Location{zone: zones[:], tx: tx}

	// Fill in the cache with information about right now,
	// since that will be the most common lookup.
	sec, _, _ := now()
	for i := range tx {
		if tx[i].when <= sec && (i+1 == len(tx) || sec < tx[i+1].when) {
			l.cacheStart = tx[i].when
			l.cacheEnd = omega
			if i+1 < len(tx) {
				l.cacheEnd = tx[i+1].when
			}
			l.cacheZone = &l.zone[tx[i].index]
		}
	}

	return l, nil
}

func loadZoneFilePlan9(name string) (*Location, error) {
	b, err := readFile(name)
	if err != nil {
		return nil, err
	}
	return loadZoneDataPlan9(string(b))
}

func initLocal() {
	t, ok := syscall.Getenv("timezone")
	if ok {
		if z, err := loadZoneDataPlan9(t); err == nil {
			localLoc = *z
			return
		}
	} else {
		if z, err := loadZoneFilePlan9("/adm/timezone/local"); err == nil {
			localLoc = *z
			localLoc.name = "Local"
			return
		}
	}

	// Fall back to UTC.
	localLoc.name = "UTC"
}
