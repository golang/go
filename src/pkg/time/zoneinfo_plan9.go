// Copyright 2011 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// Parse Plan 9 timezone(2) files.

package time

import (
	"os"
	"strconv"
	"strings"
)

func parseZones(s string) (zt []zonetime) {
	f := strings.Fields(s)
	if len(f) < 4 {
		return
	}

	// standard timezone offset
	o, err := strconv.Atoi(f[1])
	if err != nil {
		return
	}
	std := &zone{name: f[0], utcoff: o, isdst: false}

	// alternate timezone offset
	o, err = strconv.Atoi(f[3])
	if err != nil {
		return
	}
	dst := &zone{name: f[2], utcoff: o, isdst: true}

	// transition time pairs
	f = f[4:]
	for i := 0; i < len(f); i++ {
		z := std
		if i%2 == 0 {
			z = dst
		}
		t, err := strconv.Atoi(f[i])
		if err != nil {
			return nil
		}
		t -= std.utcoff
		zt = append(zt, zonetime{time: int32(t), zone: z})
	}
	return
}

func setupZone() {
	t, err := os.Getenverror("timezone")
	if err != nil {
		// do nothing: use UTC
		return
	}
	zones = parseZones(t)
}

func setupTestingZone() {
	f, err := os.Open("/adm/timezone/US_Pacific")
	if err != nil {
		return
	}
	defer f.Close()
	l, _ := f.Seek(0, 2)
	f.Seek(0, 0)
	buf := make([]byte, l)
	_, err = f.Read(buf)
	if err != nil {
		return
	}
	zones = parseZones(string(buf))
}
