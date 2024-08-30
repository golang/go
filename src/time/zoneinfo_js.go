// Copyright 2018 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

//go:build js && wasm

package time

import (
	"internal/itoa"
	"syscall/js"
)

var platformZoneSources = []string{
	"/usr/share/zoneinfo/",
	"/usr/share/lib/zoneinfo/",
	"/usr/lib/locale/TZ/",
}

func initLocal() {
	localLoc.name = "Local"

	z := zone{}
	d := js.Global().Get("Date").New()
	offset := d.Call("getTimezoneOffset").Int() * -1
	z.offset = offset * 60
	// According to https://tc39.github.io/ecma262/#sec-timezoneestring,
	// the timezone name from (new Date()).toTimeString() is an implementation-dependent
	// result, and in Google Chrome, it gives the fully expanded name rather than
	// the abbreviation.
	// Hence, we construct the name from the offset.
	z.name = "UTC"
	if offset < 0 {
		z.name += "-"
		offset *= -1
	} else {
		z.name += "+"
	}
	z.name += itoa.Itoa(offset / 60)
	min := offset % 60
	if min != 0 {
		z.name += ":" + itoa.Itoa(min)
	}
	localLoc.zone = []zone{z}
}
