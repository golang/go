// Copyright 2012 The Go Authors.  All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// +build race

package sync

import (
	"runtime"
	"unsafe"
)

const raceenabled = true

func raceAcquire(addr unsafe.Pointer) {
	runtime.RaceAcquire(addr)
}

func raceRelease(addr unsafe.Pointer) {
	runtime.RaceRelease(addr)
}

func raceReleaseMerge(addr unsafe.Pointer) {
	runtime.RaceReleaseMerge(addr)
}

func raceDisable() {
	runtime.RaceDisable()
}

func raceEnable() {
	runtime.RaceEnable()
}
