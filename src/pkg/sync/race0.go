// Copyright 2012 The Go Authors.  All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// +build !race

package sync

import (
	"unsafe"
)

const raceenabled = false

func raceAcquire(addr unsafe.Pointer) {
}

func raceRelease(addr unsafe.Pointer) {
}

func raceReleaseMerge(addr unsafe.Pointer) {
}

func raceDisable() {
}

func raceEnable() {
}

func raceRead(addr unsafe.Pointer) {
}

func raceWrite(addr unsafe.Pointer) {
}
