// Copyright 2014 The Go Authors.  All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// +build !race

// Dummy race detection API, used when not built with -race.

package runtime

import (
	"unsafe"
)

const raceenabled = false

func raceReadObjectPC(t *_type, addr unsafe.Pointer, callerpc, pc uintptr) {
}
func raceWriteObjectPC(t *_type, addr unsafe.Pointer, callerpc, pc uintptr) {
}
