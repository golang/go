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

// Because raceenabled is false, none of these functions should be called.

func raceReadObjectPC(t *_type, addr unsafe.Pointer, callerpc, pc uintptr)  { gothrow("race") }
func raceWriteObjectPC(t *_type, addr unsafe.Pointer, callerpc, pc uintptr) { gothrow("race") }
func raceinit()                                                             { gothrow("race") }
func racefini()                                                             { gothrow("race") }
func racemapshadow(addr unsafe.Pointer, size uintptr)                       { gothrow("race") }
func racewritepc(addr unsafe.Pointer, callerpc, pc uintptr)                 { gothrow("race") }
func racereadpc(addr unsafe.Pointer, callerpc, pc uintptr)                  { gothrow("race") }
func racereadrangepc(addr unsafe.Pointer, sz, callerpc, pc uintptr)         { gothrow("race") }
func racewriterangepc(addr unsafe.Pointer, sz, callerpc, pc uintptr)        { gothrow("race") }
func raceacquire(addr unsafe.Pointer)                                       { gothrow("race") }
func raceacquireg(gp *g, addr unsafe.Pointer)                               { gothrow("race") }
func racerelease(addr unsafe.Pointer)                                       { gothrow("race") }
func racereleaseg(gp *g, addr unsafe.Pointer)                               { gothrow("race") }
func racereleasemerge(addr unsafe.Pointer)                                  { gothrow("race") }
func racereleasemergeg(gp *g, addr unsafe.Pointer)                          { gothrow("race") }
func racefingo()                                                            { gothrow("race") }
func racemalloc(p unsafe.Pointer, sz uintptr)                               { gothrow("race") }
func racegostart(pc uintptr) uintptr                                        { gothrow("race"); return 0 }
func racegoend()                                                            { gothrow("race") }
