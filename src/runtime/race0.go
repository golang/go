// Copyright 2014 The Go Authors. All rights reserved.
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

func raceReadObjectPC(t *_type, addr unsafe.Pointer, callerpc, pc uintptr)  { throw("race") }
func raceWriteObjectPC(t *_type, addr unsafe.Pointer, callerpc, pc uintptr) { throw("race") }
func raceinit() (uintptr, uintptr)                                          { throw("race"); return 0, 0 }
func racefini()                                                             { throw("race") }
func raceproccreate() uintptr                                               { throw("race"); return 0 }
func raceprocdestroy(ctx uintptr)                                           { throw("race") }
func racemapshadow(addr unsafe.Pointer, size uintptr)                       { throw("race") }
func racewritepc(addr unsafe.Pointer, callerpc, pc uintptr)                 { throw("race") }
func racereadpc(addr unsafe.Pointer, callerpc, pc uintptr)                  { throw("race") }
func racereadrangepc(addr unsafe.Pointer, sz, callerpc, pc uintptr)         { throw("race") }
func racewriterangepc(addr unsafe.Pointer, sz, callerpc, pc uintptr)        { throw("race") }
func raceacquire(addr unsafe.Pointer)                                       { throw("race") }
func raceacquireg(gp *g, addr unsafe.Pointer)                               { throw("race") }
func raceacquirectx(racectx uintptr, addr unsafe.Pointer)                   { throw("race") }
func racerelease(addr unsafe.Pointer)                                       { throw("race") }
func racereleaseg(gp *g, addr unsafe.Pointer)                               { throw("race") }
func racereleasemerge(addr unsafe.Pointer)                                  { throw("race") }
func racereleasemergeg(gp *g, addr unsafe.Pointer)                          { throw("race") }
func racefingo()                                                            { throw("race") }
func racemalloc(p unsafe.Pointer, sz uintptr)                               { throw("race") }
func racefree(p unsafe.Pointer, sz uintptr)                                 { throw("race") }
func racegostart(pc uintptr) uintptr                                        { throw("race"); return 0 }
func racegoend()                                                            { throw("race") }
func racectxend(racectx uintptr)                                            { throw("race") }
