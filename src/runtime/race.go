// Copyright 2012 The Go Authors.  All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// +build race

// Public race detection API, present iff build with -race.

package runtime

import (
	"unsafe"
)

func racefini()

// RaceDisable disables handling of race events in the current goroutine.
func RaceDisable()

// RaceEnable re-enables handling of race events in the current goroutine.
func RaceEnable()

func RaceAcquire(addr unsafe.Pointer)
func RaceRelease(addr unsafe.Pointer)
func RaceReleaseMerge(addr unsafe.Pointer)

func RaceRead(addr unsafe.Pointer)
func RaceWrite(addr unsafe.Pointer)
func RaceReadRange(addr unsafe.Pointer, len int)
func RaceWriteRange(addr unsafe.Pointer, len int)

func RaceSemacquire(s *uint32)
func RaceSemrelease(s *uint32)

// private interface for the runtime
const raceenabled = true

func raceReadObjectPC(t *_type, addr unsafe.Pointer, callerpc, pc uintptr) {
	kind := t.kind & kindMask
	if kind == kindArray || kind == kindStruct {
		// for composite objects we have to read every address
		// because a write might happen to any subobject.
		racereadrangepc(addr, t.size, callerpc, pc)
	} else {
		// for non-composite objects we can read just the start
		// address, as any write must write the first byte.
		racereadpc(addr, callerpc, pc)
	}
}

func raceWriteObjectPC(t *_type, addr unsafe.Pointer, callerpc, pc uintptr) {
	kind := t.kind & kindMask
	if kind == kindArray || kind == kindStruct {
		// for composite objects we have to write every address
		// because a write might happen to any subobject.
		racewriterangepc(addr, t.size, callerpc, pc)
	} else {
		// for non-composite objects we can write just the start
		// address, as any write must write the first byte.
		racewritepc(addr, callerpc, pc)
	}
}

//go:noescape
func racereadpc(addr unsafe.Pointer, callpc, pc uintptr)

//go:noescape
func racewritepc(addr unsafe.Pointer, callpc, pc uintptr)

//go:noescape
func racereadrangepc(addr unsafe.Pointer, len uintptr, callpc, pc uintptr)

//go:noescape
func racewriterangepc(addr unsafe.Pointer, len uintptr, callpc, pc uintptr)

//go:noescape
func raceacquire(addr unsafe.Pointer)

//go:noescape
func racerelease(addr unsafe.Pointer)

//go:noescape
func raceacquireg(gp *g, addr unsafe.Pointer)

//go:noescape
func racereleaseg(gp *g, addr unsafe.Pointer)

func racefingo()

//go:noescape
func racemalloc(p unsafe.Pointer, size uintptr)

//go:noescape
func racereleasemerge(addr unsafe.Pointer)

type symbolizeContext struct {
	pc   uintptr
	fn   *byte
	file *byte
	line uintptr
	off  uintptr
	res  uintptr
}

var qq = [...]byte{'?', '?', 0}
var dash = [...]byte{'-', 0}

// Callback from C into Go, runs on g0.
func racesymbolize(ctx *symbolizeContext) {
	f := findfunc(ctx.pc)
	if f == nil {
		ctx.fn = &qq[0]
		ctx.file = &dash[0]
		ctx.line = 0
		ctx.off = ctx.pc
		ctx.res = 1
		return
	}

	ctx.fn = funcname(f)
	var file string
	ctx.line = uintptr(funcline(f, ctx.pc, &file))
	ctx.file = &bytes(file)[0] // assume NUL-terminated
	ctx.off = ctx.pc - f.entry
	ctx.res = 1
	return
}
