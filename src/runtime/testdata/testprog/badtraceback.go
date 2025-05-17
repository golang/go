// Copyright 2018 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package main

import (
	"runtime"
	"runtime/debug"
	"unsafe"
)

func init() {
	register("BadTraceback", BadTraceback)
}

func BadTraceback() {
	// Disable GC to prevent traceback at unexpected time.
	debug.SetGCPercent(-1)
	// Out of an abundance of caution, also make sure that there are
	// no GCs actively in progress.
	runtime.GC()

	// Run badLR1 on its own stack to minimize the stack size and
	// exercise the stack bounds logic in the hex dump.
	go badLR1()
	select {}
}

//go:noinline
func badLR1() {
	// We need two frames on LR machines because we'll smash this
	// frame's saved LR.
	badLR2(0)
}

//go:noinline
func badLR2(arg int) {
	// Smash the return PC or saved LR.
	lrOff := unsafe.Sizeof(uintptr(0))
	if runtime.GOARCH == "ppc64" || runtime.GOARCH == "ppc64le" {
		lrOff = 32 // FIXED_FRAME or sys.MinFrameSize
	}
	if runtime.GOARCH == "arm64" {
		// skip 8 bytes at bottom of parent frame, then point
		// to the 8 bytes of the saved PC at the top of the frame.
		lrOff = 16
	}
	lrPtr := (*uintptr)(unsafe.Pointer(uintptr(unsafe.Pointer(&arg)) - lrOff))
	*lrPtr = 0xbad

	runtime.KeepAlive(lrPtr) // prevent dead store elimination

	// Print a backtrace. This should include diagnostics for the
	// bad return PC and a hex dump.
	panic("backtrace")
}
