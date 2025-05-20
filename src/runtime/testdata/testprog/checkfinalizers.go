// Copyright 2024 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package main

import (
	"runtime"
	"runtime/debug"
	"unsafe"
)

func init() {
	register("DetectFinalizerAndCleanupLeaks", DetectFinalizerAndCleanupLeaks)
}

type tiny uint8

var tinySink *tiny

// Intended to be run only with `GODEBUG=checkfinalizers=1`.
func DetectFinalizerAndCleanupLeaks() {
	type T *int

	defer debug.SetGCPercent(debug.SetGCPercent(-1))

	// Leak a cleanup.
	cLeak := new(T)
	runtime.AddCleanup(cLeak, func(x int) {
		**cLeak = x
	}, int(0))

	// Have a regular cleanup to make sure it doesn't trip the detector.
	cNoLeak := new(T)
	runtime.AddCleanup(cNoLeak, func(_ int) {}, int(0))

	// Add a cleanup that only temporarily leaks cNoLeak.
	runtime.AddCleanup(cNoLeak, func(x int) {
		**cNoLeak = x
	}, int(0)).Stop()

	// Ensure we create an allocation into a tiny block that shares space among several values.
	var ctLeak *tiny
	for {
		tinySink = ctLeak
		ctLeak = new(tiny)
		*ctLeak = tiny(55)
		// Make sure the address is an odd value. This is sufficient to
		// be certain that we're sharing a block with another value and
		// trip the detector.
		if uintptr(unsafe.Pointer(ctLeak))%2 != 0 {
			break
		}
	}
	runtime.AddCleanup(ctLeak, func(_ struct{}) {}, struct{}{})

	// Leak a finalizer.
	fLeak := new(T)
	runtime.SetFinalizer(fLeak, func(_ *T) {
		**fLeak = 12
	})

	// Have a regular finalizer to make sure it doesn't trip the detector.
	fNoLeak := new(T)
	runtime.SetFinalizer(fNoLeak, func(x *T) {
		**x = 51
	})

	// runtime.GC here should crash.
	runtime.GC()
	println("OK")
}
