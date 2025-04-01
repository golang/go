// Copyright 2024 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package main

import (
	"runtime"
	"runtime/debug"
)

func init() {
	register("DetectFinalizerAndCleanupLeaks", DetectFinalizerAndCleanupLeaks)
}

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

	// Keep everything alive.
	runtime.KeepAlive(cLeak)
	runtime.KeepAlive(cNoLeak)
	runtime.KeepAlive(fLeak)
	runtime.KeepAlive(fNoLeak)
}
