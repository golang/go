// Copyright 2019 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package main

import (
	"runtime"
	"runtime/debug"
	"sync/atomic"
)

func init() {
	register("AsyncPreempt", AsyncPreempt)
}

func AsyncPreempt() {
	// Run with just 1 GOMAXPROCS so the runtime is required to
	// use scheduler preemption.
	runtime.GOMAXPROCS(1)
	// Disable GC so we have complete control of what we're testing.
	debug.SetGCPercent(-1)

	// Start a goroutine with no sync safe-points.
	var ready uint32
	go func() {
		for {
			atomic.StoreUint32(&ready, 1)
		}
	}()

	// Wait for the goroutine to stop passing through sync
	// safe-points.
	for atomic.LoadUint32(&ready) == 0 {
		runtime.Gosched()
	}

	// Run a GC, which will have to stop the goroutine for STW and
	// for stack scanning. If this doesn't work, the test will
	// deadlock and timeout.
	runtime.GC()

	println("OK")
}
