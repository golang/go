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
	var ready, ready2 uint32
	go func() {
		for {
			atomic.StoreUint32(&ready, 1)
			dummy()
			dummy()
		}
	}()
	// Also start one with a frameless function.
	// This is an especially interesting case for
	// LR machines.
	go func() {
		atomic.AddUint32(&ready2, 1)
		frameless()
	}()
	// Also test empty infinite loop.
	go func() {
		atomic.AddUint32(&ready2, 1)
		for {
		}
	}()

	// Wait for the goroutine to stop passing through sync
	// safe-points.
	for atomic.LoadUint32(&ready) == 0 || atomic.LoadUint32(&ready2) < 2 {
		runtime.Gosched()
	}

	// Run a GC, which will have to stop the goroutine for STW and
	// for stack scanning. If this doesn't work, the test will
	// deadlock and timeout.
	runtime.GC()

	println("OK")
}

//go:noinline
func frameless() {
	for i := int64(0); i < 1<<62; i++ {
		out += i * i * i * i * i * 12345
	}
}

var out int64

//go:noinline
func dummy() {}
