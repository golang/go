// Copyright 2025 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package main

import (
	"context"
	"log"
	"os"
	"runtime"
	"runtime/debug"
	"runtime/trace"
	"sync/atomic"
)

func init() {
	register("TraceSTW", TraceSTW)
}

// The parent writes to ping and waits for the children to write back
// via pong to show that they are running.
var ping atomic.Uint32
var pong [2]atomic.Uint32

// Tell runners to stop.
var stop atomic.Bool

func traceSTWTarget(i int) {
	for !stop.Load() {
		// Async preemption often takes 100ms+ to preempt this loop on
		// windows-386. This makes the test flaky, as the traceReadCPU
		// timer often fires by the time STW finishes, jumbling the
		// goroutine scheduling. As a workaround, ensure we have a
		// morestack call for prompt preemption.
		ensureMorestack()

		pong[i].Store(ping.Load())
	}
}

func TraceSTW() {
	ctx := context.Background()

	// The idea here is to have 2 target goroutines that are constantly
	// running. When the world restarts after STW, we expect these
	// goroutines to continue execution on the same M and P.
	//
	// Set GOMAXPROCS=4 to make room for the 2 target goroutines, 1 parent,
	// and 1 slack for potential misscheduling.
	//
	// Disable the GC because GC STW generally moves goroutines (see
	// https://go.dev/issue/65694). Alternatively, we could just ignore the
	// trace if the GC runs.
	runtime.GOMAXPROCS(4)
	debug.SetGCPercent(0)

	if err := trace.Start(os.Stdout); err != nil {
		log.Fatalf("failed to start tracing: %v", err)
	}
	defer trace.Stop()

	for i := range 2 {
		go traceSTWTarget(i)
	}

	// Wait for children to start running.
	ping.Store(1)
	for pong[0].Load() != 1 {}
	for pong[1].Load() != 1 {}

	trace.Log(ctx, "TraceSTW", "start")

	// STW
	var ms runtime.MemStats
	runtime.ReadMemStats(&ms)

	// Make sure to run long enough for the children to schedule again
	// after STW.
	ping.Store(2)
	for pong[0].Load() != 2 {}
	for pong[1].Load() != 2 {}

	trace.Log(ctx, "TraceSTW", "end")

	stop.Store(true)
}

// Manually insert a morestack call. Leaf functions can omit morestack, but
// non-leaf functions should include them.

//go:noinline
func ensureMorestack() {
	ensureMorestack1()
}

//go:noinline
func ensureMorestack1() {
}
