// Copyright 2025 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package main

import (
	"context"
	"log"
	"math/rand/v2"
	"os"
	"runtime"
	"runtime/debug"
	"runtime/metrics"
	"runtime/trace"
	"sync/atomic"
)

func init() {
	register("TraceSTW", TraceSTW)
	register("TraceGCSTW", TraceGCSTW)
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
	debug.SetGCPercent(-1)

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

// Variant of TraceSTW for GC STWs. We want the GC mark workers to start on
// previously-idle Ps, rather than bumping the current P.
func TraceGCSTW() {
	ctx := context.Background()

	// The idea here is to have 2 target goroutines that are constantly
	// running. When the world restarts after STW, we expect these
	// goroutines to continue execution on the same M and P.
	//
	// Set GOMAXPROCS=8 to make room for the 2 target goroutines, 1 parent,
	// 2 dedicated workers, and a bit of slack.
	//
	// Disable the GC initially so we can be sure it only triggers once we
	// are ready.
	runtime.GOMAXPROCS(8)
	debug.SetGCPercent(-1)

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
	triggerGC()

	// Make sure to run long enough for the children to schedule again
	// after STW. This is included for good measure, but the goroutines
	// really ought to have already scheduled since the entire GC
	// completed.
	ping.Store(2)
	for pong[0].Load() != 2 {}
	for pong[1].Load() != 2 {}

	trace.Log(ctx, "TraceSTW", "end")

	stop.Store(true)
}

func triggerGC() {
	// Allocate a bunch to trigger the GC rather than using runtime.GC. The
	// latter blocks until the GC is complete, which is convenient, but
	// messes with scheduling as it gives this P a chance to steal the
	// other goroutines before their Ps get up and running again.

	// Bring heap size up prior to enabling the GC to ensure that there is
	// a decent amount of work in case the GC triggers immediately upon
	// re-enabling.
	for range 1000 {
		alloc()
	}

	sample := make([]metrics.Sample, 1)
	sample[0].Name = "/gc/cycles/total:gc-cycles"
	metrics.Read(sample)

	start := sample[0].Value.Uint64()

	debug.SetGCPercent(100)

	// Keep allocating until the GC is complete. We really only need to
	// continue until the mark workers are scheduled, but there isn't a
	// good way to measure that.
	for {
		metrics.Read(sample)
		if sample[0].Value.Uint64() != start {
			return
		}

		alloc()
	}
}

// Allocate a tree data structure to generate plenty of scan work for the GC.

type node struct {
	children []*node
}

var gcSink node

func alloc() {
	// 10% chance of adding a node a each layer.

	curr := &gcSink
	for {
		if len(curr.children) == 0 || rand.Float32() < 0.1 {
			curr.children = append(curr.children, new(node))
			return
		}

		i := rand.IntN(len(curr.children))
		curr = curr.children[i]
	}
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
