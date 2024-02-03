// Copyright 2023 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// Tests CPU profiling.

//go:build ignore

package main

import (
	"bytes"
	"context"
	"fmt"
	"internal/profile"
	"log"
	"os"
	"runtime"
	"runtime/pprof"
	"runtime/trace"
	"strings"
	"time"
)

func main() {
	cpuBuf := new(bytes.Buffer)
	if err := pprof.StartCPUProfile(cpuBuf); err != nil {
		log.Fatalf("failed to start CPU profile: %v", err)
	}

	if err := trace.Start(os.Stdout); err != nil {
		log.Fatalf("failed to start tracing: %v", err)
	}

	dur := 100 * time.Millisecond
	func() {
		// Create a region in the execution trace. Set and clear goroutine
		// labels fully within that region, so we know that any CPU profile
		// sample with the label must also be eligible for inclusion in the
		// execution trace.
		ctx := context.Background()
		defer trace.StartRegion(ctx, "cpuHogger").End()
		pprof.Do(ctx, pprof.Labels("tracing", "on"), func(ctx context.Context) {
			cpuHogger(cpuHog1, &salt1, dur)
		})
		// Be sure the execution trace's view, when filtered to this goroutine
		// via the explicit goroutine ID in each event, gets many more samples
		// than the CPU profiler when filtered to this goroutine via labels.
		cpuHogger(cpuHog1, &salt1, dur)
	}()

	trace.Stop()
	pprof.StopCPUProfile()

	// Summarize the CPU profile to stderr so the test can check against it.

	prof, err := profile.Parse(cpuBuf)
	if err != nil {
		log.Fatalf("failed to parse CPU profile: %v", err)
	}
	// Examine the CPU profiler's view. Filter it to only include samples from
	// the single test goroutine. Use labels to execute that filter: they should
	// apply to all work done while that goroutine is getg().m.curg, and they
	// should apply to no other goroutines.
	pprofStacks := make(map[string]int)
	for _, s := range prof.Sample {
		if s.Label["tracing"] != nil {
			var fns []string
			var leaf string
			for _, loc := range s.Location {
				for _, line := range loc.Line {
					fns = append(fns, fmt.Sprintf("%s:%d", line.Function.Name, line.Line))
					leaf = line.Function.Name
				}
			}
			// runtime.sigprof synthesizes call stacks when "normal traceback is
			// impossible or has failed", using particular placeholder functions
			// to represent common failure cases. Look for those functions in
			// the leaf position as a sign that the call stack and its
			// symbolization are more complex than this test can handle.
			//
			// TODO: Make the symbolization done by the execution tracer and CPU
			// profiler match up even in these harder cases. See #53378.
			switch leaf {
			case "runtime._System", "runtime._GC", "runtime._ExternalCode", "runtime._VDSO":
				continue
			}
			stack := strings.Join(fns, "|")
			samples := int(s.Value[0])
			pprofStacks[stack] += samples
		}
	}
	for stack, samples := range pprofStacks {
		fmt.Fprintf(os.Stderr, "%s\t%d\n", stack, samples)
	}
}

func cpuHogger(f func(x int) int, y *int, dur time.Duration) {
	// We only need to get one 100 Hz clock tick, so we've got
	// a large safety buffer.
	// But do at least 500 iterations (which should take about 100ms),
	// otherwise TestCPUProfileMultithreaded can fail if only one
	// thread is scheduled during the testing period.
	t0 := time.Now()
	accum := *y
	for i := 0; i < 500 || time.Since(t0) < dur; i++ {
		accum = f(accum)
	}
	*y = accum
}

var (
	salt1 = 0
)

// The actual CPU hogging function.
// Must not call other functions nor access heap/globals in the loop,
// otherwise under race detector the samples will be in the race runtime.
func cpuHog1(x int) int {
	return cpuHog0(x, 1e5)
}

func cpuHog0(x, n int) int {
	foo := x
	for i := 0; i < n; i++ {
		if i%1000 == 0 {
			// Spend time in mcall, stored as gp.m.curg, with g0 running
			runtime.Gosched()
		}
		if foo > 0 {
			foo *= foo
		} else {
			foo *= foo + 1
		}
	}
	return foo
}
