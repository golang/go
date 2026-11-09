// Copyright 2026 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package main

import (
	"bytes"
	"fmt"
	"os"
	"runtime"
	"time"
)

// startBench starts measuring simdgen for -bench. It returns a function that
// prints the measurements to stderr as a Go benchmark result line, so the
// output of several runs can be compared with benchstat.
func startBench() func() {
	start := time.Now()
	return func() {
		wall := time.Since(start)
		var ms runtime.MemStats
		runtime.ReadMemStats(&ms)

		var buf bytes.Buffer
		fmt.Fprintf(&buf, "BenchmarkSimdgen/target=%s", *FlagArch)
		// Like the testing package, add a -GOMAXPROCS suffix unless it's 1.
		if procs := runtime.GOMAXPROCS(0); procs != 1 {
			fmt.Fprintf(&buf, "-%d", procs)
		}
		fmt.Fprintf(&buf, " 1 %d ns/op", wall.Nanoseconds())
		if t, ok := cpuTime(); ok {
			fmt.Fprintf(&buf, " %d cpu-ns/op", t.Nanoseconds())
		}
		fmt.Fprintf(&buf, " %d B/op %d allocs/op\n", ms.TotalAlloc, ms.Mallocs)
		os.Stderr.Write(buf.Bytes())
	}
}
