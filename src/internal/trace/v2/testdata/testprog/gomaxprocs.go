// Copyright 2023 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// Tests increasing and decreasing GOMAXPROCS to try and
// catch issues with stale proc state.

//go:build ignore

package main

import (
	"log"
	"os"
	"runtime"
	"runtime/trace"
	"time"
)

func main() {
	// Start a goroutine that calls runtime.GC to try and
	// introduce some interesting events in between the
	// GOMAXPROCS calls.
	go func() {
		for {
			runtime.GC()
			time.Sleep(1 * time.Millisecond)
		}
	}()

	// Start tracing.
	if err := trace.Start(os.Stdout); err != nil {
		log.Fatalf("failed to start tracing: %v", err)
	}
	// Run GOMAXPROCS a bunch of times, up and down.
	for i := 1; i <= 16; i *= 2 {
		runtime.GOMAXPROCS(i)
		time.Sleep(1 * time.Millisecond)
	}
	for i := 16; i >= 1; i /= 2 {
		runtime.GOMAXPROCS(i)
		time.Sleep(1 * time.Millisecond)
	}
	// Stop tracing.
	trace.Stop()
}
