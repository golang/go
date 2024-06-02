// Copyright 2023 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// Tests simply starting and stopping tracing multiple times.
//
// This is useful for finding bugs in trace state reset.

//go:build ignore

package main

import (
	"bytes"
	"log"
	"os"
	"runtime"
	"runtime/trace"
)

func main() {
	// Trace a few times.
	for i := 0; i < 10; i++ {
		var buf bytes.Buffer
		if err := trace.Start(&buf); err != nil {
			log.Fatalf("failed to start tracing: %v", err)
		}
		runtime.GC()
		trace.Stop()
	}

	// Start tracing again, this time writing out the result.
	if err := trace.Start(os.Stdout); err != nil {
		log.Fatalf("failed to start tracing: %v", err)
	}
	runtime.GC()
	trace.Stop()
}
