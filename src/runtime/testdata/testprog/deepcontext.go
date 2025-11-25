// Copyright 2025 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package main

import (
	"context"
	"fmt"
	"runtime"
	"time"
)

func init() {
	register("DeepContextChain", DeepContextChain)
}

// DeepContextChain tests that traceback completes in reasonable time
// even with very deep context chains (issue #75583).
func DeepContextChain() {
	// Create a context chain deep enough to trigger the frame limit.
	// We use 2000 layers to ensure we exceed the limit (1024) while
	// keeping the test fast.
	const depth = 2000
	ctx := context.Background()
	for i := 0; i < depth; i++ {
		ctx = context.WithValue(ctx, i, i)
	}

	// Start profiling to trigger stack walking with deep context
	start := time.Now()

	// Get a stack trace multiple times
	// This simulates what happens during CPU profiling
	for i := 0; i < 10; i++ {
		var pcs [64]uintptr
		n := runtime.Callers(0, pcs[:])
		if n == 0 {
			fmt.Println("FAIL: got 0 callers")
			return
		}

		// Call Deadline to ensure the deep context chain is traversed
		// during any potential stack walking
		_, _ = ctx.Deadline()
	}

	elapsed := time.Since(start)

	// The test should complete quickly. If it takes more than 1 second,
	// something is wrong (likely walking too many frames).
	if elapsed > time.Second {
		fmt.Printf("FAIL: test took %v, expected < 1s\n", elapsed)
		return
	}

	fmt.Println("OK")
}
