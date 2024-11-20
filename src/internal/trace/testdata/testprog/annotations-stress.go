// Copyright 2023 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// Tests user tasks, regions, and logging.

//go:build ignore

package main

import (
	"context"
	"fmt"
	"log"
	"os"
	"runtime/trace"
	"time"
)

func main() {
	baseCtx := context.Background()

	// Create a task that starts and ends entirely outside of the trace.
	ctx0, t0 := trace.NewTask(baseCtx, "parent")

	// Create a task that starts before the trace and ends during the trace.
	ctx1, t1 := trace.NewTask(ctx0, "type1")

	// Start tracing.
	if err := trace.Start(os.Stdout); err != nil {
		log.Fatalf("failed to start tracing: %v", err)
	}
	t1.End()

	// Create a task that starts during the trace and ends after.
	ctx2, t2 := trace.NewTask(ctx0, "type2")

	// Create a task that starts and ends during the trace.
	ctx3, t3 := trace.NewTask(baseCtx, "type3")

	// Generate some events.
	for i := 0; i < 2; i++ {
		do(baseCtx, 4)
		do(ctx0, 2)
		do(ctx1, 3)
		do(ctx2, 6)
		do(ctx3, 5)
	}

	// Finish up tasks according to their lifetime relative to the trace.
	t3.End()
	trace.Stop()
	t2.End()
	t0.End()
}

func do(ctx context.Context, k int) {
	trace.Log(ctx, "log", "before do")

	var t *trace.Task
	ctx, t = trace.NewTask(ctx, "do")
	defer t.End()

	trace.Log(ctx, "log2", "do")

	// Create a region and spawn more tasks and more workers.
	trace.WithRegion(ctx, "fanout", func() {
		for i := 0; i < k; i++ {
			go func(i int) {
				trace.WithRegion(ctx, fmt.Sprintf("region%d", i), func() {
					trace.Logf(ctx, "log", "fanout region%d", i)
					if i == 2 {
						do(ctx, 0)
						return
					}
				})
			}(i)
		}
	})

	// Sleep to let things happen, but also increase the chance that we
	// advance a generation.
	time.Sleep(10 * time.Millisecond)
}
