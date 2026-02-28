// Copyright 2023 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// Tests user tasks, regions, and logging.

//go:build ignore

package main

import (
	"context"
	"log"
	"os"
	"runtime/trace"
	"sync"
)

func main() {
	bgctx, cancel := context.WithCancel(context.Background())
	defer cancel()

	// Create a pre-existing region. This won't end up in the trace.
	preExistingRegion := trace.StartRegion(bgctx, "pre-existing region")

	// Start tracing.
	if err := trace.Start(os.Stdout); err != nil {
		log.Fatalf("failed to start tracing: %v", err)
	}

	// Beginning of traced execution.
	var wg sync.WaitGroup
	ctx, task := trace.NewTask(bgctx, "task0") // EvUserTaskCreate("task0")
	trace.StartRegion(ctx, "task0 region")

	wg.Add(1)
	go func() {
		defer wg.Done()
		defer task.End() // EvUserTaskEnd("task0")

		trace.StartRegion(ctx, "unended region")

		trace.WithRegion(ctx, "region0", func() {
			// EvUserRegionBegin("region0", start)
			trace.WithRegion(ctx, "region1", func() {
				trace.Log(ctx, "key0", "0123456789abcdef") // EvUserLog("task0", "key0", "0....f")
			})
			// EvUserRegionEnd("region0", end)
		})
	}()
	wg.Wait()

	preExistingRegion.End()
	postExistingRegion := trace.StartRegion(bgctx, "post-existing region")

	// End of traced execution.
	trace.Stop()

	postExistingRegion.End()
}
