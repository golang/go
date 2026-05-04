package executor_test

import (
	"context"
	"fmt"
	"runtime/executor"
	"time"
)

// This file mirrors specs/001-runtime-executor/quickstart.md.
//
// The example has no "// Output:" comment, so `go test` compiles
// it but does not execute it — which is what we want until the
// runtime hooks in Phase 2 make Co/Pulse/Yield functional. Once
// Phase 2 lands, an "// Output:" block will be added and T064
// (run quickstart end-to-end) will validate the example.

// Example demonstrates a fixed-budget frame loop driving three
// cooperative tasks, each of which yields at the end of its frame
// and waits for a per-task ready signal to start the next.
func Example() {
	ex := executor.New()

	for i := 0; i < 3; i++ {
		i := i
		ready := make(chan struct{}, 1)
		ready <- struct{}{}

		ex.Co(func() {
			for tick := 0; ; tick++ {
				<-ready
				fmt.Printf("task %d tick %d\n", i, tick)
				go func() { ready <- struct{}{} }()
				executor.Yield()
			}
		})
	}

	for frame := 0; frame < 5; frame++ {
		ctx, cancel := context.WithTimeout(context.Background(), 16*time.Millisecond)
		if err := ex.Pulse(ctx); err != nil {
			fmt.Println("frame", frame, "ran out of budget:", err)
		}
		cancel()
	}
}
