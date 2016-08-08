package main

import (
	"internal/trace"
	"testing"
)

// TestGoroutineCount tests runnable/running goroutine counts computed by generateTrace
// remain in the valid range.
//   - the counts must not be negative. generateTrace will return an error.
//   - the counts must not include goroutines blocked waiting on channels or in syscall.
func TestGoroutineCount(t *testing.T) {
	w := trace.NewWriter()
	w.Emit(trace.EvBatch, 0, 0)  // start of per-P batch event [pid, timestamp]
	w.Emit(trace.EvFrequency, 1) // [ticks per second]

	// In this test, we assume a valid trace contains EvGoWaiting or EvGoInSyscall
	// event for every blocked goroutine.

	// goroutine 10: blocked
	w.Emit(trace.EvGoCreate, 1, 10, 1, 1) // [timestamp, new goroutine id, new stack id, stack id]
	w.Emit(trace.EvGoWaiting, 1, 10)      // [timestamp, goroutine id]

	// goroutine 20: in syscall
	w.Emit(trace.EvGoCreate, 1, 20, 2, 1)
	w.Emit(trace.EvGoInSyscall, 1, 20) // [timestamp, goroutine id]

	// goroutine 30: runnable
	w.Emit(trace.EvGoCreate, 1, 30, 5, 1)

	w.Emit(trace.EvProcStart, 2, 0) // [timestamp, thread id]

	// goroutine 40: runnable->running->runnable
	w.Emit(trace.EvGoCreate, 1, 40, 7, 1)
	w.Emit(trace.EvGoStartLocal, 1, 40) // [timestamp, goroutine id]
	w.Emit(trace.EvGoSched, 1, 8)       // [timestamp, stack]

	events, err := trace.Parse(w, "")
	if err != nil {
		t.Fatalf("failed to parse test trace: %v", err)
	}

	params := &traceParams{
		events:  events,
		endTime: int64(1<<63 - 1),
	}

	// If the counts drop below 0, generateTrace will return an error.
	viewerData, err := generateTrace(params)
	if err != nil {
		t.Fatalf("generateTrace failed: %v", err)
	}
	for _, ev := range viewerData.Events {
		if ev.Name == "Goroutines" {
			cnt := ev.Arg.(*goroutineCountersArg)
			if cnt.Runnable+cnt.Running > 2 {
				t.Errorf("goroutine count=%+v; want no more than 2 goroutines in runnable/running state", cnt)
			}
			t.Logf("read %+v %+v", ev, cnt)
		}
	}
}
