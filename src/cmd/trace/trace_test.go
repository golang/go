package main

import (
	"internal/trace"
	"strings"
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

func TestGoroutineFilter(t *testing.T) {
	// Test that we handle state changes to selected goroutines
	// caused by events on goroutines that are not selected.

	w := trace.NewWriter()
	w.Emit(trace.EvBatch, 0, 0)  // start of per-P batch event [pid, timestamp]
	w.Emit(trace.EvFrequency, 1) // [ticks per second]

	// goroutine 10: blocked
	w.Emit(trace.EvGoCreate, 1, 10, 1, 1) // [timestamp, new goroutine id, new stack id, stack id]
	w.Emit(trace.EvGoWaiting, 1, 10)      // [timestamp, goroutine id]

	// goroutine 20: runnable->running->unblock 10
	w.Emit(trace.EvGoCreate, 1, 20, 7, 1)
	w.Emit(trace.EvGoStartLocal, 1, 20)      // [timestamp, goroutine id]
	w.Emit(trace.EvGoUnblockLocal, 1, 10, 8) // [timestamp, goroutine id, stack]
	w.Emit(trace.EvGoEnd, 1)                 // [timestamp]

	// goroutine 10: runnable->running->block
	w.Emit(trace.EvGoStartLocal, 1, 10) // [timestamp, goroutine id]
	w.Emit(trace.EvGoBlock, 1, 9)       // [timestamp, stack]

	events, err := trace.Parse(w, "")
	if err != nil {
		t.Fatalf("failed to parse test trace: %v", err)
	}

	params := &traceParams{
		events:  events,
		endTime: int64(1<<63 - 1),
		gs:      map[uint64]bool{10: true},
	}

	_, err = generateTrace(params)
	if err != nil {
		t.Fatalf("generateTrace failed: %v", err)
	}
}

func TestPreemptedMarkAssist(t *testing.T) {
	w := trace.NewWriter()
	w.Emit(trace.EvBatch, 0, 0)  // start of per-P batch event [pid, timestamp]
	w.Emit(trace.EvFrequency, 1) // [ticks per second]

	// goroutine 9999: running -> mark assisting -> preempted -> assisting -> running -> block
	w.Emit(trace.EvGoCreate, 1, 9999, 1, 1) // [timestamp, new goroutine id, new stack id, stack id]
	w.Emit(trace.EvGoStartLocal, 1, 9999)   // [timestamp, goroutine id]
	w.Emit(trace.EvGCMarkAssistStart, 1, 2) // [timestamp, stack]
	w.Emit(trace.EvGoPreempt, 1, 3)         // [timestamp, stack]
	w.Emit(trace.EvGoStartLocal, 1, 9999)   // [timestamp, goroutine id]
	w.Emit(trace.EvGCMarkAssistDone, 1)     // [timestamp]
	w.Emit(trace.EvGoBlock, 1, 4)           // [timestamp, stack]

	events, err := trace.Parse(w, "")
	if err != nil {
		t.Fatalf("failed to parse test trace: %v", err)
	}

	params := &traceParams{
		events:  events,
		endTime: int64(1<<63 - 1),
	}

	viewerData, err := generateTrace(params)
	if err != nil {
		t.Fatalf("generateTrace failed: %v", err)
	}

	marks := 0
	for _, ev := range viewerData.Events {
		if strings.Contains(ev.Name, "MARK ASSIST") {
			marks++
		}
	}
	if marks != 2 {
		t.Errorf("Got %v MARK ASSIST events, want %v", marks, 2)
	}
}
