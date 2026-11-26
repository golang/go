// Copyright 2026 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

//go:build !android && !arm && !mips && !mipsle

package runtime_test

import (
	"bytes"
	inttrace "internal/trace"
	"io"
	"runtime"
	"runtime/trace"
	"testing"
)

func TestTraceFailedPreemption(t *testing.T) {
	if trace.IsEnabled() {
		t.Skip("trace is already enabled")
	}
	// Keep background tracing and GC work off the P during each signal pair.
	defer runtime.GOMAXPROCS(runtime.GOMAXPROCS(1))

	run := func(t *testing.T) inttrace.GoID {
		t.Helper()
		runtime.LockOSThread()
		defer runtime.UnlockOSThread()
		runtime.Gosched()
		goid, ok := runtime.TraceFailedPreemptions()
		if !ok {
			t.Fatal("first pair of preemption signals was not acknowledged")
		}
		runtime.Gosched()
		if _, ok := runtime.TraceFailedPreemptions(); !ok {
			t.Fatal("second pair of preemption signals was not acknowledged")
		}
		runtime.Gosched()
		return inttrace.GoID(goid)
	}

	// Exercise the same signal path without either tracing or CPU profiling.
	goid, ok := runtime.TraceFailedPreemptions()
	if goid == 0 {
		t.Skip("asynchronous preemption disabled")
	}
	if !ok {
		t.Fatal("preemption signals were not acknowledged before tracing")
	}
	runtime.Gosched()

	t.Run("Trace", func(t *testing.T) {
		var buf bytes.Buffer
		if err := trace.Start(&buf); err != nil {
			t.Fatal(err)
		}
		defer trace.Stop()
		goid := run(t)
		trace.Stop()
		checkTraceFailedPreemptions(t, &buf, goid)
	})

	t.Run("FlightRecorder", func(t *testing.T) {
		fr := trace.NewFlightRecorder(trace.FlightRecorderConfig{})
		if err := fr.Start(); err != nil {
			t.Fatal(err)
		}
		defer fr.Stop()
		// Move past the initial generation before sending the signals.
		if _, err := fr.WriteTo(io.Discard); err != nil {
			t.Fatal(err)
		}
		goid := run(t)
		var buf bytes.Buffer
		if _, err := fr.WriteTo(&buf); err != nil {
			t.Fatal(err)
		}
		checkTraceFailedPreemptions(t, &buf, goid)
	})
}

func checkTraceFailedPreemptions(t *testing.T, r io.Reader, goid inttrace.GoID) {
	t.Helper()
	tr, err := inttrace.NewReader(r)
	if err != nil {
		t.Fatal(err)
	}
	failures, yields := 0, 0
	thread := inttrace.NoThread
	for {
		ev, err := tr.ReadEvent()
		if err == io.EOF {
			break
		}
		if err != nil {
			t.Fatal(err)
		}
		if ev.Goroutine() != goid {
			continue
		}
		if ev.Kind() == inttrace.EventGoPreemptFailed {
			// Background work can also request preemption of this G
			// outside the controlled pairs (for example, to advance
			// the trace generation).
			found := false
			for f := range ev.Stack().Frames() {
				if f.Func == "runtime.TraceFailedPreemptions" {
					found = true
					break
				}
			}
			if !found {
				continue
			}
			failures++
			if ev.Thread() == inttrace.NoThread || ev.Proc() == inttrace.NoProc {
				t.Errorf("failed preemption without a running M and P: %s", ev)
			}
			if thread != inttrace.NoThread && ev.Thread() != thread {
				t.Errorf("failed preemptions on different threads: %d and %d", thread, ev.Thread())
			}
			thread = ev.Thread()
			if yields != failures-1 {
				t.Errorf("failed preemption %d occurred after %d observed yields", failures, yields)
			}
		}
		if ev.Kind() == inttrace.EventStateTransition && failures == yields+1 {
			st := ev.StateTransition()
			if st.Resource.Kind == inttrace.ResourceGoroutine && st.Resource.Goroutine() == goid {
				from, to := st.Goroutine()
				if from == inttrace.GoRunning && to == inttrace.GoRunnable {
					yields++
				}
			}
		}
	}
	if failures != 2 || yields != 2 {
		t.Errorf("got %d failed preemptions and %d subsequent yields, want two of each", failures, yields)
	}
}
