// Copyright 2025 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package trace_test

import (
	"bytes"
	"context"
	inttrace "internal/trace"
	"internal/trace/testtrace"
	"io"
	"runtime/trace"
	"slices"
	"sync"
	"sync/atomic"
	"testing"
	"time"
)

func TestFlightRecorderDoubleStart(t *testing.T) {
	fr := trace.NewFlightRecorder(trace.FlightRecorderConfig{})
	if err := fr.Start(); err != nil {
		t.Fatalf("unexpected error on Start: %v", err)
	}
	if err := fr.Start(); err == nil {
		t.Fatalf("expected error from double Start: %v", err)
	}
	fr.Stop()
}

func TestFlightRecorderEnabled(t *testing.T) {
	fr := trace.NewFlightRecorder(trace.FlightRecorderConfig{})

	if fr.Enabled() {
		t.Fatal("flight recorder is enabled, but never started")
	}
	if err := fr.Start(); err != nil {
		t.Fatalf("unexpected error on Start: %v", err)
	}
	if !fr.Enabled() {
		t.Fatal("flight recorder is not enabled, but started")
	}
	fr.Stop()
	if fr.Enabled() {
		t.Fatal("flight recorder is enabled, but stopped")
	}
}

func TestFlightRecorderWriteToDisabled(t *testing.T) {
	var buf bytes.Buffer

	fr := trace.NewFlightRecorder(trace.FlightRecorderConfig{})
	if n, err := fr.WriteTo(&buf); err == nil {
		t.Fatalf("successfully wrote %d bytes from disabled flight recorder", n)
	}
	if err := fr.Start(); err != nil {
		t.Fatalf("unexpected error on Start: %v", err)
	}
	fr.Stop()
	if n, err := fr.WriteTo(&buf); err == nil {
		t.Fatalf("successfully wrote %d bytes from disabled flight recorder", n)
	}
}

func TestFlightRecorderConcurrentWriteTo(t *testing.T) {
	fr := trace.NewFlightRecorder(trace.FlightRecorderConfig{})
	if err := fr.Start(); err != nil {
		t.Fatalf("unexpected error on Start: %v", err)
	}

	// Start two goroutines to write snapshots.
	//
	// Most of the time one will fail and one will succeed, but we don't require this.
	// Due to a variety of factors, it's definitely possible for them both to succeed.
	// However, at least one should succeed.
	var bufs [2]bytes.Buffer
	var wg sync.WaitGroup
	var successes atomic.Uint32
	for i := range bufs {
		wg.Add(1)
		go func() {
			defer wg.Done()

			n, err := fr.WriteTo(&bufs[i])
			// TODO(go.dev/issue/63185) was an exported error. Consider refactoring.
			if err != nil && err.Error() == "call to WriteTo for trace.FlightRecorder already in progress" {
				if n != 0 {
					t.Errorf("(goroutine %d) WriteTo bytes written is non-zero for early bail out: %d", i, n)
				}
				return
			}
			if err != nil {
				t.Errorf("(goroutine %d) failed to write snapshot for unexpected reason: %v", i, err)
			}
			successes.Add(1)

			if n == 0 {
				t.Errorf("(goroutine %d) wrote invalid trace of zero bytes in size", i)
			}
			if n != int64(bufs[i].Len()) {
				t.Errorf("(goroutine %d) trace length doesn't match WriteTo result: got %d, want %d", i, n, int64(bufs[i].Len()))
			}
		}()
	}
	wg.Wait()

	// Stop tracing.
	fr.Stop()

	// Make sure at least one succeeded to write.
	if successes.Load() == 0 {
		t.Fatal("expected at least one success to write a snapshot, got zero")
	}

	// Validate the traces that came out.
	for i := range bufs {
		buf := &bufs[i]
		if buf.Len() == 0 {
			continue
		}
		testReader(t, buf.Bytes(), testtrace.ExpectSuccess())
	}
}

func TestFlightRecorder(t *testing.T) {
	testFlightRecorder(t, trace.NewFlightRecorder(trace.FlightRecorderConfig{}), func(snapshot func()) {
		snapshot()
	})
}

func TestFlightRecorderStartStop(t *testing.T) {
	fr := trace.NewFlightRecorder(trace.FlightRecorderConfig{})
	for i := 0; i < 5; i++ {
		testFlightRecorder(t, fr, func(snapshot func()) {
			snapshot()
		})
	}
}

func TestFlightRecorderLog(t *testing.T) {
	tr := testFlightRecorder(t, trace.NewFlightRecorder(trace.FlightRecorderConfig{}), func(snapshot func()) {
		trace.Log(context.Background(), "message", "hello")
		snapshot()
	})

	// Prepare to read the trace snapshot.
	r, err := inttrace.NewReader(bytes.NewReader(tr))
	if err != nil {
		t.Fatalf("unexpected error creating trace reader: %v", err)
	}

	// Find the log message in the trace.
	found := false
	for {
		ev, err := r.ReadEvent()
		if err == io.EOF {
			break
		}
		if err != nil {
			t.Fatalf("unexpected error reading trace: %v", err)
		}
		if !found && ev.Kind() == inttrace.EventLog {
			log := ev.Log()
			found = log.Category == "message" && log.Message == "hello"
		}
	}
	if !found {
		t.Errorf("failed to find expected log message (%q, %q) in snapshot", "message", "hello")
	}
}

func TestFlightRecorderGenerationCount(t *testing.T) {
	test := func(t *testing.T, fr *trace.FlightRecorder) {
		tr := testFlightRecorder(t, fr, func(snapshot func()) {
			// Sleep to let a few generations pass.
			time.Sleep(3 * time.Second)
			snapshot()
		})

		// Prepare to read the trace snapshot.
		r, err := inttrace.NewReader(bytes.NewReader(tr))
		if err != nil {
			t.Fatalf("unexpected error creating trace reader: %v", err)
		}

		// Make sure there are Sync events: at the start and end.
		var syncs []int
		evs := 0
		for {
			ev, err := r.ReadEvent()
			if err == io.EOF {
				break
			}
			if err != nil {
				t.Fatalf("unexpected error reading trace: %v", err)
			}
			if ev.Kind() == inttrace.EventSync {
				syncs = append(syncs, evs)
			}
			evs++
		}
		const wantMaxSyncs = 3
		if len(syncs) > wantMaxSyncs {
			t.Errorf("expected at most %d sync events, found %d at %d",
				wantMaxSyncs, len(syncs), syncs)
		}
		ends := []int{syncs[0], syncs[len(syncs)-1]}
		if wantEnds := []int{0, evs - 1}; !slices.Equal(wantEnds, ends) {
			t.Errorf("expected a sync event at each end of the trace, found sync events at %d instead of %d",
				ends, wantEnds)
		}
	}
	t.Run("MinAge", func(t *testing.T) {
		fr := trace.NewFlightRecorder(trace.FlightRecorderConfig{MinAge: time.Millisecond})
		test(t, fr)
	})
	t.Run("MaxBytes", func(t *testing.T) {
		fr := trace.NewFlightRecorder(trace.FlightRecorderConfig{MaxBytes: 16})
		test(t, fr)
	})
}

type flightRecorderTestFunc func(snapshot func())

func testFlightRecorder(t *testing.T, fr *trace.FlightRecorder, f flightRecorderTestFunc) []byte {
	if trace.IsEnabled() {
		t.Skip("cannot run flight recorder tests when tracing is enabled")
	}

	// Start the flight recorder.
	if err := fr.Start(); err != nil {
		t.Fatalf("unexpected error on Start: %v", err)
	}

	// Set up snapshot callback.
	var buf bytes.Buffer
	callback := func() {
		n, err := fr.WriteTo(&buf)
		if err != nil {
			t.Errorf("unexpected failure during flight recording: %v", err)
			return
		}
		if n < 16 {
			t.Errorf("expected a trace size of at least 16 bytes, got %d", n)
		}
		if n != int64(buf.Len()) {
			t.Errorf("WriteTo result doesn't match trace size: got %d, want %d", n, int64(buf.Len()))
		}
	}

	// Call the test function.
	f(callback)

	// Stop the flight recorder.
	fr.Stop()

	// Get the trace bytes; we don't want to use the Buffer as a Reader directly
	// since we may want to consume this data more than once.
	traceBytes := buf.Bytes()

	// Parse the trace to make sure it's not broken.
	testReader(t, traceBytes, testtrace.ExpectSuccess())
	return traceBytes
}

func testReader(t *testing.T, tb []byte, exp *testtrace.Expectation) {
	r, err := inttrace.NewReader(bytes.NewReader(tb))
	if err != nil {
		if err := exp.Check(err); err != nil {
			t.Error(err)
		}
		return
	}
	v := testtrace.NewValidator()
	v.SkipClockSnapshotChecks()
	for {
		ev, err := r.ReadEvent()
		if err == io.EOF {
			break
		}
		if err != nil {
			if err := exp.Check(err); err != nil {
				t.Error(err)
			}
			return
		}
		if err := v.Event(ev); err != nil {
			t.Error(err)
		}
	}
	if err := exp.Check(nil); err != nil {
		t.Error(err)
	}
	if t.Failed() || *dumpTraces {
		testtrace.Dump(t, "trace", tb, *dumpTraces)
	}
}

func TestTraceAndFlightRecorder(t *testing.T) {
	var tBuf, frBuf bytes.Buffer
	if err := trace.Start(&tBuf); err != nil {
		t.Errorf("unable to start execution tracer: %s", err)
	}
	fr := trace.NewFlightRecorder(trace.FlightRecorderConfig{MaxBytes: 16})
	fr.Start()
	fr.WriteTo(&frBuf)
	fr.Stop()
	trace.Stop()
	if tBuf.Len() == 0 || frBuf.Len() == 0 {
		t.Errorf("None of these should be equal to zero: %d %d", tBuf.Len(), frBuf.Len())
	}
	if tBuf.Len() <= frBuf.Len() {
		t.Errorf("trace should be longer than the flight recorder: trace=%d flight record=%d", tBuf.Len(), frBuf.Len())
	}
}
