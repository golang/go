// Copyright 2025 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package trace_test

import (
	"bytes"
	inttrace "internal/trace"
	"internal/trace/testtrace"
	"io"
	"runtime"
	"runtime/trace"
	"slices"
	"testing"
)

func TestSubscribers(t *testing.T) {
	validate := func(t *testing.T, source string, tr *bytes.Buffer) {
		defer func() {
			if t.Failed() {
				testtrace.Dump(t, "trace", tr.Bytes(), *dumpTraces)
			}
		}()

		// Prepare to read the trace snapshot.
		r, err := inttrace.NewReader(tr)
		if err != nil {
			t.Errorf("unexpected error creating trace reader for %s: %v", source, err)
			return
		}

		v := testtrace.NewValidator()
		// These platforms can't guarantee a monotonically increasing clock reading in a short trace.
		if runtime.GOOS == "windows" || runtime.GOARCH == "wasm" {
			v.SkipClockSnapshotChecks()
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
				t.Errorf("unexpected error reading trace for %s: %v", source, err)
			}
			if err := v.Event(ev); err != nil {
				t.Errorf("event validation failed: %s", err)
			}
			if ev.Kind() == inttrace.EventSync {
				syncs = append(syncs, evs)
			}
			evs++
		}
		if !t.Failed() {
			ends := []int{syncs[0], syncs[len(syncs)-1]}
			if wantEnds := []int{0, evs - 1}; !slices.Equal(wantEnds, ends) {
				t.Errorf("expected a sync event at each end of the trace, found sync events at %d instead of %d for %s",
					ends, wantEnds, source)
			}
		}
	}

	validateTraces := func(t *testing.T, trace, frTrace *bytes.Buffer) {
		validate(t, "tracer", trace)
		validate(t, "flightRecorder", frTrace)
	}
	startFlightRecorder := func(t *testing.T) *trace.FlightRecorder {
		fr := trace.NewFlightRecorder(trace.FlightRecorderConfig{})
		if err := fr.Start(); err != nil {
			t.Fatalf("unexpected error creating flight recorder: %v", err)
		}
		return fr
	}
	startTrace := func(t *testing.T, w io.Writer) {
		if err := trace.Start(w); err != nil {
			t.Fatalf("unexpected error starting flight recorder: %v", err)
		}
	}
	stopFlightRecorder := func(t *testing.T, fr *trace.FlightRecorder, w io.Writer) {
		if _, err := fr.WriteTo(w); err != nil {
			t.Fatalf("unexpected error writing trace from flight recorder: %v", err)
		}
		fr.Stop()
	}
	stopTrace := func() {
		trace.Stop()
	}
	t.Run("start(flight)_start(trace)_stop(trace)_stop(flight)", func(t *testing.T) {
		if trace.IsEnabled() {
			t.Skip("skipping because trace is already enabled")
		}
		frBuf := new(bytes.Buffer)
		tBuf := new(bytes.Buffer)
		fr := startFlightRecorder(t)
		defer fr.Stop()
		startTrace(t, tBuf)
		defer trace.Stop()
		stopTrace()
		stopFlightRecorder(t, fr, frBuf)
		validateTraces(t, tBuf, frBuf)
	})
	t.Run("start(trace)_start(flight)_stop(trace)_stop(flight)", func(t *testing.T) {
		if trace.IsEnabled() {
			t.Skip("skipping because trace is already enabled")
		}
		frBuf := new(bytes.Buffer)
		tBuf := new(bytes.Buffer)
		startTrace(t, tBuf)
		defer trace.Stop()
		fr := startFlightRecorder(t)
		defer fr.Stop()
		stopTrace()
		stopFlightRecorder(t, fr, frBuf)
		validateTraces(t, tBuf, frBuf)
	})
	t.Run("start(flight)_stop(flight)_start(trace)_stop(trace)", func(t *testing.T) {
		if trace.IsEnabled() {
			t.Skip("skipping because trace is already enabled")
		}
		frBuf := new(bytes.Buffer)
		tBuf := new(bytes.Buffer)
		fr := startFlightRecorder(t)
		defer fr.Stop()
		stopFlightRecorder(t, fr, frBuf)
		startTrace(t, tBuf)
		defer trace.Stop()
		stopTrace()
		validateTraces(t, tBuf, frBuf)
	})
	t.Run("start(flight)_stop(flight)_start(trace)_stop(trace)", func(t *testing.T) {
		if trace.IsEnabled() {
			t.Skip("skipping because trace is already enabled")
		}
		frBuf := new(bytes.Buffer)
		tBuf := new(bytes.Buffer)
		fr := startFlightRecorder(t)
		defer fr.Stop()
		stopFlightRecorder(t, fr, frBuf)
		startTrace(t, tBuf)
		defer trace.Stop()
		stopTrace()
		validateTraces(t, tBuf, frBuf)
	})
	t.Run("start(flight)_start(trace)_stop(flight)_stop(trace)", func(t *testing.T) {
		if trace.IsEnabled() {
			t.Skip("skipping because trace is already enabled")
		}
		frBuf := new(bytes.Buffer)
		tBuf := new(bytes.Buffer)
		fr := startFlightRecorder(t)
		defer fr.Stop()
		startTrace(t, tBuf)
		defer trace.Stop()
		stopFlightRecorder(t, fr, frBuf)
		stopTrace()
		validateTraces(t, tBuf, frBuf)
	})
}
