// Copyright 2025 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package trace

import (
	"fmt"
	"io"
	"sync"
	"time"
	_ "unsafe" // added for go linkname usage
)

// FlightRecorder represents a single consumer of a Go execution
// trace.
// It tracks a moving window over the execution trace produced by
// the runtime, always containing the most recent trace data.
//
// At most one flight recorder may be active at any given time,
// though flight recording is allowed to be concurrently active
// with a trace consumer using trace.Start.
// This restriction of only a single flight recorder may be removed
// in the future.
type FlightRecorder struct {
	err error

	// State specific to the recorder.
	header [16]byte
	active rawGeneration
	ringMu sync.Mutex
	ring   []rawGeneration
	freq   frequency // timestamp conversion factor, from the runtime

	// Externally-set options.
	targetSize   uint64
	targetPeriod time.Duration

	enabled bool       // whether the flight recorder is enabled.
	writing sync.Mutex // protects concurrent calls to WriteTo

	// The values of targetSize and targetPeriod we've committed to since the last Start.
	wantSize uint64
	wantDur  time.Duration
}

// NewFlightRecorder creates a new flight recorder from the provided configuration.
func NewFlightRecorder(cfg FlightRecorderConfig) *FlightRecorder {
	fr := new(FlightRecorder)
	if cfg.MaxBytes != 0 {
		fr.targetSize = cfg.MaxBytes
	} else {
		fr.targetSize = 10 << 20 // 10 MiB.
	}

	if cfg.MinAge != 0 {
		fr.targetPeriod = cfg.MinAge
	} else {
		fr.targetPeriod = 10 * time.Second
	}
	return fr
}

// Start activates the flight recorder and begins recording trace data.
// Only one call to trace.Start may be active at any given time.
// In addition, currently only one flight recorder may be active in the program.
// Returns an error if the flight recorder cannot be started or is already started.
func (fr *FlightRecorder) Start() error {
	if fr.enabled {
		return fmt.Errorf("cannot enable a enabled flight recorder")
	}
	fr.wantSize = fr.targetSize
	fr.wantDur = fr.targetPeriod
	fr.err = nil
	fr.freq = frequency(1.0 / (float64(runtime_traceClockUnitsPerSecond()) / 1e9))

	// Start tracing, data is sent to a recorder which forwards it to our own
	// storage.
	if err := tracing.subscribeFlightRecorder(&recorder{r: fr}); err != nil {
		return err
	}

	fr.enabled = true
	return nil
}

// Stop ends recording of trace data. It blocks until any concurrent WriteTo calls complete.
func (fr *FlightRecorder) Stop() {
	if !fr.enabled {
		return
	}
	fr.enabled = false
	tracing.unsubscribeFlightRecorder()

	// Reset all state. No need to lock because the reader has already exited.
	fr.active = rawGeneration{}
	fr.ring = nil
}

// Enabled returns true if the flight recorder is active.
// Specifically, it will return true if Start did not return an error, and Stop has not yet been called.
// It is safe to call from multiple goroutines simultaneously.
func (fr *FlightRecorder) Enabled() bool { return fr.enabled }

// WriteTo snapshots the moving window tracked by the flight recorder.
// The snapshot is expected to contain data that is up-to-date as of when WriteTo is called,
// though this is not a hard guarantee.
// Only one goroutine may execute WriteTo at a time.
// An error is returned upon failure to write to w, if another WriteTo call is already in-progress,
// or if the flight recorder is inactive.
func (fr *FlightRecorder) WriteTo(w io.Writer) (n int64, err error) {
	if !fr.enabled {
		return 0, fmt.Errorf("cannot snapshot a disabled flight recorder")
	}
	if !fr.writing.TryLock() {
		// Indicates that a call to WriteTo was made while one was already in progress.
		// If the caller of WriteTo sees this error, they should use the result from the other call to WriteTo.
		return 0, fmt.Errorf("call to WriteTo for trace.FlightRecorder already in progress")
	}
	defer fr.writing.Unlock()

	// Force a global buffer flush.
	runtime_traceAdvance(false)

	// Now that everything has been flushed and written, grab whatever we have.
	//
	// N.B. traceAdvance blocks until the tracer goroutine has actually written everything
	// out, which means the generation we just flushed must have been already been observed
	// by the recorder goroutine. Because we flushed twice, the first flush is guaranteed to
	// have been both completed *and* processed by the recorder goroutine.
	fr.ringMu.Lock()
	gens := fr.ring
	fr.ringMu.Unlock()

	// Write the header.
	nw, err := w.Write(fr.header[:])
	if err != nil {
		return int64(nw), err
	}
	n += int64(nw)

	// Write all the data.
	for _, gen := range gens {
		for _, batch := range gen.batches {
			// Write batch data.
			nw, err = w.Write(batch.data)
			n += int64(nw)
			if err != nil {
				return n, err
			}
		}
	}
	return n, nil
}

type FlightRecorderConfig struct {
	// MinAge is a lower bound on the age of an event in the flight recorder's window.
	//
	// The flight recorder will strive to promptly discard events older than the minimum age,
	// but older events may appear in the window snapshot. The age setting will always be
	// overridden by MaxBytes.
	//
	// If this is 0, the minimum age is implementation defined, but can be assumed to be on the order
	// of seconds.
	MinAge time.Duration

	// MaxBytes is an upper bound on the size of the window in bytes.
	//
	// This setting takes precedence over MinAge.
	// However, it does not make any guarantees on the size of the data WriteTo will write,
	// nor does it guarantee memory overheads will always stay below MaxBytes. Treat it
	// as a hint.
	//
	// If this is 0, the maximum size is implementation defined.
	MaxBytes uint64
}

//go:linkname runtime_traceClockUnitsPerSecond
func runtime_traceClockUnitsPerSecond() uint64

//go:linkname runtime_traceAdvance runtime.traceAdvance
func runtime_traceAdvance(stopTrace bool)
