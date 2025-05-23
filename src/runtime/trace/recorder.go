// Copyright 2025 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package trace

import (
	"fmt"
	"slices"
	"time"
	_ "unsafe" // added for go linkname usage
)

// A recorder receives bytes from the runtime tracer, processes it.
type recorder struct {
	r *FlightRecorder

	headerReceived bool
}

func (w *recorder) Write(b []byte) (n int, err error) {
	r := w.r

	defer func() {
		if err != nil {
			// Propagate errors to the flightrecorder.
			if r.err == nil {
				r.err = err
			}
		}
	}()

	if !w.headerReceived {
		if len(b) < len(r.header) {
			return 0, fmt.Errorf("expected at least %d bytes in the first write", len(r.header))
		}
		r.header = ([16]byte)(b[:16])
		n += 16
		w.headerReceived = true
	}
	if len(b) == n {
		return 0, nil
	}
	ba, gen, nb, err := readBatch(b[n:]) // Every write from the runtime is guaranteed to be a complete batch.
	if err != nil {
		return len(b) - int(nb) - n, err
	}
	n += int(nb)

	// Append the batch to the current generation.
	if r.active.gen == 0 {
		r.active.gen = gen
	}
	if r.active.minTime == 0 || r.active.minTime > r.freq.mul(ba.time) {
		r.active.minTime = r.freq.mul(ba.time)
	}
	r.active.size += len(ba.data)
	r.active.batches = append(r.active.batches, ba)

	return len(b), nil
}

func (w *recorder) endGeneration() {
	r := w.r

	// Check if we're entering a new generation.
	r.ringMu.Lock()

	// Get the current trace clock time.
	now := traceTimeNow(r.freq)

	// Add the current generation to the ring. Make sure we always have at least one
	// complete generation by putting the active generation onto the new list, regardless
	// of whatever our settings are.
	//
	// N.B. Let's completely replace the ring here, so that WriteTo can just make a copy
	// and not worry about aliasing. This creates allocations, but at a very low rate.
	newRing := []rawGeneration{r.active}
	size := r.active.size
	for i := len(r.ring) - 1; i >= 0; i-- {
		// Stop adding older generations if the new ring already exceeds the thresholds.
		// This ensures we keep generations that cross a threshold, but not any that lie
		// entirely outside it.
		if uint64(size) > r.wantSize || now.Sub(newRing[len(newRing)-1].minTime) > r.wantDur {
			break
		}
		size += r.ring[i].size
		newRing = append(newRing, r.ring[i])
	}
	slices.Reverse(newRing)
	r.ring = newRing
	r.ringMu.Unlock()

	// Start a new active generation.
	r.active = rawGeneration{}
}

type rawGeneration struct {
	gen     uint64
	size    int
	minTime eventTime
	batches []batch
}

func traceTimeNow(freq frequency) eventTime {
	return freq.mul(timestamp(runtime_traceClockNow()))
}

//go:linkname runtime_traceClockNow runtime.traceClockNow
func runtime_traceClockNow() uint64

// frequency is nanoseconds per timestamp unit.
type frequency float64

// mul multiplies an unprocessed to produce a time in nanoseconds.
func (f frequency) mul(t timestamp) eventTime {
	return eventTime(float64(t) * float64(f))
}

// eventTime is a timestamp in nanoseconds.
//
// It corresponds to the monotonic clock on the platform that the
// trace was taken, and so is possible to correlate with timestamps
// for other traces taken on the same machine using the same clock
// (i.e. no reboots in between).
//
// The actual absolute value of the timestamp is only meaningful in
// relation to other timestamps from the same clock.
//
// BUG: Timestamps coming from traces on Windows platforms are
// only comparable with timestamps from the same trace. Timestamps
// across traces cannot be compared, because the system clock is
// not used as of Go 1.22.
//
// BUG: Traces produced by Go versions 1.21 and earlier cannot be
// compared with timestamps from other traces taken on the same
// machine. This is because the system clock was not used at all
// to collect those timestamps.
type eventTime int64

// Sub subtracts t0 from t, returning the duration in nanoseconds.
func (t eventTime) Sub(t0 eventTime) time.Duration {
	return time.Duration(int64(t) - int64(t0))
}
