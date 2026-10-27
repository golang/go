// Copyright 2023 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// Trace time and clock.

package runtime

import (
	"internal/trace/tracev2"
	_ "unsafe"
)

// Timestamps in trace are produced through nanotime and divided by traceTimeDiv.
// This makes absolute values of timestamp diffs smaller, and so they are
// encoded in fewer bytes.
//
// The target resolution in all cases is 64 nanoseconds.
// This is based on the fact that fundamentally the execution tracer won't emit
// events more frequently than roughly every 200 ns or so, because that's roughly
// how long it takes to call through the scheduler.
// We could be more aggressive and bump this up to 128 ns while still getting
// useful data, but the extra bit doesn't save us that much and the headroom is
// nice to have.
const traceTimeDiv = 64

// traceTime represents a timestamp for the trace.
type traceTime uint64

// traceClockNow returns a monotonic timestamp in units of traceTimeDiv nanoseconds.
//
// nosplit because it's called from exitsyscall and various trace writing functions,
// which are nosplit.
//
// traceClockNow is called by runtime/trace and golang.org/x/exp/trace using linkname.
//
//go:linkname traceClockNow
//go:nosplit
func traceClockNow() traceTime {
	return traceTime(nanotime() / traceTimeDiv)
}

// traceClockUnitsPerSecond returns the number of trace clock units per
// second that elapse.
//
//go:linkname traceClockUnitsPerSecond runtime/trace.runtime_traceClockUnitsPerSecond
func traceClockUnitsPerSecond() uint64 {
	return 1e9 / traceTimeDiv
}

func traceSyncBatch(gen uintptr, frequency uint64) {
	w := unsafeTraceWriter(gen, nil)

	// Ensure we have a place to write to.
	w, _ = w.ensure(3 /* EvSync + EvFrequency + EvClockSnapshot */ + 5*traceBytesPerNumber /* frequency, timestamp, mono, sec, nsec */)

	// Write out the sync batch event.
	w.byte(byte(tracev2.EvSync))

	// Write out the frequency event.
	w.byte(byte(tracev2.EvFrequency))
	w.varint(frequency)

	// Write out the clock snapshot event.
	sec, nsec, mono := time_now()
	ts := traceClockNow()
	if ts <= w.traceBuf.lastTime {
		ts = w.traceBuf.lastTime + 1
	}
	tsDiff := uint64(ts - w.traceBuf.lastTime)
	w.traceBuf.lastTime = ts
	w.byte(byte(tracev2.EvClockSnapshot))
	w.varint(tsDiff)
	w.varint(uint64(mono))
	w.varint(uint64(sec))
	w.varint(uint64(nsec))

	// Immediately flush the buffer.
	systemstack(func() {
		lock(&trace.lock)
		traceBufFlush(w.traceBuf, gen)
		unlock(&trace.lock)
	})
}
