// Copyright 2024 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package runtime

// expWriter returns a traceWriter that writes into the current M's stream for
// the given experiment.
func (tl traceLocker) expWriter(exp traceExperiment) traceWriter {
	return traceWriter{traceLocker: tl, traceBuf: tl.mp.trace.buf[tl.gen%2][exp], exp: exp}
}

// unsafeTraceExpWriter produces a traceWriter for experimental trace batches
// that doesn't lock the trace. Data written to experimental batches need not
// conform to the standard trace format.
//
// It should only be used in contexts where either:
// - Another traceLocker is held.
// - trace.gen is prevented from advancing.
//
// This does not have the same stack growth restrictions as traceLocker.writer.
//
// buf may be nil.
func unsafeTraceExpWriter(gen uintptr, buf *traceBuf, exp traceExperiment) traceWriter {
	return traceWriter{traceLocker: traceLocker{gen: gen}, traceBuf: buf, exp: exp}
}

// traceExperiment is an enumeration of the different kinds of experiments supported for tracing.
type traceExperiment uint8

const (
	// traceNoExperiment indicates no experiment.
	traceNoExperiment traceExperiment = iota

	// traceExperimentAllocFree is an experiment to add alloc/free events to the trace.
	traceExperimentAllocFree

	// traceNumExperiments is the number of trace experiments (and 1 higher than
	// the highest numbered experiment).
	traceNumExperiments
)

// Experimental events.
const (
	_ traceEv = 127 + iota

	// Experimental events for ExperimentAllocFree.

	// Experimental heap span events. IDs map reversibly to base addresses.
	traceEvSpan      // heap span exists [timestamp, id, npages, type/class]
	traceEvSpanAlloc // heap span alloc [timestamp, id, npages, type/class]
	traceEvSpanFree  // heap span free [timestamp, id]

	// Experimental heap object events. IDs map reversibly to addresses.
	traceEvHeapObject      // heap object exists [timestamp, id, type]
	traceEvHeapObjectAlloc // heap object alloc [timestamp, id, type]
	traceEvHeapObjectFree  // heap object free [timestamp, id]

	// Experimental goroutine stack events. IDs map reversibly to addresses.
	traceEvGoroutineStack      // stack exists [timestamp, id, order]
	traceEvGoroutineStackAlloc // stack alloc [timestamp, id, order]
	traceEvGoroutineStackFree  // stack free [timestamp, id]
)
