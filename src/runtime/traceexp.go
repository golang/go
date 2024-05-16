// Copyright 2024 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package runtime

// traceExpWriter is a wrapper around trace writer that produces traceEvExperimentalBatch
// batches. This means that the data written to the writer need not conform to the standard
// trace format.
type traceExpWriter struct {
	traceWriter
	exp traceExperiment
}

// unsafeTraceExpWriter produces a traceExpWriter that doesn't lock the trace.
//
// It should only be used in contexts where either:
// - Another traceLocker is held.
// - trace.gen is prevented from advancing.
//
// buf may be nil.
func unsafeTraceExpWriter(gen uintptr, buf *traceBuf, exp traceExperiment) traceExpWriter {
	return traceExpWriter{traceWriter{traceLocker: traceLocker{gen: gen}, traceBuf: buf}, exp}
}

// ensure makes sure that at least maxSize bytes are available to write.
//
// Returns whether the buffer was flushed.
func (w traceExpWriter) ensure(maxSize int) (traceExpWriter, bool) {
	refill := w.traceBuf == nil || !w.available(maxSize)
	if refill {
		w.traceWriter = w.traceWriter.refill(w.exp)
	}
	return w, refill
}

// traceExperiment is an enumeration of the different kinds of experiments supported for tracing.
type traceExperiment uint8

const (
	// traceNoExperiment indicates no experiment.
	traceNoExperiment traceExperiment = iota

	// traceExperimentAllocFree is an experiment to add alloc/free events to the trace.
	traceExperimentAllocFree
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
