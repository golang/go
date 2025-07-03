// Copyright 2023 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// Trace string management.

package runtime

import "internal/trace/tracev2"

// Trace strings.

// traceStringTable is map of string -> unique ID that also manages
// writing strings out into the trace.
type traceStringTable struct {
	// lock protects buf.
	lock mutex
	buf  *traceBuf // string batches to write out to the trace.

	// tab is a mapping of string -> unique ID.
	tab traceMap
}

// put adds a string to the table, emits it, and returns a unique ID for it.
func (t *traceStringTable) put(gen uintptr, s string) uint64 {
	// Put the string in the table.
	ss := stringStructOf(&s)
	id, added := t.tab.put(ss.str, uintptr(ss.len))
	if added {
		// Write the string to the buffer.
		systemstack(func() {
			t.writeString(gen, id, s)
		})
	}
	return id
}

// emit emits a string and creates an ID for it, but doesn't add it to the table. Returns the ID.
func (t *traceStringTable) emit(gen uintptr, s string) uint64 {
	// Grab an ID and write the string to the buffer.
	id := t.tab.stealID()
	systemstack(func() {
		t.writeString(gen, id, s)
	})
	return id
}

// writeString writes the string to t.buf.
//
// Must run on the systemstack because it acquires t.lock.
//
//go:systemstack
func (t *traceStringTable) writeString(gen uintptr, id uint64, s string) {
	// Truncate the string if necessary.
	if len(s) > tracev2.MaxEventTrailerDataSize {
		s = s[:tracev2.MaxEventTrailerDataSize]
	}

	lock(&t.lock)
	w := unsafeTraceWriter(gen, t.buf)

	// Ensure we have a place to write to.
	var flushed bool
	w, flushed = w.ensure(2 + 2*traceBytesPerNumber + len(s) /* tracev2.EvStrings + tracev2.EvString + ID + len + string data */)
	if flushed {
		// Annotate the batch as containing strings.
		w.byte(byte(tracev2.EvStrings))
	}

	// Write out the string.
	w.byte(byte(tracev2.EvString))
	w.varint(id)
	w.varint(uint64(len(s)))
	w.stringData(s)

	// Store back buf in case it was updated during ensure.
	t.buf = w.traceBuf
	unlock(&t.lock)
}

// reset clears the string table and flushes any buffers it has.
//
// Must be called only once the caller is certain nothing else will be
// added to this table.
func (t *traceStringTable) reset(gen uintptr) {
	if t.buf != nil {
		systemstack(func() {
			lock(&trace.lock)
			traceBufFlush(t.buf, gen)
			unlock(&trace.lock)
		})
		t.buf = nil
	}

	// Reset the table.
	t.tab.reset()
}
