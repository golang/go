// Copyright 2023 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// Trace buffer management.

package runtime

import (
	"internal/runtime/sys"
	"unsafe"
)

// Maximum number of bytes required to encode uint64 in base-128.
const traceBytesPerNumber = 10

// traceWriter is the interface for writing all trace data.
//
// This type is passed around as a value, and all of its methods return
// a new traceWriter. This allows for chaining together calls in a fluent-style
// API. This is partly stylistic, and very slightly for performance, since
// the compiler can destructure this value and pass it between calls as
// just regular arguments. However, this style is not load-bearing, and
// we can change it if it's deemed too error-prone.
type traceWriter struct {
	traceLocker
	*traceBuf
}

// writer returns an a traceWriter that writes into the current M's stream.
//
// Once this is called, the caller must guard against stack growth until
// end is called on it. Therefore, it's highly recommended to use this
// API in a "fluent" style, for example tl.writer().event(...).end().
// Better yet, callers just looking to write events should use eventWriter
// when possible, which is a much safer wrapper around this function.
//
// nosplit to allow for safe reentrant tracing from stack growth paths.
//
//go:nosplit
func (tl traceLocker) writer() traceWriter {
	if debugTraceReentrancy {
		// Checks that the invariants of this function are being upheld.
		gp := getg()
		if gp == gp.m.curg {
			tl.mp.trace.oldthrowsplit = gp.throwsplit
			gp.throwsplit = true
		}
	}
	return traceWriter{traceLocker: tl, traceBuf: tl.mp.trace.buf[tl.gen%2]}
}

// unsafeTraceWriter produces a traceWriter that doesn't lock the trace.
//
// It should only be used in contexts where either:
// - Another traceLocker is held.
// - trace.gen is prevented from advancing.
//
// This does not have the same stack growth restrictions as traceLocker.writer.
//
// buf may be nil.
func unsafeTraceWriter(gen uintptr, buf *traceBuf) traceWriter {
	return traceWriter{traceLocker: traceLocker{gen: gen}, traceBuf: buf}
}

// event writes out the bytes of an event into the event stream.
//
// nosplit because it's part of writing an event for an M, which must not
// have any stack growth.
//
//go:nosplit
func (w traceWriter) event(ev traceEv, args ...traceArg) traceWriter {
	// N.B. Everything in this call must be nosplit to maintain
	// the stack growth related invariants for writing events.

	// Make sure we have room.
	w, _ = w.ensure(1 + (len(args)+1)*traceBytesPerNumber)

	// Compute the timestamp diff that we'll put in the trace.
	ts := traceClockNow()
	if ts <= w.traceBuf.lastTime {
		ts = w.traceBuf.lastTime + 1
	}
	tsDiff := uint64(ts - w.traceBuf.lastTime)
	w.traceBuf.lastTime = ts

	// Write out event.
	w.byte(byte(ev))
	w.varint(tsDiff)
	for _, arg := range args {
		w.varint(uint64(arg))
	}
	return w
}

// end writes the buffer back into the m.
//
// nosplit because it's part of writing an event for an M, which must not
// have any stack growth.
//
//go:nosplit
func (w traceWriter) end() {
	if w.mp == nil {
		// Tolerate a nil mp. It makes code that creates traceWriters directly
		// less error-prone.
		return
	}
	w.mp.trace.buf[w.gen%2] = w.traceBuf
	if debugTraceReentrancy {
		// The writer is no longer live, we can drop throwsplit (if it wasn't
		// already set upon entry).
		gp := getg()
		if gp == gp.m.curg {
			gp.throwsplit = w.mp.trace.oldthrowsplit
		}
	}
}

// ensure makes sure that at least maxSize bytes are available to write.
//
// Returns whether the buffer was flushed.
//
// nosplit because it's part of writing an event for an M, which must not
// have any stack growth.
//
//go:nosplit
func (w traceWriter) ensure(maxSize int) (traceWriter, bool) {
	refill := w.traceBuf == nil || !w.available(maxSize)
	if refill {
		w = w.refill(traceNoExperiment)
	}
	return w, refill
}

// flush puts w.traceBuf on the queue of full buffers.
//
// nosplit because it's part of writing an event for an M, which must not
// have any stack growth.
//
//go:nosplit
func (w traceWriter) flush() traceWriter {
	systemstack(func() {
		lock(&trace.lock)
		if w.traceBuf != nil {
			traceBufFlush(w.traceBuf, w.gen)
		}
		unlock(&trace.lock)
	})
	w.traceBuf = nil
	return w
}

// refill puts w.traceBuf on the queue of full buffers and refresh's w's buffer.
//
// exp indicates whether the refilled batch should be EvExperimentalBatch.
//
// nosplit because it's part of writing an event for an M, which must not
// have any stack growth.
//
//go:nosplit
func (w traceWriter) refill(exp traceExperiment) traceWriter {
	systemstack(func() {
		lock(&trace.lock)
		if w.traceBuf != nil {
			traceBufFlush(w.traceBuf, w.gen)
		}
		if trace.empty != nil {
			w.traceBuf = trace.empty
			trace.empty = w.traceBuf.link
			unlock(&trace.lock)
		} else {
			unlock(&trace.lock)
			w.traceBuf = (*traceBuf)(sysAlloc(unsafe.Sizeof(traceBuf{}), &memstats.other_sys))
			if w.traceBuf == nil {
				throw("trace: out of memory")
			}
		}
	})
	// Initialize the buffer.
	ts := traceClockNow()
	if ts <= w.traceBuf.lastTime {
		ts = w.traceBuf.lastTime + 1
	}
	w.traceBuf.lastTime = ts
	w.traceBuf.link = nil
	w.traceBuf.pos = 0

	// Tolerate a nil mp.
	mID := ^uint64(0)
	if w.mp != nil {
		mID = uint64(w.mp.procid)
	}

	// Write the buffer's header.
	if exp == traceNoExperiment {
		w.byte(byte(traceEvEventBatch))
	} else {
		w.byte(byte(traceEvExperimentalBatch))
		w.byte(byte(exp))
	}
	w.varint(uint64(w.gen))
	w.varint(uint64(mID))
	w.varint(uint64(ts))
	w.traceBuf.lenPos = w.varintReserve()
	return w
}

// traceBufQueue is a FIFO of traceBufs.
type traceBufQueue struct {
	head, tail *traceBuf
}

// push queues buf into queue of buffers.
func (q *traceBufQueue) push(buf *traceBuf) {
	buf.link = nil
	if q.head == nil {
		q.head = buf
	} else {
		q.tail.link = buf
	}
	q.tail = buf
}

// pop dequeues from the queue of buffers.
func (q *traceBufQueue) pop() *traceBuf {
	buf := q.head
	if buf == nil {
		return nil
	}
	q.head = buf.link
	if q.head == nil {
		q.tail = nil
	}
	buf.link = nil
	return buf
}

func (q *traceBufQueue) empty() bool {
	return q.head == nil
}

// traceBufHeader is per-P tracing buffer.
type traceBufHeader struct {
	link     *traceBuf // in trace.empty/full
	lastTime traceTime // when we wrote the last event
	pos      int       // next write offset in arr
	lenPos   int       // position of batch length value
}

// traceBuf is per-M tracing buffer.
//
// TODO(mknyszek): Rename traceBuf to traceBatch, since they map 1:1 with event batches.
type traceBuf struct {
	_ sys.NotInHeap
	traceBufHeader
	arr [64<<10 - unsafe.Sizeof(traceBufHeader{})]byte // underlying buffer for traceBufHeader.buf
}

// byte appends v to buf.
//
// nosplit because it's part of writing an event for an M, which must not
// have any stack growth.
//
//go:nosplit
func (buf *traceBuf) byte(v byte) {
	buf.arr[buf.pos] = v
	buf.pos++
}

// varint appends v to buf in little-endian-base-128 encoding.
//
// nosplit because it's part of writing an event for an M, which must not
// have any stack growth.
//
//go:nosplit
func (buf *traceBuf) varint(v uint64) {
	pos := buf.pos
	arr := buf.arr[pos : pos+traceBytesPerNumber]
	for i := range arr {
		if v < 0x80 {
			pos += i + 1
			arr[i] = byte(v)
			break
		}
		arr[i] = 0x80 | byte(v)
		v >>= 7
	}
	buf.pos = pos
}

// varintReserve reserves enough space in buf to hold any varint.
//
// Space reserved this way can be filled in with the varintAt method.
//
// nosplit because it's part of writing an event for an M, which must not
// have any stack growth.
//
//go:nosplit
func (buf *traceBuf) varintReserve() int {
	p := buf.pos
	buf.pos += traceBytesPerNumber
	return p
}

// stringData appends s's data directly to buf.
//
// nosplit because it's part of writing an event for an M, which must not
// have any stack growth.
//
//go:nosplit
func (buf *traceBuf) stringData(s string) {
	buf.pos += copy(buf.arr[buf.pos:], s)
}

// nosplit because it's part of writing an event for an M, which must not
// have any stack growth.
//
//go:nosplit
func (buf *traceBuf) available(size int) bool {
	return len(buf.arr)-buf.pos >= size
}

// varintAt writes varint v at byte position pos in buf. This always
// consumes traceBytesPerNumber bytes. This is intended for when the caller
// needs to reserve space for a varint but can't populate it until later.
// Use varintReserve to reserve this space.
//
// nosplit because it's part of writing an event for an M, which must not
// have any stack growth.
//
//go:nosplit
func (buf *traceBuf) varintAt(pos int, v uint64) {
	for i := 0; i < traceBytesPerNumber; i++ {
		if i < traceBytesPerNumber-1 {
			buf.arr[pos] = 0x80 | byte(v)
		} else {
			buf.arr[pos] = byte(v)
		}
		v >>= 7
		pos++
	}
	if v != 0 {
		throw("v could not fit in traceBytesPerNumber")
	}
}

// traceBufFlush flushes a trace buffer.
//
// Must run on the system stack because trace.lock must be held.
//
//go:systemstack
func traceBufFlush(buf *traceBuf, gen uintptr) {
	assertLockHeld(&trace.lock)

	// Write out the non-header length of the batch in the header.
	//
	// Note: the length of the header is not included to make it easier
	// to calculate this value when deserializing and reserializing the
	// trace. Varints can have additional padding of zero bits that is
	// quite difficult to preserve, and if we include the header we
	// force serializers to do more work. Nothing else actually needs
	// padding.
	buf.varintAt(buf.lenPos, uint64(buf.pos-(buf.lenPos+traceBytesPerNumber)))
	trace.full[gen%2].push(buf)

	// Notify the scheduler that there's work available and that the trace
	// reader should be scheduled.
	if !trace.workAvailable.Load() {
		trace.workAvailable.Store(true)
	}
}
