// Copyright 2023 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package trace

import (
	"cmp"
	"encoding/binary"
	"fmt"

	"internal/trace/v2/event"
	"internal/trace/v2/event/go122"
)

type batchCursor struct {
	m       ThreadID
	lastTs  Time
	idx     int       // next index into []batch
	dataOff int       // next index into batch.data
	ev      baseEvent // last read event
}

func (b *batchCursor) nextEvent(batches []batch, freq frequency) (ok bool, err error) {
	// Batches should generally always have at least one event,
	// but let's be defensive about that and accept empty batches.
	for b.idx < len(batches) && len(batches[b.idx].data) == b.dataOff {
		b.idx++
		b.dataOff = 0
		b.lastTs = 0
	}
	// Have we reached the end of the batches?
	if b.idx == len(batches) {
		return false, nil
	}
	// Initialize lastTs if it hasn't been yet.
	if b.lastTs == 0 {
		b.lastTs = freq.mul(batches[b.idx].time)
	}
	// Read an event out.
	n, tsdiff, err := readTimedBaseEvent(batches[b.idx].data[b.dataOff:], &b.ev)
	if err != nil {
		return false, err
	}
	// Complete the timestamp from the cursor's last timestamp.
	b.ev.time = freq.mul(tsdiff) + b.lastTs

	// Move the cursor's timestamp forward.
	b.lastTs = b.ev.time

	// Move the cursor forward.
	b.dataOff += n
	return true, nil
}

func (b *batchCursor) compare(a *batchCursor) int {
	return cmp.Compare(b.ev.time, a.ev.time)
}

// readTimedBaseEvent reads out the raw event data from b
// into e. It does not try to interpret the arguments
// but it does validate that the event is a regular
// event with a timestamp (vs. a structural event).
//
// It requires that the event its reading be timed, which must
// be the case for every event in a plain EventBatch.
func readTimedBaseEvent(b []byte, e *baseEvent) (int, timestamp, error) {
	// Get the event type.
	typ := event.Type(b[0])
	specs := go122.Specs()
	if int(typ) >= len(specs) {
		return 0, 0, fmt.Errorf("found invalid event type: %v", typ)
	}
	e.typ = typ

	// Get spec.
	spec := &specs[typ]
	if len(spec.Args) == 0 || !spec.IsTimedEvent {
		return 0, 0, fmt.Errorf("found event without a timestamp: type=%v", typ)
	}
	n := 1

	// Read timestamp diff.
	ts, nb := binary.Uvarint(b[n:])
	if nb <= 0 {
		return 0, 0, fmt.Errorf("found invalid uvarint for timestamp")
	}
	n += nb

	// Read the rest of the arguments.
	for i := 0; i < len(spec.Args)-1; i++ {
		arg, nb := binary.Uvarint(b[n:])
		if nb <= 0 {
			return 0, 0, fmt.Errorf("found invalid uvarint")
		}
		e.args[i] = arg
		n += nb
	}
	return n, timestamp(ts), nil
}

func heapInsert(heap []*batchCursor, bc *batchCursor) []*batchCursor {
	// Add the cursor to the end of the heap.
	heap = append(heap, bc)

	// Sift the new entry up to the right place.
	heapSiftUp(heap, len(heap)-1)
	return heap
}

func heapUpdate(heap []*batchCursor, i int) {
	// Try to sift up.
	if heapSiftUp(heap, i) != i {
		return
	}
	// Try to sift down, if sifting up failed.
	heapSiftDown(heap, i)
}

func heapRemove(heap []*batchCursor, i int) []*batchCursor {
	// Sift index i up to the root, ignoring actual values.
	for i > 0 {
		heap[(i-1)/2], heap[i] = heap[i], heap[(i-1)/2]
		i = (i - 1) / 2
	}
	// Swap the root with the last element, then remove it.
	heap[0], heap[len(heap)-1] = heap[len(heap)-1], heap[0]
	heap = heap[:len(heap)-1]
	// Sift the root down.
	heapSiftDown(heap, 0)
	return heap
}

func heapSiftUp(heap []*batchCursor, i int) int {
	for i > 0 && heap[(i-1)/2].ev.time > heap[i].ev.time {
		heap[(i-1)/2], heap[i] = heap[i], heap[(i-1)/2]
		i = (i - 1) / 2
	}
	return i
}

func heapSiftDown(heap []*batchCursor, i int) int {
	for {
		m := min3(heap, i, 2*i+1, 2*i+2)
		if m == i {
			// Heap invariant already applies.
			break
		}
		heap[i], heap[m] = heap[m], heap[i]
		i = m
	}
	return i
}

func min3(b []*batchCursor, i0, i1, i2 int) int {
	minIdx := i0
	minT := maxTime
	if i0 < len(b) {
		minT = b[i0].ev.time
	}
	if i1 < len(b) {
		if t := b[i1].ev.time; t < minT {
			minT = t
			minIdx = i1
		}
	}
	if i2 < len(b) {
		if t := b[i2].ev.time; t < minT {
			minT = t
			minIdx = i2
		}
	}
	return minIdx
}
