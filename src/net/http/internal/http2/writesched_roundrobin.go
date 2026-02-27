// Copyright 2023 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package http2

import (
	"fmt"
	"math"
)

type roundRobinWriteScheduler struct {
	// control contains control frames (SETTINGS, PING, etc.).
	control writeQueue

	// streams maps stream ID to a queue.
	streams map[uint32]*writeQueue

	// stream queues are stored in a circular linked list.
	// head is the next stream to write, or nil if there are no streams open.
	head *writeQueue

	// pool of empty queues for reuse.
	queuePool writeQueuePool
}

// newRoundRobinWriteScheduler constructs a new write scheduler.
// The round robin scheduler prioritizes control frames
// like SETTINGS and PING over DATA frames.
// When there are no control frames to send, it performs a round-robin
// selection from the ready streams.
func newRoundRobinWriteScheduler() WriteScheduler {
	ws := &roundRobinWriteScheduler{
		streams: make(map[uint32]*writeQueue),
	}
	return ws
}

func (ws *roundRobinWriteScheduler) OpenStream(streamID uint32, options OpenStreamOptions) {
	if ws.streams[streamID] != nil {
		panic(fmt.Errorf("stream %d already opened", streamID))
	}
	q := ws.queuePool.get()
	ws.streams[streamID] = q
	if ws.head == nil {
		ws.head = q
		q.next = q
		q.prev = q
	} else {
		// Queues are stored in a ring.
		// Insert the new stream before ws.head, putting it at the end of the list.
		q.prev = ws.head.prev
		q.next = ws.head
		q.prev.next = q
		q.next.prev = q
	}
}

func (ws *roundRobinWriteScheduler) CloseStream(streamID uint32) {
	q := ws.streams[streamID]
	if q == nil {
		return
	}
	if q.next == q {
		// This was the only open stream.
		ws.head = nil
	} else {
		q.prev.next = q.next
		q.next.prev = q.prev
		if ws.head == q {
			ws.head = q.next
		}
	}
	delete(ws.streams, streamID)
	ws.queuePool.put(q)
}

func (ws *roundRobinWriteScheduler) AdjustStream(streamID uint32, priority PriorityParam) {}

func (ws *roundRobinWriteScheduler) Push(wr FrameWriteRequest) {
	if wr.isControl() {
		ws.control.push(wr)
		return
	}
	q := ws.streams[wr.StreamID()]
	if q == nil {
		// This is a closed stream.
		// wr should not be a HEADERS or DATA frame.
		// We push the request onto the control queue.
		if wr.DataSize() > 0 {
			panic("add DATA on non-open stream")
		}
		ws.control.push(wr)
		return
	}
	q.push(wr)
}

func (ws *roundRobinWriteScheduler) Pop() (FrameWriteRequest, bool) {
	// Control and RST_STREAM frames first.
	if !ws.control.empty() {
		return ws.control.shift(), true
	}
	if ws.head == nil {
		return FrameWriteRequest{}, false
	}
	q := ws.head
	for {
		if wr, ok := q.consume(math.MaxInt32); ok {
			ws.head = q.next
			return wr, true
		}
		q = q.next
		if q == ws.head {
			break
		}
	}
	return FrameWriteRequest{}, false
}
