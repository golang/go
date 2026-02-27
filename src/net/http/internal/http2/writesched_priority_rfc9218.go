// Copyright 2025 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package http2

import (
	"fmt"
	"math"
)

type streamMetadata struct {
	location *writeQueue
	priority PriorityParam
}

type priorityWriteSchedulerRFC9218 struct {
	// control contains control frames (SETTINGS, PING, etc.).
	control writeQueue

	// heads contain the head of a circular list of streams.
	// We put these heads within a nested array that represents urgency and
	// incremental, as defined in
	// https://www.rfc-editor.org/rfc/rfc9218.html#name-priority-parameters.
	// 8 represents u=0 up to u=7, and 2 represents i=false and i=true.
	heads [8][2]*writeQueue

	// streams contains a mapping between each stream ID and their metadata, so
	// we can quickly locate them when needing to, for example, adjust their
	// priority.
	streams map[uint32]streamMetadata

	// queuePool are empty queues for reuse.
	queuePool writeQueuePool

	// prioritizeIncremental is used to determine whether we should prioritize
	// incremental streams or not, when urgency is the same in a given Pop()
	// call.
	prioritizeIncremental bool

	// priorityUpdateBuf is used to buffer the most recent PRIORITY_UPDATE we
	// receive per https://www.rfc-editor.org/rfc/rfc9218.html#name-the-priority_update-frame.
	priorityUpdateBuf struct {
		// streamID being 0 means that the buffer is empty. This is a safe
		// assumption as PRIORITY_UPDATE for stream 0 is a PROTOCOL_ERROR.
		streamID uint32
		priority PriorityParam
	}
}

func newPriorityWriteSchedulerRFC9218() WriteScheduler {
	ws := &priorityWriteSchedulerRFC9218{
		streams: make(map[uint32]streamMetadata),
	}
	return ws
}

func (ws *priorityWriteSchedulerRFC9218) OpenStream(streamID uint32, opt OpenStreamOptions) {
	if ws.streams[streamID].location != nil {
		panic(fmt.Errorf("stream %d already opened", streamID))
	}
	if streamID == ws.priorityUpdateBuf.streamID {
		ws.priorityUpdateBuf.streamID = 0
		opt.priority = ws.priorityUpdateBuf.priority
	}
	q := ws.queuePool.get()
	ws.streams[streamID] = streamMetadata{
		location: q,
		priority: opt.priority,
	}

	u, i := opt.priority.urgency, opt.priority.incremental
	if ws.heads[u][i] == nil {
		ws.heads[u][i] = q
		q.next = q
		q.prev = q
	} else {
		// Queues are stored in a ring.
		// Insert the new stream before ws.head, putting it at the end of the list.
		q.prev = ws.heads[u][i].prev
		q.next = ws.heads[u][i]
		q.prev.next = q
		q.next.prev = q
	}
}

func (ws *priorityWriteSchedulerRFC9218) CloseStream(streamID uint32) {
	metadata := ws.streams[streamID]
	q, u, i := metadata.location, metadata.priority.urgency, metadata.priority.incremental
	if q == nil {
		return
	}
	if q.next == q {
		// This was the only open stream.
		ws.heads[u][i] = nil
	} else {
		q.prev.next = q.next
		q.next.prev = q.prev
		if ws.heads[u][i] == q {
			ws.heads[u][i] = q.next
		}
	}
	delete(ws.streams, streamID)
	ws.queuePool.put(q)
}

func (ws *priorityWriteSchedulerRFC9218) AdjustStream(streamID uint32, priority PriorityParam) {
	metadata := ws.streams[streamID]
	q, u, i := metadata.location, metadata.priority.urgency, metadata.priority.incremental
	if q == nil {
		ws.priorityUpdateBuf.streamID = streamID
		ws.priorityUpdateBuf.priority = priority
		return
	}

	// Remove stream from current location.
	if q.next == q {
		// This was the only open stream.
		ws.heads[u][i] = nil
	} else {
		q.prev.next = q.next
		q.next.prev = q.prev
		if ws.heads[u][i] == q {
			ws.heads[u][i] = q.next
		}
	}

	// Insert stream to the new queue.
	u, i = priority.urgency, priority.incremental
	if ws.heads[u][i] == nil {
		ws.heads[u][i] = q
		q.next = q
		q.prev = q
	} else {
		// Queues are stored in a ring.
		// Insert the new stream before ws.head, putting it at the end of the list.
		q.prev = ws.heads[u][i].prev
		q.next = ws.heads[u][i]
		q.prev.next = q
		q.next.prev = q
	}

	// Update the metadata.
	ws.streams[streamID] = streamMetadata{
		location: q,
		priority: priority,
	}
}

func (ws *priorityWriteSchedulerRFC9218) Push(wr FrameWriteRequest) {
	if wr.isControl() {
		ws.control.push(wr)
		return
	}
	q := ws.streams[wr.StreamID()].location
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

func (ws *priorityWriteSchedulerRFC9218) Pop() (FrameWriteRequest, bool) {
	// Control and RST_STREAM frames first.
	if !ws.control.empty() {
		return ws.control.shift(), true
	}

	// On the next Pop(), we want to prioritize incremental if we prioritized
	// non-incremental request of the same urgency this time. Vice-versa.
	// i.e. when there are incremental and non-incremental requests at the same
	// priority, we give 50% of our bandwidth to the incremental ones in
	// aggregate and 50% to the first non-incremental one (since
	// non-incremental streams do not use round-robin writes).
	ws.prioritizeIncremental = !ws.prioritizeIncremental

	// Always prioritize lowest u (i.e. highest urgency level).
	for u := range ws.heads {
		for i := range ws.heads[u] {
			// When we want to prioritize incremental, we try to pop i=true
			// first before i=false when u is the same.
			if ws.prioritizeIncremental {
				i = (i + 1) % 2
			}
			q := ws.heads[u][i]
			if q == nil {
				continue
			}
			for {
				if wr, ok := q.consume(math.MaxInt32); ok {
					if i == 1 {
						// For incremental streams, we update head to q.next so
						// we can round-robin between multiple streams that can
						// immediately benefit from partial writes.
						ws.heads[u][i] = q.next
					} else {
						// For non-incremental streams, we try to finish one to
						// completion rather than doing round-robin. However,
						// we update head here so that if q.consume() is !ok
						// (e.g. the stream has no more frame to consume), head
						// is updated to the next q that has frames to consume
						// on future iterations. This way, we do not prioritize
						// writing to unavailable stream on next Pop() calls,
						// preventing head-of-line blocking.
						ws.heads[u][i] = q
					}
					return wr, true
				}
				q = q.next
				if q == ws.heads[u][i] {
					break
				}
			}

		}
	}
	return FrameWriteRequest{}, false
}
