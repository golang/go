// Copyright 2023 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package quic

import (
	"context"
	"sync"
	"sync/atomic"
	"time"
)

type streamsState struct {
	queue queue[*Stream] // new, peer-created streams

	// All peer-created streams.
	//
	// Implicitly created streams are included as an empty entry in the map.
	// (For example, if we receive a frame for stream 4, we implicitly create stream 0 and
	// insert an empty entry for it to the map.)
	//
	// The map value is maybeStream rather than *Stream as a reminder that values can be nil.
	streams map[streamID]maybeStream

	// Limits on the number of streams, indexed by streamType.
	localLimit  [streamTypeCount]localStreamLimits
	remoteLimit [streamTypeCount]remoteStreamLimits

	// Peer configuration provided in transport parameters.
	peerInitialMaxStreamDataRemote    [streamTypeCount]int64 // streams opened by us
	peerInitialMaxStreamDataBidiLocal int64                  // streams opened by them

	// Connection-level flow control.
	inflow  connInflow
	outflow connOutflow

	// Streams with frames to send are stored in one of two circular linked lists,
	// depending on whether they require connection-level flow control.
	needSend  atomic.Bool
	sendMu    sync.Mutex
	queueMeta streamRing // streams with any non-flow-controlled frames
	queueData streamRing // streams with only flow-controlled frames
}

// maybeStream is a possibly nil *Stream. See streamsState.streams.
type maybeStream struct {
	s *Stream
}

func (c *Conn) streamsInit() {
	c.streams.streams = make(map[streamID]maybeStream)
	c.streams.queue = newQueue[*Stream]()
	c.streams.localLimit[bidiStream].init()
	c.streams.localLimit[uniStream].init()
	c.streams.remoteLimit[bidiStream].init(c.config.maxBidiRemoteStreams())
	c.streams.remoteLimit[uniStream].init(c.config.maxUniRemoteStreams())
	c.inflowInit()
}

func (c *Conn) streamsCleanup() {
	c.streams.queue.close(errConnClosed)
	c.streams.localLimit[bidiStream].connHasClosed()
	c.streams.localLimit[uniStream].connHasClosed()
	for _, s := range c.streams.streams {
		if s.s != nil {
			s.s.connHasClosed()
		}
	}
}

// AcceptStream waits for and returns the next stream created by the peer.
func (c *Conn) AcceptStream(ctx context.Context) (*Stream, error) {
	return c.streams.queue.get(ctx)
}

// NewStream creates a stream.
//
// If the peer's maximum stream limit for the connection has been reached,
// NewStream blocks until the limit is increased or the context expires.
func (c *Conn) NewStream(ctx context.Context) (*Stream, error) {
	return c.newLocalStream(ctx, bidiStream)
}

// NewSendOnlyStream creates a unidirectional, send-only stream.
//
// If the peer's maximum stream limit for the connection has been reached,
// NewSendOnlyStream blocks until the limit is increased or the context expires.
func (c *Conn) NewSendOnlyStream(ctx context.Context) (*Stream, error) {
	return c.newLocalStream(ctx, uniStream)
}

func (c *Conn) newLocalStream(ctx context.Context, styp streamType) (*Stream, error) {
	num, err := c.streams.localLimit[styp].open(ctx, c)
	if err != nil {
		return nil, err
	}

	s := newStream(c, newStreamID(c.side, styp, num))
	s.outmaxbuf = c.config.maxStreamWriteBufferSize()
	s.outwin = c.streams.peerInitialMaxStreamDataRemote[styp]
	if styp == bidiStream {
		s.inmaxbuf = c.config.maxStreamReadBufferSize()
		s.inwin = c.config.maxStreamReadBufferSize()
	}
	s.inUnlock()
	s.outUnlock()

	// Modify c.streams on the conn's loop.
	if err := c.runOnLoop(ctx, func(now time.Time, c *Conn) {
		c.streams.streams[s.id] = maybeStream{s}
	}); err != nil {
		return nil, err
	}
	return s, nil
}

// streamFrameType identifies which direction of a stream,
// from the local perspective, a frame is associated with.
//
// For example, STREAM is a recvStream frame,
// because it carries data from the peer to us.
type streamFrameType uint8

const (
	sendStream = streamFrameType(iota) // for example, MAX_DATA
	recvStream                         // for example, STREAM_DATA_BLOCKED
)

// streamForID returns the stream with the given id.
// If the stream does not exist, it returns nil.
func (c *Conn) streamForID(id streamID) *Stream {
	return c.streams.streams[id].s
}

// streamForFrame returns the stream with the given id.
// If the stream does not exist, it may be created.
//
// streamForFrame aborts the connection if the stream id, state, and frame type don't align.
// For example, it aborts the connection with a STREAM_STATE error if a MAX_DATA frame
// is received for a receive-only stream, or if the peer attempts to create a stream that
// should be originated locally.
//
// streamForFrame returns nil if the stream no longer exists or if an error occurred.
func (c *Conn) streamForFrame(now time.Time, id streamID, ftype streamFrameType) *Stream {
	if id.streamType() == uniStream {
		if (id.initiator() == c.side) != (ftype == sendStream) {
			// Received an invalid frame for unidirectional stream.
			// For example, a RESET_STREAM frame for a send-only stream.
			c.abort(now, localTransportError{
				code:   errStreamState,
				reason: "invalid frame for unidirectional stream",
			})
			return nil
		}
	}

	ms, isOpen := c.streams.streams[id]
	if ms.s != nil {
		return ms.s
	}

	num := id.num()
	styp := id.streamType()
	if id.initiator() == c.side {
		if num < c.streams.localLimit[styp].opened {
			// This stream was created by us, and has been closed.
			return nil
		}
		// Received a frame for a stream that should be originated by us,
		// but which we never created.
		c.abort(now, localTransportError{
			code:   errStreamState,
			reason: "received frame for unknown stream",
		})
		return nil
	} else {
		// if isOpen, this is a stream that was implicitly opened by a
		// previous frame for a larger-numbered stream, but we haven't
		// actually created it yet.
		if !isOpen && num < c.streams.remoteLimit[styp].opened {
			// This stream was created by the peer, and has been closed.
			return nil
		}
	}

	prevOpened := c.streams.remoteLimit[styp].opened
	if err := c.streams.remoteLimit[styp].open(id); err != nil {
		c.abort(now, err)
		return nil
	}

	// Receiving a frame for a stream implicitly creates all streams
	// with the same initiator and type and a lower number.
	// Add a nil entry to the streams map for each implicitly created stream.
	for n := newStreamID(id.initiator(), id.streamType(), prevOpened); n < id; n += 4 {
		c.streams.streams[n] = maybeStream{}
	}

	s := newStream(c, id)
	s.inmaxbuf = c.config.maxStreamReadBufferSize()
	s.inwin = c.config.maxStreamReadBufferSize()
	if id.streamType() == bidiStream {
		s.outmaxbuf = c.config.maxStreamWriteBufferSize()
		s.outwin = c.streams.peerInitialMaxStreamDataBidiLocal
	}
	s.inUnlock()
	s.outUnlock()

	c.streams.streams[id] = maybeStream{s}
	c.streams.queue.put(s)
	return s
}

// maybeQueueStreamForSend marks a stream as containing frames that need sending.
func (c *Conn) maybeQueueStreamForSend(s *Stream, state streamState) {
	if state.wantQueue() == state.inQueue() {
		return // already on the right queue
	}
	c.streams.sendMu.Lock()
	defer c.streams.sendMu.Unlock()
	state = s.state.load() // may have changed while waiting
	c.queueStreamForSendLocked(s, state)

	c.streams.needSend.Store(true)
	c.wake()
}

// queueStreamForSendLocked moves a stream to the correct send queue,
// or removes it from all queues.
//
// state is the last known stream state.
func (c *Conn) queueStreamForSendLocked(s *Stream, state streamState) {
	for {
		wantQueue := state.wantQueue()
		inQueue := state.inQueue()
		if inQueue == wantQueue {
			return // already on the right queue
		}

		switch inQueue {
		case metaQueue:
			c.streams.queueMeta.remove(s)
		case dataQueue:
			c.streams.queueData.remove(s)
		}

		switch wantQueue {
		case metaQueue:
			c.streams.queueMeta.append(s)
			state = s.state.set(streamQueueMeta, streamQueueMeta|streamQueueData)
		case dataQueue:
			c.streams.queueData.append(s)
			state = s.state.set(streamQueueData, streamQueueMeta|streamQueueData)
		case noQueue:
			state = s.state.set(0, streamQueueMeta|streamQueueData)
		}

		// If the stream state changed while we were moving the stream,
		// we might now be on the wrong queue.
		//
		// For example:
		//   - stream has data to send: streamOutSendData|streamQueueData
		//   - appendStreamFrames sends all the data: streamQueueData
		//   - concurrently, more data is written: streamOutSendData|streamQueueData
		//   - appendStreamFrames calls us with the last state it observed
		//     (streamQueueData).
		//   - We remove the stream from the queue and observe the updated state:
		//     streamOutSendData
		//   - We realize that the stream needs to go back on the data queue.
		//
		// Go back around the loop to confirm we're on the correct queue.
	}
}

// appendStreamFrames writes stream-related frames to the current packet.
//
// It returns true if no more frames need appending,
// false if not everything fit in the current packet.
func (c *Conn) appendStreamFrames(w *packetWriter, pnum packetNumber, pto bool) bool {
	// MAX_DATA
	if !c.appendMaxDataFrame(w, pnum, pto) {
		return false
	}

	if pto {
		return c.appendStreamFramesPTO(w, pnum)
	}
	if !c.streams.needSend.Load() {
		// If queueMeta includes newly-finished streams, we may extend the peer's
		// stream limits. When there are no streams to process, add MAX_STREAMS
		// frames here. Otherwise, wait until after we've processed queueMeta.
		return c.appendMaxStreams(w, pnum, pto)
	}
	c.streams.sendMu.Lock()
	defer c.streams.sendMu.Unlock()
	// queueMeta contains streams with non-flow-controlled frames to send.
	for c.streams.queueMeta.head != nil {
		s := c.streams.queueMeta.head
		state := s.state.load()
		if state&(streamQueueMeta|streamConnRemoved) != streamQueueMeta {
			panic("BUG: queueMeta stream is not streamQueueMeta")
		}
		if state&streamInSendMeta != 0 {
			s.ingate.lock()
			ok := s.appendInFramesLocked(w, pnum, pto)
			state = s.inUnlockNoQueue()
			if !ok {
				return false
			}
			if state&streamInSendMeta != 0 {
				panic("BUG: streamInSendMeta set after successfully appending frames")
			}
		}
		if state&streamOutSendMeta != 0 {
			s.outgate.lock()
			// This might also append flow-controlled frames if we have any
			// and available conn-level quota. That's fine.
			ok := s.appendOutFramesLocked(w, pnum, pto)
			state = s.outUnlockNoQueue()
			// We're checking both ok and state, because appendOutFramesLocked
			// might have filled up the packet with flow-controlled data.
			// If so, we want to move the stream to queueData for any remaining frames.
			if !ok && state&streamOutSendMeta != 0 {
				return false
			}
			if state&streamOutSendMeta != 0 {
				panic("BUG: streamOutSendMeta set after successfully appending frames")
			}
		}
		// We've sent all frames for this stream, so remove it from the send queue.
		c.streams.queueMeta.remove(s)
		if state&(streamInDone|streamOutDone) == streamInDone|streamOutDone {
			// Stream is finished, remove it from the conn.
			state = s.state.set(streamConnRemoved, streamQueueMeta|streamConnRemoved)
			delete(c.streams.streams, s.id)

			// Record finalization of remote streams, to know when
			// to extend the peer's stream limit.
			if s.id.initiator() != c.side {
				c.streams.remoteLimit[s.id.streamType()].close()
			}
		} else {
			state = s.state.set(0, streamQueueMeta|streamConnRemoved)
		}
		// The stream may have flow-controlled data to send,
		// or something might have added non-flow-controlled frames after we
		// unlocked the stream.
		// If so, put the stream back on a queue.
		c.queueStreamForSendLocked(s, state)
	}

	// MAX_STREAMS (possibly triggered by finalization of remote streams above).
	if !c.appendMaxStreams(w, pnum, pto) {
		return false
	}

	// queueData contains streams with flow-controlled frames.
	for c.streams.queueData.head != nil {
		avail := c.streams.outflow.avail()
		if avail == 0 {
			break // no flow control quota available
		}
		s := c.streams.queueData.head
		s.outgate.lock()
		ok := s.appendOutFramesLocked(w, pnum, pto)
		state := s.outUnlockNoQueue()
		if !ok {
			// We've sent some data for this stream, but it still has more to send.
			// If the stream got a reasonable chance to put data in a packet,
			// advance sendHead to the next stream in line, to avoid starvation.
			// We'll come back to this stream after going through the others.
			//
			// If the packet was already mostly out of space, leave sendHead alone
			// and come back to this stream again on the next packet.
			if avail > 512 {
				c.streams.queueData.head = s.next
			}
			return false
		}
		if state&streamQueueData == 0 {
			panic("BUG: queueData stream is not streamQueueData")
		}
		if state&streamOutSendData != 0 {
			// We must have run out of connection-level flow control:
			// appendOutFramesLocked says it wrote all it can, but there's
			// still data to send.
			//
			// Advance sendHead to the next stream in line to avoid starvation.
			if c.streams.outflow.avail() != 0 {
				panic("BUG: streamOutSendData set and flow control available after send")
			}
			c.streams.queueData.head = s.next
			return true
		}
		c.streams.queueData.remove(s)
		state = s.state.set(0, streamQueueData)
		c.queueStreamForSendLocked(s, state)
	}
	if c.streams.queueMeta.head == nil && c.streams.queueData.head == nil {
		c.streams.needSend.Store(false)
	}
	return true
}

// appendStreamFramesPTO writes stream-related frames to the current packet
// for a PTO probe.
//
// It returns true if no more frames need appending,
// false if not everything fit in the current packet.
func (c *Conn) appendStreamFramesPTO(w *packetWriter, pnum packetNumber) bool {
	const pto = true
	if !c.appendMaxStreams(w, pnum, pto) {
		return false
	}
	c.streams.sendMu.Lock()
	defer c.streams.sendMu.Unlock()
	for _, ms := range c.streams.streams {
		s := ms.s
		if s == nil {
			continue
		}
		const pto = true
		s.ingate.lock()
		inOK := s.appendInFramesLocked(w, pnum, pto)
		s.inUnlockNoQueue()
		if !inOK {
			return false
		}

		s.outgate.lock()
		outOK := s.appendOutFramesLocked(w, pnum, pto)
		s.outUnlockNoQueue()
		if !outOK {
			return false
		}
	}
	return true
}

func (c *Conn) appendMaxStreams(w *packetWriter, pnum packetNumber, pto bool) bool {
	if !c.streams.remoteLimit[uniStream].appendFrame(w, uniStream, pnum, pto) {
		return false
	}
	if !c.streams.remoteLimit[bidiStream].appendFrame(w, bidiStream, pnum, pto) {
		return false
	}
	return true
}

// A streamRing is a circular linked list of streams.
type streamRing struct {
	head *Stream
}

// remove removes s from the ring.
// s must be on the ring.
func (r *streamRing) remove(s *Stream) {
	if s.next == s {
		r.head = nil // s was the last stream in the ring
	} else {
		s.prev.next = s.next
		s.next.prev = s.prev
		if r.head == s {
			r.head = s.next
		}
	}
}

// append places s at the last position in the ring.
// s must not be attached to any ring.
func (r *streamRing) append(s *Stream) {
	if r.head == nil {
		r.head = s
		s.next = s
		s.prev = s
	} else {
		s.prev = r.head.prev
		s.next = r.head
		s.prev.next = s
		s.next.prev = s
	}
}
