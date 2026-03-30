// Copyright 2023 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package quic

import (
	"context"
)

// Limits on the number of open streams.
// Every connection has separate limits for bidirectional and unidirectional streams.
//
// Note that the MAX_STREAMS limit includes closed as well as open streams.
// Closing a stream doesn't enable an endpoint to open a new one;
// only an increase in the MAX_STREAMS limit does.

// localStreamLimits are limits on the number of open streams created by us.
type localStreamLimits struct {
	gate   gate
	max    int64 // peer-provided MAX_STREAMS
	opened int64 // number of streams opened by us, -1 when conn is closed
}

func (lim *localStreamLimits) init() {
	lim.gate = newGate()
}

// open creates a new local stream, blocking until MAX_STREAMS quota is available.
func (lim *localStreamLimits) open(ctx context.Context, c *Conn) (num int64, err error) {
	// TODO: Send a STREAMS_BLOCKED when blocked.
	if err := lim.gate.waitAndLock(ctx); err != nil {
		return 0, err
	}
	if lim.opened < 0 {
		lim.gate.unlock(true)
		return 0, errConnClosed
	}
	num = lim.opened
	lim.opened++
	lim.gate.unlock(lim.opened < lim.max)
	return num, nil
}

// connHasClosed indicates the connection has been closed, locally or by the peer.
func (lim *localStreamLimits) connHasClosed() {
	lim.gate.lock()
	lim.opened = -1
	lim.gate.unlock(true)
}

// setMax sets the MAX_STREAMS provided by the peer.
func (lim *localStreamLimits) setMax(maxStreams int64) {
	lim.gate.lock()
	lim.max = max(lim.max, maxStreams)
	lim.gate.unlock(lim.opened < lim.max)
}

// remoteStreamLimits are limits on the number of open streams created by the peer.
type remoteStreamLimits struct {
	max     int64   // last MAX_STREAMS sent to the peer
	opened  int64   // number of streams opened by the peer (including subsequently closed ones)
	closed  int64   // number of peer streams in the "closed" state
	maxOpen int64   // how many streams we want to let the peer simultaneously open
	sendMax sentVal // set when we should send MAX_STREAMS
}

func (lim *remoteStreamLimits) init(maxOpen int64) {
	lim.maxOpen = maxOpen
	lim.max = min(maxOpen, implicitStreamLimit) // initial limit sent in transport parameters
	lim.opened = 0
}

// open handles the peer opening a new stream.
func (lim *remoteStreamLimits) open(id streamID) error {
	num := id.num()
	if num >= lim.max {
		return localTransportError{
			code:   errStreamLimit,
			reason: "stream limit exceeded",
		}
	}
	if num >= lim.opened {
		lim.opened = num + 1
		lim.maybeUpdateMax()
	}
	return nil
}

// close handles the peer closing an open stream.
func (lim *remoteStreamLimits) close() {
	lim.closed++
	lim.maybeUpdateMax()
}

// maybeUpdateMax updates the MAX_STREAMS value we will send to the peer.
func (lim *remoteStreamLimits) maybeUpdateMax() {
	newMax := min(
		// Max streams the peer can have open at once.
		lim.closed+lim.maxOpen,
		// Max streams the peer can open with a single frame.
		lim.opened+implicitStreamLimit,
	)
	avail := lim.max - lim.opened
	if newMax > lim.max && (avail < 8 || newMax-lim.max >= 2*avail) {
		// If the peer has less than 8 streams, or if increasing the peer's
		// stream limit would double it, then send a MAX_STREAMS.
		lim.max = newMax
		lim.sendMax.setUnsent()
	}
}

// appendFrame appends a MAX_STREAMS frame to the current packet, if necessary.
//
// It returns true if no more frames need appending,
// false if not everything fit in the current packet.
func (lim *remoteStreamLimits) appendFrame(w *packetWriter, typ streamType, pnum packetNumber, pto bool) bool {
	if lim.sendMax.shouldSendPTO(pto) {
		if !w.appendMaxStreamsFrame(typ, lim.max) {
			return false
		}
		lim.sendMax.setSent(pnum)
	}
	return true
}
