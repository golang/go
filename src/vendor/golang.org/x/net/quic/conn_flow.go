// Copyright 2023 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package quic

import (
	"sync/atomic"
	"time"
)

// connInflow tracks connection-level flow control for data sent by the peer to us.
//
// There are four byte offsets of significance in the stream of data received from the peer,
// each >= to the previous:
//
//   - bytes read by the user
//   - bytes received from the peer
//   - limit sent to the peer in a MAX_DATA frame
//   - potential new limit to sent to the peer
//
// We maintain a flow control window, so as bytes are read by the user
// the potential limit is extended correspondingly.
//
// We keep an atomic counter of bytes read by the user and not yet applied to the
// potential limit (credit). When this count grows large enough, we update the
// new limit to send and mark that we need to send a new MAX_DATA frame.
type connInflow struct {
	sent      sentVal // set when we need to send a MAX_DATA update to the peer
	usedLimit int64   // total bytes sent by the peer, must be less than sentLimit
	sentLimit int64   // last MAX_DATA sent to the peer
	newLimit  int64   // new MAX_DATA to send

	credit atomic.Int64 // bytes read but not yet applied to extending the flow-control window
}

func (c *Conn) inflowInit() {
	// The initial MAX_DATA limit is sent as a transport parameter.
	c.streams.inflow.sentLimit = c.config.maxConnReadBufferSize()
	c.streams.inflow.newLimit = c.streams.inflow.sentLimit
}

// handleStreamBytesReadOffLoop records that the user has consumed bytes from a stream.
// We may extend the peer's flow control window.
//
// This is called indirectly by the user, via Read or CloseRead.
func (c *Conn) handleStreamBytesReadOffLoop(n int64) {
	if n == 0 {
		return
	}
	if c.shouldUpdateFlowControl(c.streams.inflow.credit.Add(n)) {
		// We should send a MAX_DATA update to the peer.
		// Record this on the Conn's main loop.
		c.sendMsg(func(now time.Time, c *Conn) {
			// A MAX_DATA update may have already happened, so check again.
			if c.shouldUpdateFlowControl(c.streams.inflow.credit.Load()) {
				c.sendMaxDataUpdate()
			}
		})
	}
}

// handleStreamBytesReadOnLoop extends the peer's flow control window after
// data has been discarded due to a RESET_STREAM frame.
//
// This is called on the conn's loop.
func (c *Conn) handleStreamBytesReadOnLoop(n int64) {
	if c.shouldUpdateFlowControl(c.streams.inflow.credit.Add(n)) {
		c.sendMaxDataUpdate()
	}
}

func (c *Conn) sendMaxDataUpdate() {
	c.streams.inflow.sent.setUnsent()
	// Apply current credit to the limit.
	// We don't strictly need to do this here
	// since appendMaxDataFrame will do so as well,
	// but this avoids redundant trips down this path
	// if the MAX_DATA frame doesn't go out right away.
	c.streams.inflow.newLimit += c.streams.inflow.credit.Swap(0)
}

func (c *Conn) shouldUpdateFlowControl(credit int64) bool {
	return shouldUpdateFlowControl(c.config.maxConnReadBufferSize(), credit)
}

// handleStreamBytesReceived records that the peer has sent us stream data.
func (c *Conn) handleStreamBytesReceived(n int64) error {
	c.streams.inflow.usedLimit += n
	if c.streams.inflow.usedLimit > c.streams.inflow.sentLimit {
		return localTransportError{
			code:   errFlowControl,
			reason: "stream exceeded flow control limit",
		}
	}
	return nil
}

// appendMaxDataFrame appends a MAX_DATA frame to the current packet.
//
// It returns true if no more frames need appending,
// false if it could not fit a frame in the current packet.
func (c *Conn) appendMaxDataFrame(w *packetWriter, pnum packetNumber, pto bool) bool {
	if c.streams.inflow.sent.shouldSendPTO(pto) {
		// Add any unapplied credit to the new limit now.
		c.streams.inflow.newLimit += c.streams.inflow.credit.Swap(0)
		if !w.appendMaxDataFrame(c.streams.inflow.newLimit) {
			return false
		}
		c.streams.inflow.sentLimit += c.streams.inflow.newLimit
		c.streams.inflow.sent.setSent(pnum)
	}
	return true
}

// ackOrLossMaxData records the fate of a MAX_DATA frame.
func (c *Conn) ackOrLossMaxData(pnum packetNumber, fate packetFate) {
	c.streams.inflow.sent.ackLatestOrLoss(pnum, fate)
}

// connOutflow tracks connection-level flow control for data sent by us to the peer.
type connOutflow struct {
	max  int64 // largest MAX_DATA received from peer
	used int64 // total bytes of STREAM data sent to peer
}

// setMaxData updates the connection-level flow control limit
// with the initial limit conveyed in transport parameters
// or an update from a MAX_DATA frame.
func (f *connOutflow) setMaxData(maxData int64) {
	f.max = max(f.max, maxData)
}

// avail returns the number of connection-level flow control bytes available.
func (f *connOutflow) avail() int64 {
	return f.max - f.used
}

// consume records consumption of n bytes of flow.
func (f *connOutflow) consume(n int64) {
	f.used += n
}
