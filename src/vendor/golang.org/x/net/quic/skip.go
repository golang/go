// Copyright 2025 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package quic

// skipState is state for optimistic ACK defenses.
//
// An endpoint performs an optimistic ACK attack by sending acknowledgements for packets
// which it has not received, potentially convincing the sender's congestion controller to
// send at rates beyond what the network supports.
//
// We defend against this by periodically skipping packet numbers.
// Receiving an ACK for an unsent packet number is a PROTOCOL_VIOLATION error.
//
// We only skip packet numbers in the Application Data number space.
// The total data sent in the Initial/Handshake spaces should generally fit into
// the initial congestion window.
//
// https://www.rfc-editor.org/rfc/rfc9000.html#section-21.4
type skipState struct {
	// skip is the next packet number (in the Application Data space) we should skip.
	skip packetNumber

	// maxSkip is the maximum number of packets to send before skipping another number.
	// Increases over time.
	maxSkip int64
}

func (ss *skipState) init(c *Conn) {
	ss.maxSkip = 256 // skip our first packet number within this range
	ss.updateNumberSkip(c)
}

// shouldSkip returns whether we should skip the given packet number.
func (ss *skipState) shouldSkip(num packetNumber) bool {
	return ss.skip == num
}

// updateNumberSkip schedules a packet to be skipped after skipping lastSkipped.
func (ss *skipState) updateNumberSkip(c *Conn) {
	// Send at least this many packets before skipping.
	// Limits the impact of skipping a little,
	// plus allows most tests to ignore skipping.
	const minSkip = 64

	skip := minSkip + c.prng.Int64N(ss.maxSkip-minSkip)
	ss.skip += packetNumber(skip)

	// Double the size of the skip each time until we reach 128k.
	// The idea here is that an attacker needs to correctly ack ~N packets in order
	// to send an optimistic ack for another ~N packets.
	// Skipping packet numbers comes with a small cost (it causes the receiver to
	// send an immediate ACK rather than the usual delayed ACK), so we increase the
	// time between skips as a connection's lifetime grows.
	//
	// The 128k cap is arbitrary, chosen so that we skip a packet number
	// about once a second when sending full-size datagrams at 1Gbps.
	if ss.maxSkip < 128*1024 {
		ss.maxSkip *= 2
	}
}
