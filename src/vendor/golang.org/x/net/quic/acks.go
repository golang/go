// Copyright 2023 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package quic

import (
	"time"
)

// ackState tracks packets received from a peer within a number space.
// It handles packet deduplication (don't process the same packet twice) and
// determines the timing and content of ACK frames.
type ackState struct {
	seen rangeset[packetNumber]

	// The time at which we must send an ACK frame, even if we have no other data to send.
	nextAck time.Time

	// The time we received the largest-numbered packet in seen.
	maxRecvTime time.Time

	// The largest-numbered ack-eliciting packet in seen.
	maxAckEliciting packetNumber

	// The number of ack-eliciting packets in seen that we have not yet acknowledged.
	unackedAckEliciting int

	// Total ECN counters for this packet number space.
	ecn ecnCounts
}

type ecnCounts struct {
	t0 int
	t1 int
	ce int
}

// shouldProcess reports whether a packet should be handled or discarded.
func (acks *ackState) shouldProcess(num packetNumber) bool {
	if packetNumber(acks.seen.min()) > num {
		// We've discarded the state for this range of packet numbers.
		// Discard the packet rather than potentially processing a duplicate.
		// https://www.rfc-editor.org/rfc/rfc9000.html#section-13.2.3-5
		return false
	}
	if acks.seen.contains(num) {
		// Discard duplicate packets.
		return false
	}
	return true
}

// receive records receipt of a packet.
func (acks *ackState) receive(now time.Time, space numberSpace, num packetNumber, ackEliciting bool, ecn ecnBits) {
	if ackEliciting {
		acks.unackedAckEliciting++
		if acks.mustAckImmediately(space, num, ecn) {
			acks.nextAck = now
		} else if acks.nextAck.IsZero() {
			// This packet does not need to be acknowledged immediately,
			// but the ack must not be intentionally delayed by more than
			// the max_ack_delay transport parameter we sent to the peer.
			//
			// We always delay acks by the maximum allowed, less the timer
			// granularity. ("[max_ack_delay] SHOULD include the receiver's
			// expected delays in alarms firing.")
			//
			// https://www.rfc-editor.org/rfc/rfc9000#section-18.2-4.28.1
			acks.nextAck = now.Add(maxAckDelay - timerGranularity)
		}
		if num > acks.maxAckEliciting {
			acks.maxAckEliciting = num
		}
	}

	acks.seen.add(num, num+1)
	if num == acks.seen.max() {
		acks.maxRecvTime = now
	}

	switch ecn {
	case ecnECT0:
		acks.ecn.t0++
	case ecnECT1:
		acks.ecn.t1++
	case ecnCE:
		acks.ecn.ce++
	}

	// Limit the total number of ACK ranges by dropping older ranges.
	//
	// Remembering more ranges results in larger ACK frames.
	//
	// Remembering a large number of ranges could result in ACK frames becoming
	// too large to fit in a packet, in which case we will silently drop older
	// ranges during packet construction.
	//
	// Remembering fewer ranges can result in unnecessary retransmissions,
	// since we cannot accept packets older than the oldest remembered range.
	//
	// The limit here is completely arbitrary. If it seems wrong, it probably is.
	//
	// https://www.rfc-editor.org/rfc/rfc9000#section-13.2.3
	const maxAckRanges = 8
	if overflow := acks.seen.numRanges() - maxAckRanges; overflow > 0 {
		acks.seen.removeranges(0, overflow)
	}
}

// mustAckImmediately reports whether an ack-eliciting packet must be acknowledged immediately,
// or whether the ack may be deferred.
func (acks *ackState) mustAckImmediately(space numberSpace, num packetNumber, ecn ecnBits) bool {
	// https://www.rfc-editor.org/rfc/rfc9000.html#section-13.2.1
	if space != appDataSpace {
		// "[...] all ack-eliciting Initial and Handshake packets [...]"
		// https://www.rfc-editor.org/rfc/rfc9000.html#section-13.2.1-2
		return true
	}
	if num < acks.maxAckEliciting {
		// "[...] when the received packet has a packet number less than another
		// ack-eliciting packet that has been received [...]"
		// https://www.rfc-editor.org/rfc/rfc9000.html#section-13.2.1-8.1
		return true
	}
	if acks.seen.rangeContaining(acks.maxAckEliciting).end != num {
		// "[...] when the packet has a packet number larger than the highest-numbered
		// ack-eliciting packet that has been received and there are missing packets
		// between that packet and this packet."
		// https://www.rfc-editor.org/rfc/rfc9000.html#section-13.2.1-8.2
		//
		// This case is a bit tricky. Let's say we've received:
		//   0, ack-eliciting
		//   1, ack-eliciting
		//   3, NOT ack eliciting
		//
		// We have sent ACKs for 0 and 1. If we receive ack-eliciting packet 2,
		// we do not need to send an immediate ACK, because there are no missing
		// packets between it and the highest-numbered ack-eliciting packet (1).
		// If we receive ack-eliciting packet 4, we do need to send an immediate ACK,
		// because there's a gap (the missing packet 2).
		//
		// We check for this by looking up the ACK range which contains the
		// highest-numbered ack-eliciting packet: [0, 1) in the above example.
		// If the range ends just before the packet we are now processing,
		// there are no gaps. If it does not, there must be a gap.
		return true
	}
	// "[...] packets marked with the ECN Congestion Experienced (CE) codepoint
	// in the IP header SHOULD be acknowledged immediately [...]"
	// https://www.rfc-editor.org/rfc/rfc9000.html#section-13.2.1-9
	if ecn == ecnCE {
		return true
	}
	// "[...] SHOULD send an ACK frame after receiving at least two ack-eliciting packets."
	// https://www.rfc-editor.org/rfc/rfc9000.html#section-13.2.2
	//
	// This ack frequency takes a substantial toll on performance, however.
	// Follow the behavior of Google QUICHE:
	// Ack every other packet for the first 100 packets, and then ack every 10th packet.
	// This keeps ack frequency high during the beginning of slow start when CWND is
	// increasing rapidly.
	packetsBeforeAck := 2
	if acks.seen.max() > 100 {
		packetsBeforeAck = 10
	}
	return acks.unackedAckEliciting >= packetsBeforeAck
}

// shouldSendAck reports whether the connection should send an ACK frame at this time,
// in an ACK-only packet if necessary.
func (acks *ackState) shouldSendAck(now time.Time) bool {
	return !acks.nextAck.IsZero() && !acks.nextAck.After(now)
}

// acksToSend returns the set of packet numbers to ACK at this time, and the current ack delay.
// It may return acks even if shouldSendAck returns false, when there are unacked
// ack-eliciting packets whose ack is being delayed.
func (acks *ackState) acksToSend(now time.Time) (nums rangeset[packetNumber], ackDelay time.Duration) {
	if acks.nextAck.IsZero() && acks.unackedAckEliciting == 0 {
		return nil, 0
	}
	// "[...] the delays intentionally introduced between the time the packet with the
	// largest packet number is received and the time an acknowledgement is sent."
	// https://www.rfc-editor.org/rfc/rfc9000#section-13.2.5-1
	delay := now.Sub(acks.maxRecvTime)
	if delay < 0 {
		delay = 0
	}
	return acks.seen, delay
}

// sentAck records that an ACK frame has been sent.
func (acks *ackState) sentAck() {
	acks.nextAck = time.Time{}
	acks.unackedAckEliciting = 0
}

// handleAck records that an ack has been received for a ACK frame we sent
// containing the given Largest Acknowledged field.
func (acks *ackState) handleAck(largestAcked packetNumber) {
	// We can stop acking packets less or equal to largestAcked.
	// https://www.rfc-editor.org/rfc/rfc9000.html#section-13.2.4-1
	//
	// We rely on acks.seen containing the largest packet number that has been successfully
	// processed, so we retain the range containing largestAcked and discard previous ones.
	acks.seen.sub(0, acks.seen.rangeContaining(largestAcked).start)
}

// largestSeen reports the largest seen packet.
func (acks *ackState) largestSeen() packetNumber {
	return acks.seen.max()
}
