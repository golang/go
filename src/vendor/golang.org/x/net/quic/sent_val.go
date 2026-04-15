// Copyright 2023 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package quic

// A sentVal tracks sending some piece of information to the peer.
// It tracks whether the information has been sent, acked, and
// (when in-flight) the most recent packet to carry it.
//
// For example, a sentVal can track sending of a RESET_STREAM frame.
//
//   - unset: stream is active, no need to send RESET_STREAM
//   - unsent: we should send a RESET_STREAM, but have not yet
//   - sent: we have sent a RESET_STREAM, but have not received an ack
//   - received: we have sent a RESET_STREAM, and the peer has acked the packet that contained it
//
// In the "sent" state, a sentVal also tracks the latest packet number to carry
// the information. (QUIC packet numbers are always at most 62 bits in size,
// so the sentVal keeps the number in the low 62 bits and the state in the high 2 bits.)
type sentVal uint64

const (
	sentValUnset    = 0       // unset
	sentValUnsent   = 1 << 62 // set, not sent to the peer
	sentValSent     = 2 << 62 // set, sent to the peer but not yet acked; pnum is set
	sentValReceived = 3 << 62 // set, peer acked receipt

	sentValStateMask = 3 << 62
)

// isSet reports whether the value is set.
func (s sentVal) isSet() bool { return s != 0 }

// shouldSend reports whether the value is set and has not been sent to the peer.
func (s sentVal) shouldSend() bool { return s.state() == sentValUnsent }

// shouldSendPTO reports whether the value needs to be sent to the peer.
// The value needs to be sent if it is set and has not been sent.
// If pto is true, indicating that we are sending a PTO probe, the value
// should also be sent if it is set and has not been acknowledged.
func (s sentVal) shouldSendPTO(pto bool) bool {
	st := s.state()
	return st == sentValUnsent || (pto && st == sentValSent)
}

// isReceived reports whether the value has been received by the peer.
func (s sentVal) isReceived() bool { return s == sentValReceived }

// set sets the value and records that it should be sent to the peer.
// If the value has already been sent, it is not resent.
func (s *sentVal) set() {
	if *s == 0 {
		*s = sentValUnsent
	}
}

// reset sets the value to the unsent state.
func (s *sentVal) setUnsent() { *s = sentValUnsent }

// clear sets the value to the unset state.
func (s *sentVal) clear() { *s = sentValUnset }

// setSent sets the value to the send state and records the number of the most recent
// packet containing the value.
func (s *sentVal) setSent(pnum packetNumber) {
	*s = sentValSent | sentVal(pnum)
}

// setReceived sets the value to the received state.
func (s *sentVal) setReceived() { *s = sentValReceived }

// ackOrLoss reports that an acknowledgement has been received for the value,
// or that the packet carrying the value has been lost.
func (s *sentVal) ackOrLoss(pnum packetNumber, fate packetFate) {
	if fate == packetAcked {
		*s = sentValReceived
	} else if *s == sentVal(pnum)|sentValSent {
		*s = sentValUnsent
	}
}

// ackLatestOrLoss reports that an acknowledgement has been received for the value,
// or that the packet carrying the value has been lost.
// The value is set to the acked state only if pnum is the latest packet containing it.
//
// We use this to handle acks for data that varies every time it is sent.
// For example, if we send a MAX_DATA frame followed by an updated MAX_DATA value in a
// second packet, we consider the data sent only upon receiving an ack for the most
// recent value.
func (s *sentVal) ackLatestOrLoss(pnum packetNumber, fate packetFate) {
	if fate == packetAcked {
		if *s == sentVal(pnum)|sentValSent {
			*s = sentValReceived
		}
	} else {
		if *s == sentVal(pnum)|sentValSent {
			*s = sentValUnsent
		}
	}
}

func (s sentVal) state() uint64 { return uint64(s) & sentValStateMask }
