// Copyright 2024 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package quic

import "time"

type pathState struct {
	// Response to a peer's PATH_CHALLENGE.
	// This is not a sentVal, because we don't resend lost PATH_RESPONSE frames.
	// We only track the most recent PATH_CHALLENGE.
	// If the peer sends a second PATH_CHALLENGE before we respond to the first,
	// we'll drop the first response.
	sendPathResponse pathResponseType
	data             pathChallengeData
}

// pathChallengeData is data carried in a PATH_CHALLENGE or PATH_RESPONSE frame.
type pathChallengeData [64 / 8]byte

type pathResponseType uint8

const (
	pathResponseNotNeeded = pathResponseType(iota)
	pathResponseSmall     // send PATH_RESPONSE, do not expand datagram
	pathResponseExpanded  // send PATH_RESPONSE, expand datagram to 1200 bytes
)

func (c *Conn) handlePathChallenge(_ time.Time, dgram *datagram, data pathChallengeData) {
	// A PATH_RESPONSE is sent in a datagram expanded to 1200 bytes,
	// except when this would exceed the anti-amplification limit.
	//
	// Rather than maintaining anti-amplification state for each path
	// we may be sending a PATH_RESPONSE on, follow the following heuristic:
	//
	// If we receive a PATH_CHALLENGE in an expanded datagram,
	// respond with an expanded datagram.
	//
	// If we receive a PATH_CHALLENGE in a non-expanded datagram,
	// then the peer is presumably blocked by its own anti-amplification limit.
	// Respond with a non-expanded datagram. Receiving this PATH_RESPONSE
	// will validate the path to the peer, remove its anti-amplification limit,
	// and permit it to send a followup PATH_CHALLENGE in an expanded datagram.
	// https://www.rfc-editor.org/rfc/rfc9000.html#section-8.2.1
	if len(dgram.b) >= smallestMaxDatagramSize {
		c.path.sendPathResponse = pathResponseExpanded
	} else {
		c.path.sendPathResponse = pathResponseSmall
	}
	c.path.data = data
}

func (c *Conn) handlePathResponse(now time.Time, _ pathChallengeData) {
	// "If the content of a PATH_RESPONSE frame does not match the content of
	// a PATH_CHALLENGE frame previously sent by the endpoint,
	// the endpoint MAY generate a connection error of type PROTOCOL_VIOLATION."
	// https://www.rfc-editor.org/rfc/rfc9000.html#section-19.18-4
	//
	// We never send PATH_CHALLENGE frames.
	c.abort(now, localTransportError{
		code:   errProtocolViolation,
		reason: "PATH_RESPONSE received when no PATH_CHALLENGE sent",
	})
}

// appendPathFrames appends path validation related frames to the current packet.
// If the return value pad is true, then the packet should be padded to 1200 bytes.
func (c *Conn) appendPathFrames() (pad, ok bool) {
	if c.path.sendPathResponse == pathResponseNotNeeded {
		return pad, true
	}
	// We're required to send the PATH_RESPONSE on the path where the
	// PATH_CHALLENGE was received (RFC 9000, Section 8.2.2).
	//
	// At the moment, we don't support path migration and reject packets if
	// the peer changes its source address, so just sending the PATH_RESPONSE
	// in a regular datagram is fine.
	if !c.w.appendPathResponseFrame(c.path.data) {
		return pad, false
	}
	if c.path.sendPathResponse == pathResponseExpanded {
		pad = true
	}
	c.path.sendPathResponse = pathResponseNotNeeded
	return pad, true
}
