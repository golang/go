// Copyright 2023 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package quic

import (
	"time"
)

// QUIC versions.
// We only support v1 at this time.
const (
	quicVersion1 = 1
	quicVersion2 = 0x6b3343cf // https://www.rfc-editor.org/rfc/rfc9369
)

// connIDLen is the length in bytes of connection IDs chosen by this package.
// Since 1-RTT packets don't include a connection ID length field,
// we use a consistent length for all our IDs.
// https://www.rfc-editor.org/rfc/rfc9000.html#section-5.1-6
const connIDLen = 8

// Local values of various transport parameters.
// https://www.rfc-editor.org/rfc/rfc9000.html#section-18.2
const (
	defaultMaxIdleTimeout = 30 * time.Second // max_idle_timeout

	// The max_udp_payload_size transport parameter is the size of our
	// network receive buffer.
	//
	// Set this to the largest UDP packet that can be sent over
	// Ethernet without using jumbo frames: 1500 byte Ethernet frame,
	// minus 20 byte IPv4 header and 8 byte UDP header.
	//
	// The maximum possible UDP payload is 65527 bytes. Supporting this
	// without wasting memory in unused receive buffers will require some
	// care. For now, just limit ourselves to the most common case.
	maxUDPPayloadSize = 1472

	ackDelayExponent = 3                     // ack_delay_exponent
	maxAckDelay      = 25 * time.Millisecond // max_ack_delay

	// The active_conn_id_limit transport parameter is the maximum
	// number of connection IDs from the peer we're willing to store.
	//
	// maxPeerActiveConnIDLimit is the maximum number of connection IDs
	// we're willing to send to the peer.
	//
	// https://www.rfc-editor.org/rfc/rfc9000.html#section-18.2-6.2.1
	activeConnIDLimit        = 2
	maxPeerActiveConnIDLimit = 4
)

// Time limit for completing the handshake.
const defaultHandshakeTimeout = 10 * time.Second

// Keep-alive ping frequency.
const defaultKeepAlivePeriod = 0

// Local timer granularity.
// https://www.rfc-editor.org/rfc/rfc9002.html#section-6.1.2-6
const timerGranularity = 1 * time.Millisecond

// The smallest allowed maximum datagram size.
// https://www.rfc-editor.org/rfc/rfc9000#section-14
const smallestMaxDatagramSize = 1200

// Minimum size of a UDP datagram sent by a client carrying an Initial packet,
// or a server containing an ack-eliciting Initial packet.
// https://www.rfc-editor.org/rfc/rfc9000#section-14.1
const paddedInitialDatagramSize = smallestMaxDatagramSize

// Maximum number of streams of a given type which may be created.
// https://www.rfc-editor.org/rfc/rfc9000.html#section-4.6-2
const maxStreamsLimit = 1 << 60

// Maximum number of streams we will allow the peer to create implicitly.
// A stream ID that is used out of order results in all streams of that type
// with lower-numbered IDs also being opened. To limit the amount of work we
// will do in response to a single frame, we cap the peer's stream limit to
// this value.
const implicitStreamLimit = 100

// A connSide distinguishes between the client and server sides of a connection.
type connSide int8

const (
	clientSide = connSide(iota)
	serverSide
)

func (s connSide) String() string {
	switch s {
	case clientSide:
		return "client"
	case serverSide:
		return "server"
	default:
		return "BUG"
	}
}

func (s connSide) peer() connSide {
	if s == clientSide {
		return serverSide
	} else {
		return clientSide
	}
}

// A numberSpace is the context in which a packet number applies.
// https://www.rfc-editor.org/rfc/rfc9000.html#section-12.3-7
type numberSpace byte

const (
	initialSpace = numberSpace(iota)
	handshakeSpace
	appDataSpace
	numberSpaceCount
)

func (n numberSpace) String() string {
	switch n {
	case initialSpace:
		return "Initial"
	case handshakeSpace:
		return "Handshake"
	case appDataSpace:
		return "AppData"
	default:
		return "BUG"
	}
}

// A streamType is the type of a stream: bidirectional or unidirectional.
type streamType uint8

const (
	bidiStream = streamType(iota)
	uniStream
	streamTypeCount
)

func (s streamType) qlogString() string {
	switch s {
	case bidiStream:
		return "bidirectional"
	case uniStream:
		return "unidirectional"
	default:
		return "BUG"
	}
}

func (s streamType) String() string {
	switch s {
	case bidiStream:
		return "bidi"
	case uniStream:
		return "uni"
	default:
		return "BUG"
	}
}

// A streamID is a QUIC stream ID.
// https://www.rfc-editor.org/rfc/rfc9000.html#section-2.1
type streamID uint64

// The two least significant bits of a stream ID indicate the initiator
// and directionality of the stream. The upper bits are the stream number.
// Each of the four possible combinations of initiator and direction
// each has a distinct number space.
const (
	clientInitiatedStreamBit = 0x0
	serverInitiatedStreamBit = 0x1
	initiatorStreamBitMask   = 0x1

	bidiStreamBit    = 0x0
	uniStreamBit     = 0x2
	dirStreamBitMask = 0x2
)

func newStreamID(initiator connSide, typ streamType, num int64) streamID {
	id := streamID(num << 2)
	if typ == uniStream {
		id |= uniStreamBit
	}
	if initiator == serverSide {
		id |= serverInitiatedStreamBit
	}
	return id
}

func (s streamID) initiator() connSide {
	if s&initiatorStreamBitMask == serverInitiatedStreamBit {
		return serverSide
	}
	return clientSide
}

func (s streamID) num() int64 {
	return int64(s) >> 2
}

func (s streamID) streamType() streamType {
	if s&dirStreamBitMask == uniStreamBit {
		return uniStream
	}
	return bidiStream
}

// packetFate is the fate of a sent packet: Either acknowledged by the peer,
// or declared lost.
type packetFate byte

const (
	packetLost = packetFate(iota)
	packetAcked
)
