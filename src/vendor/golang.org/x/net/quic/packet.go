// Copyright 2023 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package quic

import (
	"encoding/binary"
	"fmt"

	"golang.org/x/net/internal/quic/quicwire"
)

// packetType is a QUIC packet type.
// https://www.rfc-editor.org/rfc/rfc9000.html#section-17
type packetType byte

const (
	packetTypeInvalid = packetType(iota)
	packetTypeInitial
	packetType0RTT
	packetTypeHandshake
	packetTypeRetry
	packetType1RTT
	packetTypeVersionNegotiation
)

func (p packetType) String() string {
	switch p {
	case packetTypeInitial:
		return "Initial"
	case packetType0RTT:
		return "0-RTT"
	case packetTypeHandshake:
		return "Handshake"
	case packetTypeRetry:
		return "Retry"
	case packetType1RTT:
		return "1-RTT"
	}
	return fmt.Sprintf("unknown packet type %v", byte(p))
}

func (p packetType) qlogString() string {
	switch p {
	case packetTypeInitial:
		return "initial"
	case packetType0RTT:
		return "0RTT"
	case packetTypeHandshake:
		return "handshake"
	case packetTypeRetry:
		return "retry"
	case packetType1RTT:
		return "1RTT"
	}
	return "unknown"
}

// Bits set in the first byte of a packet.
const (
	headerFormLong   = 0x80 // https://www.rfc-editor.org/rfc/rfc9000.html#section-17.2-3.2.1
	headerFormShort  = 0x00 // https://www.rfc-editor.org/rfc/rfc9000.html#section-17.3.1-4.2.1
	fixedBit         = 0x40 // https://www.rfc-editor.org/rfc/rfc9000.html#section-17.2-3.4.1
	reservedLongBits = 0x0c // https://www.rfc-editor.org/rfc/rfc9000#section-17.2-8.2.1
	reserved1RTTBits = 0x18 // https://www.rfc-editor.org/rfc/rfc9000#section-17.3.1-4.8.1
	keyPhaseBit      = 0x04 // https://www.rfc-editor.org/rfc/rfc9000#section-17.3.1-4.10.1
)

// Long Packet Type bits.
// https://www.rfc-editor.org/rfc/rfc9000.html#section-17.2-3.6.1
const (
	longPacketTypeInitial   = 0 << 4
	longPacketType0RTT      = 1 << 4
	longPacketTypeHandshake = 2 << 4
	longPacketTypeRetry     = 3 << 4
)

// Frame types.
// https://www.rfc-editor.org/rfc/rfc9000.html#section-19
const (
	frameTypePadding                    = 0x00
	frameTypePing                       = 0x01
	frameTypeAck                        = 0x02
	frameTypeAckECN                     = 0x03
	frameTypeResetStream                = 0x04
	frameTypeStopSending                = 0x05
	frameTypeCrypto                     = 0x06
	frameTypeNewToken                   = 0x07
	frameTypeStreamBase                 = 0x08 // low three bits carry stream flags
	frameTypeMaxData                    = 0x10
	frameTypeMaxStreamData              = 0x11
	frameTypeMaxStreamsBidi             = 0x12
	frameTypeMaxStreamsUni              = 0x13
	frameTypeDataBlocked                = 0x14
	frameTypeStreamDataBlocked          = 0x15
	frameTypeStreamsBlockedBidi         = 0x16
	frameTypeStreamsBlockedUni          = 0x17
	frameTypeNewConnectionID            = 0x18
	frameTypeRetireConnectionID         = 0x19
	frameTypePathChallenge              = 0x1a
	frameTypePathResponse               = 0x1b
	frameTypeConnectionCloseTransport   = 0x1c
	frameTypeConnectionCloseApplication = 0x1d
	frameTypeHandshakeDone              = 0x1e
)

// The low three bits of STREAM frames.
// https://www.rfc-editor.org/rfc/rfc9000.html#section-19.8
const (
	streamOffBit = 0x04
	streamLenBit = 0x02
	streamFinBit = 0x01
)

// Maximum length of a connection ID.
const maxConnIDLen = 20

// isLongHeader returns true if b is the first byte of a long header.
func isLongHeader(b byte) bool {
	return b&headerFormLong == headerFormLong
}

// getPacketType returns the type of a packet.
func getPacketType(b []byte) packetType {
	if len(b) == 0 {
		return packetTypeInvalid
	}
	if !isLongHeader(b[0]) {
		if b[0]&fixedBit != fixedBit {
			return packetTypeInvalid
		}
		return packetType1RTT
	}
	if len(b) < 5 {
		return packetTypeInvalid
	}
	if b[1] == 0 && b[2] == 0 && b[3] == 0 && b[4] == 0 {
		// Version Negotiation packets don't necessarily set the fixed bit.
		return packetTypeVersionNegotiation
	}
	if b[0]&fixedBit != fixedBit {
		return packetTypeInvalid
	}
	switch b[0] & 0x30 {
	case longPacketTypeInitial:
		return packetTypeInitial
	case longPacketType0RTT:
		return packetType0RTT
	case longPacketTypeHandshake:
		return packetTypeHandshake
	case longPacketTypeRetry:
		return packetTypeRetry
	}
	return packetTypeInvalid
}

// dstConnIDForDatagram returns the destination connection ID field of the
// first QUIC packet in a datagram.
func dstConnIDForDatagram(pkt []byte) (id []byte, ok bool) {
	if len(pkt) < 1 {
		return nil, false
	}
	var n int
	var b []byte
	if isLongHeader(pkt[0]) {
		if len(pkt) < 6 {
			return nil, false
		}
		n = int(pkt[5])
		b = pkt[6:]
	} else {
		n = connIDLen
		b = pkt[1:]
	}
	if len(b) < n {
		return nil, false
	}
	return b[:n], true
}

// parseVersionNegotiation parses a Version Negotiation packet.
// The returned versions is a slice of big-endian uint32s.
// It returns (nil, nil, nil) for an invalid packet.
func parseVersionNegotiation(pkt []byte) (dstConnID, srcConnID, versions []byte) {
	p, ok := parseGenericLongHeaderPacket(pkt)
	if !ok {
		return nil, nil, nil
	}
	if len(p.data)%4 != 0 {
		return nil, nil, nil
	}
	return p.dstConnID, p.srcConnID, p.data
}

// appendVersionNegotiation appends a Version Negotiation packet to pkt,
// returning the result.
func appendVersionNegotiation(pkt, dstConnID, srcConnID []byte, versions ...uint32) []byte {
	pkt = append(pkt, headerFormLong|fixedBit)      // header byte
	pkt = append(pkt, 0, 0, 0, 0)                   // Version (0 for Version Negotiation)
	pkt = quicwire.AppendUint8Bytes(pkt, dstConnID) // Destination Connection ID
	pkt = quicwire.AppendUint8Bytes(pkt, srcConnID) // Source Connection ID
	for _, v := range versions {
		pkt = binary.BigEndian.AppendUint32(pkt, v) // Supported Version
	}
	return pkt
}

// A longPacket is a long header packet.
type longPacket struct {
	ptype     packetType
	version   uint32
	num       packetNumber
	dstConnID []byte
	srcConnID []byte
	payload   []byte

	// The extra data depends on the packet type:
	//   Initial: Token.
	//   Retry: Retry token and integrity tag.
	extra []byte
}

// A shortPacket is a short header (1-RTT) packet.
type shortPacket struct {
	num     packetNumber
	payload []byte
}

// A genericLongPacket is a long header packet of an arbitrary QUIC version.
// https://www.rfc-editor.org/rfc/rfc8999#section-5.1
type genericLongPacket struct {
	version   uint32
	dstConnID []byte
	srcConnID []byte
	data      []byte
}

func parseGenericLongHeaderPacket(b []byte) (p genericLongPacket, ok bool) {
	if len(b) < 5 || !isLongHeader(b[0]) {
		return genericLongPacket{}, false
	}
	b = b[1:]
	// Version (32),
	var n int
	p.version, n = quicwire.ConsumeUint32(b)
	if n < 0 {
		return genericLongPacket{}, false
	}
	b = b[n:]
	// Destination Connection ID Length (8),
	// Destination Connection ID (0..2048),
	p.dstConnID, n = quicwire.ConsumeUint8Bytes(b)
	if n < 0 || len(p.dstConnID) > 2048/8 {
		return genericLongPacket{}, false
	}
	b = b[n:]
	// Source Connection ID Length (8),
	// Source Connection ID (0..2048),
	p.srcConnID, n = quicwire.ConsumeUint8Bytes(b)
	if n < 0 || len(p.dstConnID) > 2048/8 {
		return genericLongPacket{}, false
	}
	b = b[n:]
	p.data = b
	return p, true
}
