// Copyright 2023 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package quic

import "golang.org/x/net/internal/quic/quicwire"

// parseLongHeaderPacket parses a QUIC long header packet.
//
// It does not parse Version Negotiation packets.
//
// On input, pkt contains a long header packet (possibly followed by more packets),
// k the decryption keys for the packet, and pnumMax the largest packet number seen
// in the number space of this packet.
//
// parseLongHeaderPacket returns the parsed packet with protection removed
// and its length in bytes.
//
// It returns an empty packet and -1 if the packet could not be parsed.
func parseLongHeaderPacket(pkt []byte, k fixedKeys, pnumMax packetNumber) (p longPacket, n int) {
	if len(pkt) < 5 || !isLongHeader(pkt[0]) {
		return longPacket{}, -1
	}

	// Header Form (1) = 1,
	// Fixed Bit (1) = 1,
	// Long Packet Type (2),
	// Type-Specific Bits (4),
	b := pkt
	p.ptype = getPacketType(b)
	if p.ptype == packetTypeInvalid {
		return longPacket{}, -1
	}
	b = b[1:]
	// Version (32),
	p.version, n = quicwire.ConsumeUint32(b)
	if n < 0 {
		return longPacket{}, -1
	}
	b = b[n:]
	if p.version == 0 {
		// Version Negotiation packet; not handled here.
		return longPacket{}, -1
	}

	// Destination Connection ID Length (8),
	// Destination Connection ID (0..160),
	p.dstConnID, n = quicwire.ConsumeUint8Bytes(b)
	if n < 0 || len(p.dstConnID) > maxConnIDLen {
		return longPacket{}, -1
	}
	b = b[n:]

	// Source Connection ID Length (8),
	// Source Connection ID (0..160),
	p.srcConnID, n = quicwire.ConsumeUint8Bytes(b)
	if n < 0 || len(p.dstConnID) > maxConnIDLen {
		return longPacket{}, -1
	}
	b = b[n:]

	switch p.ptype {
	case packetTypeInitial:
		// Token Length (i),
		// Token (..),
		p.extra, n = quicwire.ConsumeVarintBytes(b)
		if n < 0 {
			return longPacket{}, -1
		}
		b = b[n:]
	case packetTypeRetry:
		// Retry Token (..),
		// Retry Integrity Tag (128),
		p.extra = b
		return p, len(pkt)
	}

	// Length (i),
	payLen, n := quicwire.ConsumeVarint(b)
	if n < 0 {
		return longPacket{}, -1
	}
	b = b[n:]
	if uint64(len(b)) < payLen {
		return longPacket{}, -1
	}

	// Packet Number (8..32),
	// Packet Payload (..),
	pnumOff := len(pkt) - len(b)
	pkt = pkt[:pnumOff+int(payLen)]

	if k.isSet() {
		var err error
		p.payload, p.num, err = k.unprotect(pkt, pnumOff, pnumMax)
		if err != nil {
			return longPacket{}, -1
		}
	}
	return p, len(pkt)
}

// skipLongHeaderPacket returns the length of the long header packet at the start of pkt,
// or -1 if the buffer does not contain a valid packet.
func skipLongHeaderPacket(pkt []byte) int {
	// Header byte, 4 bytes of version.
	n := 5
	if len(pkt) <= n {
		return -1
	}
	// Destination connection ID length, destination connection ID.
	n += 1 + int(pkt[n])
	if len(pkt) <= n {
		return -1
	}
	// Source connection ID length, source connection ID.
	n += 1 + int(pkt[n])
	if len(pkt) <= n {
		return -1
	}
	if getPacketType(pkt) == packetTypeInitial {
		// Token length, token.
		_, nn := quicwire.ConsumeVarintBytes(pkt[n:])
		if nn < 0 {
			return -1
		}
		n += nn
	}
	// Length, packet number, payload.
	_, nn := quicwire.ConsumeVarintBytes(pkt[n:])
	if nn < 0 {
		return -1
	}
	n += nn
	if len(pkt) < n {
		return -1
	}
	return n
}

// parse1RTTPacket parses a QUIC 1-RTT (short header) packet.
//
// On input, pkt contains a short header packet, k the decryption keys for the packet,
// and pnumMax the largest packet number seen in the number space of this packet.
func parse1RTTPacket(pkt []byte, k *updatingKeyPair, dstConnIDLen int, pnumMax packetNumber) (p shortPacket, err error) {
	pay, pnum, err := k.unprotect(pkt, 1+dstConnIDLen, pnumMax)
	if err != nil {
		return shortPacket{}, err
	}
	p.num = pnum
	p.payload = pay
	return p, nil
}

// Consume functions return n=-1 on conditions which result in FRAME_ENCODING_ERROR,
// which includes both general parse failures and specific violations of frame
// constraints.

func consumeAckFrame(frame []byte, f func(rangeIndex int, start, end packetNumber)) (largest packetNumber, ackDelay unscaledAckDelay, ecn ecnCounts, n int) {
	b := frame[1:] // type

	largestAck, n := quicwire.ConsumeVarint(b)
	if n < 0 {
		return 0, 0, ecnCounts{}, -1
	}
	b = b[n:]

	v, n := quicwire.ConsumeVarintInt64(b)
	if n < 0 {
		return 0, 0, ecnCounts{}, -1
	}
	b = b[n:]
	ackDelay = unscaledAckDelay(v)

	ackRangeCount, n := quicwire.ConsumeVarint(b)
	if n < 0 {
		return 0, 0, ecnCounts{}, -1
	}
	b = b[n:]

	rangeMax := packetNumber(largestAck)
	for i := uint64(0); ; i++ {
		rangeLen, n := quicwire.ConsumeVarint(b)
		if n < 0 {
			return 0, 0, ecnCounts{}, -1
		}
		b = b[n:]
		rangeMin := rangeMax - packetNumber(rangeLen)
		if rangeMin < 0 || rangeMin > rangeMax {
			return 0, 0, ecnCounts{}, -1
		}
		f(int(i), rangeMin, rangeMax+1)

		if i == ackRangeCount {
			break
		}

		gap, n := quicwire.ConsumeVarint(b)
		if n < 0 {
			return 0, 0, ecnCounts{}, -1
		}
		b = b[n:]

		rangeMax = rangeMin - packetNumber(gap) - 2
	}

	if frame[0] != frameTypeAckECN {
		return packetNumber(largestAck), ackDelay, ecnCounts{}, len(frame) - len(b)
	}

	ect0Count, n := quicwire.ConsumeVarint(b)
	if n < 0 {
		return 0, 0, ecnCounts{}, -1
	}
	b = b[n:]
	ect1Count, n := quicwire.ConsumeVarint(b)
	if n < 0 {
		return 0, 0, ecnCounts{}, -1
	}
	b = b[n:]
	ecnCECount, n := quicwire.ConsumeVarint(b)
	if n < 0 {
		return 0, 0, ecnCounts{}, -1
	}
	b = b[n:]

	ecn.t0 = int(ect0Count)
	ecn.t1 = int(ect1Count)
	ecn.ce = int(ecnCECount)

	return packetNumber(largestAck), ackDelay, ecn, len(frame) - len(b)
}

func consumeResetStreamFrame(b []byte) (id streamID, code uint64, finalSize int64, n int) {
	n = 1
	idInt, nn := quicwire.ConsumeVarint(b[n:])
	if nn < 0 {
		return 0, 0, 0, -1
	}
	n += nn
	code, nn = quicwire.ConsumeVarint(b[n:])
	if nn < 0 {
		return 0, 0, 0, -1
	}
	n += nn
	v, nn := quicwire.ConsumeVarint(b[n:])
	if nn < 0 {
		return 0, 0, 0, -1
	}
	n += nn
	finalSize = int64(v)
	return streamID(idInt), code, finalSize, n
}

func consumeStopSendingFrame(b []byte) (id streamID, code uint64, n int) {
	n = 1
	idInt, nn := quicwire.ConsumeVarint(b[n:])
	if nn < 0 {
		return 0, 0, -1
	}
	n += nn
	code, nn = quicwire.ConsumeVarint(b[n:])
	if nn < 0 {
		return 0, 0, -1
	}
	n += nn
	return streamID(idInt), code, n
}

func consumeCryptoFrame(b []byte) (off int64, data []byte, n int) {
	n = 1
	v, nn := quicwire.ConsumeVarint(b[n:])
	if nn < 0 {
		return 0, nil, -1
	}
	off = int64(v)
	n += nn
	data, nn = quicwire.ConsumeVarintBytes(b[n:])
	if nn < 0 {
		return 0, nil, -1
	}
	n += nn
	return off, data, n
}

func consumeNewTokenFrame(b []byte) (token []byte, n int) {
	n = 1
	data, nn := quicwire.ConsumeVarintBytes(b[n:])
	if nn < 0 {
		return nil, -1
	}
	if len(data) == 0 {
		return nil, -1
	}
	n += nn
	return data, n
}

func consumeStreamFrame(b []byte) (id streamID, off int64, fin bool, data []byte, n int) {
	fin = (b[0] & 0x01) != 0
	n = 1
	idInt, nn := quicwire.ConsumeVarint(b[n:])
	if nn < 0 {
		return 0, 0, false, nil, -1
	}
	n += nn
	if b[0]&0x04 != 0 {
		v, nn := quicwire.ConsumeVarint(b[n:])
		if nn < 0 {
			return 0, 0, false, nil, -1
		}
		n += nn
		off = int64(v)
	}
	if b[0]&0x02 != 0 {
		data, nn = quicwire.ConsumeVarintBytes(b[n:])
		if nn < 0 {
			return 0, 0, false, nil, -1
		}
		n += nn
	} else {
		data = b[n:]
		n += len(data)
	}
	if off+int64(len(data)) >= 1<<62 {
		return 0, 0, false, nil, -1
	}
	return streamID(idInt), off, fin, data, n
}

func consumeMaxDataFrame(b []byte) (max int64, n int) {
	n = 1
	v, nn := quicwire.ConsumeVarint(b[n:])
	if nn < 0 {
		return 0, -1
	}
	n += nn
	return int64(v), n
}

func consumeMaxStreamDataFrame(b []byte) (id streamID, max int64, n int) {
	n = 1
	v, nn := quicwire.ConsumeVarint(b[n:])
	if nn < 0 {
		return 0, 0, -1
	}
	n += nn
	id = streamID(v)
	v, nn = quicwire.ConsumeVarint(b[n:])
	if nn < 0 {
		return 0, 0, -1
	}
	n += nn
	max = int64(v)
	return id, max, n
}

func consumeMaxStreamsFrame(b []byte) (typ streamType, max int64, n int) {
	switch b[0] {
	case frameTypeMaxStreamsBidi:
		typ = bidiStream
	case frameTypeMaxStreamsUni:
		typ = uniStream
	default:
		return 0, 0, -1
	}
	n = 1
	v, nn := quicwire.ConsumeVarint(b[n:])
	if nn < 0 {
		return 0, 0, -1
	}
	n += nn
	if v > maxStreamsLimit {
		return 0, 0, -1
	}
	return typ, int64(v), n
}

func consumeStreamDataBlockedFrame(b []byte) (id streamID, max int64, n int) {
	n = 1
	v, nn := quicwire.ConsumeVarint(b[n:])
	if nn < 0 {
		return 0, 0, -1
	}
	n += nn
	id = streamID(v)
	max, nn = quicwire.ConsumeVarintInt64(b[n:])
	if nn < 0 {
		return 0, 0, -1
	}
	n += nn
	return id, max, n
}

func consumeDataBlockedFrame(b []byte) (max int64, n int) {
	n = 1
	max, nn := quicwire.ConsumeVarintInt64(b[n:])
	if nn < 0 {
		return 0, -1
	}
	n += nn
	return max, n
}

func consumeStreamsBlockedFrame(b []byte) (typ streamType, max int64, n int) {
	if b[0] == frameTypeStreamsBlockedBidi {
		typ = bidiStream
	} else {
		typ = uniStream
	}
	n = 1
	max, nn := quicwire.ConsumeVarintInt64(b[n:])
	if nn < 0 {
		return 0, 0, -1
	}
	n += nn
	return typ, max, n
}

func consumeNewConnectionIDFrame(b []byte) (seq, retire int64, connID []byte, resetToken statelessResetToken, n int) {
	n = 1
	var nn int
	seq, nn = quicwire.ConsumeVarintInt64(b[n:])
	if nn < 0 {
		return 0, 0, nil, statelessResetToken{}, -1
	}
	n += nn
	retire, nn = quicwire.ConsumeVarintInt64(b[n:])
	if nn < 0 {
		return 0, 0, nil, statelessResetToken{}, -1
	}
	n += nn
	if seq < retire {
		return 0, 0, nil, statelessResetToken{}, -1
	}
	connID, nn = quicwire.ConsumeVarintBytes(b[n:])
	if nn < 0 {
		return 0, 0, nil, statelessResetToken{}, -1
	}
	if len(connID) < 1 || len(connID) > 20 {
		return 0, 0, nil, statelessResetToken{}, -1
	}
	n += nn
	if len(b[n:]) < len(resetToken) {
		return 0, 0, nil, statelessResetToken{}, -1
	}
	copy(resetToken[:], b[n:])
	n += len(resetToken)
	return seq, retire, connID, resetToken, n
}

func consumeRetireConnectionIDFrame(b []byte) (seq int64, n int) {
	n = 1
	var nn int
	seq, nn = quicwire.ConsumeVarintInt64(b[n:])
	if nn < 0 {
		return 0, -1
	}
	n += nn
	return seq, n
}

func consumePathChallengeFrame(b []byte) (data pathChallengeData, n int) {
	n = 1
	nn := copy(data[:], b[n:])
	if nn != len(data) {
		return data, -1
	}
	n += nn
	return data, n
}

func consumePathResponseFrame(b []byte) (data pathChallengeData, n int) {
	return consumePathChallengeFrame(b) // identical frame format
}

func consumeConnectionCloseTransportFrame(b []byte) (code transportError, frameType uint64, reason string, n int) {
	n = 1
	var nn int
	var codeInt uint64
	codeInt, nn = quicwire.ConsumeVarint(b[n:])
	if nn < 0 {
		return 0, 0, "", -1
	}
	code = transportError(codeInt)
	n += nn
	frameType, nn = quicwire.ConsumeVarint(b[n:])
	if nn < 0 {
		return 0, 0, "", -1
	}
	n += nn
	reasonb, nn := quicwire.ConsumeVarintBytes(b[n:])
	if nn < 0 {
		return 0, 0, "", -1
	}
	n += nn
	reason = string(reasonb)
	return code, frameType, reason, n
}

func consumeConnectionCloseApplicationFrame(b []byte) (code uint64, reason string, n int) {
	n = 1
	var nn int
	code, nn = quicwire.ConsumeVarint(b[n:])
	if nn < 0 {
		return 0, "", -1
	}
	n += nn
	reasonb, nn := quicwire.ConsumeVarintBytes(b[n:])
	if nn < 0 {
		return 0, "", -1
	}
	n += nn
	reason = string(reasonb)
	return code, reason, n
}
