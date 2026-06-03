// Copyright 2023 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package quic

import (
	"encoding/binary"

	"golang.org/x/net/internal/quic/quicwire"
)

// A packetWriter constructs QUIC datagrams.
//
// A datagram consists of one or more packets.
// A packet consists of a header followed by one or more frames.
//
// Packets are written in three steps:
// - startProtectedLongHeaderPacket or start1RTT packet prepare the packet;
// - append*Frame appends frames to the payload; and
// - finishProtectedLongHeaderPacket or finish1RTT finalize the packet.
//
// The start functions are efficient, so we can start speculatively
// writing a packet before we know whether we have any frames to
// put in it. The finish functions will abandon the packet if the
// payload contains no data.
type packetWriter struct {
	dgramLim int // max datagram size
	pktLim   int // max packet size
	pktOff   int // offset of the start of the current packet
	payOff   int // offset of the payload of the current packet
	b        []byte
	sent     *sentPacket
}

// reset prepares to write a datagram of at most lim bytes.
func (w *packetWriter) reset(lim int) {
	if cap(w.b) < lim {
		w.b = make([]byte, 0, lim)
	}
	w.dgramLim = lim
	w.b = w.b[:0]
}

// datagram returns the current datagram.
func (w *packetWriter) datagram() []byte {
	return w.b
}

// packetLen returns the size of the current packet.
func (w *packetWriter) packetLen() int {
	return len(w.b[w.pktOff:]) + aeadOverhead
}

// payload returns the payload of the current packet.
func (w *packetWriter) payload() []byte {
	return w.b[w.payOff:]
}

func (w *packetWriter) abandonPacket() {
	w.b = w.b[:w.payOff]
	w.sent.reset()
}

// startProtectedLongHeaderPacket starts writing an Initial, 0-RTT, or Handshake packet.
func (w *packetWriter) startProtectedLongHeaderPacket(pnumMaxAcked packetNumber, p longPacket) {
	if w.sent == nil {
		w.sent = newSentPacket()
	}
	w.pktOff = len(w.b)
	hdrSize := 1 // packet type
	hdrSize += 4 // version
	hdrSize += 1 + len(p.dstConnID)
	hdrSize += 1 + len(p.srcConnID)
	switch p.ptype {
	case packetTypeInitial:
		hdrSize += quicwire.SizeVarint(uint64(len(p.extra))) + len(p.extra)
	}
	hdrSize += 2 // length, hardcoded to a 2-byte varint
	pnumOff := len(w.b) + hdrSize
	hdrSize += packetNumberLength(p.num, pnumMaxAcked)
	payOff := len(w.b) + hdrSize
	// Check if we have enough space to hold the packet, including the header,
	// header protection sample (RFC 9001, section 5.4.2), and encryption overhead.
	if pnumOff+4+headerProtectionSampleSize+aeadOverhead >= w.dgramLim {
		// Set the limit on the packet size to be the current write buffer length,
		// ensuring that any writes to the payload fail.
		w.payOff = len(w.b)
		w.pktLim = len(w.b)
		return
	}
	w.payOff = payOff
	w.pktLim = w.dgramLim - aeadOverhead
	// We hardcode the payload length field to be 2 bytes, which limits the payload
	// (including the packet number) to 16383 bytes (the largest 2-byte QUIC varint).
	//
	// Most networks don't support datagrams over 1472 bytes, and even Ethernet
	// jumbo frames are generally only about 9000 bytes.
	if lim := pnumOff + 16383 - aeadOverhead; lim < w.pktLim {
		w.pktLim = lim
	}
	w.b = w.b[:payOff]
}

// finishProtectedLongHeaderPacket finishes writing an Initial, 0-RTT, or Handshake packet,
// canceling the packet if it contains no payload.
// It returns a sentPacket describing the packet, or nil if no packet was written.
func (w *packetWriter) finishProtectedLongHeaderPacket(pnumMaxAcked packetNumber, k fixedKeys, p longPacket) *sentPacket {
	if len(w.b) == w.payOff {
		// The payload is empty, so just abandon the packet.
		w.b = w.b[:w.pktOff]
		return nil
	}
	pnumLen := packetNumberLength(p.num, pnumMaxAcked)
	plen := w.padPacketLength(pnumLen)
	hdr := w.b[:w.pktOff]
	var typeBits byte
	switch p.ptype {
	case packetTypeInitial:
		typeBits = longPacketTypeInitial
	case packetType0RTT:
		typeBits = longPacketType0RTT
	case packetTypeHandshake:
		typeBits = longPacketTypeHandshake
	case packetTypeRetry:
		typeBits = longPacketTypeRetry
	}
	hdr = append(hdr, headerFormLong|fixedBit|typeBits|byte(pnumLen-1))
	hdr = binary.BigEndian.AppendUint32(hdr, p.version)
	hdr = quicwire.AppendUint8Bytes(hdr, p.dstConnID)
	hdr = quicwire.AppendUint8Bytes(hdr, p.srcConnID)
	switch p.ptype {
	case packetTypeInitial:
		hdr = quicwire.AppendVarintBytes(hdr, p.extra) // token
	}

	// Packet length, always encoded as a 2-byte varint.
	hdr = append(hdr, 0x40|byte(plen>>8), byte(plen))

	pnumOff := len(hdr)
	hdr = appendPacketNumber(hdr, p.num, pnumMaxAcked)

	k.protect(hdr[w.pktOff:], w.b[len(hdr):], pnumOff-w.pktOff, p.num)
	return w.finish(p.ptype, p.num)
}

// start1RTTPacket starts writing a 1-RTT (short header) packet.
func (w *packetWriter) start1RTTPacket(pnum, pnumMaxAcked packetNumber, dstConnID []byte) {
	if w.sent == nil {
		w.sent = newSentPacket()
	}
	w.pktOff = len(w.b)
	hdrSize := 1 // packet type
	hdrSize += len(dstConnID)
	// Ensure we have enough space to hold the packet, including the header,
	// header protection sample (RFC 9001, section 5.4.2), and encryption overhead.
	if len(w.b)+hdrSize+4+headerProtectionSampleSize+aeadOverhead >= w.dgramLim {
		w.payOff = len(w.b)
		w.pktLim = len(w.b)
		return
	}
	hdrSize += packetNumberLength(pnum, pnumMaxAcked)
	w.payOff = len(w.b) + hdrSize
	w.pktLim = w.dgramLim - aeadOverhead
	w.b = w.b[:w.payOff]
}

// finish1RTTPacket finishes writing a 1-RTT packet,
// canceling the packet if it contains no payload.
// It returns a sentPacket describing the packet, or nil if no packet was written.
func (w *packetWriter) finish1RTTPacket(pnum, pnumMaxAcked packetNumber, dstConnID []byte, k *updatingKeyPair) *sentPacket {
	if len(w.b) == w.payOff {
		// The payload is empty, so just abandon the packet.
		w.b = w.b[:w.pktOff]
		return nil
	}
	// TODO: Spin
	pnumLen := packetNumberLength(pnum, pnumMaxAcked)
	hdr := w.b[:w.pktOff]
	hdr = append(hdr, 0x40|byte(pnumLen-1))
	hdr = append(hdr, dstConnID...)
	pnumOff := len(hdr)
	hdr = appendPacketNumber(hdr, pnum, pnumMaxAcked)
	w.padPacketLength(pnumLen)
	k.protect(hdr[w.pktOff:], w.b[len(hdr):], pnumOff-w.pktOff, pnum)
	return w.finish(packetType1RTT, pnum)
}

// padPacketLength pads out the payload of the current packet to the minimum size,
// and returns the combined length of the packet number and payload (used for the Length
// field of long header packets).
func (w *packetWriter) padPacketLength(pnumLen int) int {
	plen := len(w.b) - w.payOff + pnumLen + aeadOverhead
	// "To ensure that sufficient data is available for sampling, packets are
	// padded so that the combined lengths of the encoded packet number and
	// protected payload is at least 4 bytes longer than the sample required
	// for header protection."
	// https://www.rfc-editor.org/rfc/rfc9001.html#section-5.4.2
	for plen < 4+headerProtectionSampleSize {
		w.b = append(w.b, 0)
		plen++
	}
	return plen
}

// finish finishes the current packet after protection is applied.
func (w *packetWriter) finish(ptype packetType, pnum packetNumber) *sentPacket {
	w.b = w.b[:len(w.b)+aeadOverhead]
	w.sent.size = len(w.b) - w.pktOff
	w.sent.ptype = ptype
	w.sent.num = pnum
	sent := w.sent
	w.sent = nil
	return sent
}

// avail reports how many more bytes may be written to the current packet.
func (w *packetWriter) avail() int {
	return w.pktLim - len(w.b)
}

// appendPaddingTo appends PADDING frames until the total datagram size
// (including AEAD overhead of the current packet) is n.
func (w *packetWriter) appendPaddingTo(n int) {
	n -= aeadOverhead
	lim := w.pktLim
	if n < lim {
		lim = n
	}
	if len(w.b) >= lim {
		return
	}
	for len(w.b) < lim {
		w.b = append(w.b, frameTypePadding)
	}
	// Packets are considered in flight when they contain a PADDING frame.
	// https://www.rfc-editor.org/rfc/rfc9002.html#section-2-3.6.1
	w.sent.inFlight = true
}

func (w *packetWriter) appendPingFrame() (added bool) {
	if len(w.b) >= w.pktLim {
		return false
	}
	w.b = append(w.b, frameTypePing)
	w.sent.markAckEliciting() // no need to record the frame itself
	return true
}

// appendAckFrame appends an ACK frame to the payload.
// It includes at least the most recent range in the rangeset
// (the range with the largest packet numbers),
// followed by as many additional ranges as fit within the packet.
//
// We always place ACK frames at the start of packets,
// we limit the number of ack ranges retained, and
// we set a minimum packet payload size.
// As a result, appendAckFrame will rarely if ever drop ranges
// in practice.
//
// In the event that ranges are dropped, the impact is limited
// to the peer potentially failing to receive an acknowledgement
// for an older packet during a period of high packet loss or
// reordering. This may result in unnecessary retransmissions.
func (w *packetWriter) appendAckFrame(seen rangeset[packetNumber], delay unscaledAckDelay, ecn ecnCounts) (added bool) {
	if len(seen) == 0 {
		return false
	}
	var (
		largest    = uint64(seen.max())
		firstRange = uint64(seen[len(seen)-1].size() - 1)
	)
	var ecnLen int
	ackType := byte(frameTypeAck)
	if (ecn != ecnCounts{}) {
		// "Even if an endpoint does not set an ECT field in packets it sends,
		// the endpoint MUST provide feedback about ECN markings it receives, if
		// these are accessible."
		// https://www.rfc-editor.org/rfc/rfc9000.html#section-13.4.1-2
		ecnLen = quicwire.SizeVarint(uint64(ecn.ce)) + quicwire.SizeVarint(uint64(ecn.t0)) + quicwire.SizeVarint(uint64(ecn.t1))
		ackType = frameTypeAckECN
	}
	if w.avail() < 1+quicwire.SizeVarint(largest)+quicwire.SizeVarint(uint64(delay))+1+quicwire.SizeVarint(firstRange)+ecnLen {
		return false
	}
	w.b = append(w.b, ackType)
	w.b = quicwire.AppendVarint(w.b, largest)
	w.b = quicwire.AppendVarint(w.b, uint64(delay))
	// The range count is technically a varint, but we'll reserve a single byte for it
	// and never add more than 62 ranges (the maximum varint that fits in a byte).
	rangeCountOff := len(w.b)
	w.b = append(w.b, 0)
	w.b = quicwire.AppendVarint(w.b, firstRange)
	rangeCount := byte(0)
	for i := len(seen) - 2; i >= 0; i-- {
		gap := uint64(seen[i+1].start - seen[i].end - 1)
		size := uint64(seen[i].size() - 1)
		if w.avail() < quicwire.SizeVarint(gap)+quicwire.SizeVarint(size)+ecnLen || rangeCount > 62 {
			break
		}
		w.b = quicwire.AppendVarint(w.b, gap)
		w.b = quicwire.AppendVarint(w.b, size)
		rangeCount++
	}
	w.b[rangeCountOff] = rangeCount
	if ackType == frameTypeAckECN {
		w.b = quicwire.AppendVarint(w.b, uint64(ecn.t0))
		w.b = quicwire.AppendVarint(w.b, uint64(ecn.t1))
		w.b = quicwire.AppendVarint(w.b, uint64(ecn.ce))
	}
	w.sent.appendNonAckElicitingFrame(ackType)
	w.sent.appendInt(uint64(seen.max()))
	return true
}

func (w *packetWriter) appendNewTokenFrame(token []byte) (added bool) {
	if w.avail() < 1+quicwire.SizeVarint(uint64(len(token)))+len(token) {
		return false
	}
	w.b = append(w.b, frameTypeNewToken)
	w.b = quicwire.AppendVarintBytes(w.b, token)
	return true
}

func (w *packetWriter) appendResetStreamFrame(id streamID, code uint64, finalSize int64) (added bool) {
	if w.avail() < 1+quicwire.SizeVarint(uint64(id))+quicwire.SizeVarint(code)+quicwire.SizeVarint(uint64(finalSize)) {
		return false
	}
	w.b = append(w.b, frameTypeResetStream)
	w.b = quicwire.AppendVarint(w.b, uint64(id))
	w.b = quicwire.AppendVarint(w.b, code)
	w.b = quicwire.AppendVarint(w.b, uint64(finalSize))
	w.sent.appendAckElicitingFrame(frameTypeResetStream)
	w.sent.appendInt(uint64(id))
	return true
}

func (w *packetWriter) appendStopSendingFrame(id streamID, code uint64) (added bool) {
	if w.avail() < 1+quicwire.SizeVarint(uint64(id))+quicwire.SizeVarint(code) {
		return false
	}
	w.b = append(w.b, frameTypeStopSending)
	w.b = quicwire.AppendVarint(w.b, uint64(id))
	w.b = quicwire.AppendVarint(w.b, code)
	w.sent.appendAckElicitingFrame(frameTypeStopSending)
	w.sent.appendInt(uint64(id))
	return true
}

// appendCryptoFrame appends a CRYPTO frame.
// It returns a []byte into which the data should be written and whether a frame was added.
// The returned []byte may be smaller than size if the packet cannot hold all the data.
func (w *packetWriter) appendCryptoFrame(off int64, size int) (_ []byte, added bool) {
	max := w.avail()
	max -= 1                                 // frame type
	max -= quicwire.SizeVarint(uint64(off))  // offset
	max -= quicwire.SizeVarint(uint64(size)) // maximum length
	if max <= 0 {
		return nil, false
	}
	if max < size {
		size = max
	}
	w.b = append(w.b, frameTypeCrypto)
	w.b = quicwire.AppendVarint(w.b, uint64(off))
	w.b = quicwire.AppendVarint(w.b, uint64(size))
	start := len(w.b)
	w.b = w.b[:start+size]
	w.sent.appendAckElicitingFrame(frameTypeCrypto)
	w.sent.appendOffAndSize(off, size)
	return w.b[start:][:size], true
}

// appendStreamFrame appends a STREAM frame.
// It returns a []byte into which the data should be written and whether a frame was added.
// The returned []byte may be smaller than size if the packet cannot hold all the data.
func (w *packetWriter) appendStreamFrame(id streamID, off int64, size int, fin bool) (_ []byte, added bool) {
	typ := uint8(frameTypeStreamBase | streamLenBit)
	max := w.avail()
	max -= 1 // frame type
	max -= quicwire.SizeVarint(uint64(id))
	if off != 0 {
		max -= quicwire.SizeVarint(uint64(off))
		typ |= streamOffBit
	}
	max -= quicwire.SizeVarint(uint64(size)) // maximum length
	if max < 0 || (max == 0 && size > 0) {
		return nil, false
	}
	if max < size {
		size = max
	} else if fin {
		typ |= streamFinBit
	}
	w.b = append(w.b, typ)
	w.b = quicwire.AppendVarint(w.b, uint64(id))
	if off != 0 {
		w.b = quicwire.AppendVarint(w.b, uint64(off))
	}
	w.b = quicwire.AppendVarint(w.b, uint64(size))
	start := len(w.b)
	w.b = w.b[:start+size]
	w.sent.appendAckElicitingFrame(typ & (frameTypeStreamBase | streamFinBit))
	w.sent.appendInt(uint64(id))
	w.sent.appendOffAndSize(off, size)
	return w.b[start:][:size], true
}

func (w *packetWriter) appendMaxDataFrame(max int64) (added bool) {
	if w.avail() < 1+quicwire.SizeVarint(uint64(max)) {
		return false
	}
	w.b = append(w.b, frameTypeMaxData)
	w.b = quicwire.AppendVarint(w.b, uint64(max))
	w.sent.appendAckElicitingFrame(frameTypeMaxData)
	return true
}

func (w *packetWriter) appendMaxStreamDataFrame(id streamID, max int64) (added bool) {
	if w.avail() < 1+quicwire.SizeVarint(uint64(id))+quicwire.SizeVarint(uint64(max)) {
		return false
	}
	w.b = append(w.b, frameTypeMaxStreamData)
	w.b = quicwire.AppendVarint(w.b, uint64(id))
	w.b = quicwire.AppendVarint(w.b, uint64(max))
	w.sent.appendAckElicitingFrame(frameTypeMaxStreamData)
	w.sent.appendInt(uint64(id))
	return true
}

func (w *packetWriter) appendMaxStreamsFrame(streamType streamType, max int64) (added bool) {
	if w.avail() < 1+quicwire.SizeVarint(uint64(max)) {
		return false
	}
	var typ byte
	if streamType == bidiStream {
		typ = frameTypeMaxStreamsBidi
	} else {
		typ = frameTypeMaxStreamsUni
	}
	w.b = append(w.b, typ)
	w.b = quicwire.AppendVarint(w.b, uint64(max))
	w.sent.appendAckElicitingFrame(typ)
	return true
}

func (w *packetWriter) appendDataBlockedFrame(max int64) (added bool) {
	if w.avail() < 1+quicwire.SizeVarint(uint64(max)) {
		return false
	}
	w.b = append(w.b, frameTypeDataBlocked)
	w.b = quicwire.AppendVarint(w.b, uint64(max))
	w.sent.appendAckElicitingFrame(frameTypeDataBlocked)
	return true
}

func (w *packetWriter) appendStreamDataBlockedFrame(id streamID, max int64) (added bool) {
	if w.avail() < 1+quicwire.SizeVarint(uint64(id))+quicwire.SizeVarint(uint64(max)) {
		return false
	}
	w.b = append(w.b, frameTypeStreamDataBlocked)
	w.b = quicwire.AppendVarint(w.b, uint64(id))
	w.b = quicwire.AppendVarint(w.b, uint64(max))
	w.sent.appendAckElicitingFrame(frameTypeStreamDataBlocked)
	w.sent.appendInt(uint64(id))
	return true
}

func (w *packetWriter) appendStreamsBlockedFrame(typ streamType, max int64) (added bool) {
	if w.avail() < 1+quicwire.SizeVarint(uint64(max)) {
		return false
	}
	var ftype byte
	if typ == bidiStream {
		ftype = frameTypeStreamsBlockedBidi
	} else {
		ftype = frameTypeStreamsBlockedUni
	}
	w.b = append(w.b, ftype)
	w.b = quicwire.AppendVarint(w.b, uint64(max))
	w.sent.appendAckElicitingFrame(ftype)
	return true
}

func (w *packetWriter) appendNewConnectionIDFrame(seq, retirePriorTo int64, connID []byte, token [16]byte) (added bool) {
	if w.avail() < 1+quicwire.SizeVarint(uint64(seq))+quicwire.SizeVarint(uint64(retirePriorTo))+1+len(connID)+len(token) {
		return false
	}
	w.b = append(w.b, frameTypeNewConnectionID)
	w.b = quicwire.AppendVarint(w.b, uint64(seq))
	w.b = quicwire.AppendVarint(w.b, uint64(retirePriorTo))
	w.b = quicwire.AppendUint8Bytes(w.b, connID)
	w.b = append(w.b, token[:]...)
	w.sent.appendAckElicitingFrame(frameTypeNewConnectionID)
	w.sent.appendInt(uint64(seq))
	return true
}

func (w *packetWriter) appendRetireConnectionIDFrame(seq int64) (added bool) {
	if w.avail() < 1+quicwire.SizeVarint(uint64(seq)) {
		return false
	}
	w.b = append(w.b, frameTypeRetireConnectionID)
	w.b = quicwire.AppendVarint(w.b, uint64(seq))
	w.sent.appendAckElicitingFrame(frameTypeRetireConnectionID)
	w.sent.appendInt(uint64(seq))
	return true
}

func (w *packetWriter) appendPathChallengeFrame(data pathChallengeData) (added bool) {
	if w.avail() < 1+8 {
		return false
	}
	w.b = append(w.b, frameTypePathChallenge)
	w.b = append(w.b, data[:]...)
	w.sent.markAckEliciting() // no need to record the frame itself
	return true
}

func (w *packetWriter) appendPathResponseFrame(data pathChallengeData) (added bool) {
	if w.avail() < 1+8 {
		return false
	}
	w.b = append(w.b, frameTypePathResponse)
	w.b = append(w.b, data[:]...)
	w.sent.markAckEliciting() // no need to record the frame itself
	return true
}

// appendConnectionCloseTransportFrame appends a CONNECTION_CLOSE frame
// carrying a transport error code.
func (w *packetWriter) appendConnectionCloseTransportFrame(code transportError, frameType uint64, reason string) (added bool) {
	if w.avail() < 1+quicwire.SizeVarint(uint64(code))+quicwire.SizeVarint(frameType)+quicwire.SizeVarint(uint64(len(reason)))+len(reason) {
		return false
	}
	w.b = append(w.b, frameTypeConnectionCloseTransport)
	w.b = quicwire.AppendVarint(w.b, uint64(code))
	w.b = quicwire.AppendVarint(w.b, frameType)
	w.b = quicwire.AppendVarintBytes(w.b, []byte(reason))
	// We don't record CONNECTION_CLOSE frames in w.sent, since they are never acked or
	// detected as lost.
	return true
}

// appendConnectionCloseApplicationFrame appends a CONNECTION_CLOSE frame
// carrying an application protocol error code.
func (w *packetWriter) appendConnectionCloseApplicationFrame(code uint64, reason string) (added bool) {
	if w.avail() < 1+quicwire.SizeVarint(code)+quicwire.SizeVarint(uint64(len(reason)))+len(reason) {
		return false
	}
	w.b = append(w.b, frameTypeConnectionCloseApplication)
	w.b = quicwire.AppendVarint(w.b, code)
	w.b = quicwire.AppendVarintBytes(w.b, []byte(reason))
	// We don't record CONNECTION_CLOSE frames in w.sent, since they are never acked or
	// detected as lost.
	return true
}

func (w *packetWriter) appendHandshakeDoneFrame() (added bool) {
	if w.avail() < 1 {
		return false
	}
	w.b = append(w.b, frameTypeHandshakeDone)
	w.sent.appendAckElicitingFrame(frameTypeHandshakeDone)
	return true
}
