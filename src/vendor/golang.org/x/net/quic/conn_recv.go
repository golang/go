// Copyright 2023 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package quic

import (
	"bytes"
	"encoding/binary"
	"errors"
	"time"
)

func (c *Conn) handleDatagram(now time.Time, dgram *datagram) (handled bool) {
	if !c.localAddr.IsValid() {
		// We don't have any way to tell in the general case what address we're
		// sending packets from. Set our address from the destination address of
		// the first packet received from the peer.
		c.localAddr = dgram.localAddr
	}
	if dgram.peerAddr.IsValid() && dgram.peerAddr != c.peerAddr {
		if c.side == clientSide {
			// "If a client receives packets from an unknown server address,
			// the client MUST discard these packets."
			// https://www.rfc-editor.org/rfc/rfc9000#section-9-6
			return false
		}
		// We currently don't support connection migration,
		// so for now the server also drops packets from an unknown address.
		return false
	}
	buf := dgram.b
	c.loss.datagramReceived(now, len(buf))
	if c.isDraining() {
		return false
	}
	for len(buf) > 0 {
		var n int
		ptype := getPacketType(buf)
		switch ptype {
		case packetTypeInitial:
			if c.side == serverSide && len(dgram.b) < paddedInitialDatagramSize {
				// Discard client-sent Initial packets in too-short datagrams.
				// https://www.rfc-editor.org/rfc/rfc9000#section-14.1-4
				return false
			}
			n = c.handleLongHeader(now, dgram, ptype, initialSpace, c.keysInitial.r, buf)
		case packetTypeHandshake:
			n = c.handleLongHeader(now, dgram, ptype, handshakeSpace, c.keysHandshake.r, buf)
		case packetType1RTT:
			n = c.handle1RTT(now, dgram, buf)
		case packetTypeRetry:
			c.handleRetry(now, buf)
			return true
		case packetTypeVersionNegotiation:
			c.handleVersionNegotiation(now, buf)
			return true
		default:
			n = -1
		}
		if n <= 0 {
			// We don't expect to get a stateless reset with a valid
			// destination connection ID, since the sender of a stateless
			// reset doesn't know what the connection ID is.
			//
			// We're required to perform this check anyway.
			//
			// "[...] the comparison MUST be performed when the first packet
			// in an incoming datagram [...] cannot be decrypted."
			// https://www.rfc-editor.org/rfc/rfc9000#section-10.3.1-2
			if len(buf) == len(dgram.b) && len(buf) > statelessResetTokenLen {
				var token statelessResetToken
				copy(token[:], buf[len(buf)-len(token):])
				if c.handleStatelessReset(now, token) {
					return true
				}
			}
			// Invalid data at the end of a datagram is ignored.
			return false
		}
		c.idleHandlePacketReceived(now)
		buf = buf[n:]
	}
	return true
}

func (c *Conn) handleLongHeader(now time.Time, dgram *datagram, ptype packetType, space numberSpace, k fixedKeys, buf []byte) int {
	if !k.isSet() {
		return skipLongHeaderPacket(buf)
	}

	pnumMax := c.acks[space].largestSeen()
	p, n := parseLongHeaderPacket(buf, k, pnumMax)
	if n < 0 {
		return -1
	}
	if buf[0]&reservedLongBits != 0 {
		// Reserved header bits must be 0.
		// https://www.rfc-editor.org/rfc/rfc9000#section-17.2-8.2.1
		c.abort(now, localTransportError{
			code:   errProtocolViolation,
			reason: "reserved header bits are not zero",
		})
		return -1
	}
	if p.version != quicVersion1 {
		// The peer has changed versions on us mid-handshake?
		c.abort(now, localTransportError{
			code:   errProtocolViolation,
			reason: "protocol version changed during handshake",
		})
		return -1
	}

	if !c.acks[space].shouldProcess(p.num) {
		return n
	}

	if logPackets {
		logInboundLongPacket(c, p)
	}
	if c.logEnabled(QLogLevelPacket) {
		c.logLongPacketReceived(p, buf[:n])
	}
	c.connIDState.handlePacket(c, p.ptype, p.srcConnID)
	ackEliciting := c.handleFrames(now, dgram, ptype, space, p.payload)
	c.acks[space].receive(now, space, p.num, ackEliciting, dgram.ecn)
	if p.ptype == packetTypeHandshake && c.side == serverSide {
		c.loss.validateClientAddress()

		// "[...] a server MUST discard Initial keys when it first successfully
		// processes a Handshake packet [...]"
		// https://www.rfc-editor.org/rfc/rfc9001#section-4.9.1-2
		c.discardKeys(now, initialSpace)
	}
	return n
}

func (c *Conn) handle1RTT(now time.Time, dgram *datagram, buf []byte) int {
	if !c.keysAppData.canRead() {
		// 1-RTT packets extend to the end of the datagram,
		// so skip the remainder of the datagram if we can't parse this.
		return len(buf)
	}

	pnumMax := c.acks[appDataSpace].largestSeen()
	p, err := parse1RTTPacket(buf, &c.keysAppData, connIDLen, pnumMax)
	if err != nil {
		// A localTransportError terminates the connection.
		// Other errors indicate an unparsable packet, but otherwise may be ignored.
		if _, ok := err.(localTransportError); ok {
			c.abort(now, err)
		}
		return -1
	}
	if buf[0]&reserved1RTTBits != 0 {
		// Reserved header bits must be 0.
		// https://www.rfc-editor.org/rfc/rfc9000#section-17.3.1-4.8.1
		c.abort(now, localTransportError{
			code:   errProtocolViolation,
			reason: "reserved header bits are not zero",
		})
		return -1
	}

	if !c.acks[appDataSpace].shouldProcess(p.num) {
		return len(buf)
	}

	if logPackets {
		logInboundShortPacket(c, p)
	}
	if c.logEnabled(QLogLevelPacket) {
		c.log1RTTPacketReceived(p, buf)
	}
	ackEliciting := c.handleFrames(now, dgram, packetType1RTT, appDataSpace, p.payload)
	c.acks[appDataSpace].receive(now, appDataSpace, p.num, ackEliciting, dgram.ecn)
	return len(buf)
}

func (c *Conn) handleRetry(now time.Time, pkt []byte) {
	if c.side != clientSide {
		return // clients don't send Retry packets
	}
	// "After the client has received and processed an Initial or Retry packet
	// from the server, it MUST discard any subsequent Retry packets that it receives."
	// https://www.rfc-editor.org/rfc/rfc9000#section-17.2.5.2-1
	if !c.keysInitial.canRead() {
		return // discarded Initial keys, connection is already established
	}
	if c.acks[initialSpace].seen.numRanges() != 0 {
		return // processed at least one packet
	}
	if c.retryToken != nil {
		return // received a Retry already
	}
	// "Clients MUST discard Retry packets that have a Retry Integrity Tag
	// that cannot be validated."
	// https://www.rfc-editor.org/rfc/rfc9000#section-17.2.5.2-2
	p, ok := parseRetryPacket(pkt, c.connIDState.originalDstConnID)
	if !ok {
		return
	}
	// "A client MUST discard a Retry packet with a zero-length Retry Token field."
	// https://www.rfc-editor.org/rfc/rfc9000#section-17.2.5.2-2
	if len(p.token) == 0 {
		return
	}
	c.retryToken = cloneBytes(p.token)
	c.connIDState.handleRetryPacket(p.srcConnID)
	c.keysInitial = initialKeys(p.srcConnID, c.side)
	// We need to resend any data we've already sent in Initial packets.
	// We must not reuse already sent packet numbers.
	c.loss.discardPackets(initialSpace, c.log, c.handleAckOrLoss)
	// TODO: Discard 0-RTT packets as well, once we support 0-RTT.
	if c.testHooks != nil {
		c.testHooks.init(false)
	}
}

var errVersionNegotiation = errors.New("server does not support QUIC version 1")

func (c *Conn) handleVersionNegotiation(now time.Time, pkt []byte) {
	if c.side != clientSide {
		return // servers don't handle Version Negotiation packets
	}
	// "A client MUST discard any Version Negotiation packet if it has
	// received and successfully processed any other packet [...]"
	// https://www.rfc-editor.org/rfc/rfc9000#section-6.2-2
	if !c.keysInitial.canRead() {
		return // discarded Initial keys, connection is already established
	}
	if c.acks[initialSpace].seen.numRanges() != 0 {
		return // processed at least one packet
	}
	_, srcConnID, versions := parseVersionNegotiation(pkt)
	if len(c.connIDState.remote) < 1 || !bytes.Equal(c.connIDState.remote[0].cid, srcConnID) {
		return // Source Connection ID doesn't match what we sent
	}
	for len(versions) >= 4 {
		ver := binary.BigEndian.Uint32(versions)
		if ver == 1 {
			// "A client MUST discard a Version Negotiation packet that lists
			// the QUIC version selected by the client."
			// https://www.rfc-editor.org/rfc/rfc9000#section-6.2-2
			return
		}
		versions = versions[4:]
	}
	// "A client that supports only this version of QUIC MUST
	// abandon the current connection attempt if it receives
	// a Version Negotiation packet, [with the two exceptions handled above]."
	// https://www.rfc-editor.org/rfc/rfc9000#section-6.2-2
	c.abortImmediately(now, errVersionNegotiation)
}

func (c *Conn) handleFrames(now time.Time, dgram *datagram, ptype packetType, space numberSpace, payload []byte) (ackEliciting bool) {
	if len(payload) == 0 {
		// "An endpoint MUST treat receipt of a packet containing no frames
		// as a connection error of type PROTOCOL_VIOLATION."
		// https://www.rfc-editor.org/rfc/rfc9000#section-12.4-3
		c.abort(now, localTransportError{
			code:   errProtocolViolation,
			reason: "packet contains no frames",
		})
		return false
	}
	// frameOK verifies that ptype is one of the packets in mask.
	frameOK := func(c *Conn, ptype, mask packetType) (ok bool) {
		if ptype&mask == 0 {
			// "An endpoint MUST treat receipt of a frame in a packet type
			// that is not permitted as a connection error of type
			// PROTOCOL_VIOLATION."
			// https://www.rfc-editor.org/rfc/rfc9000#section-12.4-3
			c.abort(now, localTransportError{
				code:   errProtocolViolation,
				reason: "frame not allowed in packet",
			})
			return false
		}
		return true
	}
	// Packet masks from RFC 9000 Table 3.
	// https://www.rfc-editor.org/rfc/rfc9000#table-3
	const (
		IH_1 = packetTypeInitial | packetTypeHandshake | packetType1RTT
		__01 = packetType0RTT | packetType1RTT
		___1 = packetType1RTT
	)
	hasCrypto := false
	for len(payload) > 0 {
		switch payload[0] {
		case frameTypePadding, frameTypeAck, frameTypeAckECN,
			frameTypeConnectionCloseTransport, frameTypeConnectionCloseApplication:
		default:
			ackEliciting = true
		}
		n := -1
		switch payload[0] {
		case frameTypePadding:
			// PADDING is OK in all spaces.
			n = 1
		case frameTypePing:
			// PING is OK in all spaces.
			//
			// A PING frame causes us to respond with an ACK by virtue of being
			// an ack-eliciting frame, but requires no other action.
			n = 1
		case frameTypeAck, frameTypeAckECN:
			if !frameOK(c, ptype, IH_1) {
				return
			}
			n = c.handleAckFrame(now, space, payload)
		case frameTypeResetStream:
			if !frameOK(c, ptype, __01) {
				return
			}
			n = c.handleResetStreamFrame(now, space, payload)
		case frameTypeStopSending:
			if !frameOK(c, ptype, __01) {
				return
			}
			n = c.handleStopSendingFrame(now, space, payload)
		case frameTypeCrypto:
			if !frameOK(c, ptype, IH_1) {
				return
			}
			hasCrypto = true
			n = c.handleCryptoFrame(now, space, payload)
		case frameTypeNewToken:
			if !frameOK(c, ptype, ___1) {
				return
			}
			_, n = consumeNewTokenFrame(payload)
		case 0x08, 0x09, 0x0a, 0x0b, 0x0c, 0x0d, 0x0e, 0x0f: // STREAM
			if !frameOK(c, ptype, __01) {
				return
			}
			n = c.handleStreamFrame(now, space, payload)
		case frameTypeMaxData:
			if !frameOK(c, ptype, __01) {
				return
			}
			n = c.handleMaxDataFrame(now, payload)
		case frameTypeMaxStreamData:
			if !frameOK(c, ptype, __01) {
				return
			}
			n = c.handleMaxStreamDataFrame(now, payload)
		case frameTypeMaxStreamsBidi, frameTypeMaxStreamsUni:
			if !frameOK(c, ptype, __01) {
				return
			}
			n = c.handleMaxStreamsFrame(now, payload)
		case frameTypeDataBlocked:
			if !frameOK(c, ptype, __01) {
				return
			}
			_, n = consumeDataBlockedFrame(payload)
		case frameTypeStreamsBlockedBidi, frameTypeStreamsBlockedUni:
			if !frameOK(c, ptype, __01) {
				return
			}
			_, _, n = consumeStreamsBlockedFrame(payload)
		case frameTypeStreamDataBlocked:
			if !frameOK(c, ptype, __01) {
				return
			}
			_, _, n = consumeStreamDataBlockedFrame(payload)
		case frameTypeNewConnectionID:
			if !frameOK(c, ptype, __01) {
				return
			}
			n = c.handleNewConnectionIDFrame(now, space, payload)
		case frameTypeRetireConnectionID:
			if !frameOK(c, ptype, __01) {
				return
			}
			n = c.handleRetireConnectionIDFrame(now, space, payload)
		case frameTypePathChallenge:
			if !frameOK(c, ptype, __01) {
				return
			}
			n = c.handlePathChallengeFrame(now, dgram, space, payload)
		case frameTypePathResponse:
			if !frameOK(c, ptype, ___1) {
				return
			}
			n = c.handlePathResponseFrame(now, space, payload)
		case frameTypeConnectionCloseTransport:
			// Transport CONNECTION_CLOSE is OK in all spaces.
			n = c.handleConnectionCloseTransportFrame(now, payload)
		case frameTypeConnectionCloseApplication:
			if !frameOK(c, ptype, __01) {
				return
			}
			n = c.handleConnectionCloseApplicationFrame(now, payload)
		case frameTypeHandshakeDone:
			if !frameOK(c, ptype, ___1) {
				return
			}
			n = c.handleHandshakeDoneFrame(now, space, payload)
		}
		if n < 0 {
			c.abort(now, localTransportError{
				code:   errFrameEncoding,
				reason: "frame encoding error",
			})
			return false
		}
		payload = payload[n:]
	}
	if hasCrypto {
		// Process TLS events after handling all frames in a packet.
		// TLS events can cause us to drop state for a number space,
		// so do that last, to avoid handling frames differently
		// depending on whether they come before or after a CRYPTO frame.
		if err := c.handleTLSEvents(now); err != nil {
			c.abort(now, err)
		}
	}
	return ackEliciting
}

func (c *Conn) handleAckFrame(now time.Time, space numberSpace, payload []byte) int {
	c.loss.receiveAckStart()
	largest, ackDelay, ecn, n := consumeAckFrame(payload, func(rangeIndex int, start, end packetNumber) {
		if err := c.loss.receiveAckRange(now, space, rangeIndex, start, end, c.handleAckOrLoss); err != nil {
			c.abort(now, err)
			return
		}
	})
	// TODO: Make use of ECN feedback.
	// https://www.rfc-editor.org/rfc/rfc9000.html#section-19.3.2
	_ = ecn
	// Prior to receiving the peer's transport parameters, we cannot
	// interpret the ACK Delay field because we don't know the ack_delay_exponent
	// to apply.
	//
	// For servers, we should always know the ack_delay_exponent because the
	// client's transport parameters are carried in its Initial packets and we
	// won't send an ack-eliciting Initial packet until after receiving the last
	// client Initial packet.
	//
	// For clients, we won't receive the server's transport parameters until handling
	// its Handshake flight, which will probably happen after reading its ACK for our
	// Initial packet(s). However, the peer's acknowledgement delay cannot reduce our
	// adjusted RTT sample below min_rtt, and min_rtt is generally going to be set
	// by the packet containing the ACK for our Initial flight. Therefore, the
	// ACK Delay for an ACK in the Initial space is likely to be ignored anyway.
	//
	// Long story short, setting the delay to 0 prior to reading transport parameters
	// is usually going to have no effect, will have only a minor effect in the rare
	// cases when it happens, and there aren't any good alternatives anyway since we
	// can't interpret the ACK Delay field without knowing the exponent.
	var delay time.Duration
	if c.peerAckDelayExponent >= 0 {
		delay = ackDelay.Duration(uint8(c.peerAckDelayExponent))
	}
	c.loss.receiveAckEnd(now, c.log, space, delay, c.handleAckOrLoss)
	if space == appDataSpace {
		c.keysAppData.handleAckFor(largest)
	}
	return n
}

func (c *Conn) handleMaxDataFrame(now time.Time, payload []byte) int {
	maxData, n := consumeMaxDataFrame(payload)
	if n < 0 {
		return -1
	}
	c.streams.outflow.setMaxData(maxData)
	return n
}

func (c *Conn) handleMaxStreamDataFrame(now time.Time, payload []byte) int {
	id, maxStreamData, n := consumeMaxStreamDataFrame(payload)
	if n < 0 {
		return -1
	}
	if s := c.streamForFrame(now, id, sendStream); s != nil {
		if err := s.handleMaxStreamData(maxStreamData); err != nil {
			c.abort(now, err)
			return -1
		}
	}
	return n
}

func (c *Conn) handleMaxStreamsFrame(now time.Time, payload []byte) int {
	styp, max, n := consumeMaxStreamsFrame(payload)
	if n < 0 {
		return -1
	}
	c.streams.localLimit[styp].setMax(max)
	return n
}

func (c *Conn) handleResetStreamFrame(now time.Time, space numberSpace, payload []byte) int {
	id, code, finalSize, n := consumeResetStreamFrame(payload)
	if n < 0 {
		return -1
	}
	if s := c.streamForFrame(now, id, recvStream); s != nil {
		if err := s.handleReset(code, finalSize); err != nil {
			c.abort(now, err)
		}
	}
	return n
}

func (c *Conn) handleStopSendingFrame(now time.Time, space numberSpace, payload []byte) int {
	id, code, n := consumeStopSendingFrame(payload)
	if n < 0 {
		return -1
	}
	if s := c.streamForFrame(now, id, sendStream); s != nil {
		if err := s.handleStopSending(code); err != nil {
			c.abort(now, err)
		}
	}
	return n
}

func (c *Conn) handleCryptoFrame(now time.Time, space numberSpace, payload []byte) int {
	off, data, n := consumeCryptoFrame(payload)
	err := c.handleCrypto(now, space, off, data)
	if err != nil {
		c.abort(now, err)
		return -1
	}
	return n
}

func (c *Conn) handleStreamFrame(now time.Time, space numberSpace, payload []byte) int {
	id, off, fin, b, n := consumeStreamFrame(payload)
	if n < 0 {
		return -1
	}
	if s := c.streamForFrame(now, id, recvStream); s != nil {
		if err := s.handleData(off, b, fin); err != nil {
			c.abort(now, err)
		}
	}
	return n
}

func (c *Conn) handleNewConnectionIDFrame(now time.Time, space numberSpace, payload []byte) int {
	seq, retire, connID, resetToken, n := consumeNewConnectionIDFrame(payload)
	if n < 0 {
		return -1
	}
	if err := c.connIDState.handleNewConnID(c, seq, retire, connID, resetToken); err != nil {
		c.abort(now, err)
	}
	return n
}

func (c *Conn) handleRetireConnectionIDFrame(now time.Time, space numberSpace, payload []byte) int {
	seq, n := consumeRetireConnectionIDFrame(payload)
	if n < 0 {
		return -1
	}
	if err := c.connIDState.handleRetireConnID(c, seq); err != nil {
		c.abort(now, err)
	}
	return n
}

func (c *Conn) handlePathChallengeFrame(now time.Time, dgram *datagram, space numberSpace, payload []byte) int {
	data, n := consumePathChallengeFrame(payload)
	if n < 0 {
		return -1
	}
	c.handlePathChallenge(now, dgram, data)
	return n
}

func (c *Conn) handlePathResponseFrame(now time.Time, space numberSpace, payload []byte) int {
	data, n := consumePathResponseFrame(payload)
	if n < 0 {
		return -1
	}
	c.handlePathResponse(now, data)
	return n
}

func (c *Conn) handleConnectionCloseTransportFrame(now time.Time, payload []byte) int {
	code, _, reason, n := consumeConnectionCloseTransportFrame(payload)
	if n < 0 {
		return -1
	}
	c.handlePeerConnectionClose(now, peerTransportError{code: code, reason: reason})
	return n
}

func (c *Conn) handleConnectionCloseApplicationFrame(now time.Time, payload []byte) int {
	code, reason, n := consumeConnectionCloseApplicationFrame(payload)
	if n < 0 {
		return -1
	}
	c.handlePeerConnectionClose(now, &ApplicationError{Code: code, Reason: reason})
	return n
}

func (c *Conn) handleHandshakeDoneFrame(now time.Time, space numberSpace, payload []byte) int {
	if c.side == serverSide {
		// Clients should never send HANDSHAKE_DONE.
		// https://www.rfc-editor.org/rfc/rfc9000#section-19.20-4
		c.abort(now, localTransportError{
			code:   errProtocolViolation,
			reason: "client sent HANDSHAKE_DONE",
		})
		return -1
	}
	if c.isAlive() {
		c.confirmHandshake(now)
	}
	return 1
}

var errStatelessReset = errors.New("received stateless reset")

func (c *Conn) handleStatelessReset(now time.Time, resetToken statelessResetToken) (valid bool) {
	if !c.connIDState.isValidStatelessResetToken(resetToken) {
		return false
	}
	c.setFinalError(errStatelessReset)
	c.enterDraining(now)
	return true
}
