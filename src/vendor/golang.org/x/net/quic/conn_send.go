// Copyright 2023 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package quic

import (
	"crypto/tls"
	"errors"
	"time"
)

// maybeSend sends datagrams, if possible.
//
// If sending is blocked by pacing, it returns the next time
// a datagram may be sent.
//
// If sending is blocked indefinitely, it returns the zero Time.
func (c *Conn) maybeSend(now time.Time) (next time.Time) {
	// Assumption: The congestion window is not underutilized.
	// If congestion control, pacing, and anti-amplification all permit sending,
	// but we have no packet to send, then we will declare the window underutilized.
	underutilized := false
	defer func() {
		c.loss.cc.setUnderutilized(c.log, underutilized)
	}()

	// Send one datagram on each iteration of this loop,
	// until we hit a limit or run out of data to send.
	//
	// For each number space where we have write keys,
	// attempt to construct a packet in that space.
	// If the packet contains no frames (we have no data in need of sending),
	// abandon the packet.
	//
	// Speculatively constructing packets means we don't need
	// separate code paths for "do we have data to send?" and
	// "send the data" that need to be kept in sync.
	for {
		limit, next := c.loss.sendLimit(now)
		if limit == ccBlocked {
			// If anti-amplification blocks sending, then no packet can be sent.
			return next
		}
		if !c.sendOK(now) {
			return time.Time{}
		}
		// We may still send ACKs, even if congestion control or pacing limit sending.

		// Prepare to write a datagram of at most maxSendSize bytes.
		c.w.reset(c.loss.maxSendSize())

		dstConnID, ok := c.connIDState.dstConnID()
		if !ok {
			// It is currently not possible for us to end up without a connection ID,
			// but handle the case anyway.
			return time.Time{}
		}

		// Initial packet.
		pad := false
		var sentInitial *sentPacket
		if c.keysInitial.canWrite() {
			pnumMaxAcked := c.loss.spaces[initialSpace].maxAcked
			pnum := c.loss.nextNumber(initialSpace)
			p := longPacket{
				ptype:     packetTypeInitial,
				version:   quicVersion1,
				num:       pnum,
				dstConnID: dstConnID,
				srcConnID: c.connIDState.srcConnID(),
				extra:     c.retryToken,
			}
			c.w.startProtectedLongHeaderPacket(pnumMaxAcked, p)
			c.appendFrames(now, initialSpace, pnum, limit)
			if logPackets {
				logSentPacket(c, packetTypeInitial, pnum, p.srcConnID, p.dstConnID, c.w.payload())
			}
			if c.logEnabled(QLogLevelPacket) && len(c.w.payload()) > 0 {
				c.logPacketSent(packetTypeInitial, pnum, p.srcConnID, p.dstConnID, c.w.packetLen(), c.w.payload())
			}
			sentInitial = c.w.finishProtectedLongHeaderPacket(pnumMaxAcked, c.keysInitial.w, p)
			if sentInitial != nil {
				// Client initial packets and ack-eliciting server initial packaets
				// need to be sent in a datagram padded to at least 1200 bytes.
				// We can't add the padding yet, however, since we may want to
				// coalesce additional packets with this one.
				if c.side == clientSide || sentInitial.ackEliciting {
					pad = true
				}
			}
		}

		// Handshake packet.
		if c.keysHandshake.canWrite() {
			pnumMaxAcked := c.loss.spaces[handshakeSpace].maxAcked
			pnum := c.loss.nextNumber(handshakeSpace)
			p := longPacket{
				ptype:     packetTypeHandshake,
				version:   quicVersion1,
				num:       pnum,
				dstConnID: dstConnID,
				srcConnID: c.connIDState.srcConnID(),
			}
			c.w.startProtectedLongHeaderPacket(pnumMaxAcked, p)
			c.appendFrames(now, handshakeSpace, pnum, limit)
			if logPackets {
				logSentPacket(c, packetTypeHandshake, pnum, p.srcConnID, p.dstConnID, c.w.payload())
			}
			if c.logEnabled(QLogLevelPacket) && len(c.w.payload()) > 0 {
				c.logPacketSent(packetTypeHandshake, pnum, p.srcConnID, p.dstConnID, c.w.packetLen(), c.w.payload())
			}
			if sent := c.w.finishProtectedLongHeaderPacket(pnumMaxAcked, c.keysHandshake.w, p); sent != nil {
				c.packetSent(now, handshakeSpace, sent)
				if c.side == clientSide {
					// "[...] a client MUST discard Initial keys when it first
					// sends a Handshake packet [...]"
					// https://www.rfc-editor.org/rfc/rfc9001.html#section-4.9.1-2
					c.discardKeys(now, initialSpace)
				}
			}
		}

		// 1-RTT packet.
		if c.keysAppData.canWrite() {
			pnumMaxAcked := c.loss.spaces[appDataSpace].maxAcked
			pnum := c.loss.nextNumber(appDataSpace)
			c.w.start1RTTPacket(pnum, pnumMaxAcked, dstConnID)
			c.appendFrames(now, appDataSpace, pnum, limit)
			if pad && len(c.w.payload()) > 0 {
				// 1-RTT packets have no length field and extend to the end
				// of the datagram, so if we're sending a datagram that needs
				// padding we need to add it inside the 1-RTT packet.
				c.w.appendPaddingTo(paddedInitialDatagramSize)
				pad = false
			}
			if logPackets {
				logSentPacket(c, packetType1RTT, pnum, nil, dstConnID, c.w.payload())
			}
			if c.logEnabled(QLogLevelPacket) && len(c.w.payload()) > 0 {
				c.logPacketSent(packetType1RTT, pnum, nil, dstConnID, c.w.packetLen(), c.w.payload())
			}
			if sent := c.w.finish1RTTPacket(pnum, pnumMaxAcked, dstConnID, &c.keysAppData); sent != nil {
				c.packetSent(now, appDataSpace, sent)
				if c.skip.shouldSkip(pnum + 1) {
					c.loss.skipNumber(now, appDataSpace)
					c.skip.updateNumberSkip(c)
				}
			}
		}

		buf := c.w.datagram()
		if len(buf) == 0 {
			if limit == ccOK {
				// We have nothing to send, and congestion control does not
				// block sending. The congestion window is underutilized.
				underutilized = true
			}
			return next
		}

		if sentInitial != nil {
			if pad {
				// Pad out the datagram with zeros, coalescing the Initial
				// packet with invalid packets that will be ignored by the peer.
				// https://www.rfc-editor.org/rfc/rfc9000.html#section-14.1-1
				for len(buf) < paddedInitialDatagramSize {
					buf = append(buf, 0)
					// Technically this padding isn't in any packet, but
					// account it to the Initial packet in this datagram
					// for purposes of flow control and loss recovery.
					sentInitial.size++
					sentInitial.inFlight = true
				}
			}
			// If we're a client and this Initial packet is coalesced
			// with a Handshake packet, then we've discarded Initial keys
			// since constructing the packet and shouldn't record it as in-flight.
			if c.keysInitial.canWrite() {
				c.packetSent(now, initialSpace, sentInitial)
			}
		}

		c.endpoint.sendDatagram(datagram{
			b:        buf,
			peerAddr: c.peerAddr,
		})
	}
}

func (c *Conn) packetSent(now time.Time, space numberSpace, sent *sentPacket) {
	c.idleHandlePacketSent(now, sent)
	c.loss.packetSent(now, c.log, space, sent)
}

func (c *Conn) appendFrames(now time.Time, space numberSpace, pnum packetNumber, limit ccLimit) {
	if c.lifetime.localErr != nil {
		c.appendConnectionCloseFrame(now, space, c.lifetime.localErr)
		return
	}

	shouldSendAck := c.acks[space].shouldSendAck(now)
	if limit != ccOK {
		// ACKs are not limited by congestion control.
		if shouldSendAck && c.appendAckFrame(now, space) {
			c.acks[space].sentAck()
		}
		return
	}
	// We want to send an ACK frame if the ack controller wants to send a frame now,
	// OR if we are sending a packet anyway and have ack-eliciting packets which we
	// have not yet acked.
	//
	// We speculatively add ACK frames here, to put them at the front of the packet
	// to avoid truncation.
	//
	// After adding all frames, if we don't need to send an ACK frame and have not
	// added any other frames, we abandon the packet.
	if c.appendAckFrame(now, space) {
		defer func() {
			// All frames other than ACK and PADDING are ack-eliciting,
			// so if the packet is ack-eliciting we've added additional
			// frames to it.
			if !shouldSendAck && !c.w.sent.ackEliciting {
				// There's nothing in this packet but ACK frames, and
				// we don't want to send an ACK-only packet at this time.
				// Abandoning the packet means we wrote an ACK frame for
				// nothing, but constructing the frame is cheap.
				c.w.abandonPacket()
				return
			}
			// Either we are willing to send an ACK-only packet,
			// or we've added additional frames.
			c.acks[space].sentAck()
			if !c.w.sent.ackEliciting && c.shouldMakePacketAckEliciting() {
				c.w.appendPingFrame()
			}
		}()
	}
	if limit != ccOK {
		return
	}
	pto := c.loss.ptoExpired

	// TODO: Add all the other frames we can send.

	// CRYPTO
	c.crypto[space].dataToSend(pto, func(off, size int64) int64 {
		b, _ := c.w.appendCryptoFrame(off, int(size))
		c.crypto[space].sendData(off, b)
		return int64(len(b))
	})

	// Test-only PING frames.
	if space == c.testSendPingSpace && c.testSendPing.shouldSendPTO(pto) {
		if !c.w.appendPingFrame() {
			return
		}
		c.testSendPing.setSent(pnum)
	}

	if space == appDataSpace {
		// HANDSHAKE_DONE
		if c.handshakeConfirmed.shouldSendPTO(pto) {
			if !c.w.appendHandshakeDoneFrame() {
				return
			}
			c.handshakeConfirmed.setSent(pnum)
		}

		// NEW_CONNECTION_ID, RETIRE_CONNECTION_ID
		if !c.connIDState.appendFrames(c, pnum, pto) {
			return
		}

		// PATH_RESPONSE
		if pad, ok := c.appendPathFrames(); !ok {
			return
		} else if pad {
			defer c.w.appendPaddingTo(smallestMaxDatagramSize)
		}

		// All stream-related frames. This should come last in the packet,
		// so large amounts of STREAM data don't crowd out other frames
		// we may need to send.
		if !c.appendStreamFrames(&c.w, pnum, pto) {
			return
		}

		if !c.appendKeepAlive(now) {
			return
		}
	}

	// If this is a PTO probe and we haven't added an ack-eliciting frame yet,
	// add a PING to make this an ack-eliciting probe.
	//
	// Technically, there are separate PTO timers for each number space.
	// When a PTO timer expires, we MUST send an ack-eliciting packet in the
	// timer's space. We SHOULD send ack-eliciting packets in every other space
	// with in-flight data. (RFC 9002, section 6.2.4)
	//
	// What we actually do is send a single datagram containing an ack-eliciting packet
	// for every space for which we have keys.
	//
	// We fill the PTO probe packets with new or unacknowledged data. For example,
	// a PTO probe sent for the Initial space will generally retransmit previously
	// sent but unacknowledged CRYPTO data.
	//
	// When sending a PTO probe datagram containing multiple packets, it is
	// possible that an earlier packet will fill up the datagram, leaving no
	// space for the remaining probe packet(s). This is not a problem in practice.
	//
	// A client discards Initial keys when it first sends a Handshake packet
	// (RFC 9001 Section 4.9.1). Handshake keys are discarded when the handshake
	// is confirmed (RFC 9001 Section  4.9.2). The PTO timer is not set for the
	// Application Data packet number space until the handshake is confirmed
	// (RFC 9002 Section 6.2.1). Therefore, the only times a PTO probe can fire
	// while data for multiple spaces is in flight are:
	//
	// - a server's Initial or Handshake timers can fire while Initial and Handshake
	//   data is in flight; and
	//
	// - a client's Handshake timer can fire while Handshake and Application Data
	//   data is in flight.
	//
	// It is theoretically possible for a server's Initial CRYPTO data to overflow
	// the maximum datagram size, but unlikely in practice; this space contains
	// only the ServerHello TLS message, which is small. It's also unlikely that
	// the Handshake PTO probe will fire while Initial data is in flight (this
	// requires not just that the Initial CRYPTO data completely fill a datagram,
	// but a quite specific arrangement of lost and retransmitted packets.)
	// We don't bother worrying about this case here, since the worst case is
	// that we send a PTO probe for the in-flight Initial data and drop the
	// Handshake probe.
	//
	// If a client's Handshake PTO timer fires while Application Data data is in
	// flight, it is possible that the resent Handshake CRYPTO data will crowd
	// out the probe for the Application Data space. However, since this probe is
	// optional (recall that the Application Data PTO timer is never set until
	// after Handshake keys have been discarded), dropping it is acceptable.
	if pto && !c.w.sent.ackEliciting {
		c.w.appendPingFrame()
	}
}

// shouldMakePacketAckEliciting is called when sending a packet containing nothing but an ACK frame.
// It reports whether we should add a PING frame to the packet to make it ack-eliciting.
func (c *Conn) shouldMakePacketAckEliciting() bool {
	if c.keysAppData.needAckEliciting() {
		// The peer has initiated a key update.
		// We haven't sent them any packets yet in the new phase.
		// Make this an ack-eliciting packet.
		// Their ack of this packet will complete the key update.
		return true
	}
	if c.loss.consecutiveNonAckElicitingPackets >= 19 {
		// We've sent a run of non-ack-eliciting packets.
		// Add in an ack-eliciting one every once in a while so the peer
		// lets us know which ones have arrived.
		//
		// Google QUICHE injects a PING after sending 19 packets. We do the same.
		//
		// https://www.rfc-editor.org/rfc/rfc9000#section-13.2.4-2
		return true
	}
	// TODO: Consider making every packet sent when in PTO ack-eliciting to speed up recovery.
	return false
}

func (c *Conn) appendAckFrame(now time.Time, space numberSpace) bool {
	seen, delay := c.acks[space].acksToSend(now)
	if len(seen) == 0 {
		return false
	}
	d := unscaledAckDelayFromDuration(delay, ackDelayExponent)
	return c.w.appendAckFrame(seen, d, c.acks[space].ecn)
}

func (c *Conn) appendConnectionCloseFrame(now time.Time, space numberSpace, err error) {
	c.sentConnectionClose(now)
	switch e := err.(type) {
	case localTransportError:
		c.w.appendConnectionCloseTransportFrame(e.code, 0, e.reason)
	case *ApplicationError:
		if space != appDataSpace {
			// "CONNECTION_CLOSE frames signaling application errors (type 0x1d)
			// MUST only appear in the application data packet number space."
			// https://www.rfc-editor.org/rfc/rfc9000#section-12.5-2.2
			c.w.appendConnectionCloseTransportFrame(errApplicationError, 0, "")
		} else {
			c.w.appendConnectionCloseApplicationFrame(e.Code, e.Reason)
		}
	default:
		// TLS alerts are sent using error codes [0x0100,0x01ff).
		// https://www.rfc-editor.org/rfc/rfc9000#section-20.1-2.36.1
		var alert tls.AlertError
		switch {
		case errors.As(err, &alert):
			// tls.AlertError is a uint8, so this can't exceed 0x01ff.
			code := errTLSBase + transportError(alert)
			c.w.appendConnectionCloseTransportFrame(code, 0, "")
		default:
			c.w.appendConnectionCloseTransportFrame(errInternal, 0, "")
		}
	}
}
