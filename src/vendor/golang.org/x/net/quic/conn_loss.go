// Copyright 2023 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package quic

import "fmt"

// handleAckOrLoss deals with the final fate of a packet we sent:
// Either the peer acknowledges it, or we declare it lost.
//
// In order to handle packet loss, we must retain any information sent to the peer
// until the peer has acknowledged it.
//
// When information is acknowledged, we can discard it.
//
// When information is lost, we mark it for retransmission.
// See RFC 9000, Section 13.3 for a complete list of information which is retransmitted on loss.
// https://www.rfc-editor.org/rfc/rfc9000#section-13.3
func (c *Conn) handleAckOrLoss(space numberSpace, sent *sentPacket, fate packetFate) {
	if fate == packetLost && c.logEnabled(QLogLevelPacket) {
		c.logPacketLost(space, sent)
	}

	// The list of frames in a sent packet is marshaled into a buffer in the sentPacket
	// by the packetWriter. Unmarshal that buffer here. This code must be kept in sync with
	// packetWriter.append*.
	//
	// A sent packet meets its fate (acked or lost) only once, so it's okay to consume
	// the sentPacket's buffer here.
	for !sent.done() {
		switch f := sent.next(); f {
		default:
			panic(fmt.Sprintf("BUG: unhandled acked/lost frame type %x", f))
		case frameTypeAck, frameTypeAckECN:
			// Unlike most information, loss of an ACK frame does not trigger
			// retransmission. ACKs are sent in response to ack-eliciting packets,
			// and always contain the latest information available.
			//
			// Acknowledgement of an ACK frame may allow us to discard information
			// about older packets.
			largest := packetNumber(sent.nextInt())
			if fate == packetAcked {
				c.acks[space].handleAck(largest)
			}
		case frameTypeCrypto:
			start, end := sent.nextRange()
			c.crypto[space].ackOrLoss(start, end, fate)
		case frameTypeMaxData:
			c.ackOrLossMaxData(sent.num, fate)
		case frameTypeResetStream,
			frameTypeStopSending,
			frameTypeMaxStreamData,
			frameTypeStreamDataBlocked:
			id := streamID(sent.nextInt())
			s := c.streamForID(id)
			if s == nil {
				continue
			}
			s.ackOrLoss(sent.num, f, fate)
		case frameTypeStreamBase,
			frameTypeStreamBase | streamFinBit:
			id := streamID(sent.nextInt())
			start, end := sent.nextRange()
			s := c.streamForID(id)
			if s == nil {
				continue
			}
			fin := f&streamFinBit != 0
			s.ackOrLossData(sent.num, start, end, fin, fate)
		case frameTypeMaxStreamsBidi:
			c.streams.remoteLimit[bidiStream].sendMax.ackLatestOrLoss(sent.num, fate)
		case frameTypeMaxStreamsUni:
			c.streams.remoteLimit[uniStream].sendMax.ackLatestOrLoss(sent.num, fate)
		case frameTypeNewConnectionID:
			seq := int64(sent.nextInt())
			c.connIDState.ackOrLossNewConnectionID(sent.num, seq, fate)
		case frameTypeRetireConnectionID:
			seq := int64(sent.nextInt())
			c.connIDState.ackOrLossRetireConnectionID(sent.num, seq, fate)
		case frameTypeHandshakeDone:
			c.handshakeConfirmed.ackOrLoss(sent.num, fate)
		}
	}
}
