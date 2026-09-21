// Copyright 2023 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package quic

import (
	"context"
	"log/slog"
	"math"
	"time"
)

type lossState struct {
	side connSide

	// True when the handshake is confirmed.
	// https://www.rfc-editor.org/rfc/rfc9001#section-4.1.2
	handshakeConfirmed bool

	// Peer's max_ack_delay transport parameter.
	// https://www.rfc-editor.org/rfc/rfc9000.html#section-18.2-4.28.1
	maxAckDelay time.Duration

	// Time of the next event: PTO expiration (if ptoTimerArmed is true),
	// or loss detection.
	// The connection must call lossState.advance when the timer expires.
	timer time.Time

	// True when the PTO timer is set.
	ptoTimerArmed bool

	// True when the PTO timer has expired and a probe packet has not yet been sent.
	ptoExpired bool

	// Count of PTO expirations since the lack received acknowledgement.
	// https://www.rfc-editor.org/rfc/rfc9002#section-6.2.1-9
	ptoBackoffCount int

	// Anti-amplification limit: Three times the amount of data received from
	// the peer, less the amount of data sent.
	//
	// Set to antiAmplificationUnlimited (MaxInt) to disable the limit.
	// The limit is always disabled for clients, and for servers after the
	// peer's address is validated.
	//
	// Anti-amplification is per-address; this will need to change if/when we
	// support address migration.
	//
	// https://www.rfc-editor.org/rfc/rfc9000#section-8-2
	antiAmplificationLimit int

	// Count of non-ack-eliciting packets (ACKs) sent since the last ack-eliciting one.
	consecutiveNonAckElicitingPackets int

	rtt   rttState
	pacer pacerState
	cc    *ccReno

	// Per-space loss detection state.
	spaces [numberSpaceCount]struct {
		sentPacketList
		maxAcked         packetNumber
		lastAckEliciting packetNumber
	}

	// Temporary state used when processing an ACK frame.
	ackFrameRTT                  time.Duration // RTT from latest packet in frame
	ackFrameContainsAckEliciting bool          // newly acks an ack-eliciting packet?
}

const antiAmplificationUnlimited = math.MaxInt

func (c *lossState) init(side connSide, maxDatagramSize int, now time.Time) {
	c.side = side
	if side == clientSide {
		// Clients don't have an anti-amplification limit.
		c.antiAmplificationLimit = antiAmplificationUnlimited
	}
	c.rtt.init()
	c.cc = newReno(maxDatagramSize)
	c.pacer.init(now, c.cc.congestionWindow, timerGranularity)

	// Peer's assumed max_ack_delay, prior to receiving transport parameters.
	// https://www.rfc-editor.org/rfc/rfc9000#section-18.2
	c.maxAckDelay = 25 * time.Millisecond

	for space := range c.spaces {
		c.spaces[space].maxAcked = -1
		c.spaces[space].lastAckEliciting = -1
	}
}

// setMaxAckDelay sets the max_ack_delay transport parameter received from the peer.
func (c *lossState) setMaxAckDelay(d time.Duration) {
	if d >= (1<<14)*time.Millisecond {
		// Values of 2^14 or greater are invalid.
		// https://www.rfc-editor.org/rfc/rfc9000.html#section-18.2-4.28.1
		return
	}
	c.maxAckDelay = d
}

// confirmHandshake indicates the handshake has been confirmed.
func (c *lossState) confirmHandshake() {
	c.handshakeConfirmed = true
}

// validateClientAddress disables the anti-amplification limit after
// a server validates a client's address.
func (c *lossState) validateClientAddress() {
	c.antiAmplificationLimit = antiAmplificationUnlimited
}

// minDatagramSize is the minimum datagram size permitted by
// anti-amplification protection.
//
// Defining a minimum size avoids the case where, say, anti-amplification
// technically allows us to send a 1-byte datagram, but no such datagram
// can be constructed.
const minPacketSize = 128

type ccLimit int

const (
	ccOK      = ccLimit(iota) // OK to send
	ccBlocked                 // sending blocked by anti-amplification
	ccLimited                 // sending blocked by congestion control
	ccPaced                   // sending allowed by congestion, but delayed by pacer
)

// sendLimit reports whether sending is possible at this time.
// When sending is pacing limited, it returns the next time a packet may be sent.
func (c *lossState) sendLimit(now time.Time) (limit ccLimit, next time.Time) {
	if c.antiAmplificationLimit < minPacketSize {
		// When at the anti-amplification limit, we may not send anything.
		return ccBlocked, time.Time{}
	}
	if c.ptoExpired {
		// On PTO expiry, send a probe.
		return ccOK, time.Time{}
	}
	if !c.cc.canSend() {
		// Congestion control blocks sending.
		return ccLimited, time.Time{}
	}
	if c.cc.bytesInFlight == 0 {
		// If no bytes are in flight, send packet unpaced.
		return ccOK, time.Time{}
	}
	canSend, next := c.pacer.canSend(now)
	if !canSend {
		// Pacer blocks sending.
		return ccPaced, next
	}
	return ccOK, time.Time{}
}

// maxSendSize reports the maximum datagram size that may be sent.
func (c *lossState) maxSendSize() int {
	return min(c.antiAmplificationLimit, c.cc.maxDatagramSize)
}

// advance is called when time passes.
// The lossf function is called for each packet newly detected as lost.
func (c *lossState) advance(now time.Time, lossf func(numberSpace, *sentPacket, packetFate)) {
	c.pacer.advance(now, c.cc.congestionWindow, c.rtt.smoothedRTT)
	if c.ptoTimerArmed && !c.timer.IsZero() && !c.timer.After(now) {
		c.ptoExpired = true
		c.timer = time.Time{}
		c.ptoBackoffCount++
	}
	c.detectLoss(now, lossf)
}

// nextNumber returns the next packet number to use in a space.
func (c *lossState) nextNumber(space numberSpace) packetNumber {
	return c.spaces[space].nextNum
}

// skipNumber skips a packet number as a defense against optimistic ACK attacks.
func (c *lossState) skipNumber(now time.Time, space numberSpace) {
	sent := newSentPacket()
	sent.num = c.spaces[space].nextNum
	sent.time = now
	sent.state = sentPacketUnsent
	c.spaces[space].add(sent)
}

// packetSent records a sent packet.
func (c *lossState) packetSent(now time.Time, log *slog.Logger, space numberSpace, sent *sentPacket) {
	sent.time = now
	c.spaces[space].add(sent)
	size := sent.size
	if c.antiAmplificationLimit != antiAmplificationUnlimited {
		c.antiAmplificationLimit = max(0, c.antiAmplificationLimit-size)
	}
	if sent.inFlight {
		c.cc.packetSent(now, log, space, sent)
		c.pacer.packetSent(now, size, c.cc.congestionWindow, c.rtt.smoothedRTT)
		if sent.ackEliciting {
			c.spaces[space].lastAckEliciting = sent.num
			c.ptoExpired = false // reset expired PTO timer after sending probe
		}
		c.scheduleTimer(now)
		if logEnabled(log, QLogLevelPacket) {
			logBytesInFlight(log, c.cc.bytesInFlight)
		}
	}
	if sent.ackEliciting {
		c.consecutiveNonAckElicitingPackets = 0
	} else {
		c.consecutiveNonAckElicitingPackets++
	}
}

// datagramReceived records a datagram (not packet!) received from the peer.
func (c *lossState) datagramReceived(now time.Time, size int) {
	if c.antiAmplificationLimit != antiAmplificationUnlimited {
		c.antiAmplificationLimit += 3 * size
		// Reset the PTO timer, possibly to a point in the past, in which
		// case the caller should execute it immediately.
		// https://www.rfc-editor.org/rfc/rfc9002.html#section-6.2.2.1-2
		c.scheduleTimer(now)
		if c.ptoTimerArmed && !c.timer.IsZero() && !c.timer.After(now) {
			c.ptoExpired = true
			c.timer = time.Time{}
		}
	}
}

// receiveAckStart starts processing an ACK frame.
// Call receiveAckRange for each range in the frame.
// Call receiveAckFrameEnd after all ranges are processed.
func (c *lossState) receiveAckStart() {
	c.ackFrameContainsAckEliciting = false
	c.ackFrameRTT = -1
}

// receiveAckRange processes a range within an ACK frame.
// The ackf function is called for each newly-acknowledged packet.
func (c *lossState) receiveAckRange(now time.Time, space numberSpace, rangeIndex int, start, end packetNumber, ackf func(numberSpace, *sentPacket, packetFate)) error {
	// Limit our range to the intersection of the ACK range and
	// the in-flight packets we have state for.
	if s := c.spaces[space].start(); start < s {
		start = s
	}
	if e := c.spaces[space].end(); end > e {
		return localTransportError{
			code:   errProtocolViolation,
			reason: "acknowledgement for unsent packet",
		}
	}
	if start >= end {
		return nil
	}
	if rangeIndex == 0 {
		// If the latest packet in the ACK frame is newly-acked,
		// record the RTT in c.ackFrameRTT.
		sent := c.spaces[space].num(end - 1)
		if sent.state == sentPacketSent {
			c.ackFrameRTT = max(0, now.Sub(sent.time))
		}
	}
	for pnum := start; pnum < end; pnum++ {
		sent := c.spaces[space].num(pnum)
		if sent.state == sentPacketUnsent {
			return localTransportError{
				code:   errProtocolViolation,
				reason: "acknowledgement for unsent packet",
			}
		}
		if sent.state != sentPacketSent {
			continue
		}
		// This is a newly-acknowledged packet.
		if pnum > c.spaces[space].maxAcked {
			c.spaces[space].maxAcked = pnum
		}
		sent.state = sentPacketAcked
		c.cc.packetAcked(now, sent)
		ackf(space, sent, packetAcked)
		if sent.ackEliciting {
			c.ackFrameContainsAckEliciting = true
		}
	}
	return nil
}

// receiveAckEnd finishes processing an ack frame.
// The lossf function is called for each packet newly detected as lost.
func (c *lossState) receiveAckEnd(now time.Time, log *slog.Logger, space numberSpace, ackDelay time.Duration, lossf func(numberSpace, *sentPacket, packetFate)) {
	c.spaces[space].sentPacketList.clean()
	// Update the RTT sample when the largest acknowledged packet in the ACK frame
	// is newly acknowledged, and at least one newly acknowledged packet is ack-eliciting.
	// https://www.rfc-editor.org/rfc/rfc9002.html#section-5.1-2.2
	if c.ackFrameRTT >= 0 && c.ackFrameContainsAckEliciting {
		c.rtt.updateSample(now, c.handshakeConfirmed, space, c.ackFrameRTT, ackDelay, c.maxAckDelay)
	}
	// Reset the PTO backoff.
	// Exception: A client does not reset the backoff on acks for Initial packets.
	// https://www.rfc-editor.org/rfc/rfc9002.html#section-6.2.1-9
	if !(c.side == clientSide && space == initialSpace) {
		c.ptoBackoffCount = 0
	}
	// If the client has set a PTO timer with no packets in flight
	// we want to restart that timer now. Clearing c.timer does this.
	// https://www.rfc-editor.org/rfc/rfc9002.html#section-6.2.2.1-3
	c.timer = time.Time{}
	c.detectLoss(now, lossf)
	c.cc.packetBatchEnd(now, log, space, &c.rtt, c.maxAckDelay)

	if logEnabled(log, QLogLevelPacket) {
		var ssthresh slog.Attr
		if c.cc.slowStartThreshold != math.MaxInt {
			ssthresh = slog.Int("ssthresh", c.cc.slowStartThreshold)
		}
		log.LogAttrs(context.Background(), QLogLevelPacket,
			"recovery:metrics_updated",
			slog.Duration("min_rtt", c.rtt.minRTT),
			slog.Duration("smoothed_rtt", c.rtt.smoothedRTT),
			slog.Duration("latest_rtt", c.rtt.latestRTT),
			slog.Duration("rtt_variance", c.rtt.rttvar),
			slog.Int("congestion_window", c.cc.congestionWindow),
			slog.Int("bytes_in_flight", c.cc.bytesInFlight),
			ssthresh,
		)
	}
}

// discardPackets declares that packets within a number space will not be delivered
// and that data contained in them should be resent.
// For example, after receiving a Retry packet we discard already-sent Initial packets.
func (c *lossState) discardPackets(space numberSpace, log *slog.Logger, lossf func(numberSpace, *sentPacket, packetFate)) {
	for i := 0; i < c.spaces[space].size; i++ {
		sent := c.spaces[space].nth(i)
		if sent.state != sentPacketSent {
			// This should not be possible, since we only discard packets
			// in spaces which have never received an ack, but check anyway.
			continue
		}
		sent.state = sentPacketLost
		c.cc.packetDiscarded(sent)
		lossf(numberSpace(space), sent, packetLost)
	}
	c.spaces[space].clean()
	if logEnabled(log, QLogLevelPacket) {
		logBytesInFlight(log, c.cc.bytesInFlight)
	}
}

// discardKeys is called when dropping packet protection keys for a number space.
func (c *lossState) discardKeys(now time.Time, log *slog.Logger, space numberSpace) {
	// https://www.rfc-editor.org/rfc/rfc9002.html#section-6.4
	for i := 0; i < c.spaces[space].size; i++ {
		sent := c.spaces[space].nth(i)
		if sent.state != sentPacketSent {
			continue
		}
		c.cc.packetDiscarded(sent)
	}
	c.spaces[space].discard()
	c.spaces[space].maxAcked = -1
	c.spaces[space].lastAckEliciting = -1
	c.scheduleTimer(now)
	if logEnabled(log, QLogLevelPacket) {
		logBytesInFlight(log, c.cc.bytesInFlight)
	}
}

func (c *lossState) lossDuration() time.Duration {
	// https://www.rfc-editor.org/rfc/rfc9002.html#section-6.1.2
	return max((9*max(c.rtt.smoothedRTT, c.rtt.latestRTT))/8, timerGranularity)
}

func (c *lossState) detectLoss(now time.Time, lossf func(numberSpace, *sentPacket, packetFate)) {
	// https://www.rfc-editor.org/rfc/rfc9002.html#section-6.1.1-1
	const lossThreshold = 3

	lossTime := now.Add(-c.lossDuration())
	for space := numberSpace(0); space < numberSpaceCount; space++ {
		for i := 0; i < c.spaces[space].size; i++ {
			sent := c.spaces[space].nth(i)
			if sent.state != sentPacketSent {
				continue
			}
			// RFC 9002 Section 6.1 states that a packet is only declared lost if it
			// is "in flight", which excludes packets that contain only ACK frames.
			// However, we need some way to determine when to drop state for ACK-only
			// packets, and the loss algorithm in Appendix A handles loss detection of
			// not-in-flight packets identically to all others, so we do the same here.
			switch {
			case c.spaces[space].maxAcked-sent.num >= lossThreshold:
				// Packet threshold
				// https://www.rfc-editor.org/rfc/rfc9002.html#section-6.1.1
				fallthrough
			case sent.num <= c.spaces[space].maxAcked && !sent.time.After(lossTime):
				// Time threshold
				// https://www.rfc-editor.org/rfc/rfc9002.html#section-6.1.2
				sent.state = sentPacketLost
				lossf(space, sent, packetLost)
				if sent.inFlight {
					c.cc.packetLost(now, space, sent, &c.rtt)
				}
			}
			if sent.state != sentPacketLost {
				break
			}
		}
		c.spaces[space].clean()
	}
	c.scheduleTimer(now)
}

// scheduleTimer sets the loss or PTO timer.
//
// The connection is responsible for arranging for advance to be called after
// the timer expires.
//
// The timer may be set to a point in the past, in which advance should be called
// immediately. We don't do this here, because executing the timer can cause
// packet loss events, and it's simpler for the connection if loss events only
// occur when advancing time.
func (c *lossState) scheduleTimer(now time.Time) {
	c.ptoTimerArmed = false

	// Loss timer for sent packets.
	// The loss timer is only started once a later packet has been acknowledged,
	// and takes precedence over the PTO timer.
	// https://www.rfc-editor.org/rfc/rfc9002.html#section-6.1.2
	var oldestPotentiallyLost time.Time
	for space := numberSpace(0); space < numberSpaceCount; space++ {
		if c.spaces[space].size > 0 && c.spaces[space].start() <= c.spaces[space].maxAcked {
			firstTime := c.spaces[space].nth(0).time
			if oldestPotentiallyLost.IsZero() || firstTime.Before(oldestPotentiallyLost) {
				oldestPotentiallyLost = firstTime
			}
		}
	}
	if !oldestPotentiallyLost.IsZero() {
		c.timer = oldestPotentiallyLost.Add(c.lossDuration())
		return
	}

	// PTO timer.
	if c.ptoExpired {
		// PTO timer has expired, don't restart it until we send a probe.
		c.timer = time.Time{}
		return
	}
	if c.antiAmplificationLimit >= 0 && c.antiAmplificationLimit < minPacketSize {
		// Server is at its anti-amplification limit and can't send any more data.
		// https://www.rfc-editor.org/rfc/rfc9002.html#section-6.2.2.1-1
		c.timer = time.Time{}
		return
	}
	// Timer starts at the most recently sent ack-eliciting packet.
	// Prior to confirming the handshake, we consider the Initial and Handshake
	// number spaces; after, we consider only Application Data.
	var last time.Time
	if !c.handshakeConfirmed {
		for space := initialSpace; space <= handshakeSpace; space++ {
			sent := c.spaces[space].num(c.spaces[space].lastAckEliciting)
			if sent == nil {
				continue
			}
			if last.IsZero() || last.After(sent.time) {
				last = sent.time
			}
		}
	} else {
		sent := c.spaces[appDataSpace].num(c.spaces[appDataSpace].lastAckEliciting)
		if sent != nil {
			last = sent.time
		}
	}
	if last.IsZero() &&
		c.side == clientSide &&
		c.spaces[handshakeSpace].maxAcked < 0 &&
		!c.handshakeConfirmed {
		// The client must always set a PTO timer prior to receiving an ack for a
		// handshake packet or the handshake being confirmed.
		// https://www.rfc-editor.org/rfc/rfc9002.html#section-6.2.2.1
		if !c.timer.IsZero() {
			// If c.timer is non-zero here, we've already set the PTO timer and
			// should leave it as-is rather than moving it forward.
			c.ptoTimerArmed = true
			return
		}
		last = now
	} else if last.IsZero() {
		c.timer = time.Time{}
		return
	}
	c.timer = last.Add(c.ptoPeriod())
	c.ptoTimerArmed = true
}

func (c *lossState) ptoPeriod() time.Duration {
	// https://www.rfc-editor.org/rfc/rfc9002.html#section-6.2.1
	return c.ptoBasePeriod() << c.ptoBackoffCount
}

func (c *lossState) ptoBasePeriod() time.Duration {
	// https://www.rfc-editor.org/rfc/rfc9002.html#section-6.2.1
	pto := c.rtt.smoothedRTT + max(4*c.rtt.rttvar, timerGranularity)
	if c.handshakeConfirmed {
		// The max_ack_delay is the maximum amount of time the peer might delay sending
		// an ack to us. We only take it into account for the Application Data space.
		// https://www.rfc-editor.org/rfc/rfc9002.html#section-6.2.1-4
		pto += c.maxAckDelay
	}
	return pto
}

func logBytesInFlight(log *slog.Logger, bytesInFlight int) {
	log.LogAttrs(context.Background(), QLogLevelPacket,
		"recovery:metrics_updated",
		slog.Int("bytes_in_flight", bytesInFlight),
	)
}
