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

// ccReno is the NewReno-based congestion controller defined in RFC 9002.
// https://www.rfc-editor.org/rfc/rfc9002.html#section-7
type ccReno struct {
	maxDatagramSize int

	// Maximum number of bytes allowed to be in flight.
	congestionWindow int

	// Sum of size of all packets that contain at least one ack-eliciting
	// or PADDING frame (i.e., any non-ACK frame), and have neither been
	// acknowledged nor declared lost.
	bytesInFlight int

	// When the congestion window is below the slow start threshold,
	// the controller is in slow start.
	slowStartThreshold int

	// The time the current recovery period started, or zero when not
	// in a recovery period.
	recoveryStartTime time.Time

	// Accumulated count of bytes acknowledged in congestion avoidance.
	congestionPendingAcks int

	// When entering a recovery period, we are allowed to send one packet
	// before reducing the congestion window. sendOnePacketInRecovery is
	// true if we haven't sent that packet yet.
	sendOnePacketInRecovery bool

	// inRecovery is set when we are in the recovery state.
	inRecovery bool

	// underutilized is set if the congestion window is underutilized
	// due to insufficient application data, flow control limits, or
	// anti-amplification limits.
	underutilized bool

	// ackLastLoss is the sent time of the newest lost packet processed
	// in the current batch.
	ackLastLoss time.Time

	// Data tracking the duration of the most recently handled sequence of
	// contiguous lost packets. If this exceeds the persistent congestion duration,
	// persistent congestion is declared.
	//
	// https://www.rfc-editor.org/rfc/rfc9002#section-7.6
	persistentCongestion [numberSpaceCount]struct {
		start time.Time    // send time of first lost packet
		end   time.Time    // send time of last lost packet
		next  packetNumber // one plus the number of the last lost packet
	}
}

func newReno(maxDatagramSize int) *ccReno {
	c := &ccReno{
		maxDatagramSize: maxDatagramSize,
	}

	// https://www.rfc-editor.org/rfc/rfc9002.html#section-7.2-1
	c.congestionWindow = min(10*maxDatagramSize, max(14720, c.minimumCongestionWindow()))

	// https://www.rfc-editor.org/rfc/rfc9002.html#section-7.3.1-1
	c.slowStartThreshold = math.MaxInt

	for space := range c.persistentCongestion {
		c.persistentCongestion[space].next = -1
	}
	return c
}

// canSend reports whether the congestion controller permits sending
// a maximum-size datagram at this time.
//
// "An endpoint MUST NOT send a packet if it would cause bytes_in_flight [...]
// to be larger than the congestion window [...]"
// https://www.rfc-editor.org/rfc/rfc9002#section-7-7
//
// For simplicity and efficiency, we don't permit sending undersized datagrams.
func (c *ccReno) canSend() bool {
	if c.sendOnePacketInRecovery {
		return true
	}
	return c.bytesInFlight+c.maxDatagramSize <= c.congestionWindow
}

// setUnderutilized indicates that the congestion window is underutilized.
//
// The congestion window is underutilized if bytes in flight is smaller than
// the congestion window and sending is not pacing limited; that is, the
// congestion controller permits sending data, but no data is sent.
//
// https://www.rfc-editor.org/rfc/rfc9002#section-7.8
func (c *ccReno) setUnderutilized(log *slog.Logger, v bool) {
	if c.underutilized == v {
		return
	}
	oldState := c.state()
	c.underutilized = v
	if logEnabled(log, QLogLevelPacket) {
		logCongestionStateUpdated(log, oldState, c.state())
	}
}

// packetSent indicates that a packet has been sent.
func (c *ccReno) packetSent(now time.Time, log *slog.Logger, space numberSpace, sent *sentPacket) {
	if !sent.inFlight {
		return
	}
	c.bytesInFlight += sent.size
	if c.sendOnePacketInRecovery {
		c.sendOnePacketInRecovery = false
	}
}

// Acked and lost packets are processed in batches
// resulting from either a received ACK frame or
// the loss detection timer expiring.
//
// A batch consists of zero or more calls to packetAcked and packetLost,
// followed by a single call to packetBatchEnd.
//
// Acks may be reported in any order, but lost packets must
// be reported in strictly increasing order.

// packetAcked indicates that a packet has been newly acknowledged.
func (c *ccReno) packetAcked(now time.Time, sent *sentPacket) {
	if !sent.inFlight {
		return
	}
	c.bytesInFlight -= sent.size

	if c.underutilized {
		// https://www.rfc-editor.org/rfc/rfc9002.html#section-7.8
		return
	}
	if sent.time.Before(c.recoveryStartTime) {
		// In recovery, and this packet was sent before we entered recovery.
		// (If this packet was sent after we entered recovery, receiving an ack
		// for it moves us out of recovery into congestion avoidance.)
		// https://www.rfc-editor.org/rfc/rfc9002.html#section-7.3.2
		return
	}
	c.congestionPendingAcks += sent.size
}

// packetLost indicates that a packet has been newly marked as lost.
// Lost packets must be reported in increasing order.
func (c *ccReno) packetLost(now time.Time, space numberSpace, sent *sentPacket, rtt *rttState) {
	// Record state to check for persistent congestion.
	// https://www.rfc-editor.org/rfc/rfc9002#section-7.6
	//
	// Note that this relies on always receiving loss events in increasing order:
	// All packets prior to the one we're examining now have either been
	// acknowledged or declared lost.
	isValidPersistentCongestionSample := (sent.ackEliciting &&
		!rtt.firstSampleTime.IsZero() &&
		!sent.time.Before(rtt.firstSampleTime))
	if isValidPersistentCongestionSample {
		// This packet either extends an existing range of lost packets,
		// or starts a new one.
		if sent.num != c.persistentCongestion[space].next {
			c.persistentCongestion[space].start = sent.time
		}
		c.persistentCongestion[space].end = sent.time
		c.persistentCongestion[space].next = sent.num + 1
	} else {
		// This packet cannot establish persistent congestion on its own.
		// However, if we have an existing range of lost packets,
		// this does not break it.
		if sent.num == c.persistentCongestion[space].next {
			c.persistentCongestion[space].next = sent.num + 1
		}
	}

	if !sent.inFlight {
		return
	}
	c.bytesInFlight -= sent.size
	if sent.time.After(c.ackLastLoss) {
		c.ackLastLoss = sent.time
	}
}

// packetBatchEnd is called at the end of processing a batch of acked or lost packets.
func (c *ccReno) packetBatchEnd(now time.Time, log *slog.Logger, space numberSpace, rtt *rttState, maxAckDelay time.Duration) {
	if logEnabled(log, QLogLevelPacket) {
		oldState := c.state()
		defer func() { logCongestionStateUpdated(log, oldState, c.state()) }()
	}
	if !c.ackLastLoss.IsZero() && !c.ackLastLoss.Before(c.recoveryStartTime) {
		// Enter the recovery state.
		// https://www.rfc-editor.org/rfc/rfc9002.html#section-7.3.2
		c.recoveryStartTime = now
		c.slowStartThreshold = c.congestionWindow / 2
		c.congestionWindow = max(c.slowStartThreshold, c.minimumCongestionWindow())
		c.sendOnePacketInRecovery = true
		// Clear congestionPendingAcks to avoid increasing the congestion
		// window based on acks in a frame that sends us into recovery.
		c.congestionPendingAcks = 0
		c.inRecovery = true
	} else if c.congestionPendingAcks > 0 {
		// We are in slow start or congestion avoidance.
		c.inRecovery = false
		if c.congestionWindow < c.slowStartThreshold {
			// When the congestion window is less than the slow start threshold,
			// we are in slow start and increase the window by the number of
			// bytes acknowledged.
			d := min(c.slowStartThreshold-c.congestionWindow, c.congestionPendingAcks)
			c.congestionWindow += d
			c.congestionPendingAcks -= d
		}
		// When the congestion window is at or above the slow start threshold,
		// we are in congestion avoidance.
		//
		// RFC 9002 does not specify an algorithm here. The following is
		// the recommended algorithm from RFC 5681, in which we increment
		// the window by the maximum datagram size every time the number
		// of bytes acknowledged reaches cwnd.
		for c.congestionPendingAcks > c.congestionWindow {
			c.congestionPendingAcks -= c.congestionWindow
			c.congestionWindow += c.maxDatagramSize
		}
	}
	if !c.ackLastLoss.IsZero() {
		// Check for persistent congestion.
		// https://www.rfc-editor.org/rfc/rfc9002.html#section-7.6
		//
		// "A sender [...] MAY use state for just the packet number space that
		// was acknowledged."
		// https://www.rfc-editor.org/rfc/rfc9002#section-7.6.2-5
		//
		// For simplicity, we consider each number space independently.
		const persistentCongestionThreshold = 3
		d := (rtt.smoothedRTT + max(4*rtt.rttvar, timerGranularity) + maxAckDelay) *
			persistentCongestionThreshold
		start := c.persistentCongestion[space].start
		end := c.persistentCongestion[space].end
		if end.Sub(start) >= d {
			c.congestionWindow = c.minimumCongestionWindow()
			c.recoveryStartTime = time.Time{}
			rtt.establishPersistentCongestion()
		}
	}
	c.ackLastLoss = time.Time{}
}

// packetDiscarded indicates that the keys for a packet's space have been discarded.
func (c *ccReno) packetDiscarded(sent *sentPacket) {
	// https://www.rfc-editor.org/rfc/rfc9002#section-6.2.2-3
	if sent.inFlight {
		c.bytesInFlight -= sent.size
	}
}

func (c *ccReno) minimumCongestionWindow() int {
	// https://www.rfc-editor.org/rfc/rfc9002.html#section-7.2-4
	return 2 * c.maxDatagramSize
}

func logCongestionStateUpdated(log *slog.Logger, oldState, newState congestionState) {
	if oldState == newState {
		return
	}
	log.LogAttrs(context.Background(), QLogLevelPacket,
		"recovery:congestion_state_updated",
		slog.String("old", oldState.String()),
		slog.String("new", newState.String()),
	)
}

type congestionState string

func (s congestionState) String() string { return string(s) }

const (
	congestionSlowStart           = congestionState("slow_start")
	congestionCongestionAvoidance = congestionState("congestion_avoidance")
	congestionApplicationLimited  = congestionState("application_limited")
	congestionRecovery            = congestionState("recovery")
)

func (c *ccReno) state() congestionState {
	switch {
	case c.inRecovery:
		return congestionRecovery
	case c.underutilized:
		return congestionApplicationLimited
	case c.congestionWindow < c.slowStartThreshold:
		return congestionSlowStart
	default:
		return congestionCongestionAvoidance
	}
}
