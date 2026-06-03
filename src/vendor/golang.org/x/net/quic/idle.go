// Copyright 2023 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package quic

import (
	"time"
)

// idleState tracks connection idle events.
//
// Before the handshake is confirmed, the idle timeout is Config.HandshakeTimeout.
//
// After the handshake is confirmed, the idle timeout is
// the minimum of Config.MaxIdleTimeout and the peer's max_idle_timeout transport parameter.
//
// If KeepAlivePeriod is set, keep-alive pings are sent.
// Keep-alives are only sent after the handshake is confirmed.
//
// https://www.rfc-editor.org/rfc/rfc9000#section-10.1
type idleState struct {
	// idleDuration is the negotiated idle timeout for the connection.
	idleDuration time.Duration

	// idleTimeout is the time at which the connection will be closed due to inactivity.
	idleTimeout time.Time

	// nextTimeout is the time of the next idle event.
	// If nextTimeout == idleTimeout, this is the idle timeout.
	// Otherwise, this is the keep-alive timeout.
	nextTimeout time.Time

	// sentSinceLastReceive is set if we have sent an ack-eliciting packet
	// since the last time we received and processed a packet from the peer.
	sentSinceLastReceive bool
}

// receivePeerMaxIdleTimeout handles the peer's max_idle_timeout transport parameter.
func (c *Conn) receivePeerMaxIdleTimeout(peerMaxIdleTimeout time.Duration) {
	localMaxIdleTimeout := c.config.maxIdleTimeout()
	switch {
	case localMaxIdleTimeout == 0:
		c.idle.idleDuration = peerMaxIdleTimeout
	case peerMaxIdleTimeout == 0:
		c.idle.idleDuration = localMaxIdleTimeout
	default:
		c.idle.idleDuration = min(localMaxIdleTimeout, peerMaxIdleTimeout)
	}
}

func (c *Conn) idleHandlePacketReceived(now time.Time) {
	if !c.handshakeConfirmed.isSet() {
		return
	}
	// "An endpoint restarts its idle timer when a packet from its peer is
	// received and processed successfully."
	// https://www.rfc-editor.org/rfc/rfc9000#section-10.1-3
	c.idle.sentSinceLastReceive = false
	c.restartIdleTimer(now)
}

func (c *Conn) idleHandlePacketSent(now time.Time, sent *sentPacket) {
	// "An endpoint also restarts its idle timer when sending an ack-eliciting packet
	// if no other ack-eliciting packets have been sent since
	// last receiving and processing a packet."
	// https://www.rfc-editor.org/rfc/rfc9000#section-10.1-3
	if c.idle.sentSinceLastReceive || !sent.ackEliciting || !c.handshakeConfirmed.isSet() {
		return
	}
	c.idle.sentSinceLastReceive = true
	c.restartIdleTimer(now)
}

func (c *Conn) restartIdleTimer(now time.Time) {
	if !c.isAlive() {
		// Connection is closing, disable timeouts.
		c.idle.idleTimeout = time.Time{}
		c.idle.nextTimeout = time.Time{}
		return
	}
	var idleDuration time.Duration
	if c.handshakeConfirmed.isSet() {
		idleDuration = c.idle.idleDuration
	} else {
		idleDuration = c.config.handshakeTimeout()
	}
	if idleDuration == 0 {
		c.idle.idleTimeout = time.Time{}
	} else {
		// "[...] endpoints MUST increase the idle timeout period to be
		// at least three times the current Probe Timeout (PTO)."
		// https://www.rfc-editor.org/rfc/rfc9000#section-10.1-4
		idleDuration = max(idleDuration, 3*c.loss.ptoPeriod())
		c.idle.idleTimeout = now.Add(idleDuration)
	}
	// Set the time of our next event:
	// The idle timer if no keep-alive is set, or the keep-alive timer if one is.
	c.idle.nextTimeout = c.idle.idleTimeout
	keepAlive := c.config.keepAlivePeriod()
	switch {
	case !c.handshakeConfirmed.isSet():
		// We do not send keep-alives before the handshake is complete.
	case keepAlive <= 0:
		// Keep-alives are not enabled.
	case c.idle.sentSinceLastReceive:
		// We have sent an ack-eliciting packet to the peer.
		// If they don't acknowledge it, loss detection will follow up with PTO probes,
		// which will function as keep-alives.
		// We don't need to send further pings.
	case idleDuration == 0:
		// The connection does not have a negotiated idle timeout.
		// Send keep-alives anyway, since they may be required to keep middleboxes
		// from losing state.
		c.idle.nextTimeout = now.Add(keepAlive)
	default:
		// Schedule our next keep-alive.
		// If our configured keep-alive period is greater than half the negotiated
		// connection idle timeout, we reduce the keep-alive period to half
		// the idle timeout to ensure we have time for the ping to arrive.
		c.idle.nextTimeout = now.Add(min(keepAlive, idleDuration/2))
	}
}

func (c *Conn) appendKeepAlive(now time.Time) bool {
	if c.idle.nextTimeout.IsZero() || c.idle.nextTimeout.After(now) {
		return true // timer has not expired
	}
	if c.idle.nextTimeout.Equal(c.idle.idleTimeout) {
		return true // no keepalive timer set, only idle
	}
	if c.idle.sentSinceLastReceive {
		return true // already sent an ack-eliciting packet
	}
	if c.w.sent.ackEliciting {
		return true // this packet is already ack-eliciting
	}
	// Send an ack-eliciting PING frame to the peer to keep the connection alive.
	return c.w.appendPingFrame()
}

var errHandshakeTimeout error = localTransportError{
	code:   errConnectionRefused,
	reason: "handshake timeout",
}

func (c *Conn) idleAdvance(now time.Time) (shouldExit bool) {
	if c.idle.idleTimeout.IsZero() || now.Before(c.idle.idleTimeout) {
		return false
	}
	c.idle.idleTimeout = time.Time{}
	c.idle.nextTimeout = time.Time{}
	if !c.handshakeConfirmed.isSet() {
		// Handshake timeout has expired.
		// If we're a server, we're refusing the too-slow client.
		// If we're a client, we're giving up.
		// In either case, we're going to send a CONNECTION_CLOSE frame and
		// enter the closing state rather than unceremoniously dropping the connection,
		// since the peer might still be trying to complete the handshake.
		c.abort(now, errHandshakeTimeout)
		return false
	}
	// Idle timeout has expired.
	//
	// "[...] the connection is silently closed and its state is discarded [...]"
	// https://www.rfc-editor.org/rfc/rfc9000#section-10.1-1
	return true
}
