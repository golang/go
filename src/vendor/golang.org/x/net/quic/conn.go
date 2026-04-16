// Copyright 2023 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package quic

import (
	"context"
	cryptorand "crypto/rand"
	"crypto/tls"
	"errors"
	"fmt"
	"log/slog"
	"math/rand/v2"
	"net/netip"
	"time"
)

// A Conn is a QUIC connection.
//
// Multiple goroutines may invoke methods on a Conn simultaneously.
type Conn struct {
	side      connSide
	endpoint  *Endpoint
	config    *Config
	testHooks connTestHooks
	peerAddr  netip.AddrPort
	localAddr netip.AddrPort
	prng      *rand.Rand

	msgc  chan any
	donec chan struct{} // closed when conn loop exits

	w           packetWriter
	acks        [numberSpaceCount]ackState // indexed by number space
	lifetime    lifetimeState
	idle        idleState
	connIDState connIDState
	loss        lossState
	streams     streamsState
	path        pathState
	skip        skipState

	// Packet protection keys, CRYPTO streams, and TLS state.
	keysInitial   fixedKeyPair
	keysHandshake fixedKeyPair
	keysAppData   updatingKeyPair
	crypto        [numberSpaceCount]cryptoStream
	tls           *tls.QUICConn

	// retryToken is the token provided by the peer in a Retry packet.
	retryToken []byte

	// handshakeConfirmed is set when the handshake is confirmed.
	// For server connections, it tracks sending HANDSHAKE_DONE.
	handshakeConfirmed sentVal

	peerAckDelayExponent int8 // -1 when unknown

	// Tests only: Send a PING in a specific number space.
	testSendPingSpace numberSpace
	testSendPing      sentVal

	log *slog.Logger
}

// connTestHooks override conn behavior in tests.
type connTestHooks interface {
	// init is called after a conn is created.
	init(first bool)

	// handleTLSEvent is called with each TLS event.
	handleTLSEvent(tls.QUICEvent)

	// newConnID is called to generate a new connection ID.
	// Permits tests to generate consistent connection IDs rather than random ones.
	newConnID(seq int64) ([]byte, error)
}

// newServerConnIDs is connection IDs associated with a new server connection.
type newServerConnIDs struct {
	srcConnID         []byte // source from client's current Initial
	dstConnID         []byte // destination from client's current Initial
	originalDstConnID []byte // destination from client's first Initial
	retrySrcConnID    []byte // source from server's Retry
}

func newConn(now time.Time, side connSide, cids newServerConnIDs, peerHostname string, peerAddr netip.AddrPort, config *Config, e *Endpoint) (conn *Conn, _ error) {
	c := &Conn{
		side:                 side,
		endpoint:             e,
		config:               config,
		peerAddr:             unmapAddrPort(peerAddr),
		donec:                make(chan struct{}),
		peerAckDelayExponent: -1,
	}
	defer func() {
		// If we hit an error in newConn, close donec so tests don't get stuck waiting for it.
		// This is only relevant if we've got a bug, but it makes tracking that bug down
		// much easier.
		if conn == nil {
			close(c.donec)
		}
	}()

	// A one-element buffer allows us to wake a Conn's event loop as a
	// non-blocking operation.
	c.msgc = make(chan any, 1)

	if e.testHooks != nil {
		e.testHooks.newConn(c)
	}

	// initialConnID is the connection ID used to generate Initial packet protection keys.
	var initialConnID []byte
	if c.side == clientSide {
		if err := c.connIDState.initClient(c); err != nil {
			return nil, err
		}
		initialConnID, _ = c.connIDState.dstConnID()
	} else {
		initialConnID = cids.originalDstConnID
		if cids.retrySrcConnID != nil {
			initialConnID = cids.retrySrcConnID
		}
		if err := c.connIDState.initServer(c, cids); err != nil {
			return nil, err
		}
	}

	// A per-conn ChaCha8 PRNG is probably more than we need,
	// but at least it's fairly small.
	var seed [32]byte
	if _, err := cryptorand.Read(seed[:]); err != nil {
		panic(err)
	}
	c.prng = rand.New(rand.NewChaCha8(seed))

	// TODO: PMTU discovery.
	c.logConnectionStarted(cids.originalDstConnID, peerAddr)
	c.keysAppData.init()
	c.loss.init(c.side, smallestMaxDatagramSize, now)
	c.streamsInit()
	c.lifetimeInit()
	c.restartIdleTimer(now)
	c.skip.init(c)

	if err := c.startTLS(now, initialConnID, peerHostname, transportParameters{
		initialSrcConnID:               c.connIDState.srcConnID(),
		originalDstConnID:              cids.originalDstConnID,
		retrySrcConnID:                 cids.retrySrcConnID,
		ackDelayExponent:               ackDelayExponent,
		maxUDPPayloadSize:              maxUDPPayloadSize,
		maxAckDelay:                    maxAckDelay,
		disableActiveMigration:         true,
		initialMaxData:                 config.maxConnReadBufferSize(),
		initialMaxStreamDataBidiLocal:  config.maxStreamReadBufferSize(),
		initialMaxStreamDataBidiRemote: config.maxStreamReadBufferSize(),
		initialMaxStreamDataUni:        config.maxStreamReadBufferSize(),
		initialMaxStreamsBidi:          c.streams.remoteLimit[bidiStream].max,
		initialMaxStreamsUni:           c.streams.remoteLimit[uniStream].max,
		activeConnIDLimit:              activeConnIDLimit,
	}); err != nil {
		return nil, err
	}

	if c.testHooks != nil {
		c.testHooks.init(true)
	}
	go c.loop(now)
	return c, nil
}

func (c *Conn) String() string {
	return fmt.Sprintf("quic.Conn(%v,->%v)", c.side, c.peerAddr)
}

// LocalAddr returns the local network address, if known.
func (c *Conn) LocalAddr() netip.AddrPort {
	return c.localAddr
}

// RemoteAddr returns the remote network address, if known.
func (c *Conn) RemoteAddr() netip.AddrPort {
	return c.peerAddr
}

// ConnectionState returns basic TLS details about the connection.
func (c *Conn) ConnectionState() tls.ConnectionState {
	return c.tls.ConnectionState()
}

// confirmHandshake is called when the handshake is confirmed.
// https://www.rfc-editor.org/rfc/rfc9001#section-4.1.2
func (c *Conn) confirmHandshake(now time.Time) {
	// If handshakeConfirmed is unset, the handshake is not confirmed.
	// If it is unsent, the handshake is confirmed and we need to send a HANDSHAKE_DONE.
	// If it is sent, we have sent a HANDSHAKE_DONE.
	// If it is received, the handshake is confirmed and we do not need to send anything.
	if c.handshakeConfirmed.isSet() {
		return // already confirmed
	}
	if c.side == serverSide {
		// When the server confirms the handshake, it sends a HANDSHAKE_DONE.
		c.handshakeConfirmed.setUnsent()
		c.endpoint.serverConnEstablished(c)
	} else {
		// The client never sends a HANDSHAKE_DONE, so we set handshakeConfirmed
		// to the received state, indicating that the handshake is confirmed and we
		// don't need to send anything.
		c.handshakeConfirmed.setReceived()
	}
	c.restartIdleTimer(now)
	c.loss.confirmHandshake()
	// "An endpoint MUST discard its Handshake keys when the TLS handshake is confirmed"
	// https://www.rfc-editor.org/rfc/rfc9001#section-4.9.2-1
	c.discardKeys(now, handshakeSpace)
}

// discardKeys discards unused packet protection keys.
// https://www.rfc-editor.org/rfc/rfc9001#section-4.9
func (c *Conn) discardKeys(now time.Time, space numberSpace) {
	if err := c.crypto[space].discardKeys(); err != nil {
		c.abort(now, err)
	}
	switch space {
	case initialSpace:
		c.keysInitial.discard()
	case handshakeSpace:
		c.keysHandshake.discard()
	}
	c.loss.discardKeys(now, c.log, space)
}

// receiveTransportParameters applies transport parameters sent by the peer.
func (c *Conn) receiveTransportParameters(p transportParameters) error {
	isRetry := c.retryToken != nil
	if err := c.connIDState.validateTransportParameters(c, isRetry, p); err != nil {
		return err
	}
	c.streams.outflow.setMaxData(p.initialMaxData)
	c.streams.localLimit[bidiStream].setMax(p.initialMaxStreamsBidi)
	c.streams.localLimit[uniStream].setMax(p.initialMaxStreamsUni)
	c.streams.peerInitialMaxStreamDataBidiLocal = p.initialMaxStreamDataBidiLocal
	c.streams.peerInitialMaxStreamDataRemote[bidiStream] = p.initialMaxStreamDataBidiRemote
	c.streams.peerInitialMaxStreamDataRemote[uniStream] = p.initialMaxStreamDataUni
	c.receivePeerMaxIdleTimeout(p.maxIdleTimeout)
	c.peerAckDelayExponent = p.ackDelayExponent
	c.loss.setMaxAckDelay(p.maxAckDelay)
	if err := c.connIDState.setPeerActiveConnIDLimit(c, p.activeConnIDLimit); err != nil {
		return err
	}
	if p.preferredAddrConnID != nil {
		var (
			seq           int64 = 1 // sequence number of this conn id is 1
			retirePriorTo int64 = 0 // retire nothing
			resetToken    [16]byte
		)
		copy(resetToken[:], p.preferredAddrResetToken)
		if err := c.connIDState.handleNewConnID(c, seq, retirePriorTo, p.preferredAddrConnID, resetToken); err != nil {
			return err
		}
	}
	// TODO: stateless_reset_token
	// TODO: max_udp_payload_size
	// TODO: disable_active_migration
	// TODO: preferred_address
	return nil
}

type (
	timerEvent struct{}
	wakeEvent  struct{}
)

var errIdleTimeout = errors.New("idle timeout")

// loop is the connection main loop.
//
// Except where otherwise noted, all connection state is owned by the loop goroutine.
//
// The loop processes messages from c.msgc and timer events.
// Other goroutines may examine or modify conn state by sending the loop funcs to execute.
func (c *Conn) loop(now time.Time) {
	defer c.cleanup()

	// The connection timer sends a message to the connection loop on expiry.
	// We need to give it an expiry when creating it, so set the initial timeout to
	// an arbitrary large value. The timer will be reset before this expires (and it
	// isn't a problem if it does anyway).
	var lastTimeout time.Time
	timer := time.AfterFunc(1*time.Hour, func() {
		c.sendMsg(timerEvent{})
	})
	defer timer.Stop()

	for c.lifetime.state != connStateDone {
		sendTimeout := c.maybeSend(now) // try sending

		// Note that we only need to consider the ack timer for the App Data space,
		// since the Initial and Handshake spaces always ack immediately.
		nextTimeout := sendTimeout
		nextTimeout = firstTime(nextTimeout, c.idle.nextTimeout)
		if c.isAlive() {
			nextTimeout = firstTime(nextTimeout, c.loss.timer)
			nextTimeout = firstTime(nextTimeout, c.acks[appDataSpace].nextAck)
		} else {
			nextTimeout = firstTime(nextTimeout, c.lifetime.drainEndTime)
		}

		var m any
		if !nextTimeout.IsZero() && nextTimeout.Before(now) {
			// A connection timer has expired.
			now = time.Now()
			m = timerEvent{}
		} else {
			// Reschedule the connection timer if necessary
			// and wait for the next event.
			if !nextTimeout.Equal(lastTimeout) && !nextTimeout.IsZero() {
				// Resetting a timer created with time.AfterFunc guarantees
				// that the timer will run again. We might generate a spurious
				// timer event under some circumstances, but that's okay.
				timer.Reset(nextTimeout.Sub(now))
				lastTimeout = nextTimeout
			}
			m = <-c.msgc
			now = time.Now()
		}
		switch m := m.(type) {
		case *datagram:
			if !c.handleDatagram(now, m) {
				if c.logEnabled(QLogLevelPacket) {
					c.logPacketDropped(m)
				}
			}
			m.recycle()
		case timerEvent:
			// A connection timer has expired.
			if c.idleAdvance(now) {
				// The connection idle timer has expired.
				c.abortImmediately(now, errIdleTimeout)
				return
			}
			c.loss.advance(now, c.handleAckOrLoss)
			if c.lifetimeAdvance(now) {
				// The connection has completed the draining period,
				// and may be shut down.
				return
			}
		case wakeEvent:
			// We're being woken up to try sending some frames.
		case func(time.Time, *Conn):
			// Send a func to msgc to run it on the main Conn goroutine
			m(now, c)
		case func(now, next time.Time, _ *Conn):
			// Send a func to msgc to run it on the main Conn goroutine
			m(now, nextTimeout, c)
		default:
			panic(fmt.Sprintf("quic: unrecognized conn message %T", m))
		}
	}
}

func (c *Conn) cleanup() {
	c.logConnectionClosed()
	c.endpoint.connDrained(c)
	c.tls.Close()
	close(c.donec)
}

// sendMsg sends a message to the conn's loop.
// It does not wait for the message to be processed.
// The conn may close before processing the message, in which case it is lost.
func (c *Conn) sendMsg(m any) {
	select {
	case c.msgc <- m:
	case <-c.donec:
	}
}

// wake wakes up the conn's loop.
func (c *Conn) wake() {
	select {
	case c.msgc <- wakeEvent{}:
	default:
	}
}

// runOnLoop executes a function within the conn's loop goroutine.
func (c *Conn) runOnLoop(ctx context.Context, f func(now time.Time, c *Conn)) error {
	donec := make(chan struct{})
	msg := func(now time.Time, c *Conn) {
		defer close(donec)
		f(now, c)
	}
	c.sendMsg(msg)
	select {
	case <-donec:
	case <-c.donec:
		return errors.New("quic: connection closed")
	}
	return nil
}

func (c *Conn) waitOnDone(ctx context.Context, ch <-chan struct{}) error {
	// Check the channel before the context.
	// We always prefer to return results when available,
	// even when provided with an already-canceled context.
	select {
	case <-ch:
		return nil
	default:
	}
	select {
	case <-ch:
	case <-ctx.Done():
		return ctx.Err()
	}
	return nil
}

// firstTime returns the earliest non-zero time, or zero if both times are zero.
func firstTime(a, b time.Time) time.Time {
	switch {
	case a.IsZero():
		return b
	case b.IsZero():
		return a
	case a.Before(b):
		return a
	default:
		return b
	}
}
