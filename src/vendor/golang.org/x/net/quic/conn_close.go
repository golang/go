// Copyright 2023 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package quic

import (
	"context"
	"errors"
	"time"
)

// connState is the state of a connection.
type connState int

const (
	// A connection is alive when it is first created.
	connStateAlive = connState(iota)

	// The connection has received a CONNECTION_CLOSE frame from the peer,
	// and has not yet sent a CONNECTION_CLOSE in response.
	//
	// We will send a CONNECTION_CLOSE, and then enter the draining state.
	connStatePeerClosed

	// The connection is in the closing state.
	//
	// We will send CONNECTION_CLOSE frames to the peer
	// (once upon entering the closing state, and possibly again in response to peer packets).
	//
	// If we receive a CONNECTION_CLOSE from the peer, we will enter the draining state.
	// Otherwise, we will eventually time out and move to the done state.
	//
	// https://www.rfc-editor.org/rfc/rfc9000#section-10.2.1
	connStateClosing

	// The connection is in the draining state.
	//
	// We will neither send packets nor process received packets.
	// When the drain timer expires, we move to the done state.
	//
	// https://www.rfc-editor.org/rfc/rfc9000#section-10.2.2
	connStateDraining

	// The connection is done, and the conn loop will exit.
	connStateDone
)

// lifetimeState tracks the state of a connection.
//
// This is fairly coupled to the rest of a Conn, but putting it in a struct of its own helps
// reason about operations that cause state transitions.
type lifetimeState struct {
	state connState

	readyc chan struct{} // closed when TLS handshake completes
	donec  chan struct{} // closed when finalErr is set

	localErr error // error sent to the peer
	finalErr error // error sent by the peer, or transport error; set before closing donec

	connCloseSentTime time.Time     // send time of last CONNECTION_CLOSE frame
	connCloseDelay    time.Duration // delay until next CONNECTION_CLOSE frame sent
	drainEndTime      time.Time     // time the connection exits the draining state
}

func (c *Conn) lifetimeInit() {
	c.lifetime.readyc = make(chan struct{})
	c.lifetime.donec = make(chan struct{})
}

var (
	errNoPeerResponse = errors.New("peer did not respond to CONNECTION_CLOSE")
	errConnClosed     = errors.New("connection closed")
)

// advance is called when time passes.
func (c *Conn) lifetimeAdvance(now time.Time) (done bool) {
	if c.lifetime.drainEndTime.IsZero() || c.lifetime.drainEndTime.After(now) {
		return false
	}
	// The connection drain period has ended, and we can shut down.
	// https://www.rfc-editor.org/rfc/rfc9000.html#section-10.2-7
	c.lifetime.drainEndTime = time.Time{}
	if c.lifetime.state != connStateDraining {
		// We were in the closing state, waiting for a CONNECTION_CLOSE from the peer.
		c.setFinalError(errNoPeerResponse)
	}
	c.setState(now, connStateDone)
	return true
}

// setState sets the conn state.
func (c *Conn) setState(now time.Time, state connState) {
	if c.lifetime.state == state {
		return
	}
	c.lifetime.state = state
	switch state {
	case connStateClosing, connStateDraining:
		if c.lifetime.drainEndTime.IsZero() {
			c.lifetime.drainEndTime = now.Add(3 * c.loss.ptoBasePeriod())
		}
	case connStateDone:
		c.setFinalError(nil)
	}
	if state != connStateAlive {
		c.streamsCleanup()
	}
}

// handshakeDone is called when the TLS handshake completes.
func (c *Conn) handshakeDone() {
	close(c.lifetime.readyc)
}

// isDraining reports whether the conn is in the draining state.
//
// The draining state is entered once an endpoint receives a CONNECTION_CLOSE frame.
// The endpoint will no longer send any packets, but we retain knowledge of the connection
// until the end of the drain period to ensure we discard packets for the connection
// rather than treating them as starting a new connection.
//
// https://www.rfc-editor.org/rfc/rfc9000.html#section-10.2.2
func (c *Conn) isDraining() bool {
	switch c.lifetime.state {
	case connStateDraining, connStateDone:
		return true
	}
	return false
}

// isAlive reports whether the conn is handling packets.
func (c *Conn) isAlive() bool {
	return c.lifetime.state == connStateAlive
}

// sendOK reports whether the conn can send frames at this time.
func (c *Conn) sendOK(now time.Time) bool {
	switch c.lifetime.state {
	case connStateAlive:
		return true
	case connStatePeerClosed:
		if c.lifetime.localErr == nil {
			// We're waiting for the user to close the connection, providing us with
			// a final status to send to the peer.
			return false
		}
		// We should send a CONNECTION_CLOSE.
		return true
	case connStateClosing:
		if c.lifetime.connCloseSentTime.IsZero() {
			return true
		}
		maxRecvTime := c.acks[initialSpace].maxRecvTime
		if t := c.acks[handshakeSpace].maxRecvTime; t.After(maxRecvTime) {
			maxRecvTime = t
		}
		if t := c.acks[appDataSpace].maxRecvTime; t.After(maxRecvTime) {
			maxRecvTime = t
		}
		if maxRecvTime.Before(c.lifetime.connCloseSentTime.Add(c.lifetime.connCloseDelay)) {
			// After sending CONNECTION_CLOSE, ignore packets from the peer for
			// a delay. On the next packet received after the delay, send another
			// CONNECTION_CLOSE.
			return false
		}
		return true
	case connStateDraining:
		// We are in the draining state, and will send no more packets.
		return false
	case connStateDone:
		return false
	default:
		panic("BUG: unhandled connection state")
	}
}

// sentConnectionClose reports that the conn has sent a CONNECTION_CLOSE to the peer.
func (c *Conn) sentConnectionClose(now time.Time) {
	switch c.lifetime.state {
	case connStatePeerClosed:
		c.enterDraining(now)
	}
	if c.lifetime.connCloseSentTime.IsZero() {
		// Set the initial delay before we will send another CONNECTION_CLOSE.
		//
		// RFC 9000 states that we should rate limit CONNECTION_CLOSE frames,
		// but leaves the implementation of the limit up to us. Here, we start
		// with the same delay as the PTO timer (RFC 9002, Section 6.2.1),
		// not including max_ack_delay, and double it on every CONNECTION_CLOSE sent.
		c.lifetime.connCloseDelay = c.loss.rtt.smoothedRTT + max(4*c.loss.rtt.rttvar, timerGranularity)
	} else if !c.lifetime.connCloseSentTime.Equal(now) {
		// If connCloseSentTime == now, we're sending two CONNECTION_CLOSE frames
		// coalesced into the same datagram. We only want to increase the delay once.
		c.lifetime.connCloseDelay *= 2
	}
	c.lifetime.connCloseSentTime = now
}

// handlePeerConnectionClose handles a CONNECTION_CLOSE from the peer.
func (c *Conn) handlePeerConnectionClose(now time.Time, err error) {
	c.setFinalError(err)
	switch c.lifetime.state {
	case connStateAlive:
		c.setState(now, connStatePeerClosed)
	case connStatePeerClosed:
		// Duplicate CONNECTION_CLOSE, ignore.
	case connStateClosing:
		if c.lifetime.connCloseSentTime.IsZero() {
			c.setState(now, connStatePeerClosed)
		} else {
			c.setState(now, connStateDraining)
		}
	case connStateDraining:
	case connStateDone:
	}
}

// setFinalError records the final connection status we report to the user.
func (c *Conn) setFinalError(err error) {
	select {
	case <-c.lifetime.donec:
		return // already set
	default:
	}
	c.lifetime.finalErr = err
	close(c.lifetime.donec)
}

// finalError returns the final connection status reported to the user,
// or nil if a final status has not yet been set.
func (c *Conn) finalError() error {
	select {
	case <-c.lifetime.donec:
		return c.lifetime.finalErr
	default:
	}
	return nil
}

func (c *Conn) waitReady(ctx context.Context) error {
	select {
	case <-c.lifetime.readyc:
		return nil
	case <-c.lifetime.donec:
		return c.lifetime.finalErr
	default:
	}
	select {
	case <-c.lifetime.readyc:
		return nil
	case <-c.lifetime.donec:
		return c.lifetime.finalErr
	case <-ctx.Done():
		return ctx.Err()
	}
}

// Close closes the connection.
//
// Close is equivalent to:
//
//	conn.Abort(nil)
//	err := conn.Wait(context.Background())
func (c *Conn) Close() error {
	c.Abort(nil)
	<-c.lifetime.donec
	return c.lifetime.finalErr
}

// Wait waits for the peer to close the connection.
//
// If the connection is closed locally and the peer does not close its end of the connection,
// Wait will return with a non-nil error after the drain period expires.
//
// If the peer closes the connection with a NO_ERROR transport error, Wait returns nil.
// If the peer closes the connection with an application error, Wait returns an ApplicationError
// containing the peer's error code and reason.
// If the peer closes the connection with any other status, Wait returns a non-nil error.
func (c *Conn) Wait(ctx context.Context) error {
	if err := c.waitOnDone(ctx, c.lifetime.donec); err != nil {
		return err
	}
	return c.lifetime.finalErr
}

// Abort closes the connection and returns immediately.
//
// If err is nil, Abort sends a transport error of NO_ERROR to the peer.
// If err is an ApplicationError, Abort sends its error code and text.
// Otherwise, Abort sends a transport error of APPLICATION_ERROR with the error's text.
func (c *Conn) Abort(err error) {
	if err == nil {
		err = localTransportError{code: errNo}
	}
	c.sendMsg(func(now time.Time, c *Conn) {
		c.enterClosing(now, err)
	})
}

// abort terminates a connection with an error.
func (c *Conn) abort(now time.Time, err error) {
	c.setFinalError(err) // this error takes precedence over the peer's CONNECTION_CLOSE
	c.enterClosing(now, err)
}

// abortImmediately terminates a connection.
// The connection does not send a CONNECTION_CLOSE, and skips the draining period.
func (c *Conn) abortImmediately(now time.Time, err error) {
	c.setFinalError(err)
	c.setState(now, connStateDone)
}

// enterClosing starts an immediate close.
// We will send a CONNECTION_CLOSE to the peer and wait for their response.
func (c *Conn) enterClosing(now time.Time, err error) {
	switch c.lifetime.state {
	case connStateAlive:
		c.lifetime.localErr = err
		c.setState(now, connStateClosing)
	case connStatePeerClosed:
		c.lifetime.localErr = err
	}
}

// enterDraining moves directly to the draining state, without sending a CONNECTION_CLOSE.
func (c *Conn) enterDraining(now time.Time) {
	switch c.lifetime.state {
	case connStateAlive, connStatePeerClosed, connStateClosing:
		c.setState(now, connStateDraining)
	}
}

// exit fully terminates a connection immediately.
func (c *Conn) exit() {
	c.sendMsg(func(now time.Time, c *Conn) {
		c.abortImmediately(now, errors.New("connection closed"))
	})
}
