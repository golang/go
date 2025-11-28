// Copyright 2023 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package tls

import (
	"context"
	"errors"
	"fmt"
)

// QUICEncryptionLevel represents a QUIC encryption level used to transmit
// handshake messages.
type QUICEncryptionLevel int

const (
	QUICEncryptionLevelInitial = QUICEncryptionLevel(iota)
	QUICEncryptionLevelEarly
	QUICEncryptionLevelHandshake
	QUICEncryptionLevelApplication
)

func (l QUICEncryptionLevel) String() string {
	switch l {
	case QUICEncryptionLevelInitial:
		return "Initial"
	case QUICEncryptionLevelEarly:
		return "Early"
	case QUICEncryptionLevelHandshake:
		return "Handshake"
	case QUICEncryptionLevelApplication:
		return "Application"
	default:
		return fmt.Sprintf("QUICEncryptionLevel(%v)", int(l))
	}
}

// A QUICConn represents a connection which uses a QUIC implementation as the underlying
// transport as described in RFC 9001.
//
// Methods of QUICConn are not safe for concurrent use.
type QUICConn struct {
	conn *Conn

	sessionTicketSent bool
}

// A QUICConfig configures a [QUICConn].
type QUICConfig struct {
	TLSConfig *Config

	// EnableSessionEvents may be set to true to enable the
	// [QUICStoreSession] and [QUICResumeSession] events for client connections.
	// When this event is enabled, sessions are not automatically
	// stored in the client session cache.
	// The application should use [QUICConn.StoreSession] to store sessions.
	EnableSessionEvents bool
}

// A QUICEventKind is a type of operation on a QUIC connection.
type QUICEventKind int

const (
	// QUICNoEvent indicates that there are no events available.
	QUICNoEvent QUICEventKind = iota

	// QUICSetReadSecret and QUICSetWriteSecret provide the read and write
	// secrets for a given encryption level.
	// QUICEvent.Level, QUICEvent.Data, and QUICEvent.Suite are set.
	//
	// Secrets for the Initial encryption level are derived from the initial
	// destination connection ID, and are not provided by the QUICConn.
	QUICSetReadSecret
	QUICSetWriteSecret

	// QUICWriteData provides data to send to the peer in CRYPTO frames.
	// QUICEvent.Data is set.
	QUICWriteData

	// QUICTransportParameters provides the peer's QUIC transport parameters.
	// QUICEvent.Data is set.
	QUICTransportParameters

	// QUICTransportParametersRequired indicates that the caller must provide
	// QUIC transport parameters to send to the peer. The caller should set
	// the transport parameters with QUICConn.SetTransportParameters and call
	// QUICConn.NextEvent again.
	//
	// If transport parameters are set before calling QUICConn.Start, the
	// connection will never generate a QUICTransportParametersRequired event.
	QUICTransportParametersRequired

	// QUICRejectedEarlyData indicates that the server rejected 0-RTT data even
	// if we offered it. It's returned before QUICEncryptionLevelApplication
	// keys are returned.
	// This event only occurs on client connections.
	QUICRejectedEarlyData

	// QUICHandshakeDone indicates that the TLS handshake has completed.
	QUICHandshakeDone

	// QUICResumeSession indicates that a client is attempting to resume a previous session.
	// [QUICEvent.SessionState] is set.
	//
	// For client connections, this event occurs when the session ticket is selected.
	// For server connections, this event occurs when receiving the client's session ticket.
	//
	// The application may set [QUICEvent.SessionState.EarlyData] to false before the
	// next call to [QUICConn.NextEvent] to decline 0-RTT even if the session supports it.
	QUICResumeSession

	// QUICStoreSession indicates that the server has provided state permitting
	// the client to resume the session.
	// [QUICEvent.SessionState] is set.
	// The application should use [QUICConn.StoreSession] session to store the [SessionState].
	// The application may modify the [SessionState] before storing it.
	// This event only occurs on client connections.
	QUICStoreSession

	// QUICErrorEvent indicates that a fatal error has occurred.
	// The handshake cannot proceed and the connection must be closed.
	// QUICEvent.Err is set.
	QUICErrorEvent
)

// A QUICEvent is an event occurring on a QUIC connection.
//
// The type of event is specified by the Kind field.
// The contents of the other fields are kind-specific.
type QUICEvent struct {
	Kind QUICEventKind

	// Set for QUICSetReadSecret, QUICSetWriteSecret, and QUICWriteData.
	Level QUICEncryptionLevel

	// Set for QUICTransportParameters, QUICSetReadSecret, QUICSetWriteSecret, and QUICWriteData.
	// The contents are owned by crypto/tls, and are valid until the next NextEvent call.
	Data []byte

	// Set for QUICSetReadSecret and QUICSetWriteSecret.
	Suite uint16

	// Set for QUICResumeSession and QUICStoreSession.
	SessionState *SessionState

	// Set for QUICErrorEvent.
	// The error will wrap AlertError.
	Err error
}

type quicState struct {
	events    []QUICEvent
	nextEvent int

	// eventArr is a statically allocated event array, large enough to handle
	// the usual maximum number of events resulting from a single call: transport
	// parameters, Initial data, Early read secret, Handshake write and read
	// secrets, Handshake data, Application write secret, Application data.
	eventArr [8]QUICEvent

	started  bool
	signalc  chan struct{}   // handshake data is available to be read
	blockedc chan struct{}   // handshake is waiting for data, closed when done
	cancelc  <-chan struct{} // handshake has been canceled
	cancel   context.CancelFunc

	waitingForDrain bool
	errorReturned   bool

	// readbuf is shared between HandleData and the handshake goroutine.
	// HandshakeCryptoData passes ownership to the handshake goroutine by
	// reading from signalc, and reclaims ownership by reading from blockedc.
	readbuf []byte

	transportParams []byte // to send to the peer

	enableSessionEvents bool
}

// QUICClient returns a new TLS client side connection using QUICTransport as the
// underlying transport. The config cannot be nil.
//
// The config's MinVersion must be at least TLS 1.3.
func QUICClient(config *QUICConfig) *QUICConn {
	return newQUICConn(Client(nil, config.TLSConfig), config)
}

// QUICServer returns a new TLS server side connection using QUICTransport as the
// underlying transport. The config cannot be nil.
//
// The config's MinVersion must be at least TLS 1.3.
func QUICServer(config *QUICConfig) *QUICConn {
	return newQUICConn(Server(nil, config.TLSConfig), config)
}

func newQUICConn(conn *Conn, config *QUICConfig) *QUICConn {
	conn.quic = &quicState{
		signalc:             make(chan struct{}),
		blockedc:            make(chan struct{}),
		enableSessionEvents: config.EnableSessionEvents,
	}
	conn.quic.events = conn.quic.eventArr[:0]
	return &QUICConn{
		conn: conn,
	}
}

// Start starts the client or server handshake protocol.
// It may produce connection events, which may be read with [QUICConn.NextEvent].
//
// Start must be called at most once.
func (q *QUICConn) Start(ctx context.Context) error {
	if q.conn.quic.started {
		return quicError(errors.New("tls: Start called more than once"))
	}
	q.conn.quic.started = true
	if q.conn.config.MinVersion < VersionTLS13 {
		return quicError(errors.New("tls: Config MinVersion must be at least TLS 1.3"))
	}
	go q.conn.HandshakeContext(ctx)
	if _, ok := <-q.conn.quic.blockedc; !ok {
		return q.conn.handshakeErr
	}
	return nil
}

// NextEvent returns the next event occurring on the connection.
// It returns an event with a Kind of [QUICNoEvent] when no events are available.
func (q *QUICConn) NextEvent() QUICEvent {
	qs := q.conn.quic
	if last := qs.nextEvent - 1; last >= 0 && len(qs.events[last].Data) > 0 {
		// Write over some of the previous event's data,
		// to catch callers erroneously retaining it.
		qs.events[last].Data[0] = 0
	}
	if qs.nextEvent >= len(qs.events) && qs.waitingForDrain {
		qs.waitingForDrain = false
		<-qs.signalc
		<-qs.blockedc
	}
	if err := q.conn.handshakeErr; err != nil {
		if qs.errorReturned {
			return QUICEvent{Kind: QUICNoEvent}
		}
		qs.errorReturned = true
		qs.events = nil
		qs.nextEvent = 0
		return QUICEvent{Kind: QUICErrorEvent, Err: q.conn.handshakeErr}
	}
	if qs.nextEvent >= len(qs.events) {
		qs.events = qs.events[:0]
		qs.nextEvent = 0
		return QUICEvent{Kind: QUICNoEvent}
	}
	e := qs.events[qs.nextEvent]
	qs.events[qs.nextEvent] = QUICEvent{} // zero out references to data
	qs.nextEvent++
	return e
}

// Close closes the connection and stops any in-progress handshake.
func (q *QUICConn) Close() error {
	if q.conn.quic.cancel == nil {
		return nil // never started
	}
	q.conn.quic.cancel()
	for range q.conn.quic.blockedc {
		// Wait for the handshake goroutine to return.
	}
	return q.conn.handshakeErr
}

// HandleData handles handshake bytes received from the peer.
// It may produce connection events, which may be read with [QUICConn.NextEvent].
func (q *QUICConn) HandleData(level QUICEncryptionLevel, data []byte) error {
	c := q.conn
	if c.in.level != level {
		return quicError(c.in.setErrorLocked(errors.New("tls: handshake data received at wrong level")))
	}
	c.quic.readbuf = data
	<-c.quic.signalc
	_, ok := <-c.quic.blockedc
	if ok {
		// The handshake goroutine is waiting for more data.
		return nil
	}
	// The handshake goroutine has exited.
	c.handshakeMutex.Lock()
	defer c.handshakeMutex.Unlock()
	c.hand.Write(c.quic.readbuf)
	c.quic.readbuf = nil
	for q.conn.hand.Len() >= 4 && q.conn.handshakeErr == nil {
		b := q.conn.hand.Bytes()
		n := int(b[1])<<16 | int(b[2])<<8 | int(b[3])
		if n > maxHandshake {
			q.conn.handshakeErr = fmt.Errorf("tls: handshake message of length %d bytes exceeds maximum of %d bytes", n, maxHandshake)
			break
		}
		if len(b) < 4+n {
			return nil
		}
		if err := q.conn.handlePostHandshakeMessage(); err != nil {
			q.conn.handshakeErr = err
		}
	}
	if q.conn.handshakeErr != nil {
		return quicError(q.conn.handshakeErr)
	}
	return nil
}

type QUICSessionTicketOptions struct {
	// EarlyData specifies whether the ticket may be used for 0-RTT.
	EarlyData bool
	Extra     [][]byte
}

// SendSessionTicket sends a session ticket to the client.
// It produces connection events, which may be read with [QUICConn.NextEvent].
// Currently, it can only be called once.
func (q *QUICConn) SendSessionTicket(opts QUICSessionTicketOptions) error {
	c := q.conn
	if c.config.SessionTicketsDisabled {
		return nil
	}
	if !c.isHandshakeComplete.Load() {
		return quicError(errors.New("tls: SendSessionTicket called before handshake completed"))
	}
	if c.isClient {
		return quicError(errors.New("tls: SendSessionTicket called on the client"))
	}
	if q.sessionTicketSent {
		return quicError(errors.New("tls: SendSessionTicket called multiple times"))
	}
	q.sessionTicketSent = true
	return quicError(c.sendSessionTicket(opts.EarlyData, opts.Extra))
}

// StoreSession stores a session previously received in a QUICStoreSession event
// in the ClientSessionCache.
// The application may process additional events or modify the SessionState
// before storing the session.
func (q *QUICConn) StoreSession(session *SessionState) error {
	c := q.conn
	if !c.isClient {
		return quicError(errors.New("tls: StoreSessionTicket called on the server"))
	}
	cacheKey := c.clientSessionCacheKey()
	if cacheKey == "" {
		return nil
	}
	cs := &ClientSessionState{session: session}
	c.config.ClientSessionCache.Put(cacheKey, cs)
	return nil
}

// ConnectionState returns basic TLS details about the connection.
func (q *QUICConn) ConnectionState() ConnectionState {
	return q.conn.ConnectionState()
}

// SetTransportParameters sets the transport parameters to send to the peer.
//
// Server connections may delay setting the transport parameters until after
// receiving the client's transport parameters. See [QUICTransportParametersRequired].
func (q *QUICConn) SetTransportParameters(params []byte) {
	if params == nil {
		params = []byte{}
	}
	q.conn.quic.transportParams = params
	if q.conn.quic.started {
		<-q.conn.quic.signalc
		<-q.conn.quic.blockedc
	}
}

// quicError ensures err is an AlertError.
// If err is not already, quicError wraps it with alertInternalError.
func quicError(err error) error {
	if err == nil {
		return nil
	}
	if _, ok := errors.AsType[AlertError](err); ok {
		return err
	}
	a, ok := errors.AsType[alert](err)
	if !ok {
		a = alertInternalError
	}
	// Return an error wrapping the original error and an AlertError.
	// Truncate the text of the alert to 0 characters.
	return fmt.Errorf("%w%.0w", err, AlertError(a))
}

func (c *Conn) quicReadHandshakeBytes(n int) error {
	for c.hand.Len() < n {
		if err := c.quicWaitForSignal(); err != nil {
			return err
		}
	}
	return nil
}

func (c *Conn) quicSetReadSecret(level QUICEncryptionLevel, suite uint16, secret []byte) {
	c.quic.events = append(c.quic.events, QUICEvent{
		Kind:  QUICSetReadSecret,
		Level: level,
		Suite: suite,
		Data:  secret,
	})
}

func (c *Conn) quicSetWriteSecret(level QUICEncryptionLevel, suite uint16, secret []byte) {
	c.quic.events = append(c.quic.events, QUICEvent{
		Kind:  QUICSetWriteSecret,
		Level: level,
		Suite: suite,
		Data:  secret,
	})
}

func (c *Conn) quicWriteCryptoData(level QUICEncryptionLevel, data []byte) {
	var last *QUICEvent
	if len(c.quic.events) > 0 {
		last = &c.quic.events[len(c.quic.events)-1]
	}
	if last == nil || last.Kind != QUICWriteData || last.Level != level {
		c.quic.events = append(c.quic.events, QUICEvent{
			Kind:  QUICWriteData,
			Level: level,
		})
		last = &c.quic.events[len(c.quic.events)-1]
	}
	last.Data = append(last.Data, data...)
}

func (c *Conn) quicResumeSession(session *SessionState) error {
	c.quic.events = append(c.quic.events, QUICEvent{
		Kind:         QUICResumeSession,
		SessionState: session,
	})
	c.quic.waitingForDrain = true
	for c.quic.waitingForDrain {
		if err := c.quicWaitForSignal(); err != nil {
			return err
		}
	}
	return nil
}

func (c *Conn) quicStoreSession(session *SessionState) {
	c.quic.events = append(c.quic.events, QUICEvent{
		Kind:         QUICStoreSession,
		SessionState: session,
	})
}

func (c *Conn) quicSetTransportParameters(params []byte) {
	c.quic.events = append(c.quic.events, QUICEvent{
		Kind: QUICTransportParameters,
		Data: params,
	})
}

func (c *Conn) quicGetTransportParameters() ([]byte, error) {
	if c.quic.transportParams == nil {
		c.quic.events = append(c.quic.events, QUICEvent{
			Kind: QUICTransportParametersRequired,
		})
	}
	for c.quic.transportParams == nil {
		if err := c.quicWaitForSignal(); err != nil {
			return nil, err
		}
	}
	return c.quic.transportParams, nil
}

func (c *Conn) quicHandshakeComplete() {
	c.quic.events = append(c.quic.events, QUICEvent{
		Kind: QUICHandshakeDone,
	})
}

func (c *Conn) quicRejectedEarlyData() {
	c.quic.events = append(c.quic.events, QUICEvent{
		Kind: QUICRejectedEarlyData,
	})
}

// quicWaitForSignal notifies the QUICConn that handshake progress is blocked,
// and waits for a signal that the handshake should proceed.
//
// The handshake may become blocked waiting for handshake bytes
// or for the user to provide transport parameters.
func (c *Conn) quicWaitForSignal() error {
	// Drop the handshake mutex while blocked to allow the user
	// to call ConnectionState before the handshake completes.
	c.handshakeMutex.Unlock()
	defer c.handshakeMutex.Lock()
	// Send on blockedc to notify the QUICConn that the handshake is blocked.
	// Exported methods of QUICConn wait for the handshake to become blocked
	// before returning to the user.
	select {
	case c.quic.blockedc <- struct{}{}:
	case <-c.quic.cancelc:
		return c.sendAlertLocked(alertCloseNotify)
	}
	// The QUICConn reads from signalc to notify us that the handshake may
	// be able to proceed. (The QUICConn reads, because we close signalc to
	// indicate that the handshake has completed.)
	select {
	case c.quic.signalc <- struct{}{}:
		c.hand.Write(c.quic.readbuf)
		c.quic.readbuf = nil
	case <-c.quic.cancelc:
		return c.sendAlertLocked(alertCloseNotify)
	}
	return nil
}
