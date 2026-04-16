// Copyright 2023 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package quic

import (
	"context"
	"crypto/rand"
	"errors"
	"net"
	"net/netip"
	"sync"
	"sync/atomic"
	"time"
)

// An Endpoint handles QUIC traffic on a network address.
// It can accept inbound connections or create outbound ones.
//
// Multiple goroutines may invoke methods on an Endpoint simultaneously.
type Endpoint struct {
	listenConfig *Config
	packetConn   packetConn
	testHooks    endpointTestHooks
	resetGen     statelessResetTokenGenerator
	retry        retryState

	acceptQueue queue[*Conn] // new inbound connections
	connsMap    connsMap     // only accessed by the listen loop

	connsMu sync.Mutex
	conns   map[*Conn]struct{}
	closing bool          // set when Close is called
	closec  chan struct{} // closed when the listen loop exits
}

type endpointTestHooks interface {
	newConn(c *Conn)
}

// A packetConn is the interface to sending and receiving UDP packets.
type packetConn interface {
	Close() error
	LocalAddr() netip.AddrPort
	Read(f func(*datagram))
	Write(datagram) error
}

// Listen listens on a local network address.
//
// The config is used to for connections accepted by the endpoint.
// If the config is nil, the endpoint will not accept connections.
func Listen(network, address string, listenConfig *Config) (*Endpoint, error) {
	if listenConfig != nil && listenConfig.TLSConfig == nil {
		return nil, errors.New("TLSConfig is not set")
	}
	a, err := net.ResolveUDPAddr(network, address)
	if err != nil {
		return nil, err
	}
	udpConn, err := net.ListenUDP(network, a)
	if err != nil {
		return nil, err
	}
	pc, err := newNetUDPConn(udpConn)
	if err != nil {
		return nil, err
	}
	return newEndpoint(pc, listenConfig, nil)
}

// NewEndpoint creates an endpoint using a net.PacketConn as the underlying transport.
//
// If the PacketConn is not a *net.UDPConn, the endpoint may be slower and lack
// access to some features of the network.
func NewEndpoint(conn net.PacketConn, config *Config) (*Endpoint, error) {
	var pc packetConn
	var err error
	switch conn := conn.(type) {
	case *net.UDPConn:
		pc, err = newNetUDPConn(conn)
	default:
		pc, err = newNetPacketConn(conn)
	}
	if err != nil {
		return nil, err
	}
	return newEndpoint(pc, config, nil)
}

func newEndpoint(pc packetConn, config *Config, hooks endpointTestHooks) (*Endpoint, error) {
	e := &Endpoint{
		listenConfig: config,
		packetConn:   pc,
		testHooks:    hooks,
		conns:        make(map[*Conn]struct{}),
		acceptQueue:  newQueue[*Conn](),
		closec:       make(chan struct{}),
	}
	var statelessResetKey [32]byte
	if config != nil {
		statelessResetKey = config.StatelessResetKey
	}
	e.resetGen.init(statelessResetKey)
	e.connsMap.init()
	if config != nil && config.RequireAddressValidation {
		if err := e.retry.init(); err != nil {
			return nil, err
		}
	}
	go e.listen()
	return e, nil
}

// LocalAddr returns the local network address.
func (e *Endpoint) LocalAddr() netip.AddrPort {
	return e.packetConn.LocalAddr()
}

// Close closes the Endpoint.
// Any blocked operations on the Endpoint or associated Conns and Stream will be unblocked
// and return errors.
//
// Close aborts every open connection.
// Data in stream read and write buffers is discarded.
// It waits for the peers of any open connection to acknowledge the connection has been closed.
func (e *Endpoint) Close(ctx context.Context) error {
	e.acceptQueue.close(errors.New("endpoint closed"))

	// It isn't safe to call Conn.Abort or conn.exit with connsMu held,
	// so copy the list of conns.
	var conns []*Conn
	e.connsMu.Lock()
	if !e.closing {
		e.closing = true // setting e.closing prevents new conns from being created
		for c := range e.conns {
			conns = append(conns, c)
		}
		if len(e.conns) == 0 {
			e.packetConn.Close()
		}
	}
	e.connsMu.Unlock()

	for _, c := range conns {
		c.Abort(localTransportError{code: errNo})
	}
	select {
	case <-e.closec:
	case <-ctx.Done():
		for _, c := range conns {
			c.exit()
		}
		return ctx.Err()
	}
	return nil
}

// Accept waits for and returns the next connection.
func (e *Endpoint) Accept(ctx context.Context) (*Conn, error) {
	return e.acceptQueue.get(ctx)
}

// Dial creates and returns a connection to a network address.
// The config cannot be nil.
func (e *Endpoint) Dial(ctx context.Context, network, address string, config *Config) (*Conn, error) {
	u, err := net.ResolveUDPAddr(network, address)
	if err != nil {
		return nil, err
	}
	addr := u.AddrPort()
	addr = netip.AddrPortFrom(addr.Addr().Unmap(), addr.Port())
	c, err := e.newConn(time.Now(), config, clientSide, newServerConnIDs{}, address, addr)
	if err != nil {
		return nil, err
	}
	if err := c.waitReady(ctx); err != nil {
		c.Abort(nil)
		return nil, err
	}
	return c, nil
}

func (e *Endpoint) newConn(now time.Time, config *Config, side connSide, cids newServerConnIDs, peerHostname string, peerAddr netip.AddrPort) (*Conn, error) {
	e.connsMu.Lock()
	defer e.connsMu.Unlock()
	if e.closing {
		return nil, errors.New("endpoint closed")
	}
	c, err := newConn(now, side, cids, peerHostname, peerAddr, config, e)
	if err != nil {
		return nil, err
	}
	e.conns[c] = struct{}{}
	return c, nil
}

// serverConnEstablished is called by a conn when the handshake completes
// for an inbound (serverSide) connection.
func (e *Endpoint) serverConnEstablished(c *Conn) {
	e.acceptQueue.put(c)
}

// connDrained is called by a conn when it leaves the draining state,
// either when the peer acknowledges connection closure or the drain timeout expires.
func (e *Endpoint) connDrained(c *Conn) {
	var cids [][]byte
	for i := range c.connIDState.local {
		cids = append(cids, c.connIDState.local[i].cid)
	}
	var tokens []statelessResetToken
	for i := range c.connIDState.remote {
		tokens = append(tokens, c.connIDState.remote[i].resetToken)
	}
	e.connsMap.updateConnIDs(func(conns *connsMap) {
		for _, cid := range cids {
			conns.retireConnID(c, cid)
		}
		for _, token := range tokens {
			conns.retireResetToken(c, token)
		}
	})
	e.connsMu.Lock()
	defer e.connsMu.Unlock()
	delete(e.conns, c)
	if e.closing && len(e.conns) == 0 {
		e.packetConn.Close()
	}
}

func (e *Endpoint) listen() {
	defer close(e.closec)
	e.packetConn.Read(func(m *datagram) {
		if e.connsMap.updateNeeded.Load() {
			e.connsMap.applyUpdates()
		}
		e.handleDatagram(m)
	})
}

func (e *Endpoint) handleDatagram(m *datagram) {
	dstConnID, ok := dstConnIDForDatagram(m.b)
	if !ok {
		m.recycle()
		return
	}
	c := e.connsMap.byConnID[string(dstConnID)]
	if c == nil {
		// TODO: Move this branch into a separate goroutine to avoid blocking
		// the endpoint while processing packets.
		e.handleUnknownDestinationDatagram(m)
		return
	}

	// TODO: This can block the endpoint while waiting for the conn to accept the dgram.
	// Think about buffering between the receive loop and the conn.
	c.sendMsg(m)
}

func (e *Endpoint) handleUnknownDestinationDatagram(m *datagram) {
	defer func() {
		if m != nil {
			m.recycle()
		}
	}()
	const minimumValidPacketSize = 21
	if len(m.b) < minimumValidPacketSize {
		return
	}
	now := time.Now()
	// Check to see if this is a stateless reset.
	var token statelessResetToken
	copy(token[:], m.b[len(m.b)-len(token):])
	if c := e.connsMap.byResetToken[token]; c != nil {
		c.sendMsg(func(now time.Time, c *Conn) {
			c.handleStatelessReset(now, token)
		})
		return
	}
	// If this is a 1-RTT packet, there's nothing productive we can do with it.
	// Send a stateless reset if possible.
	if !isLongHeader(m.b[0]) {
		e.maybeSendStatelessReset(m.b, m.peerAddr)
		return
	}
	p, ok := parseGenericLongHeaderPacket(m.b)
	if !ok || len(m.b) < paddedInitialDatagramSize {
		return
	}
	switch p.version {
	case quicVersion1:
	case 0:
		// Version Negotiation for an unknown connection.
		return
	default:
		// Unknown version.
		e.sendVersionNegotiation(p, m.peerAddr)
		return
	}
	if getPacketType(m.b) != packetTypeInitial {
		// This packet isn't trying to create a new connection.
		// It might be associated with some connection we've lost state for.
		// We are technically permitted to send a stateless reset for
		// a long-header packet, but this isn't generally useful. See:
		// https://www.rfc-editor.org/rfc/rfc9000#section-10.3-16
		return
	}
	if e.listenConfig == nil {
		// We are not configured to accept connections.
		return
	}
	cids := newServerConnIDs{
		srcConnID: p.srcConnID,
		dstConnID: p.dstConnID,
	}
	if e.listenConfig.RequireAddressValidation {
		var ok bool
		cids.retrySrcConnID = p.dstConnID
		cids.originalDstConnID, ok = e.validateInitialAddress(now, p, m.peerAddr)
		if !ok {
			return
		}
	} else {
		cids.originalDstConnID = p.dstConnID
	}
	var err error
	c, err := e.newConn(now, e.listenConfig, serverSide, cids, "", m.peerAddr)
	if err != nil {
		// The accept queue is probably full.
		// We could send a CONNECTION_CLOSE to the peer to reject the connection.
		// Currently, we just drop the datagram.
		// https://www.rfc-editor.org/rfc/rfc9000.html#section-5.2.2-5
		return
	}
	c.sendMsg(m)
	m = nil // don't recycle, sendMsg takes ownership
}

func (e *Endpoint) maybeSendStatelessReset(b []byte, peerAddr netip.AddrPort) {
	if !e.resetGen.canReset {
		// Config.StatelessResetKey isn't set, so we don't send stateless resets.
		return
	}
	// The smallest possible valid packet a peer can send us is:
	//   1 byte of header
	//   connIDLen bytes of destination connection ID
	//   1 byte of packet number
	//   1 byte of payload
	//   16 bytes AEAD expansion
	if len(b) < 1+connIDLen+1+1+16 {
		return
	}
	// TODO: Rate limit stateless resets.
	cid := b[1:][:connIDLen]
	token := e.resetGen.tokenForConnID(cid)
	// We want to generate a stateless reset that is as short as possible,
	// but long enough to be difficult to distinguish from a 1-RTT packet.
	//
	// The minimal 1-RTT packet is:
	//   1 byte of header
	//   0-20 bytes of destination connection ID
	//   1-4 bytes of packet number
	//   1 byte of payload
	//   16 bytes AEAD expansion
	//
	// Assuming the maximum possible connection ID and packet number size,
	// this gives 1 + 20 + 4 + 1 + 16 = 42 bytes.
	//
	// We also must generate a stateless reset that is shorter than the datagram
	// we are responding to, in order to ensure that reset loops terminate.
	//
	// See: https://www.rfc-editor.org/rfc/rfc9000#section-10.3
	size := min(len(b)-1, 42)
	// Reuse the input buffer for generating the stateless reset.
	b = b[:size]
	rand.Read(b[:len(b)-statelessResetTokenLen])
	b[0] &^= headerFormLong // clear long header bit
	b[0] |= fixedBit        // set fixed bit
	copy(b[len(b)-statelessResetTokenLen:], token[:])
	e.sendDatagram(datagram{
		b:        b,
		peerAddr: peerAddr,
	})
}

func (e *Endpoint) sendVersionNegotiation(p genericLongPacket, peerAddr netip.AddrPort) {
	m := newDatagram()
	m.b = appendVersionNegotiation(m.b[:0], p.srcConnID, p.dstConnID, quicVersion1)
	m.peerAddr = peerAddr
	e.sendDatagram(*m)
	m.recycle()
}

func (e *Endpoint) sendConnectionClose(in genericLongPacket, peerAddr netip.AddrPort, code transportError) {
	keys := initialKeys(in.dstConnID, serverSide)
	var w packetWriter
	p := longPacket{
		ptype:     packetTypeInitial,
		version:   quicVersion1,
		num:       0,
		dstConnID: in.srcConnID,
		srcConnID: in.dstConnID,
	}
	const pnumMaxAcked = 0
	w.reset(paddedInitialDatagramSize)
	w.startProtectedLongHeaderPacket(pnumMaxAcked, p)
	w.appendConnectionCloseTransportFrame(code, 0, "")
	w.finishProtectedLongHeaderPacket(pnumMaxAcked, keys.w, p)
	buf := w.datagram()
	if len(buf) == 0 {
		return
	}
	e.sendDatagram(datagram{
		b:        buf,
		peerAddr: peerAddr,
	})
}

func (e *Endpoint) sendDatagram(dgram datagram) error {
	return e.packetConn.Write(dgram)
}

// A connsMap is an endpoint's mapping of conn ids and reset tokens to conns.
type connsMap struct {
	byConnID     map[string]*Conn
	byResetToken map[statelessResetToken]*Conn

	updateMu     sync.Mutex
	updateNeeded atomic.Bool
	updates      []func(*connsMap)
}

func (m *connsMap) init() {
	m.byConnID = map[string]*Conn{}
	m.byResetToken = map[statelessResetToken]*Conn{}
}

func (m *connsMap) addConnID(c *Conn, cid []byte) {
	m.byConnID[string(cid)] = c
}

func (m *connsMap) retireConnID(c *Conn, cid []byte) {
	delete(m.byConnID, string(cid))
}

func (m *connsMap) addResetToken(c *Conn, token statelessResetToken) {
	m.byResetToken[token] = c
}

func (m *connsMap) retireResetToken(c *Conn, token statelessResetToken) {
	delete(m.byResetToken, token)
}

func (m *connsMap) updateConnIDs(f func(*connsMap)) {
	m.updateMu.Lock()
	defer m.updateMu.Unlock()
	m.updates = append(m.updates, f)
	m.updateNeeded.Store(true)
}

// applyUpdates is called by the datagram receive loop to update its connection ID map.
func (m *connsMap) applyUpdates() {
	m.updateMu.Lock()
	defer m.updateMu.Unlock()
	for _, f := range m.updates {
		f(m)
	}
	clear(m.updates)
	m.updates = m.updates[:0]
	m.updateNeeded.Store(false)
}
