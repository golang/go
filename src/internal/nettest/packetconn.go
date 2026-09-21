// Copyright 2026 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package nettest

import (
	"context"
	"errors"
	"internal/gate"
	"net"
	"os"
	"slices"
	"sync"
	"time"
)

// A PacketNet is a group of communicating [PacketConn]s.
type PacketNet struct {
	mu    sync.Mutex
	conns map[netAddr]*PacketConn
}

type netAddr struct {
	network string
	addr    string
}

// NewPacketNet returns a new PacketNet.
func NewPacketNet() *PacketNet {
	return &PacketNet{
		conns: make(map[netAddr]*PacketConn),
	}
}

// NewConn returns a new [PacketConn] listening on the given address.
// It returns an error if there is an existing listener on this address.
func (n *PacketNet) NewConn(a net.Addr) (*PacketConn, error) {
	n.mu.Lock()
	defer n.mu.Unlock()
	addrKey := netAddr{a.Network(), a.String()}
	if _, ok := n.conns[addrKey]; ok {
		return nil, &net.OpError{
			Op:   "listen",
			Net:  "udp",
			Addr: a,
			Err:  errors.New("address is in use"),
		}
	}
	p := &PacketConn{
		gate: gate.New(false),
		addr: a,
		net:  n,
	}
	n.conns[addrKey] = p
	return p, nil
}

type PacketConn struct {
	gate         gate.Gate
	queue        queue[*packet]
	closed       bool
	readErr      error
	writeErr     error
	closeErr     error
	readDeadline connDeadline

	net  *PacketNet
	addr net.Addr
}

type packet struct {
	b   []byte
	src net.Addr
}

// ReadFrom reads a packet from the connection, copying the payload into b.
func (p *PacketConn) ReadFrom(b []byte) (n int, addr net.Addr, err error) {
	p.gate.WaitAndLock(context.Background())
	defer p.unlock()

	switch {
	case p.closed:
		err = net.ErrClosed
	case p.readDeadline.expired:
		err = os.ErrDeadlineExceeded
	case p.queue.len() == 0 && p.readErr != nil:
		err = p.readErr
	}
	if err != nil {
		return 0, nil, &net.OpError{
			Op:   "read",
			Net:  "udp",
			Addr: p.addr,
			Err:  err,
		}
	}
	pkt := p.queue.pop()
	n = copy(b, pkt.b)
	return n, pkt.src, nil
}

// WriteTo writes a packet with payload b to addr.
// addr must be a [*net.UDPAddr].
//
// WriteTo appends the packet to the recipient's receive buffer.
// If no recipient is listening on addr or if the recipient's
// receive buffer is full, the packet is silently discarded.
func (p *PacketConn) WriteTo(b []byte, addr net.Addr) (n int, err error) {
	p.gate.Lock()
	switch {
	case p.closed:
		err = net.ErrClosed
	case p.writeErr != nil:
		err = p.writeErr
	}
	p.unlock()
	if err != nil {
		return 0, &net.OpError{
			Op:     "write",
			Net:    "udp",
			Source: p.addr,
			Addr:   addr,
			Err:    err,
		}
	}

	p.net.mu.Lock()
	dst := p.net.conns[netAddr{addr.Network(), addr.String()}]
	p.net.mu.Unlock()

	if dst == nil {
		// There is no PacketConn listening on the destination address,
		// and the packet falls silently into the void.
		return len(b), nil
	}

	dst.lock()
	if !dst.closed {
		dst.queue.push(&packet{b: slices.Clone(b), src: p.addr})
	}
	dst.unlock()
	return len(b), nil
}

// Close closes the connection.
func (p *PacketConn) Close() error {
	p.net.mu.Lock()
	delete(p.net.conns, netAddr{p.addr.Network(), p.addr.String()})
	p.net.mu.Unlock()

	p.lock()
	defer p.unlock()
	err := p.closeErr
	p.closed = true
	p.readErr = net.ErrClosed
	p.writeErr = net.ErrClosed
	p.closeErr = net.ErrClosed
	if err != nil {
		return &net.OpError{
			Op:   "close",
			Net:  "udp",
			Addr: p.addr,
			Err:  err,
		}
	}
	return err
}

// LocalAddr returns the (fake) local network address.
func (p *PacketConn) LocalAddr() net.Addr {
	p.lock()
	defer p.unlock()
	return p.addr
}

// SetReadDeadline sets the read deadline for the connection.
// PacketConns have no write deadline.
func (p *PacketConn) SetDeadline(t time.Time) error {
	return p.SetReadDeadline(t)
}

// SetReadDeadline sets the read deadline for the connection.
func (p *PacketConn) SetReadDeadline(t time.Time) error {
	p.readDeadline.setDeadline(p, t)
	return nil
}

// SetWriteDeadline has no effect.
// Writes to PacketConns never block.
func (p *PacketConn) SetWriteDeadline(t time.Time) error {
	return nil
}

// SetReadError causes any currently blocked and future ReadFrom calls to return
// a net.OpError wrapping err. It does not affect the other half of the connection.
// Reads will return any buffered data before returning the error,
// including data written after the error is set.
// A nil error restores the usual behavior.
func (c *PacketConn) SetReadError(err error) {
	c.lock()
	defer c.unlock()
	c.readErr = err
}

// SetWriteError causes any currently blocked and future WriteTo calls to return
// a net.OpError wrapping err. It does not affect the other half of the connection.
// Writes will not write data while an error is set.
// A nil error restores the usual behavior.
func (c *PacketConn) SetWriteError(err error) {
	c.lock()
	defer c.unlock()
	c.writeErr = err
}

// SetCloseError sets the error returned by Close.
// Close still closes the connection.
// A nil error restores the usual behavior.
func (c *PacketConn) SetCloseError(err error) {
	c.lock()
	defer c.unlock()
	c.closeErr = err
}

// CanRead reports whether [ReadFrom] can return at least one byte or an error.
// If [ReadFrom] would block, CanRead returns false.
func (p *PacketConn) CanRead() bool {
	p.lock()
	defer p.unlock()
	return p.canReadLocked()
}

func (p *PacketConn) canReadLocked() bool {
	return p.queue.len() > 0 || p.readDeadline.expired || p.closed || p.readErr != nil
}

// IsClosed reports whether the connection has been closed.
func (p *PacketConn) IsClosed() bool {
	p.lock()
	defer p.unlock()
	return p.closed
}

func (p *PacketConn) lock() {
	p.gate.Lock()
}

func (p *PacketConn) unlock() {
	p.gate.Unlock(p.canReadLocked())
}
