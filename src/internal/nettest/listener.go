// Copyright 2026 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package nettest

import (
	"context"
	"internal/gate"
	"net"
	"net/netip"
)

// Listener is an in-memory test implementation of net.Listener.
type Listener struct {
	gate      gate.Gate
	queue     queue[*Conn]
	closed    bool
	acceptErr error
	closeErr  error

	addr     net.Addr
	nextaddr netip.AddrPort
}

// NewListener returns a new Listener.
func NewListener() *Listener {
	return &Listener{
		gate:     gate.New(false),
		addr:     net.TCPAddrFromAddrPort(netip.MustParseAddrPort("127.0.0.1:1000")),
		nextaddr: netip.MustParseAddrPort("127.0.0.1:10001"),
	}
}

// Close closes the listener.
// Any blocked Accept operations will be unblocked and return errors.
func (li *Listener) Close() error {
	li.lock()
	defer li.unlock()
	err := li.closeErr
	li.closed = true
	li.acceptErr = net.ErrClosed
	li.closeErr = net.ErrClosed
	if err != nil {
		return &net.OpError{
			Op:   "close",
			Net:  "tcp",
			Addr: li.addr,
			Err:  err,
		}
	}
	return err
}

// Addr returns the listener's network address.
//
// The address is always a *net.TCPAddr.
func (li *Listener) Addr() net.Addr {
	li.lock()
	defer li.unlock()
	return li.addr
}

// SetAddr sets the listener's network address.
func (li *Listener) SetAddr(addr net.Addr) {
	li.lock()
	defer li.unlock()
	li.addr = addr
}

// NewConn returns a new connection to the listener.
//
// Accept will return the other side of the conn.
func (li *Listener) NewConn() *Conn {
	return li.NewConnConfig(func(*Conn) {})
}

// NewConnConfig returns a new connection to the listener.
//
// The function f is called with the new client connection.
// After f returns, Accept will return the other side of the connection.
//
// For example, to create a connection from a specific IP address:
//
//	conn := li.NewConnConfig(func(conn *nettest.Conn) {
//		conn.SetLocalAddr(net.TCPAddrFromAddrPort(netip.MustParseAddrPort("10.0.0.1:1234")))
//	})
func (li *Listener) NewConnConfig(f func(*Conn)) *Conn {
	li.lock()
	defer li.unlock()
	cli, srv := newConnPair(
		net.TCPAddrFromAddrPort(li.nextaddr),
		li.addr,
	)
	li.nextaddr = netip.AddrPortFrom(li.nextaddr.Addr(), li.nextaddr.Port()+1)
	f(cli)
	li.queue.push(srv)
	return cli
}

// Accept waits for and returns the next connection to the listener.
//
// The connections returned by Accept are always [*Conn]s.
func (li *Listener) Accept() (net.Conn, error) {
	li.gate.WaitAndLock(context.Background())
	defer li.unlock()
	if li.acceptErr != nil && li.queue.len() == 0 {
		return nil, &net.OpError{
			Op:   "accept",
			Net:  "tcp",
			Addr: li.addr,
			Err:  li.acceptErr,
		}
	}
	return li.queue.pop(), nil
}

// SetAcceptError causes any currently blocked and future Accept calls to return
// a net.OpError wrapping err.
// Accept will return any available connections before returning the error,
// including connections created after the error is set.
// A nil error restores the usual behavior.
func (li *Listener) SetAcceptError(err error) {
	li.gate.Lock()
	defer li.unlock()
	if !li.closed {
		li.acceptErr = err
	}
}

// SetCloseError sets the error returned by Close.
// Close still closes the listener.
// A nil error restores the usual behavior.
func (li *Listener) SetCloseError(err error) {
	li.gate.Lock()
	defer li.unlock()
	if !li.closed {
		li.closeErr = err
	}
}

func (li *Listener) lock() {
	li.gate.Lock()
}

func (li *Listener) unlock() {
	li.gate.Unlock(li.acceptErr != nil || li.queue.len() > 0)
}
