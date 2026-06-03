// Copyright 2026 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package nettest

import (
	"bytes"
	"errors"
	"io"
	"math"
	"net"
	"net/netip"
	"os"
	"time"
)

// Conn is an in-memory test implementation of net.Conn.
type Conn struct {
	// Conns come in pairs.
	// Writes to one Conn are read by its peer, and vice-versa.
	//
	// A connHalf handles one direction of data flow.
	// A Conn consists of read and write halves.
	// A Conn's peer has the same halves, only swapped.
	//
	// A Conn reads from r and writes to w.
	r, w *connHalf

	// peer is the other endpoint.
	peer *Conn
}

// NewConnPair returns a pair of connected Conns.
func NewConnPair() (*Conn, *Conn) {
	return newConnPair(
		net.TCPAddrFromAddrPort(netip.MustParseAddrPort("127.0.0.1:10000")),
		net.TCPAddrFromAddrPort(netip.MustParseAddrPort("127.0.0.1:10001")),
	)
}

func newConnPair(addr1, addr2 net.Addr) (*Conn, *Conn) {
	h1 := newConnHalf(addr1)
	h2 := newConnHalf(addr2)
	c1 := &Conn{r: h1, w: h2}
	c2 := &Conn{r: h2, w: h1}
	c1.peer = c2
	c2.peer = c1
	c1.SetReadBufferSize(-1)
	c2.SetReadBufferSize(-1)
	return c1, c2
}

// Peer returns the other end of the connection.
func (c *Conn) Peer() *Conn {
	return c.peer
}

// Read reads data from the connection.
func (c *Conn) Read(b []byte) (n int, err error) {
	n, err = c.r.read(b)
	if err != nil && err != io.EOF {
		err = &net.OpError{
			Op:     "read",
			Net:    "tcp",
			Source: c.RemoteAddr(),
			Addr:   c.LocalAddr(),
			Err:    err,
		}
	}
	return n, err
}

// CanRead reports whether Read can proceed without blocking.
func (c *Conn) CanRead() bool {
	return c.r.canRead()
}

// Write writes data to the connection.
func (c *Conn) Write(b []byte) (n int, err error) {
	n, err = c.w.write(b)
	if err != nil {
		err = &net.OpError{
			Op:     "write",
			Net:    "tcp",
			Source: c.LocalAddr(),
			Addr:   c.RemoteAddr(),
			Err:    err,
		}
	}
	return n, err
}

// IsClosed reports whether the connection has been closed.
// A connection is closed if [CloseRead] and [CloseWrite] are both called,
// or if [Close] is called.
//
// To identify when the other side of the Conn has been closed,
// use Conn.Peer().IsClosed().
func (c *Conn) IsClosed() bool {
	c.r.lock()
	readClosed := c.r.readClosed
	c.r.unlock()
	c.w.lock()
	writeClosed := c.w.writeClosed
	c.w.unlock()
	return readClosed && writeClosed
}

var errClosedByPeer = errors.New("connection closed by peer")

// CloseRead shuts down the reading side of the connection.
func (c *Conn) CloseRead() error {
	c.r.lock()
	defer c.r.unlock()
	c.r.buf.Reset() // discard unread data
	c.r.readClosed = true
	return nil
}

// CloseWrite shuts down the writing side of the connection.
func (c *Conn) CloseWrite() error {
	c.w.lock()
	defer c.w.unlock()
	c.w.writeClosed = true
	return nil
}

// Close closes the connection.
func (c *Conn) Close() error {
	c.r.lock()
	readClosed := c.r.readClosed
	c.r.buf.Reset() // discard unread data
	c.r.readClosed = true
	err := c.r.closeErr
	c.r.unlock()

	c.w.lock()
	writeClosed := c.w.writeClosed
	c.w.writeClosed = true
	c.w.unlock()

	if readClosed && writeClosed {
		err = net.ErrClosed
	}
	if err != nil {
		err = &net.OpError{
			Op:   "close",
			Net:  "tcp",
			Addr: c.LocalAddr(),
			Err:  err,
		}
	}
	return err
}

// SetCloseError sets the error returned by Close.
// Close still closes the connection.
// A nil error restores the usual behavior.
func (c *Conn) SetCloseError(err error) {
	c.r.lock()
	c.r.closeErr = err
	c.r.unlock()
}

// LocalAddr returns the (fake) local network address.
func (c *Conn) LocalAddr() net.Addr {
	c.r.lock()
	defer c.r.unlock()
	return c.r.addr
}

// SetLocalAddr sets the local address.
//
// To set the remote address, set the local address of Conn's peer.
func (c *Conn) SetLocalAddr(addr net.Addr) {
	c.r.lock()
	defer c.r.unlock()
	c.r.addr = addr
}

// LocalAddr returns the (fake) remote network address.
func (c *Conn) RemoteAddr() net.Addr {
	c.r.lock()
	defer c.r.unlock()
	return c.w.addr
}

// SetDeadline sets the read and write deadlines for the connection.
func (c *Conn) SetDeadline(t time.Time) error {
	c.SetReadDeadline(t)
	c.SetWriteDeadline(t)
	return nil
}

// SetReadDeadline sets the read deadline for the connection.
func (c *Conn) SetReadDeadline(t time.Time) error {
	c.r.readDeadline.setDeadline(c.r, t)
	return nil
}

// SetWriteDeadline sets the write deadline for the connection.
func (c *Conn) SetWriteDeadline(t time.Time) error {
	c.w.writeDeadline.setDeadline(c.w, t)
	return nil
}

// SetReadBufferSize sets the connection's read buffer.
// Writes to the other end of the connection will block so long as the buffer is full.
// Setting the size to 0 blocks all writes until the size is increased.
func (c *Conn) SetReadBufferSize(size int) {
	if size < 0 {
		size = math.MaxInt
	}
	c.r.setBufferSize(size)
}

// SetReadError causes any currently blocked and future Read calls to return
// a net.OpError wrapping err. It does not affect the other half of the connection.
// Reads will return any buffered data before returning the error,
// including data written after the error is set and io.EOF after the other end is closed.
// A nil error restores the usual behavior.
func (c *Conn) SetReadError(err error) {
	c.r.lock()
	defer c.r.unlock()
	c.r.readErr = err
}

// SetWriteError causes any currently blocked and future Write calls to return
// a net.OpError wrapping err. It does not affect the other half of the connection.
// Writes will not write data to the connection buffer while an error is set.
// A nil error restores the usual behavior.
func (c *Conn) SetWriteError(err error) {
	c.w.lock()
	defer c.w.unlock()
	c.w.writeErr = err
}

// connHalf is one direction data flow in a Conn.
// The connHalf contains a buffer.
// Writes to the connHalf push to the buffer and reads pull from it.
type connHalf struct {
	addr net.Addr

	// A half can be readable and/or writable.
	//
	// These four channels act as a lock,
	// and allow waiting for readability/writability.
	// When the half is unlocked, exactly one channel contains a value.
	// When the half is locked, all channels are empty.
	lockr  chan struct{} // readable
	lockw  chan struct{} // writable
	lockrw chan struct{} // readable and writable
	lockc  chan struct{} // neither readable nor writable

	// Read and write timeouts.
	readDeadline, writeDeadline connDeadline

	bufMax int // maximum buffer size
	buf    bytes.Buffer

	readClosed, writeClosed bool
	readErr, writeErr       error // errors returned by reads/writes
	closeErr                error // error returned by closing the conn reading from this half
}

func newConnHalf(addr net.Addr) *connHalf {
	h := &connHalf{
		addr:   addr,
		lockw:  make(chan struct{}, 1),
		lockr:  make(chan struct{}, 1),
		lockrw: make(chan struct{}, 1),
		lockc:  make(chan struct{}, 1),
		bufMax: math.MaxInt, // unlimited
	}
	h.unlock()
	return h
}

// lock locks h.
func (h *connHalf) lock() {
	select {
	case <-h.lockw: // writable
	case <-h.lockr: // readable
	case <-h.lockrw: // readable and writable
	case <-h.lockc: // neither readable nor writable
	}
}

// unlock unlocks h.
func (h *connHalf) unlock() {
	canRead := h.canReadLocked()
	canWrite := h.canWriteLocked()
	switch {
	case canRead && canWrite:
		h.lockrw <- struct{}{} // readable and writable
	case canRead:
		h.lockr <- struct{}{} // readable
	case canWrite:
		h.lockw <- struct{}{} // writable
	default:
		h.lockc <- struct{}{} // neither readable nor writable
	}
}

func (h *connHalf) canRead() bool {
	h.lock()
	defer h.unlock()
	return h.canReadLocked()
}

func (h *connHalf) canReadLocked() bool {
	return h.readErr != nil || h.readDeadline.expired || h.buf.Len() > 0 || h.readClosed || h.writeClosed
}

func (h *connHalf) canWriteLocked() bool {
	return h.writeErr != nil || h.writeDeadline.expired || h.bufMax > h.buf.Len() || h.readClosed || h.writeClosed
}

// waitAndLockForRead waits until h is readable and locks it.
func (h *connHalf) waitAndLockForRead() {
	select {
	case <-h.lockr:
		// readable
	case <-h.lockrw:
		// readable and writable
	}
}

// waitAndLockForWrite waits until h is writable and locks it.
func (h *connHalf) waitAndLockForWrite() {
	select {
	case <-h.lockw:
		// writable
	case <-h.lockrw:
		// readable and writable
	}
}

func (h *connHalf) read(b []byte) (n int, err error) {
	h.waitAndLockForRead()
	defer h.unlock()
	if h.readClosed {
		return 0, net.ErrClosed
	}
	if h.readDeadline.expired {
		return 0, os.ErrDeadlineExceeded
	}
	if h.buf.Len() > 0 {
		return h.buf.Read(b)
	}
	if h.writeClosed {
		return 0, io.EOF
	}
	return 0, h.readErr
}

func (h *connHalf) setBufferSize(size int) {
	h.lock()
	defer h.unlock()
	h.bufMax = size
}

func (h *connHalf) write(b []byte) (n int, err error) {
	for n < len(b) {
		nn, err := h.writePartial(b[n:])
		n += nn
		if err != nil {
			return n, err
		}
	}
	return n, nil
}

func (h *connHalf) writePartial(b []byte) (n int, err error) {
	h.waitAndLockForWrite()
	defer h.unlock()
	if h.writeClosed {
		return 0, net.ErrClosed
	}
	if h.writeDeadline.expired {
		return 0, os.ErrDeadlineExceeded
	}
	if h.readClosed {
		return 0, errClosedByPeer
	}
	if h.writeErr != nil {
		return 0, h.writeErr
	}
	writeMax := h.bufMax - h.buf.Len()
	if writeMax < len(b) {
		b = b[:writeMax]
	}
	return h.buf.Write(b)
}

type connDeadline struct {
	timer   *time.Timer
	expired bool
}

type locker interface {
	lock()
	unlock()
}

func (d *connDeadline) setDeadline(mu locker, t time.Time) {
	mu.lock()
	defer mu.unlock()
	if d.timer != nil {
		d.timer.Stop()
		d.timer = nil
	}
	if t.IsZero() {
		// No deadline.
		d.expired = false
		return
	}
	expiry := time.Until(t)
	if expiry <= 0 {
		// Deadline has already passed.
		d.expired = true
		return
	}
	// Deadline is in the future.
	d.expired = false
	var timer *time.Timer
	timer = time.AfterFunc(expiry, func() {
		mu.lock()
		defer mu.unlock()
		if d.timer == timer {
			d.timer = nil
			d.expired = true
		}
	})
	d.timer = timer
}
