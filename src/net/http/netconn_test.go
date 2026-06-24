// Copyright 2024 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package http_test

import (
	"bytes"
	"context"
	"internal/synctest"
	"io"
	"math"
	"net"
	"net/netip"
	"os"
	"sync"
	"time"
)

func fakeNetListen() *fakeNetListener {
	li := &fakeNetListener{
		setc:    make(chan struct{}, 1),
		unsetc:  make(chan struct{}, 1),
		addr:    netip.MustParseAddrPort("127.0.0.1:8000"),
		locPort: 10000,
	}
	li.unsetc <- struct{}{}
	return li
}

type fakeNetListener struct {
	setc, unsetc chan struct{}
	queue        []net.Conn
	closed       bool
	addr         netip.AddrPort
	locPort      uint16

	onDial  func()             // called when making a new connection
	onClose func(*fakeNetConn) // called when closing a connection

	trackConns bool // set this to record all created conns
	conns      []*fakeNetConn
}

func (li *fakeNetListener) lock() {
	select {
	case <-li.setc:
	case <-li.unsetc:
	}
}

func (li *fakeNetListener) unlock() {
	if li.closed || len(li.queue) > 0 {
		li.setc <- struct{}{}
	} else {
		li.unsetc <- struct{}{}
	}
}

func (li *fakeNetListener) connect() *fakeNetConn {
	if li.onDial != nil {
		li.onDial()
	}
	li.lock()
	defer li.unlock()
	locAddr := netip.AddrPortFrom(netip.AddrFrom4([4]byte{127, 0, 0, 1}), li.locPort)
	li.locPort++
	c0, c1 := fakeNetPipe(li.addr, locAddr)
	c0.onClose = li.onClose
	c1.onClose = li.onClose
	li.queue = append(li.queue, c0)
	if li.trackConns {
		li.conns = append(li.conns, c0)
	}
	return c1
}

func (li *fakeNetListener) Accept() (net.Conn, error) {
	<-li.setc
	defer li.unlock()
	if li.closed {
		return nil, net.ErrClosed
	}
	c := li.queue[0]
	li.queue = li.queue[1:]
	return c, nil
}

func (li *fakeNetListener) Close() error {
	li.lock()
	defer li.unlock()
	li.closed = true
	return nil
}

func (li *fakeNetListener) Addr() net.Addr {
	return net.TCPAddrFromAddrPort(li.addr)
}

// fakeNetPipe creates an in-memory, full duplex network connection.
//
// Unlike net.Pipe, the connection is not synchronous.
// Writes are made to a buffer, and return immediately.
// By default, the buffer size is unlimited.
func fakeNetPipe(s1ap, s2ap netip.AddrPort) (r, w *fakeNetConn) {
	s1addr := net.TCPAddrFromAddrPort(s1ap)
	s2addr := net.TCPAddrFromAddrPort(s2ap)
	s1 := newSynctestNetConnHalf(s1addr)
	s2 := newSynctestNetConnHalf(s2addr)
	c1 := &fakeNetConn{loc: s1, rem: s2}
	c2 := &fakeNetConn{loc: s2, rem: s1}
	c1.peer = c2
	c2.peer = c1
	return c1, c2
}

// A fakeNetConn is one endpoint of the connection created by fakeNetPipe.
type fakeNetConn struct {
	// local and remote connection halves.
	// Each half contains a buffer.
	// Reads pull from the local buffer, and writes push to the remote buffer.
	loc, rem *fakeNetConnHalf

	// When set, synctest.Wait is automatically called before reads and after writes.
	autoWait bool

	// peer is the other endpoint.
	peer *fakeNetConn

	onClose func(*fakeNetConn) // called when closing
}

// Read reads data from the connection.
func (c *fakeNetConn) Read(b []byte) (n int, err error) {
	if c.autoWait {
		synctest.Wait()
	}
	return c.loc.read(b)
}

// Peek returns the available unread read buffer,
// without consuming its contents.
func (c *fakeNetConn) Peek() []byte {
	if c.autoWait {
		synctest.Wait()
	}
	return c.loc.peek()
}

// Write writes data to the connection.
func (c *fakeNetConn) Write(b []byte) (n int, err error) {
	if c.autoWait {
		defer synctest.Wait()
	}
	return c.rem.write(b)
}

// IsClosed reports whether the peer has closed its end of the connection.
func (c *fakeNetConn) IsClosedByPeer() bool {
	if c.autoWait {
		synctest.Wait()
	}
	c.rem.lock()
	defer c.rem.unlock()
	// If the remote half of the conn is returning ErrClosed,
	// the peer has closed the connection.
	return c.rem.readErr == net.ErrClosed
}

// Close closes the connection.
func (c *fakeNetConn) Close() error {
	if c.onClose != nil {
		c.onClose(c)
	}
	// Local half of the conn is now closed.
	c.loc.lock()
	c.loc.writeErr = net.ErrClosed
	c.loc.readErr = net.ErrClosed
	c.loc.buf.Reset()
	c.loc.unlock()
	// Remote half of the connection reads EOF after reading any remaining data.
	c.rem.lock()
	if c.rem.readErr == nil {
		c.rem.readErr = io.EOF
	}
	c.rem.writeErr = net.ErrClosed
	c.rem.unlock()
	if c.autoWait {
		synctest.Wait()
	}
	return nil
}

// LocalAddr returns the (fake) local network address.
func (c *fakeNetConn) LocalAddr() net.Addr {
	return c.loc.addr
}

// LocalAddr returns the (fake) remote network address.
func (c *fakeNetConn) RemoteAddr() net.Addr {
	return c.rem.addr
}

// SetDeadline sets the read and write deadlines for the connection.
func (c *fakeNetConn) SetDeadline(t time.Time) error {
	c.SetReadDeadline(t)
	c.SetWriteDeadline(t)
	return nil
}

// SetReadDeadline sets the read deadline for the connection.
func (c *fakeNetConn) SetReadDeadline(t time.Time) error {
	c.loc.rctx.setDeadline(t)
	return nil
}

// SetWriteDeadline sets the write deadline for the connection.
func (c *fakeNetConn) SetWriteDeadline(t time.Time) error {
	c.rem.wctx.setDeadline(t)
	return nil
}

// SetReadBufferSize sets the read buffer limit for the connection.
// Writes by the peer will block so long as the buffer is full.
func (c *fakeNetConn) SetReadBufferSize(size int) {
	c.loc.setReadBufferSize(size)
}

// fakeNetConnHalf is one data flow in the connection created by fakeNetPipe.
// Each half contains a buffer. Writes to the half push to the buffer, and reads pull from it.
type fakeNetConnHalf struct {
	addr net.Addr

	// Read and write timeouts.
	rctx, wctx deadlineContext

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

	bufMax   int // maximum buffer size
	buf      bytes.Buffer
	readErr  error // error returned by reads
	writeErr error // error returned by writes
}

func newSynctestNetConnHalf(addr net.Addr) *fakeNetConnHalf {
	h := &fakeNetConnHalf{
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
func (h *fakeNetConnHalf) lock() {
	select {
	case <-h.lockw: // writable
	case <-h.lockr: // readable
	case <-h.lockrw: // readable and writable
	case <-h.lockc: // neither readable nor writable
	}
}

// h unlocks h.
func (h *fakeNetConnHalf) unlock() {
	canRead := h.readErr != nil || h.buf.Len() > 0
	canWrite := h.writeErr != nil || h.bufMax > h.buf.Len()
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

// waitAndLockForRead waits until h is readable and locks it.
func (h *fakeNetConnHalf) waitAndLockForRead() error {
	// First a non-blocking select to see if we can make immediate progress.
	// This permits using a canceled context for a non-blocking operation.
	select {
	case <-h.lockr:
		return nil // readable
	case <-h.lockrw:
		return nil // readable and writable
	default:
	}
	ctx := h.rctx.context()
	select {
	case <-h.lockr:
		return nil // readable
	case <-h.lockrw:
		return nil // readable and writable
	case <-ctx.Done():
		return context.Cause(ctx)
	}
}

// waitAndLockForWrite waits until h is writable and locks it.
func (h *fakeNetConnHalf) waitAndLockForWrite() error {
	// First a non-blocking select to see if we can make immediate progress.
	// This permits using a canceled context for a non-blocking operation.
	select {
	case <-h.lockw:
		return nil // writable
	case <-h.lockrw:
		return nil // readable and writable
	default:
	}
	ctx := h.wctx.context()
	select {
	case <-h.lockw:
		return nil // writable
	case <-h.lockrw:
		return nil // readable and writable
	case <-ctx.Done():
		return context.Cause(ctx)
	}
}

func (h *fakeNetConnHalf) peek() []byte {
	h.lock()
	defer h.unlock()
	return h.buf.Bytes()
}

func (h *fakeNetConnHalf) read(b []byte) (n int, err error) {
	if err := h.waitAndLockForRead(); err != nil {
		return 0, err
	}
	defer h.unlock()
	if h.buf.Len() == 0 && h.readErr != nil {
		return 0, h.readErr
	}
	return h.buf.Read(b)
}

func (h *fakeNetConnHalf) setReadBufferSize(size int) {
	h.lock()
	defer h.unlock()
	h.bufMax = size
}

func (h *fakeNetConnHalf) write(b []byte) (n int, err error) {
	for n < len(b) {
		nn, err := h.writePartial(b[n:])
		n += nn
		if err != nil {
			return n, err
		}
	}
	return n, nil
}

func (h *fakeNetConnHalf) writePartial(b []byte) (n int, err error) {
	if err := h.waitAndLockForWrite(); err != nil {
		return 0, err
	}
	defer h.unlock()
	if h.writeErr != nil {
		return 0, h.writeErr
	}
	writeMax := h.bufMax - h.buf.Len()
	if writeMax < len(b) {
		b = b[:writeMax]
	}
	return h.buf.Write(b)
}

// deadlineContext converts a changable deadline (as in net.Conn.SetDeadline) into a Context.
type deadlineContext struct {
	mu     sync.Mutex
	ctx    context.Context
	cancel context.CancelCauseFunc
	timer  *time.Timer
}

// context returns a Context which expires when the deadline does.
func (t *deadlineContext) context() context.Context {
	t.mu.Lock()
	defer t.mu.Unlock()
	if t.ctx == nil {
		t.ctx, t.cancel = context.WithCancelCause(context.Background())
	}
	return t.ctx
}

// setDeadline sets the current deadline.
func (t *deadlineContext) setDeadline(deadline time.Time) {
	t.mu.Lock()
	defer t.mu.Unlock()
	// If t.ctx is non-nil and t.cancel is nil, then t.ctx was canceled
	// and we should create a new one.
	if t.ctx == nil || t.cancel == nil {
		t.ctx, t.cancel = context.WithCancelCause(context.Background())
	}
	// Stop any existing deadline from expiring.
	if t.timer != nil {
		t.timer.Stop()
	}
	if deadline.IsZero() {
		// No deadline.
		return
	}
	now := time.Now()
	if !deadline.After(now) {
		// Deadline has already expired.
		t.cancel(os.ErrDeadlineExceeded)
		t.cancel = nil
		return
	}
	if t.timer != nil {
		// Reuse existing deadline timer.
		t.timer.Reset(deadline.Sub(now))
		return
	}
	// Create a new timer to cancel the context at the deadline.
	t.timer = time.AfterFunc(deadline.Sub(now), func() {
		t.mu.Lock()
		defer t.mu.Unlock()
		t.cancel(os.ErrDeadlineExceeded)
		t.cancel = nil
	})
}
