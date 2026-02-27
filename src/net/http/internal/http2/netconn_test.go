// Copyright 2024 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package http2_test

import (
	"bytes"
	"context"
	"errors"
	"io"
	"math"
	"net"
	"net/netip"
	"os"
	"sync"
	"testing/synctest"
	"time"
)

// synctestNetPipe creates an in-memory, full duplex network connection.
// Read and write timeouts are managed by the synctest group.
//
// Unlike net.Pipe, the connection is not synchronous.
// Writes are made to a buffer, and return immediately.
// By default, the buffer size is unlimited.
func synctestNetPipe() (r, w *synctestNetConn) {
	s1addr := net.TCPAddrFromAddrPort(netip.MustParseAddrPort("127.0.0.1:8000"))
	s2addr := net.TCPAddrFromAddrPort(netip.MustParseAddrPort("127.0.0.1:8001"))
	s1 := newSynctestNetConnHalf(s1addr)
	s2 := newSynctestNetConnHalf(s2addr)
	r = &synctestNetConn{loc: s1, rem: s2}
	w = &synctestNetConn{loc: s2, rem: s1}
	r.peer = w
	w.peer = r
	return r, w
}

// A synctestNetConn is one endpoint of the connection created by synctestNetPipe.
type synctestNetConn struct {
	// local and remote connection halves.
	// Each half contains a buffer.
	// Reads pull from the local buffer, and writes push to the remote buffer.
	loc, rem *synctestNetConnHalf

	// When set, group.Wait is automatically called before reads and after writes.
	autoWait bool

	// peer is the other endpoint.
	peer *synctestNetConn
}

// Read reads data from the connection.
func (c *synctestNetConn) Read(b []byte) (n int, err error) {
	if c.autoWait {
		synctest.Wait()
	}
	return c.loc.read(b)
}

// Peek returns the available unread read buffer,
// without consuming its contents.
func (c *synctestNetConn) Peek() []byte {
	if c.autoWait {
		synctest.Wait()
	}
	return c.loc.peek()
}

// Write writes data to the connection.
func (c *synctestNetConn) Write(b []byte) (n int, err error) {
	if c.autoWait {
		defer synctest.Wait()
	}
	return c.rem.write(b)
}

// IsClosedByPeer reports whether the peer has closed its end of the connection.
func (c *synctestNetConn) IsClosedByPeer() bool {
	if c.autoWait {
		synctest.Wait()
	}
	return c.loc.isClosedByPeer()
}

// Close closes the connection.
func (c *synctestNetConn) Close() error {
	c.loc.setWriteError(errors.New("connection closed by peer"))
	c.rem.setReadError(io.EOF)
	if c.autoWait {
		synctest.Wait()
	}
	return nil
}

// LocalAddr returns the (fake) local network address.
func (c *synctestNetConn) LocalAddr() net.Addr {
	return c.loc.addr
}

// RemoteAddr returns the (fake) remote network address.
func (c *synctestNetConn) RemoteAddr() net.Addr {
	return c.rem.addr
}

// SetDeadline sets the read and write deadlines for the connection.
func (c *synctestNetConn) SetDeadline(t time.Time) error {
	c.SetReadDeadline(t)
	c.SetWriteDeadline(t)
	return nil
}

// SetReadDeadline sets the read deadline for the connection.
func (c *synctestNetConn) SetReadDeadline(t time.Time) error {
	c.loc.rctx.setDeadline(t)
	return nil
}

// SetWriteDeadline sets the write deadline for the connection.
func (c *synctestNetConn) SetWriteDeadline(t time.Time) error {
	c.rem.wctx.setDeadline(t)
	return nil
}

// SetReadBufferSize sets the read buffer limit for the connection.
// Writes by the peer will block so long as the buffer is full.
func (c *synctestNetConn) SetReadBufferSize(size int) {
	c.loc.setReadBufferSize(size)
}

// synctestNetConnHalf is one data flow in the connection created by synctestNetPipe.
// Each half contains a buffer. Writes to the half push to the buffer, and reads pull from it.
type synctestNetConnHalf struct {
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

func newSynctestNetConnHalf(addr net.Addr) *synctestNetConnHalf {
	h := &synctestNetConnHalf{
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

func (h *synctestNetConnHalf) lock() {
	select {
	case <-h.lockw:
	case <-h.lockr:
	case <-h.lockrw:
	case <-h.lockc:
	}
}

func (h *synctestNetConnHalf) unlock() {
	canRead := h.readErr != nil || h.buf.Len() > 0
	canWrite := h.writeErr != nil || h.bufMax > h.buf.Len()
	switch {
	case canRead && canWrite:
		h.lockrw <- struct{}{}
	case canRead:
		h.lockr <- struct{}{}
	case canWrite:
		h.lockw <- struct{}{}
	default:
		h.lockc <- struct{}{}
	}
}

func (h *synctestNetConnHalf) readWaitAndLock() error {
	select {
	case <-h.lockr:
		return nil
	case <-h.lockrw:
		return nil
	default:
	}
	ctx := h.rctx.context()
	select {
	case <-h.lockr:
		return nil
	case <-h.lockrw:
		return nil
	case <-ctx.Done():
		return context.Cause(ctx)
	}
}

func (h *synctestNetConnHalf) writeWaitAndLock() error {
	select {
	case <-h.lockw:
		return nil
	case <-h.lockrw:
		return nil
	default:
	}
	ctx := h.wctx.context()
	select {
	case <-h.lockw:
		return nil
	case <-h.lockrw:
		return nil
	case <-ctx.Done():
		return context.Cause(ctx)
	}
}

func (h *synctestNetConnHalf) peek() []byte {
	h.lock()
	defer h.unlock()
	return h.buf.Bytes()
}

func (h *synctestNetConnHalf) isClosedByPeer() bool {
	h.lock()
	defer h.unlock()
	return h.readErr != nil
}

func (h *synctestNetConnHalf) read(b []byte) (n int, err error) {
	if err := h.readWaitAndLock(); err != nil {
		return 0, err
	}
	defer h.unlock()
	if h.buf.Len() == 0 && h.readErr != nil {
		return 0, h.readErr
	}
	return h.buf.Read(b)
}

func (h *synctestNetConnHalf) setReadBufferSize(size int) {
	h.lock()
	defer h.unlock()
	h.bufMax = size
}

func (h *synctestNetConnHalf) write(b []byte) (n int, err error) {
	for n < len(b) {
		nn, err := h.writePartial(b[n:])
		n += nn
		if err != nil {
			return n, err
		}
	}
	return n, nil
}

func (h *synctestNetConnHalf) writePartial(b []byte) (n int, err error) {
	if err := h.writeWaitAndLock(); err != nil {
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

func (h *synctestNetConnHalf) setReadError(err error) {
	h.lock()
	defer h.unlock()
	if h.readErr == nil {
		h.readErr = err
	}
}

func (h *synctestNetConnHalf) setWriteError(err error) {
	h.lock()
	defer h.unlock()
	if h.writeErr == nil {
		h.writeErr = err
	}
}

// deadlineContext converts a changeable deadline (as in net.Conn.SetDeadline) into a Context.
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
	if !deadline.After(time.Now()) {
		// Deadline has already expired.
		t.cancel(os.ErrDeadlineExceeded)
		t.cancel = nil
		return
	}
	if t.timer != nil {
		// Reuse existing deadline timer.
		t.timer.Reset(deadline.Sub(time.Now()))
		return
	}
	// Create a new timer to cancel the context at the deadline.
	t.timer = time.AfterFunc(deadline.Sub(time.Now()), func() {
		t.mu.Lock()
		defer t.mu.Unlock()
		t.cancel(os.ErrDeadlineExceeded)
		t.cancel = nil
	})
}
