// Copyright 2020 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package jsonrpc2

import (
	"context"
	"io"
	"runtime"
	"strings"
	"sync"
	"syscall"
	"time"

	errors "golang.org/x/xerrors"
)

// Listener is implemented by protocols to accept new inbound connections.
type Listener interface {
	// Accept an inbound connection to a server.
	// It must block until an inbound connection is made, or the listener is
	// shut down.
	Accept(context.Context) (io.ReadWriteCloser, error)

	// Close is used to ask a listener to stop accepting new connections.
	Close() error

	// Dialer returns a dialer that can be used to connect to this listener
	// locally.
	// If a listener does not implement this it will return a nil.
	Dialer() Dialer
}

// Dialer is used by clients to dial a server.
type Dialer interface {
	// Dial returns a new communication byte stream to a listening server.
	Dial(ctx context.Context) (io.ReadWriteCloser, error)
}

// Server is a running server that is accepting incoming connections.
type Server struct {
	listener Listener
	binder   Binder
	async    async
}

// Dial uses the dialer to make a new connection, wraps the returned
// reader and writer using the framer to make a stream, and then builds
// a connection on top of that stream using the binder.
func Dial(ctx context.Context, dialer Dialer, binder Binder) (*Connection, error) {
	// dial a server
	rwc, err := dialer.Dial(ctx)
	if err != nil {
		return nil, err
	}
	return newConnection(ctx, rwc, binder)
}

// Serve starts a new server listening for incoming connections and returns
// it.
// This returns a fully running and connected server, it does not block on
// the listener.
// You can call Wait to block on the server, or Shutdown to get the sever to
// terminate gracefully.
// To notice incoming connections, use an intercepting Binder.
func Serve(ctx context.Context, listener Listener, binder Binder) (*Server, error) {
	server := &Server{
		listener: listener,
		binder:   binder,
	}
	server.async.init()
	go server.run(ctx)
	return server, nil
}

// Wait returns only when the server has shut down.
func (s *Server) Wait() error {
	return s.async.wait()
}

// run accepts incoming connections from the listener,
// If IdleTimeout is non-zero, run exits after there are no clients for this
// duration, otherwise it exits only on error.
func (s *Server) run(ctx context.Context) {
	defer s.async.done()
	var activeConns []*Connection
	for {
		// we never close the accepted connection, we rely on the other end
		// closing or the socket closing itself naturally
		rwc, err := s.listener.Accept(ctx)
		if err != nil {
			if !isClosingError(err) {
				s.async.setError(err)
			}
			// we are done generating new connections for good
			break
		}

		// see if any connections were closed while we were waiting
		activeConns = onlyActive(activeConns)

		// a new inbound connection,
		conn, err := newConnection(ctx, rwc, s.binder)
		if err != nil {
			if !isClosingError(err) {
				s.async.setError(err)
			}
			continue
		}
		activeConns = append(activeConns, conn)
	}

	// wait for all active conns to finish
	for _, c := range activeConns {
		c.Wait()
	}
}

func onlyActive(conns []*Connection) []*Connection {
	i := 0
	for _, c := range conns {
		if !c.async.isDone() {
			conns[i] = c
			i++
		}
	}
	// trim the slice down
	return conns[:i]
}

// isClosingError reports if the error occurs normally during the process of
// closing a network connection. It uses imperfect heuristics that err on the
// side of false negatives, and should not be used for anything critical.
func isClosingError(err error) bool {
	if err == nil {
		return false
	}
	// Fully unwrap the error, so the following tests work.
	for wrapped := err; wrapped != nil; wrapped = errors.Unwrap(err) {
		err = wrapped
	}

	// Was it based on an EOF error?
	if err == io.EOF {
		return true
	}

	// Was it based on a closed pipe?
	if err == io.ErrClosedPipe {
		return true
	}

	// Per https://github.com/golang/go/issues/4373, this error string should not
	// change. This is not ideal, but since the worst that could happen here is
	// some superfluous logging, it is acceptable.
	if err.Error() == "use of closed network connection" {
		return true
	}

	if runtime.GOOS == "plan9" {
		// Error reading from a closed connection.
		if err == syscall.EINVAL {
			return true
		}
		// Error trying to accept a new connection from a closed listener.
		if strings.HasSuffix(err.Error(), " listen hungup") {
			return true
		}
	}
	return false
}

// NewIdleListener wraps a listener with an idle timeout.
// When there are no active connections for at least the timeout duration a
// call to accept will fail with ErrIdleTimeout.
func NewIdleListener(timeout time.Duration, wrap Listener) Listener {
	l := &idleListener{
		timeout:    timeout,
		wrapped:    wrap,
		newConns:   make(chan *idleCloser),
		closed:     make(chan struct{}),
		wasTimeout: make(chan struct{}),
	}
	go l.run()
	return l
}

type idleListener struct {
	wrapped    Listener
	timeout    time.Duration
	newConns   chan *idleCloser
	closed     chan struct{}
	wasTimeout chan struct{}
	closeOnce  sync.Once
}

type idleCloser struct {
	wrapped   io.ReadWriteCloser
	closed    chan struct{}
	closeOnce sync.Once
}

func (c *idleCloser) Read(p []byte) (int, error) {
	n, err := c.wrapped.Read(p)
	if err != nil && isClosingError(err) {
		c.closeOnce.Do(func() { close(c.closed) })
	}
	return n, err
}

func (c *idleCloser) Write(p []byte) (int, error) {
	// we do not close on write failure, we rely on the wrapped writer to do that
	// if it is appropriate, which we will detect in the next read.
	return c.wrapped.Write(p)
}

func (c *idleCloser) Close() error {
	// we rely on closing the wrapped stream to signal to the next read that we
	// are closed, rather than triggering the closed signal directly
	return c.wrapped.Close()
}

func (l *idleListener) Accept(ctx context.Context) (io.ReadWriteCloser, error) {
	rwc, err := l.wrapped.Accept(ctx)
	if err != nil {
		if isClosingError(err) {
			// underlying listener was closed
			l.closeOnce.Do(func() { close(l.closed) })
			// was it closed because of the idle timeout?
			select {
			case <-l.wasTimeout:
				err = ErrIdleTimeout
			default:
			}
		}
		return nil, err
	}
	conn := &idleCloser{
		wrapped: rwc,
		closed:  make(chan struct{}),
	}
	l.newConns <- conn
	return conn, err
}

func (l *idleListener) Close() error {
	defer l.closeOnce.Do(func() { close(l.closed) })
	return l.wrapped.Close()
}

func (l *idleListener) Dialer() Dialer {
	return l.wrapped.Dialer()
}

func (l *idleListener) run() {
	var conns []*idleCloser
	for {
		var firstClosed chan struct{} // left at nil if there are no active conns
		var timeout <-chan time.Time  // left at nil if there are  active conns
		if len(conns) > 0 {
			firstClosed = conns[0].closed
		} else {
			timeout = time.After(l.timeout)
		}
		select {
		case <-l.closed:
			// the main listener closed, no need to keep going
			return
		case conn := <-l.newConns:
			// a new conn arrived, add it to the list
			conns = append(conns, conn)
		case <-timeout:
			// we timed out, only happens when there are no active conns
			// close the underlying listener, and allow the normal closing process to happen
			close(l.wasTimeout)
			l.wrapped.Close()
		case <-firstClosed:
			// a conn closed, remove it from the active list
			conns = conns[:copy(conns, conns[1:])]
		}
	}
}
