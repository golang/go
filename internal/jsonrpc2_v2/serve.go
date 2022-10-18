// Copyright 2020 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package jsonrpc2

import (
	"context"
	"fmt"
	"io"
	"runtime"
	"sync"
	"sync/atomic"
	"time"
)

// Listener is implemented by protocols to accept new inbound connections.
type Listener interface {
	// Accept accepts an inbound connection to a server.
	// It blocks until either an inbound connection is made, or the listener is closed.
	Accept(context.Context) (io.ReadWriteCloser, error)

	// Close closes the listener.
	// Any blocked Accept or Dial operations will unblock and return errors.
	Close() error

	// Dialer returns a dialer that can be used to connect to this listener
	// locally.
	// If a listener does not implement this it will return nil.
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
	async    *async

	shutdownOnce sync.Once
	closing      int32 // atomic: set to nonzero when Shutdown is called
}

// Dial uses the dialer to make a new connection, wraps the returned
// reader and writer using the framer to make a stream, and then builds
// a connection on top of that stream using the binder.
//
// The returned Connection will operate independently using the Preempter and/or
// Handler provided by the Binder, and will release its own resources when the
// connection is broken, but the caller may Close it earlier to stop accepting
// (or sending) new requests.
func Dial(ctx context.Context, dialer Dialer, binder Binder) (*Connection, error) {
	// dial a server
	rwc, err := dialer.Dial(ctx)
	if err != nil {
		return nil, err
	}
	return newConnection(ctx, rwc, binder, nil), nil
}

// NewServer starts a new server listening for incoming connections and returns
// it.
// This returns a fully running and connected server, it does not block on
// the listener.
// You can call Wait to block on the server, or Shutdown to get the sever to
// terminate gracefully.
// To notice incoming connections, use an intercepting Binder.
func NewServer(ctx context.Context, listener Listener, binder Binder) *Server {
	server := &Server{
		listener: listener,
		binder:   binder,
		async:    newAsync(),
	}
	go server.run(ctx)
	return server
}

// Wait returns only when the server has shut down.
func (s *Server) Wait() error {
	return s.async.wait()
}

// Shutdown informs the server to stop accepting new connections.
func (s *Server) Shutdown() {
	s.shutdownOnce.Do(func() {
		atomic.StoreInt32(&s.closing, 1)
		s.listener.Close()
	})
}

// run accepts incoming connections from the listener,
// If IdleTimeout is non-zero, run exits after there are no clients for this
// duration, otherwise it exits only on error.
func (s *Server) run(ctx context.Context) {
	defer s.async.done()

	var activeConns sync.WaitGroup
	for {
		rwc, err := s.listener.Accept(ctx)
		if err != nil {
			// Only Shutdown closes the listener. If we get an error after Shutdown is
			// called, assume that that was the cause and don't report the error;
			// otherwise, report the error in case it is unexpected.
			if atomic.LoadInt32(&s.closing) == 0 {
				s.async.setError(err)
			}
			// We are done generating new connections for good.
			break
		}

		// A new inbound connection.
		activeConns.Add(1)
		_ = newConnection(ctx, rwc, s.binder, activeConns.Done) // unregisters itself when done
	}
	activeConns.Wait()
}

// NewIdleListener wraps a listener with an idle timeout.
//
// When there are no active connections for at least the timeout duration,
// calls to Accept will fail with ErrIdleTimeout.
//
// A connection is considered inactive as soon as its Close method is called.
func NewIdleListener(timeout time.Duration, wrap Listener) Listener {
	l := &idleListener{
		wrapped:   wrap,
		timeout:   timeout,
		active:    make(chan int, 1),
		timedOut:  make(chan struct{}),
		idleTimer: make(chan *time.Timer, 1),
	}
	l.idleTimer <- time.AfterFunc(l.timeout, l.timerExpired)
	return l
}

type idleListener struct {
	wrapped Listener
	timeout time.Duration

	// Only one of these channels is receivable at any given time.
	active    chan int         // count of active connections; closed when Close is called if not timed out
	timedOut  chan struct{}    // closed when the idle timer expires
	idleTimer chan *time.Timer // holds the timer only when idle
}

// Accept accepts an incoming connection.
//
// If an incoming connection is accepted concurrent to the listener being closed
// due to idleness, the new connection is immediately closed.
func (l *idleListener) Accept(ctx context.Context) (io.ReadWriteCloser, error) {
	rwc, err := l.wrapped.Accept(ctx)

	select {
	case n, ok := <-l.active:
		if err != nil {
			if ok {
				l.active <- n
			}
			return nil, err
		}
		if ok {
			l.active <- n + 1
		} else {
			// l.wrapped.Close Close has been called, but Accept returned a
			// connection. This race can occur with concurrent Accept and Close calls
			// with any net.Listener, and it is benign: since the listener was closed
			// explicitly, it can't have also timed out.
		}
		return l.newConn(rwc), nil

	case <-l.timedOut:
		if err == nil {
			// Keeping the connection open would leave the listener simultaneously
			// active and closed due to idleness, which would be contradictory and
			// confusing. Close the connection and pretend that it never happened.
			rwc.Close()
		} else {
			// In theory the timeout could have raced with an unrelated error return
			// from Accept. However, ErrIdleTimeout is arguably still valid (since we
			// would have closed due to the timeout independent of the error), and the
			// harm from returning a spurious ErrIdleTimeout is negliglible anyway.
		}
		return nil, ErrIdleTimeout

	case timer := <-l.idleTimer:
		if err != nil {
			// The idle timer doesn't run until it receives itself from the idleTimer
			// channel, so it can't have called l.wrapped.Close yet and thus err can't
			// be ErrIdleTimeout. Leave the idle timer as it was and return whatever
			// error we got.
			l.idleTimer <- timer
			return nil, err
		}

		if !timer.Stop() {
			// Failed to stop the timer — the timer goroutine is in the process of
			// firing. Send the timer back to the timer goroutine so that it can
			// safely close the timedOut channel, and then wait for the listener to
			// actually be closed before we return ErrIdleTimeout.
			l.idleTimer <- timer
			rwc.Close()
			<-l.timedOut
			return nil, ErrIdleTimeout
		}

		l.active <- 1
		return l.newConn(rwc), nil
	}
}

func (l *idleListener) Close() error {
	select {
	case _, ok := <-l.active:
		if ok {
			close(l.active)
		}

	case <-l.timedOut:
		// Already closed by the timer; take care not to double-close if the caller
		// only explicitly invokes this Close method once, since the io.Closer
		// interface explicitly leaves doubled Close calls undefined.
		return ErrIdleTimeout

	case timer := <-l.idleTimer:
		if !timer.Stop() {
			// Couldn't stop the timer. It shouldn't take long to run, so just wait
			// (so that the Listener is guaranteed to be closed before we return)
			// and pretend that this call happened afterward.
			// That way we won't leak any timers or goroutines when Close returns.
			l.idleTimer <- timer
			<-l.timedOut
			return ErrIdleTimeout
		}
		close(l.active)
	}

	return l.wrapped.Close()
}

func (l *idleListener) Dialer() Dialer {
	return l.wrapped.Dialer()
}

func (l *idleListener) timerExpired() {
	select {
	case n, ok := <-l.active:
		if ok {
			panic(fmt.Sprintf("jsonrpc2: idleListener idle timer fired with %d connections still active", n))
		} else {
			panic("jsonrpc2: Close finished with idle timer still running")
		}

	case <-l.timedOut:
		panic("jsonrpc2: idleListener idle timer fired more than once")

	case <-l.idleTimer:
		// The timer for this very call!
	}

	// Close the Listener with all channels still blocked to ensure that this call
	// to l.wrapped.Close doesn't race with the one in l.Close.
	defer close(l.timedOut)
	l.wrapped.Close()
}

func (l *idleListener) connClosed() {
	select {
	case n, ok := <-l.active:
		if !ok {
			// l is already closed, so it can't close due to idleness,
			// and we don't need to track the number of active connections any more.
			return
		}
		n--
		if n == 0 {
			l.idleTimer <- time.AfterFunc(l.timeout, l.timerExpired)
		} else {
			l.active <- n
		}

	case <-l.timedOut:
		panic("jsonrpc2: idleListener idle timer fired before last active connection was closed")

	case <-l.idleTimer:
		panic("jsonrpc2: idleListener idle timer active before last active connection was closed")
	}
}

type idleListenerConn struct {
	wrapped   io.ReadWriteCloser
	l         *idleListener
	closeOnce sync.Once
}

func (l *idleListener) newConn(rwc io.ReadWriteCloser) *idleListenerConn {
	c := &idleListenerConn{
		wrapped: rwc,
		l:       l,
	}

	// A caller that forgets to call Close may disrupt the idleListener's
	// accounting, even though the file descriptor for the underlying connection
	// may eventually be garbage-collected anyway.
	//
	// Set a (best-effort) finalizer to verify that a Close call always occurs.
	// (We will clear the finalizer explicitly in Close.)
	runtime.SetFinalizer(c, func(c *idleListenerConn) {
		panic("jsonrpc2: IdleListener connection became unreachable without a call to Close")
	})

	return c
}

func (c *idleListenerConn) Read(p []byte) (int, error)  { return c.wrapped.Read(p) }
func (c *idleListenerConn) Write(p []byte) (int, error) { return c.wrapped.Write(p) }

func (c *idleListenerConn) Close() error {
	defer c.closeOnce.Do(func() {
		c.l.connClosed()
		runtime.SetFinalizer(c, nil)
	})
	return c.wrapped.Close()
}
