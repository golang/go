// Copyright 2020 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package jsonrpc2

import (
	"context"
	"io"
	"time"

	"golang.org/x/tools/internal/event"
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
	options  ServeOptions // a copy of the config that started this server
	async    async
}

// ServeOptions holds the options to the Serve function.
//TODO: kill ServeOptions and push timeout into the listener
type ServeOptions struct {
	// IdleTimeout is the maximum amount of time to remain idle and running.
	IdleTimeout time.Duration
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
func Serve(ctx context.Context, listener Listener, binder Binder, options ServeOptions) (*Server, error) {
	server := &Server{
		listener: listener,
		binder:   binder,
		options:  options,
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
	// Max duration: ~290 years; surely that's long enough.
	const forever = 1<<63 - 1
	idleTimeout := s.options.IdleTimeout
	if idleTimeout <= 0 {
		idleTimeout = forever
	}
	idleTimer := time.NewTimer(idleTimeout)

	// run a goroutine that listens for incoming connections and posts them
	// back to the worker
	newStreams := make(chan io.ReadWriteCloser)
	go func() {
		for {
			// we never close the accepted connection, we rely on the other end
			// closing or the socket closing itself naturally
			rwc, err := s.listener.Accept(ctx)
			if err != nil {
				if !isClosingError(err) {
					event.Error(ctx, "Accept", err)
				}
				// signal we are done generating new connections for good
				close(newStreams)
				return
			}
			newStreams <- rwc
		}
	}()

	closedConns := make(chan struct{})
	activeConns := 0
	lnClosed := false
	for {
		select {
		case rwc := <-newStreams:
			// whatever happes we are not idle anymore
			idleTimer.Stop()
			if rwc == nil {
				// the net listener has been closed
				lnClosed = true
				if activeConns == 0 {
					// accept is done and there are no active connections, so just stop now
					return
				}
				// replace the channel with one that will never trigger
				// this is save because the only writer has already quit
				newStreams = nil
				// and then wait for all active connections to stop
				continue
			}
			// a new inbound connection,
			conn, err := newConnection(ctx, rwc, s.binder)
			if err != nil {
				if !isClosingError(err) {
					event.Error(ctx, "NewConn", err)
				}
				continue
			}
			// register the new conn as active
			activeConns++
			// wrap the conn in a close monitor
			//TODO: we do this to maintain our active count correctly, is there a better way?
			go func() {
				err := conn.Wait()
				if err != nil && !isClosingError(err) {
					event.Error(ctx, "closed a connection", err)
				}
				closedConns <- struct{}{}
			}()
		case <-closedConns:
			activeConns--
			if activeConns == 0 {
				// no more active connections, restart the idle timer
				if lnClosed {
					// we can never get a new connection, so we are done
					return
				}
				// we are idle, but might get a new connection still
				idleTimer.Reset(idleTimeout)
			}
		case <-idleTimer.C:
			// no activity for a while, time to stop serving
			s.async.setError(ErrIdleTimeout)
			return
		case <-ctx.Done():
			s.async.setError(ctx.Err())
			return
		}
	}
}

// isClosingError reports if the error occurs normally during the process of
// closing a network connection. It uses imperfect heuristics that err on the
// side of false negatives, and should not be used for anything critical.
func isClosingError(err error) bool {
	if err == nil {
		return false
	}
	// fully unwrap the error, so the following tests work
	for wrapped := err; wrapped != nil; wrapped = errors.Unwrap(err) {
		err = wrapped
	}

	// was it based on an EOF error?
	if err == io.EOF {
		return true
	}

	// Per https://github.com/golang/go/issues/4373, this error string should not
	// change. This is not ideal, but since the worst that could happen here is
	// some superfluous logging, it is acceptable.
	if err.Error() == "use of closed network connection" {
		return true
	}

	return false
}
