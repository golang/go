// Copyright 2020 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package jsonrpc2

import (
	"context"
	"errors"
	"fmt"
	"io"
	"net"
	"os"
	"time"

	"golang.org/x/tools/internal/event"
)

// NOTE: This file provides an experimental API for serving multiple remote
// jsonrpc2 clients over the network. For now, it is intentionally similar to
// net/http, but that may change in the future as we figure out the correct
// semantics.

// A StreamServer is used to serve incoming jsonrpc2 clients communicating over
// a newly created connection.
type StreamServer interface {
	ServeStream(context.Context, Conn) error
}

// The ServerFunc type is an adapter that implements the StreamServer interface
// using an ordinary function.
type ServerFunc func(context.Context, Conn) error

// ServeStream calls f(ctx, s).
func (f ServerFunc) ServeStream(ctx context.Context, c Conn) error {
	return f(ctx, c)
}

// HandlerServer returns a StreamServer that handles incoming streams using the
// provided handler.
func HandlerServer(h Handler) StreamServer {
	return ServerFunc(func(ctx context.Context, conn Conn) error {
		conn.Go(ctx, h)
		<-conn.Done()
		return conn.Err()
	})
}

// ListenAndServe starts an jsonrpc2 server on the given address.  If
// idleTimeout is non-zero, ListenAndServe exits after there are no clients for
// this duration, otherwise it exits only on error.
func ListenAndServe(ctx context.Context, network, addr string, server StreamServer, idleTimeout time.Duration) error {
	ln, err := net.Listen(network, addr)
	if err != nil {
		return err
	}
	defer ln.Close()
	if network == "unix" {
		defer os.Remove(addr)
	}
	return Serve(ctx, ln, server, idleTimeout)
}

// Serve accepts incoming connections from the network, and handles them using
// the provided server. If idleTimeout is non-zero, ListenAndServe exits after
// there are no clients for this duration, otherwise it exits only on error.
func Serve(ctx context.Context, ln net.Listener, server StreamServer, idleTimeout time.Duration) error {
	ctx, cancel := context.WithCancel(ctx)
	defer cancel()
	// Max duration: ~290 years; surely that's long enough.
	const forever = 1<<63 - 1
	if idleTimeout <= 0 {
		idleTimeout = forever
	}
	connTimer := time.NewTimer(idleTimeout)

	newConns := make(chan net.Conn)
	doneListening := make(chan error)
	closedConns := make(chan error)

	go func() {
		for {
			nc, err := ln.Accept()
			if err != nil {
				select {
				case doneListening <- fmt.Errorf("Accept(): %w", err):
				case <-ctx.Done():
				}
				return
			}
			newConns <- nc
		}
	}()

	activeConns := 0
	for {
		select {
		case netConn := <-newConns:
			activeConns++
			connTimer.Stop()
			stream := NewHeaderStream(netConn)
			go func() {
				conn := NewConn(stream)
				closedConns <- server.ServeStream(ctx, conn)
				stream.Close()
			}()
		case err := <-doneListening:
			return err
		case err := <-closedConns:
			if !isClosingError(err) {
				event.Error(ctx, "closed a connection", err)
			}
			activeConns--
			if activeConns == 0 {
				connTimer.Reset(idleTimeout)
			}
		case <-connTimer.C:
			return ErrIdleTimeout
		case <-ctx.Done():
			return ctx.Err()
		}
	}
}

// isClosingError reports if the error occurs normally during the process of
// closing a network connection. It uses imperfect heuristics that err on the
// side of false negatives, and should not be used for anything critical.
func isClosingError(err error) bool {
	if errors.Is(err, io.EOF) {
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
