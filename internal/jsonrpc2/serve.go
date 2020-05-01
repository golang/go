// Copyright 2020 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package jsonrpc2

import (
	"context"
	"fmt"
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
// a newly created stream.
type StreamServer interface {
	ServeStream(context.Context, Stream) error
}

// The ServerFunc type is an adapter that implements the StreamServer interface
// using an ordinary function.
type ServerFunc func(context.Context, Stream) error

// ServeStream calls f(ctx, s).
func (f ServerFunc) ServeStream(ctx context.Context, s Stream) error {
	return f(ctx, s)
}

// HandlerServer returns a StreamServer that handles incoming streams using the
// provided handler.
func HandlerServer(h Handler) StreamServer {
	return ServerFunc(func(ctx context.Context, s Stream) error {
		conn := NewConn(s)
		return conn.Run(ctx, h)
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
				case doneListening <- fmt.Errorf("Accept(): %v", err):
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
			stream := NewHeaderStream(netConn, netConn)
			go func() {
				closedConns <- server.ServeStream(ctx, stream)
				stream.Close()
			}()
		case err := <-doneListening:
			return err
		case err := <-closedConns:
			event.Error(ctx, "closed a connection", err)
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
