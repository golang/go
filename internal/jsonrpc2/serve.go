// Copyright 2020 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package jsonrpc2

import (
	"context"
	"net"
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
		conn.AddHandler(h)
		return conn.Run(ctx)
	})
}

// ListenAndServe starts an jsonrpc2 server on the given address. It exits only
// on error.
func ListenAndServe(ctx context.Context, addr string, server StreamServer) error {
	ln, err := net.Listen("tcp", addr)
	if err != nil {
		return err
	}
	return Serve(ctx, ln, server)
}

// Serve accepts incoming connections from the network, and handles them using
// the provided server. It exits only on error.
func Serve(ctx context.Context, ln net.Listener, server StreamServer) error {
	for {
		netConn, err := ln.Accept()
		if err != nil {
			return err
		}
		stream := NewHeaderStream(netConn, netConn)
		go server.ServeStream(ctx, stream)
	}
}
