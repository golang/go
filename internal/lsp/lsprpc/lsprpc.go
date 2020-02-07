// Copyright 2020 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// Package lsprpc implements a jsonrpc2.StreamServer that may be used to
// serve the LSP on a jsonrpc2 channel.
package lsprpc

import (
	"context"
	"fmt"
	"net"

	"golang.org/x/sync/errgroup"
	"golang.org/x/tools/internal/jsonrpc2"
	"golang.org/x/tools/internal/lsp"
	"golang.org/x/tools/internal/lsp/protocol"
	"golang.org/x/tools/internal/lsp/source"
)

// The StreamServer type is a jsonrpc2.StreamServer that handles incoming
// streams as a new LSP session, using a shared cache.
type StreamServer struct {
	withTelemetry bool

	// accept is mutable for testing.
	accept func(protocol.Client) protocol.Server
}

// NewStreamServer creates a StreamServer using the shared cache. If
// withTelemetry is true, each session is instrumented with telemetry that
// records RPC statistics.
func NewStreamServer(cache source.Cache, withTelemetry bool) *StreamServer {
	s := &StreamServer{
		withTelemetry: withTelemetry,
	}
	s.accept = func(c protocol.Client) protocol.Server {
		session := cache.NewSession()
		return lsp.NewServer(session, c)
	}
	return s
}

// ServeStream implements the jsonrpc2.StreamServer interface, by handling
// incoming streams using a new lsp server.
func (s *StreamServer) ServeStream(ctx context.Context, stream jsonrpc2.Stream) error {
	conn := jsonrpc2.NewConn(stream)
	client := protocol.ClientDispatcher(conn)
	server := s.accept(client)
	conn.AddHandler(protocol.ServerHandler(server))
	conn.AddHandler(protocol.Canceller{})
	if s.withTelemetry {
		conn.AddHandler(telemetryHandler{})
	}
	return conn.Run(protocol.WithClient(ctx, client))
}

// A Forwarder is a jsonrpc2.StreamServer that handles an LSP stream by
// forwarding it to a remote. This is used when the gopls process started by
// the editor is in the `-remote` mode, which means it finds and connects to a
// separate gopls daemon. In these cases, we still want the forwarder gopls to
// be instrumented with telemetry, and want to be able to in some cases hijack
// the jsonrpc2 connection with the daemon.
type Forwarder struct {
	remote        string
	withTelemetry bool
}

// NewForwarder creates a new Forwarder, ready to forward connections to the
// given remote.
func NewForwarder(remote string, withTelemetry bool) *Forwarder {
	return &Forwarder{
		remote:        remote,
		withTelemetry: withTelemetry,
	}
}

// ServeStream dials the forwarder remote and binds the remote to serve the LSP
// on the incoming stream.
func (f *Forwarder) ServeStream(ctx context.Context, stream jsonrpc2.Stream) error {
	clientConn := jsonrpc2.NewConn(stream)
	client := protocol.ClientDispatcher(clientConn)

	netConn, err := net.Dial("tcp", f.remote)
	if err != nil {
		return fmt.Errorf("forwarder: dialing remote: %v", err)
	}
	serverConn := jsonrpc2.NewConn(jsonrpc2.NewHeaderStream(netConn, netConn))
	server := protocol.ServerDispatcher(serverConn)

	// Forward between connections.
	serverConn.AddHandler(protocol.ClientHandler(client))
	serverConn.AddHandler(protocol.Canceller{})
	clientConn.AddHandler(protocol.ServerHandler(server))
	clientConn.AddHandler(protocol.Canceller{})
	if f.withTelemetry {
		clientConn.AddHandler(telemetryHandler{})
	}

	g, ctx := errgroup.WithContext(ctx)
	g.Go(func() error {
		return serverConn.Run(ctx)
	})
	g.Go(func() error {
		return clientConn.Run(ctx)
	})
	return g.Wait()
}
