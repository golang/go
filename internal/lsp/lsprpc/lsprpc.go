// Copyright 2020 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// Package lsprpc implements a jsonrpc2.StreamServer that may be used to
// serve the LSP on a jsonrpc2 channel.
package lsprpc

import (
	"context"

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
