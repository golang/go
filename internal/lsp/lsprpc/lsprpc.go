// Copyright 2020 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// Package lsprpc implements a jsonrpc2.StreamServer that may be used to
// serve the LSP on a jsonrpc2 channel.
package lsprpc

import (
	"context"
	"fmt"
	"log"
	"net"
	"os"
	"time"

	"golang.org/x/sync/errgroup"
	"golang.org/x/tools/internal/jsonrpc2"
	"golang.org/x/tools/internal/lsp"
	"golang.org/x/tools/internal/lsp/cache"
	"golang.org/x/tools/internal/lsp/protocol"
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
func NewStreamServer(cache *cache.Cache, withTelemetry bool) *StreamServer {
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
	network, addr string

	// Configuration. Right now, not all of this may be customizable, but in the
	// future it probably will be.
	withTelemetry bool
	dialTimeout   time.Duration
	retries       int
}

// NewForwarder creates a new Forwarder, ready to forward connections to the
// remote server specified by network and addr.
func NewForwarder(network, addr string, withTelemetry bool) *Forwarder {
	return &Forwarder{
		network:       network,
		addr:          addr,
		withTelemetry: withTelemetry,
		dialTimeout:   1 * time.Second,
		retries:       5,
	}
}

// ServeStream dials the forwarder remote and binds the remote to serve the LSP
// on the incoming stream.
func (f *Forwarder) ServeStream(ctx context.Context, stream jsonrpc2.Stream) error {
	clientConn := jsonrpc2.NewConn(stream)
	client := protocol.ClientDispatcher(clientConn)

	var (
		netConn net.Conn
		err     error
	)
	// Sometimes the forwarder will be started immediately after the server is
	// started. To account for these cases, add in some simple retrying.
	// Note that the number of total attempts is f.retries + 1.
	for attempt := 0; attempt <= f.retries; attempt++ {
		startDial := time.Now()
		netConn, err = net.DialTimeout(f.network, f.addr, f.dialTimeout)
		if err == nil {
			break
		}
		log.Printf("failed an attempt to connect to remote: %v\n", err)
		// In case our failure was a fast-failure, ensure we wait at least
		// f.dialTimeout before trying again.
		if attempt != f.retries {
			time.Sleep(f.dialTimeout - time.Since(startDial))
		}
	}
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
	clientConn.AddHandler(forwarderHandler{})
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

// ForwarderExitFunc is used to exit the forwarder process. It is mutable for
// testing purposes.
var ForwarderExitFunc = os.Exit

// forwarderHandler intercepts 'exit' messages to prevent the shared gopls
// instance from exiting. In the future it may also intercept 'shutdown' to
// provide more graceful shutdown of the client connection.
type forwarderHandler struct {
	jsonrpc2.EmptyHandler
}

func (forwarderHandler) Deliver(ctx context.Context, r *jsonrpc2.Request, delivered bool) bool {
	// TODO(golang.org/issues/34111): we should more gracefully disconnect here,
	// once that process exists.
	if r.Method == "exit" {
		ForwarderExitFunc(0)
		// Still return true here to prevent the message from being delivered: in
		// tests, ForwarderExitFunc may be overridden to something that doesn't
		// exit the process.
		return true
	}
	return false
}
