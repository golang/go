// Copyright 2020 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// Package servertest provides utilities for running tests against a remote LSP
// server.
package servertest

import (
	"context"
	"fmt"
	"net"

	"golang.org/x/tools/internal/jsonrpc2"
)

// Server is a helper for executing tests against a remote jsonrpc2 connection.
// Once initialized, its Addr field may be used to connect a jsonrpc2 client.
type Server struct {
	Addr string

	ln net.Listener
}

// NewServer returns a new test server listening on local tcp port and serving
// incoming jsonrpc2 streams using the provided stream server. It panics on any
// error.
func NewServer(ctx context.Context, server jsonrpc2.StreamServer) *Server {
	ln, err := net.Listen("tcp", "127.0.0.1:0")
	if err != nil {
		panic(fmt.Sprintf("servertest: failed to listen: %v", err))
	}
	go jsonrpc2.Serve(ctx, ln, server)
	return &Server{Addr: ln.Addr().String(), ln: ln}
}

// Connect dials the test server and returns a jsonrpc2 Connection that is
// ready for use.
func (s *Server) Connect(ctx context.Context) *jsonrpc2.Conn {
	netConn, err := net.Dial("tcp", s.Addr)
	if err != nil {
		panic(fmt.Sprintf("servertest: failed to connect to test instance: %v", err))
	}
	conn := jsonrpc2.NewConn(jsonrpc2.NewHeaderStream(netConn, netConn))
	go conn.Run(ctx)
	return conn
}

// Close is a placeholder for proper test server shutdown.
// TODO: implement proper shutdown, which gracefully closes existing
// connections to the test server.
func (s *Server) Close() error {
	return nil
}
