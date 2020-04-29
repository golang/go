// Copyright 2020 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// Package servertest provides utilities for running tests against a remote LSP
// server.
package servertest

import (
	"context"
	"fmt"
	"io"
	"net"
	"sync"

	"golang.org/x/tools/internal/jsonrpc2"
)

// Connector is the interface used to connect to a server.
type Connector interface {
	Connect(context.Context) *jsonrpc2.Conn
}

// TCPServer is a helper for executing tests against a remote jsonrpc2
// connection. Once initialized, its Addr field may be used to connect a
// jsonrpc2 client.
type TCPServer struct {
	Addr string

	ln  net.Listener
	cls *closerList
}

// NewTCPServer returns a new test server listening on local tcp port and
// serving incoming jsonrpc2 streams using the provided stream server. It
// panics on any error.
func NewTCPServer(ctx context.Context, server jsonrpc2.StreamServer) *TCPServer {
	ln, err := net.Listen("tcp", "127.0.0.1:0")
	if err != nil {
		panic(fmt.Sprintf("servertest: failed to listen: %v", err))
	}
	go jsonrpc2.Serve(ctx, ln, server, 0)
	return &TCPServer{Addr: ln.Addr().String(), ln: ln, cls: &closerList{}}
}

// Connect dials the test server and returns a jsonrpc2 Connection that is
// ready for use.
func (s *TCPServer) Connect(ctx context.Context) *jsonrpc2.Conn {
	netConn, err := net.Dial("tcp", s.Addr)
	if err != nil {
		panic(fmt.Sprintf("servertest: failed to connect to test instance: %v", err))
	}
	s.cls.add(func() {
		netConn.Close()
	})
	return jsonrpc2.NewConn(jsonrpc2.NewHeaderStream(netConn, netConn))
}

// Close closes all connected pipes.
func (s *TCPServer) Close() error {
	s.cls.closeAll()
	return nil
}

// PipeServer is a test server that handles connections over io.Pipes.
type PipeServer struct {
	server jsonrpc2.StreamServer
	cls    *closerList
}

// NewPipeServer returns a test server that can be connected to via io.Pipes.
func NewPipeServer(ctx context.Context, server jsonrpc2.StreamServer) *PipeServer {
	return &PipeServer{server: server, cls: &closerList{}}
}

// Connect creates new io.Pipes and binds them to the underlying StreamServer.
func (s *PipeServer) Connect(ctx context.Context) *jsonrpc2.Conn {
	// Pipes connect like this:
	// Client🡒(sWriter)🡒(sReader)🡒Server
	//       🡔(cReader)🡐(cWriter)🡗
	sReader, sWriter := io.Pipe()
	cReader, cWriter := io.Pipe()
	s.cls.add(func() {
		sReader.Close()
		sWriter.Close()
		cReader.Close()
		cWriter.Close()
	})
	serverStream := jsonrpc2.NewRawStream(sReader, cWriter)
	go s.server.ServeStream(ctx, serverStream)

	clientStream := jsonrpc2.NewRawStream(cReader, sWriter)
	return jsonrpc2.NewConn(clientStream)
}

// Close closes all connected pipes.
func (s *PipeServer) Close() error {
	s.cls.closeAll()
	return nil
}

// closerList tracks closers to run when a testserver is closed.  This is a
// convenience, so that callers don't have to worry about closing each
// connection.
type closerList struct {
	mu      sync.Mutex
	closers []func()
}

func (l *closerList) add(closer func()) {
	l.mu.Lock()
	defer l.mu.Unlock()
	l.closers = append(l.closers, closer)
}

func (l *closerList) closeAll() {
	l.mu.Lock()
	defer l.mu.Unlock()
	for _, closer := range l.closers {
		closer()
	}
}
