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
	"strings"
	"sync"

	"golang.org/x/tools/internal/jsonrpc2"
)

// Connector is the interface used to connect to a server.
type Connector interface {
	Connect(context.Context) jsonrpc2.Conn
}

// TCPServer is a helper for executing tests against a remote jsonrpc2
// connection. Once initialized, its Addr field may be used to connect a
// jsonrpc2 client.
type TCPServer struct {
	*connList

	Addr string

	ln     net.Listener
	framer jsonrpc2.Framer
}

// NewTCPServer returns a new test server listening on local tcp port and
// serving incoming jsonrpc2 streams using the provided stream server. It
// panics on any error.
func NewTCPServer(ctx context.Context, server jsonrpc2.StreamServer, framer jsonrpc2.Framer) *TCPServer {
	ln, err := net.Listen("tcp", "127.0.0.1:0")
	if err != nil {
		panic(fmt.Sprintf("servertest: failed to listen: %v", err))
	}
	if framer == nil {
		framer = jsonrpc2.NewHeaderStream
	}
	go jsonrpc2.Serve(ctx, ln, server, 0)
	return &TCPServer{Addr: ln.Addr().String(), ln: ln, framer: framer, connList: &connList{}}
}

// Connect dials the test server and returns a jsonrpc2 Connection that is
// ready for use.
func (s *TCPServer) Connect(_ context.Context) jsonrpc2.Conn {
	netConn, err := net.Dial("tcp", s.Addr)
	if err != nil {
		panic(fmt.Sprintf("servertest: failed to connect to test instance: %v", err))
	}
	conn := jsonrpc2.NewConn(s.framer(netConn))
	s.add(conn)
	return conn
}

// PipeServer is a test server that handles connections over io.Pipes.
type PipeServer struct {
	*connList
	server jsonrpc2.StreamServer
	framer jsonrpc2.Framer
}

// NewPipeServer returns a test server that can be connected to via io.Pipes.
func NewPipeServer(server jsonrpc2.StreamServer, framer jsonrpc2.Framer) *PipeServer {
	if framer == nil {
		framer = jsonrpc2.NewRawStream
	}
	return &PipeServer{server: server, framer: framer, connList: &connList{}}
}

// Connect creates new io.Pipes and binds them to the underlying StreamServer.
func (s *PipeServer) Connect(ctx context.Context) jsonrpc2.Conn {
	sPipe, cPipe := net.Pipe()
	serverStream := s.framer(sPipe)
	serverConn := jsonrpc2.NewConn(serverStream)
	s.add(serverConn)
	go s.server.ServeStream(ctx, serverConn)

	clientStream := s.framer(cPipe)
	clientConn := jsonrpc2.NewConn(clientStream)
	s.add(clientConn)
	return clientConn
}

// connList tracks closers to run when a testserver is closed.  This is a
// convenience, so that callers don't have to worry about closing each
// connection.
type connList struct {
	mu    sync.Mutex
	conns []jsonrpc2.Conn
}

func (l *connList) add(conn jsonrpc2.Conn) {
	l.mu.Lock()
	defer l.mu.Unlock()
	l.conns = append(l.conns, conn)
}

func (l *connList) Close() error {
	l.mu.Lock()
	defer l.mu.Unlock()
	var errmsgs []string
	for _, conn := range l.conns {
		if err := conn.Close(); err != nil {
			errmsgs = append(errmsgs, err.Error())
		}
	}
	if len(errmsgs) > 0 {
		return fmt.Errorf("closing errors:\n%s", strings.Join(errmsgs, "\n"))
	}
	return nil
}
