// Copyright 2026 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package http

import (
	"context"
	"crypto/tls"
	"net"
)

// http3Server is an HTTP/3 server implementation.
// x/net/http3 registers an implementation of this interface by passing it to Server.Serve.
type http3Server interface {
	ServeHTTP3(context.Context, net.PacketConn, *tls.Config, Handler) error
	Shutdown(ctx context.Context) error
}

func (s *Server) setHTTP3Server(h3 http3Server) {
	if s.h3Server != nil {
		panic("http: HTTP/3 Server already registered")
	}
	s.h3Server = h3
}

// http3PacketConn is a net.Listener we pass to Server.BaseContext for HTTP/3 ports.
type http3PacketConn struct {
	conn net.PacketConn
}

func (c http3PacketConn) Accept() (net.Conn, error) { return nil, net.ErrClosed }
func (c http3PacketConn) Close() error              { return c.conn.Close() }
func (c http3PacketConn) Addr() net.Addr            { return c.conn.LocalAddr() }

func (s *Server) serveHTTP3(conn net.PacketConn, certFile, keyFile string) error {
	baseCtx := context.Background()
	if s.BaseContext != nil {
		// The Server.BaseContext hook receives a net.Listener.
		// For HTTP/3 connections, provide it with a Listener that has the same Addr
		// as the PacketConn we're actually serving from.
		baseCtx = s.BaseContext(http3PacketConn{conn})
	}
	baseCtx = context.WithValue(baseCtx, ServerContextKey, s)

	tlsConfig, err := s.setupTLSConfig(certFile, keyFile, []string{"h3"})
	if err != nil {
		return err
	}
	return s.h3Server.ServeHTTP3(baseCtx, conn, tlsConfig, serverHandler{s})
}
