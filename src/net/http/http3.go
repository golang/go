// Copyright 2026 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package http

import (
	"context"
	"crypto/tls"
	"net"
	"net/url"
)

type http3Transport interface {
	// DialClientConn creates a new client connection to address.
	//
	// If proxy is non-nil, the connection should use the provided proxy.
	// If HTTP/3 proxies are not supported, DialClientConn should return
	// an error wrapping [errors.ErrUnsupported].
	//
	// The RoundTripper returned by DialClientConn must also implement the
	// genericClientConn interface.
	//
	// The client connection should arrange to call internalStateHook
	// when the connection closes, when requests complete, and when the
	// connection concurrency limit changes.
	//
	// The client connection must call the internal state hook when
	// the connection state changes asynchronously, such as when a request completes.
	//
	// The internal state hook need not be called after synchronous changes
	// to the state: Close, Reserve, Release, and RoundTrip calls
	// which don't start a request do not need to call the hook.
	DialClientConn(ctx context.Context, address string, proxy *url.URL, tlsConfig *tls.Config, internalStateHook func() error) (RoundTripper, error)
	Close() error
}

// http3Server is an HTTP/3 server implementation. net/http/internal/http3
// registers an implementation of this interface by passing it to Server.Serve.
type http3Server interface {
	ServeHTTP3(context.Context, net.PacketConn, *tls.Config) error
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
	return s.h3Server.ServeHTTP3(baseCtx, conn, tlsConfig)
}
