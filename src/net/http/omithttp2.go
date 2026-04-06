// Copyright 2019 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

//go:build nethttpomithttp2

package http

import (
	"context"
	"errors"
	"net"
)

func init() {
	omitBundledHTTP2 = true
}

const noHTTP2 = "no bundled HTTP/2" // should never see this

func (s *Server) configureHTTP2()                                                                   {}
func (s *Server) serveHTTP2Conn(ctx context.Context, nc net.Conn, h Handler, sawClientPreface bool) {}

func (t *Transport) configureHTTP2(protocols Protocols) {}
func (t *Transport) http2AddConn(scheme, authority string, nc net.Conn) (RoundTripper, error) {
	return nil, errors.ErrUnsupported
}
func (t *Transport) http2ExternalDial(ctx context.Context, cm connectMethod) (RoundTripper, error) {
	return nil, errors.ErrUnsupported
}
func (t *Transport) http2NewClientConn(nc net.Conn, internalStateHook func()) (RoundTripper, error) {
	return nil, errors.ErrUnsupported
}
func (t *Transport) http2NewClientConnFromContext(ctx context.Context) (*ClientConn, error) {
	return nil, errors.ErrUnsupported
}

type http2Server struct{}

type http2Transport struct{}
type http2ExternalTransportConfig interface {
	ExternalRoundTrip() bool
	RoundTrip(*Request) (*Response, error)
	Registered(*Transport)
	unimplementable()
}

func (*http2Transport) CloseIdleConnections()            {}
func (*http2Transport) IdleConnStrsForTesting() []string { return nil }

type http2RoundTripper struct{}

func (http2RoundTripper) RoundTrip(*Request) (*Response, error) { panic(noHTTP2) }
