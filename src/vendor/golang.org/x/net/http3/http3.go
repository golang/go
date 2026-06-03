// Copyright 2026 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package http3

import (
	"net/http"
	"time"
	_ "unsafe" // for linkname

	. "golang.org/x/net/internal/http3"
	"golang.org/x/net/quic"
)

// Be extra generous with the handshake timeout. On some builders, the default
// handshake timeout seems to be insufficient, causing rare test flakes.
const handshakeTimeout = 1 * time.Minute

//go:linkname registerHTTP3Server net/http_test.registerHTTP3Server
func registerHTTP3Server(s *http.Server) <-chan *quic.Endpoint {
	endpointCh := make(chan *quic.Endpoint)
	RegisterServer(s, ServerOpts{
		ListenQUIC: func(addr string, config *quic.Config) (*quic.Endpoint, error) {
			e, err := quic.Listen("udp", addr, config)
			endpointCh <- e
			return e, err
		},
		QUICConfig: &quic.Config{HandshakeTimeout: handshakeTimeout},
	})
	return endpointCh
}

//go:linkname registerHTTP3Transport net/http_test.registerHTTP3Transport
func registerHTTP3Transport(tr *http.Transport) <-chan *quic.Endpoint {
	endpointCh := make(chan *quic.Endpoint)
	RegisterTransport(tr, TransportOpts{
		ListenQUIC: func(addr string, config *quic.Config) (*quic.Endpoint, error) {
			e, err := quic.Listen("udp", addr, config)
			endpointCh <- e
			return e, err
		},
		QUICConfig: &quic.Config{HandshakeTimeout: handshakeTimeout},
	})
	return endpointCh
}
