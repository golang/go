// Copyright 2026 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package http3

import (
	"net/http"
	_ "unsafe" // for linkname

	. "golang.org/x/net/internal/http3"
	"golang.org/x/net/quic"
)

//go:linkname registerHTTP3Server net/http_test.registerHTTP3Server
func registerHTTP3Server(s *http.Server) <-chan string {
	listenAddr := make(chan string)
	RegisterServer(s, ServerOpts{
		ListenQUIC: func(addr string, config *quic.Config) (*quic.Endpoint, error) {
			e, err := quic.Listen("udp", addr, config)
			listenAddr <- e.LocalAddr().String()
			return e, err
		},
	})
	return listenAddr
}

//go:linkname registerHTTP3Transport net/http_test.registerHTTP3Transport
func registerHTTP3Transport(tr *http.Transport) {
	RegisterTransport(tr)
}
