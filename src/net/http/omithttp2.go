// Copyright 2019 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

//go:build nethttpomithttp2

package http

func init() {
	omitBundledHTTP2 = true
}

const noHTTP2 = "no bundled HTTP/2" // should never see this

func (s *Server) configureHTTP2()                       {}
func (t *Transport) configureHTTP2(protocols Protocols) {}

type http2Server struct{}

type http2RoundTripper struct{}

func (http2RoundTripper) RoundTrip(*Request) (*Response, error) { panic(noHTTP2) }
