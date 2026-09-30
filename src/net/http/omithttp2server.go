// Copyright 2026 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

//go:build nethttpomithttp2 || nethttpomithttp2server

package http

import (
	"context"
	"net"
)

// omitHTTP2Server is true when the nethttpomithttp2 or
// nethttpomithttp2server build tag is set, which means http2server.go
// isn't compiled in and we shouldn't try to use it.
const omitHTTP2Server = true

func (s *Server) configureHTTP2()                               {}
func (s *Server) setHTTP2Config(conf http2ExternalServerConfig) {}
func (s *Server) serveHTTP2Conn(ctx context.Context, nc net.Conn, h Handler, sawClientPreface bool, upgradeReq *Request, settings []byte, onClose func()) {
}

type http2Server struct{}
type http2ExternalServerConfig interface {
	unimplementable()
}
