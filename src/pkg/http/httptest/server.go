// Copyright 2011 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// Implementation of TempServer

package httptest

import (
	"fmt"
	"http"
	"net"
)

// A Server is an HTTP server listening on a system-chosen port on the
// local loopback interface, for use in end-to-end HTTP tests.
type Server struct {
	URL      string // base URL of form http://ipaddr:port with no trailing slash
	Listener net.Listener
}

// NewServer starts and returns a new Server.
// The caller should call Close when finished, to shut it down.
func NewServer(handler http.Handler) *Server {
	ts := new(Server)
	l, err := net.Listen("tcp", "127.0.0.1:0")
	if err != nil {
		if l, err = net.Listen("tcp6", "[::1]:0"); err != nil {
			panic(fmt.Sprintf("httptest: failed to listen on a port: %v", err))
		}
	}
	ts.Listener = l
	ts.URL = "http://" + l.Addr().String()
	server := &http.Server{Handler: handler}
	go server.Serve(l)
	return ts
}

// Close shuts down the temporary server.
func (s *Server) Close() {
	s.Listener.Close()
}
