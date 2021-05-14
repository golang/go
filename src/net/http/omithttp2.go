// Copyright 2019 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

//go:build nethttpomithttp2
// +build nethttpomithttp2

package http

import (
	"errors"
	"sync"
	"time"
)

func init() {
	omitBundledHTTP2 = true
}

const noHTTP2 = "no bundled HTTP/2" // should never see this

var http2errRequestCanceled = errors.New("net/http: request canceled")

var http2goAwayTimeout = 1 * time.Second

const http2NextProtoTLS = "h2"

type http2Transport struct {
	MaxHeaderListSize uint32
	ConnPool          interface{}
}

func (*http2Transport) RoundTrip(*Request) (*Response, error) { panic(noHTTP2) }
func (*http2Transport) CloseIdleConnections()                 {}

type http2noDialH2RoundTripper struct{}

func (http2noDialH2RoundTripper) RoundTrip(*Request) (*Response, error) { panic(noHTTP2) }

type http2noDialClientConnPool struct {
	http2clientConnPool http2clientConnPool
}

type http2clientConnPool struct {
	mu    *sync.Mutex
	conns map[string][]struct{}
}

func http2configureTransports(*Transport) (*http2Transport, error) { panic(noHTTP2) }

func http2isNoCachedConnError(err error) bool {
	_, ok := err.(interface{ IsHTTP2NoCachedConnError() })
	return ok
}

type http2Server struct {
	NewWriteScheduler func() http2WriteScheduler
}

type http2WriteScheduler interface{}

func http2NewPriorityWriteScheduler(interface{}) http2WriteScheduler { panic(noHTTP2) }

func http2ConfigureServer(s *Server, conf *http2Server) error { panic(noHTTP2) }

var http2ErrNoCachedConn = http2noCachedConnError{}

type http2noCachedConnError struct{}

func (http2noCachedConnError) IsHTTP2NoCachedConnError() {}

func (http2noCachedConnError) Error() string { return "http2: no cached connection was available" }
