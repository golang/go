// Copyright 2016 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// Package httptest provides utilities for HTTP testing.
package httptest

import (
	"bufio"
	"bytes"
	"context"
	"crypto/tls"
	"io"
	"net/http"
	"strings"
)

// NewRequest wraps NewRequestWithContext using context.Background.
func NewRequest(method, target string, body io.Reader) *http.Request {
	return NewRequestWithContext(context.Background(), method, target, body)
}

// NewRequestWithContext returns a new incoming server Request, suitable
// for passing to an [http.Handler] for testing.
//
// The target is the RFC 7230 "request-target": it may be either a
// path or an absolute URL. If target is an absolute URL, the host name
// from the URL is used. Otherwise, "example.com" is used.
//
// The TLS field is set to a non-nil dummy value if target has scheme
// "https".
//
// The Request.Proto is always HTTP/1.1.
//
// An empty method means "GET".
//
// The provided body may be nil. If the body is of type *bytes.Reader,
// *strings.Reader, or *bytes.Buffer, the Request.ContentLength is
// set.
//
// NewRequest panics on error for ease of use in testing, where a
// panic is acceptable.
//
// To generate a client HTTP request instead of a server request, see
// the NewRequest function in the net/http package.
func NewRequestWithContext(ctx context.Context, method, target string, body io.Reader) *http.Request {
	if method == "" {
		method = "GET"
	}
	req, err := http.ReadRequest(bufio.NewReader(strings.NewReader(method + " " + target + " HTTP/1.0\r\n\r\n")))
	if err != nil {
		panic("invalid NewRequest arguments; " + err.Error())
	}
	req = req.WithContext(ctx)

	// HTTP/1.0 was used above to avoid needing a Host field. Change it to 1.1 here.
	req.Proto = "HTTP/1.1"
	req.ProtoMinor = 1
	req.Close = false

	if body != nil {
		switch v := body.(type) {
		case *bytes.Buffer:
			req.ContentLength = int64(v.Len())
		case *bytes.Reader:
			req.ContentLength = int64(v.Len())
		case *strings.Reader:
			req.ContentLength = int64(v.Len())
		default:
			req.ContentLength = -1
		}
		if rc, ok := body.(io.ReadCloser); ok {
			req.Body = rc
		} else {
			req.Body = io.NopCloser(body)
		}
	}

	// 192.0.2.0/24 is "TEST-NET" in RFC 5737 for use solely in
	// documentation and example source code and should not be
	// used publicly.
	req.RemoteAddr = "192.0.2.1:1234"

	if req.Host == "" {
		req.Host = "example.com"
	}

	if strings.HasPrefix(target, "https://") {
		req.TLS = &tls.ConnectionState{
			Version:           tls.VersionTLS12,
			HandshakeComplete: true,
			ServerName:        req.Host,
		}
	}

	return req
}
