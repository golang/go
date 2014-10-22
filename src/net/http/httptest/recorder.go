// Copyright 2011 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// Package httptest provides utilities for HTTP testing.
package httptest

import (
	"bytes"
	"net/http"
)

// ResponseRecorder is an implementation of http.ResponseWriter that
// records its mutations for later inspection in tests.
type ResponseRecorder struct {
	Code      int           // the HTTP response code from WriteHeader
	HeaderMap http.Header   // the HTTP response headers
	Body      *bytes.Buffer // if non-nil, the bytes.Buffer to append written data to
	Flushed   bool

	wroteHeader bool
}

// NewRecorder returns an initialized ResponseRecorder.
func NewRecorder() *ResponseRecorder {
	return &ResponseRecorder{
		HeaderMap: make(http.Header),
		Body:      new(bytes.Buffer),
		Code:      200,
	}
}

// DefaultRemoteAddr is the default remote address to return in RemoteAddr if
// an explicit DefaultRemoteAddr isn't set on ResponseRecorder.
const DefaultRemoteAddr = "1.2.3.4"

// Header returns the response headers.
func (rw *ResponseRecorder) Header() http.Header {
	m := rw.HeaderMap
	if m == nil {
		m = make(http.Header)
		rw.HeaderMap = m
	}
	return m
}

// Write always succeeds and writes to rw.Body, if not nil.
func (rw *ResponseRecorder) Write(buf []byte) (int, error) {
	if !rw.wroteHeader {
		rw.WriteHeader(200)
	}
	if rw.Body != nil {
		rw.Body.Write(buf)
	}
	return len(buf), nil
}

// WriteHeader sets rw.Code.
func (rw *ResponseRecorder) WriteHeader(code int) {
	if !rw.wroteHeader {
		rw.Code = code
	}
	rw.wroteHeader = true
}

// Flush sets rw.Flushed to true.
func (rw *ResponseRecorder) Flush() {
	if !rw.wroteHeader {
		rw.WriteHeader(200)
	}
	rw.Flushed = true
}
