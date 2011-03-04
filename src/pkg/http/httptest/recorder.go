// Copyright 2011 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// The httptest package provides utilities for HTTP testing.
package httptest

import (
	"bufio"
	"bytes"
	"http"
	"io"
	"os"
)

// ResponseRecorder is an implementation of http.ResponseWriter that
// records its mutations for later inspection in tests.
//
// Note that Hijack is not implemented and simply panics.
type ResponseRecorder struct {
	Code    int           // the HTTP response code from WriteHeader
	Header  http.Header   // if non-nil, the headers to populate
	Body    *bytes.Buffer // if non-nil, the bytes.Buffer to append written data to
	Flushed bool

	FakeRemoteAddr string // the fake RemoteAddr to return, or "" for DefaultRemoteAddr
	FakeUsingTLS   bool   // whether to return true from the UsingTLS method
}

// NewRecorder returns an initialized ResponseRecorder.
func NewRecorder() *ResponseRecorder {
	return &ResponseRecorder{
		Header: http.Header(make(map[string][]string)),
		Body:   new(bytes.Buffer),
	}
}

// DefaultRemoteAddr is the default remote address to return in RemoteAddr if
// an explicit DefaultRemoteAddr isn't set on ResponseRecorder.
const DefaultRemoteAddr = "1.2.3.4"

// RemoteAddr returns the value of rw.FakeRemoteAddr, if set, else
// returns DefaultRemoteAddr.
func (rw *ResponseRecorder) RemoteAddr() string {
	if rw.FakeRemoteAddr != "" {
		return rw.FakeRemoteAddr
	}
	return DefaultRemoteAddr
}

// UsingTLS returns the fake value in rw.FakeUsingTLS
func (rw *ResponseRecorder) UsingTLS() bool {
	return rw.FakeUsingTLS
}

// SetHeader populates rw.Header, if non-nil.
func (rw *ResponseRecorder) SetHeader(k, v string) {
	if rw.Header != nil {
		if v == "" {
			rw.Header.Del(k)
		} else {
			rw.Header.Set(k, v)
		}
	}
}

// Write always succeeds and writes to rw.Body, if not nil.
func (rw *ResponseRecorder) Write(buf []byte) (int, os.Error) {
	if rw.Body != nil {
		rw.Body.Write(buf)
	}
	return len(buf), nil
}

// WriteHeader sets rw.Code.
func (rw *ResponseRecorder) WriteHeader(code int) {
	rw.Code = code
}

// Flush sets rw.Flushed to true.
func (rw *ResponseRecorder) Flush() {
	rw.Flushed = true
}

// Hijack is not implemented in ResponseRecorder and instead panics.
func (rw *ResponseRecorder) Hijack() (io.ReadWriteCloser, *bufio.ReadWriter, os.Error) {
	panic("Hijack not implemented in ResponseRecorder")
}
