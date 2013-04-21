// Copyright 2012 The Go Authors.  All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package http_test

import (
	. "net/http"
	"net/http/httptest"
	"net/url"
	"testing"
)

var serveMuxRegister = []struct {
	pattern string
	h       Handler
}{
	{"/dir/", serve(200)},
	{"/search", serve(201)},
	{"codesearch.google.com/search", serve(202)},
	{"codesearch.google.com/", serve(203)},
}

// serve returns a handler that sends a response with the given code.
func serve(code int) HandlerFunc {
	return func(w ResponseWriter, r *Request) {
		w.WriteHeader(code)
	}
}

var serveMuxTests = []struct {
	method  string
	host    string
	path    string
	code    int
	pattern string
}{
	{"GET", "google.com", "/", 404, ""},
	{"GET", "google.com", "/dir", 301, "/dir/"},
	{"GET", "google.com", "/dir/", 200, "/dir/"},
	{"GET", "google.com", "/dir/file", 200, "/dir/"},
	{"GET", "google.com", "/search", 201, "/search"},
	{"GET", "google.com", "/search/", 404, ""},
	{"GET", "google.com", "/search/foo", 404, ""},
	{"GET", "codesearch.google.com", "/search", 202, "codesearch.google.com/search"},
	{"GET", "codesearch.google.com", "/search/", 203, "codesearch.google.com/"},
	{"GET", "codesearch.google.com", "/search/foo", 203, "codesearch.google.com/"},
	{"GET", "codesearch.google.com", "/", 203, "codesearch.google.com/"},
	{"GET", "images.google.com", "/search", 201, "/search"},
	{"GET", "images.google.com", "/search/", 404, ""},
	{"GET", "images.google.com", "/search/foo", 404, ""},
	{"GET", "google.com", "/../search", 301, "/search"},
	{"GET", "google.com", "/dir/..", 301, ""},
	{"GET", "google.com", "/dir/..", 301, ""},
	{"GET", "google.com", "/dir/./file", 301, "/dir/"},

	// The /foo -> /foo/ redirect applies to CONNECT requests
	// but the path canonicalization does not.
	{"CONNECT", "google.com", "/dir", 301, "/dir/"},
	{"CONNECT", "google.com", "/../search", 404, ""},
	{"CONNECT", "google.com", "/dir/..", 200, "/dir/"},
	{"CONNECT", "google.com", "/dir/..", 200, "/dir/"},
	{"CONNECT", "google.com", "/dir/./file", 200, "/dir/"},
}

func TestServeMuxHandler(t *testing.T) {
	mux := NewServeMux()
	for _, e := range serveMuxRegister {
		mux.Handle(e.pattern, e.h)
	}

	for _, tt := range serveMuxTests {
		r := &Request{
			Method: tt.method,
			Host:   tt.host,
			URL: &url.URL{
				Path: tt.path,
			},
		}
		h, pattern := mux.Handler(r)
		rr := httptest.NewRecorder()
		h.ServeHTTP(rr, r)
		if pattern != tt.pattern || rr.Code != tt.code {
			t.Errorf("%s %s %s = %d, %q, want %d, %q", tt.method, tt.host, tt.path, rr.Code, pattern, tt.code, tt.pattern)
		}
	}
}

func TestServerRedirect(t *testing.T) {
	// This used to crash. It's not valid input (bad path), but it
	// shouldn't crash.
	rr := httptest.NewRecorder()
	req := &Request{
		Method: "GET",
		URL: &url.URL{
			Scheme: "http",
			Path:   "not-empty-but-no-leading-slash", // bogus
		},
	}
	Redirect(rr, req, "", 304)
	if rr.Code != 304 {
		t.Errorf("Code = %d; want 304", rr.Code)
	}
}
