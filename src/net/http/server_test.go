// Copyright 2018 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// Server unit tests

package http

import (
	"fmt"
	"net/url"
	"testing"
	"time"
)

func TestServerTLSHandshakeTimeout(t *testing.T) {
	tests := []struct {
		s    *Server
		want time.Duration
	}{
		{
			s:    &Server{},
			want: 0,
		},
		{
			s: &Server{
				ReadTimeout: -1,
			},
			want: 0,
		},
		{
			s: &Server{
				ReadTimeout: 5 * time.Second,
			},
			want: 5 * time.Second,
		},
		{
			s: &Server{
				ReadTimeout:  5 * time.Second,
				WriteTimeout: -1,
			},
			want: 5 * time.Second,
		},
		{
			s: &Server{
				ReadTimeout:  5 * time.Second,
				WriteTimeout: 4 * time.Second,
			},
			want: 4 * time.Second,
		},
		{
			s: &Server{
				ReadTimeout:       5 * time.Second,
				ReadHeaderTimeout: 2 * time.Second,
				WriteTimeout:      4 * time.Second,
			},
			want: 2 * time.Second,
		},
	}
	for i, tt := range tests {
		got := tt.s.tlsHandshakeTimeout()
		if got != tt.want {
			t.Errorf("%d. got %v; want %v", i, got, tt.want)
		}
	}
}

type handler struct{ i int }

func (handler) ServeHTTP(ResponseWriter, *Request) {}

func TestServeMuxFindHandler(t *testing.T) {
	mux := NewServeMux()
	for _, ph := range []struct {
		pat string
		h   Handler
	}{
		{"/", &handler{1}},
		{"GET /", &handler{11}},
		{"/foo/", &handler{2}},
		{"/foo", &handler{3}},
		{"/bar/", &handler{4}},
		{"//foo", &handler{5}},
	} {
		mux.Handle(ph.pat, ph.h)
	}

	for _, test := range []struct {
		method      string
		path        string
		wantHandler string
	}{
		{"GET", "/", "&http.handler{i:11}"},
		{"GET", "//", `&http.redirectHandler{url:"/", code:301}`},
		{"GET", "/foo/../bar/./..//baz", `&http.redirectHandler{url:"/baz", code:301}`},
		{"GET", "/foo", "&http.handler{i:3}"},
		{"GET", "/foo/x", "&http.handler{i:2}"},
		{"GET", "/bar/x", "&http.handler{i:4}"},
		{"GET", "/bar", `&http.redirectHandler{url:"/bar/", code:301}`},
		{"CONNECT", "/", "&http.handler{i:1}"},
		{"CONNECT", "//", "&http.handler{i:1}"},
		{"CONNECT", "//foo", "&http.handler{i:5}"},
		{"CONNECT", "/foo/../bar/./..//baz", "&http.handler{i:2}"},
		{"CONNECT", "/foo", "&http.handler{i:3}"},
		{"CONNECT", "/foo/x", "&http.handler{i:2}"},
		{"CONNECT", "/bar/x", "&http.handler{i:4}"},
		{"CONNECT", "/bar", `&http.redirectHandler{url:"/bar/", code:301}`},
	} {
		var r Request
		r.Method = test.method
		r.Host = "example.com"
		r.URL = &url.URL{Path: test.path}
		gotH, _ := mux.handler(&r)
		got := fmt.Sprintf("%#v", gotH)
		if got != test.wantHandler {
			t.Errorf("%s %q: got %q, want %q", test.method, test.path, got, test.wantHandler)
		}
	}
}

func TestServeMuxEmpty(t *testing.T) {
	// Verify that a ServeMux with nothing registered
	// doesn't panic.
	mux := NewServeMux()
	var r Request
	r.Method = "GET"
	r.Host = "example.com"
	r.URL = &url.URL{Path: "/"}
	_, p := mux.Handler(&r)
	if p != "" {
		t.Errorf(`got %q, want ""`, p)
	}
}

func TestServeMuxHandleStatic(t *testing.T) {
	t.Run("1.21", func(t *testing.T) { testServeMuxHandleStatic(t, true) })
	t.Run("latest", func(t *testing.T) { testServeMuxHandleStatic(t, false) })
}


func testServeMuxHandleStatic(t *testing.T, test121 bool) {
	defer func(u bool) { use121 = u }(use121)
	use121 = test121

	type patternData struct {
		path    string
		pattern string
	}
	serveMuxTests := []patternData{
		{"/", "/"},
		{"/index", "/index"},
		{"/home", "/home"},
		{"/about", "/about"},
		{"/contact", "/contact"},
		{"/robots.txt", "/robots.txt"},
		{"/products/", "/products/"},
		{"/products/1", "/products/1"},
		{"/products/2", "/products/2"},
		{"/products/3", "/products/3"},
		{"/products/3/image.jpg", "/products/3/image.jpg"},
	}

	ok := func(w ResponseWriter, r *Request) {}
	mux := NewServeMux()
	for _, route := range serveMuxTests {
		mux.HandleFunc(route.path, ok)
	}
	mux.Handle("/src", HandlerFunc(ok))

	serveMuxTests = append(serveMuxTests, 
		patternData{"/pprof", "/"}, 
		patternData{"/products/4", "/products/"},
	)
	for _, route := range serveMuxTests {
		r := &Request{
			Method: "GET",
			URL:    &url.URL{Path: route.path},
		}
		_, pattern := mux.Handler(r)
		mux.ServeHTTP(nil, r)
		if pattern != route.pattern {
			t.Errorf("%q, want %q", pattern, route.pattern)
		}
	}

}
