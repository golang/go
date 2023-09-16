// Copyright 2018 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// Server unit tests

package http

import (
	"fmt"
	"net/url"
	"regexp"
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

func TestFindHandler(t *testing.T) {
	mux := NewServeMux()
	for _, ph := range []struct {
		pat string
		h   Handler
	}{
		{"/", &handler{1}},
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
		{"GET", "/", "&http.handler{i:1}"},
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
		gotH, _, _, _ := mux.findHandler(&r)
		got := fmt.Sprintf("%#v", gotH)
		if got != test.wantHandler {
			t.Errorf("%s %q: got %q, want %q", test.method, test.path, got, test.wantHandler)
		}
	}
}

func TestRegisterErr(t *testing.T) {
	mux := NewServeMux()
	h := &handler{}
	mux.Handle("/a", h)

	for _, test := range []struct {
		pattern    string
		handler    Handler
		wantRegexp string
	}{
		{"", h, "invalid pattern"},
		{"/", nil, "nil handler"},
		{"/", HandlerFunc(nil), "nil handler"},
		{"/{x", h, `parsing "/\{x": bad wildcard segment`},
		{"/a", h, `conflicts with pattern.* \(registered at .*/server_test.go:\d+`},
	} {
		t.Run(fmt.Sprintf("%s:%#v", test.pattern, test.handler), func(t *testing.T) {
			err := mux.registerErr(test.pattern, test.handler)
			if err == nil {
				t.Fatal("got nil error")
			}
			re := regexp.MustCompile(test.wantRegexp)
			if g := err.Error(); !re.MatchString(g) {
				t.Errorf("\ngot %q\nwant string matching %q", g, test.wantRegexp)
			}
		})
	}
}

func TestExactMatch(t *testing.T) {
	for _, test := range []struct {
		pattern string
		path    string
		want    bool
	}{
		{"", "/a", false},
		{"/", "/a", false},
		{"/a", "/a", true},
		{"/a/{x...}", "/a/b", false},
		{"/a/{x}", "/a/b", true},
		{"/a/b/", "/a/b/", true},
		{"/a/b/{$}", "/a/b/", true},
		{"/a/", "/a/b/", false},
	} {
		var n *routingNode
		if test.pattern != "" {
			pat := mustParsePattern(t, test.pattern)
			n = &routingNode{pattern: pat}
		}
		got := exactMatch(n, test.path)
		if got != test.want {
			t.Errorf("%q, %s: got %t, want %t", test.pattern, test.path, got, test.want)
		}
	}
}

func BenchmarkServerMatch(b *testing.B) {
	fn := func(w ResponseWriter, r *Request) {
		fmt.Fprintf(w, "OK")
	}
	mux := NewServeMux()
	mux.HandleFunc("/", fn)
	mux.HandleFunc("/index", fn)
	mux.HandleFunc("/home", fn)
	mux.HandleFunc("/about", fn)
	mux.HandleFunc("/contact", fn)
	mux.HandleFunc("/robots.txt", fn)
	mux.HandleFunc("/products/", fn)
	mux.HandleFunc("/products/1", fn)
	mux.HandleFunc("/products/2", fn)
	mux.HandleFunc("/products/3", fn)
	mux.HandleFunc("/products/3/image.jpg", fn)
	mux.HandleFunc("/admin", fn)
	mux.HandleFunc("/admin/products/", fn)
	mux.HandleFunc("/admin/products/create", fn)
	mux.HandleFunc("/admin/products/update", fn)
	mux.HandleFunc("/admin/products/delete", fn)

	paths := []string{"/", "/notfound", "/admin/", "/admin/foo", "/contact", "/products",
		"/products/", "/products/3/image.jpg"}
	b.StartTimer()
	for i := 0; i < b.N; i++ {
		r, err := NewRequest("GET", "http://example.com/"+paths[i%len(paths)], nil)
		if err != nil {
			b.Fatal(err)
		}
		if h, p, _, _ := mux.findHandler(r); h != nil && p == "" {
			b.Error("impossible")
		}
	}
	b.StopTimer()
}
