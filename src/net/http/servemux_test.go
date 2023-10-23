// Copyright 2009 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package http_test

import (
	"fmt"
	. "net/http"
	"net/http/httptest"
	"net/url"
	"testing"
)

func TestServeMuxHandleRegisterPanic(t *testing.T) {
	ok := serve(200)
	serveMuxRegister := []struct {
		method  string
		path    string
		handler Handler
		result  string
	}{
		{"", "", ok, "http: invalid pattern"},
		{"", "/", nil, "http: nil handler"},
		{",", "", ok, "http: invalid pattern"},
		{"LOCK", "/", ok, "http: invalid method LOCK"},
		{"GET", "/get", ok, "<nil>"},
		{"GET", "/get", ok, "http: multiple registrations for GET /get"},
		{"ANY", "/any", ok, "<nil>"},
		{"ANY", "/any", ok, "http: multiple registrations for /any"},
	}

	mux := NewServeMux()
	for _, route := range serveMuxRegister {
		pattern := route.path
		if route.method != "" {
			pattern = route.method + " " + pattern
		}

		func() {
			defer func() {
				e := fmt.Sprint(recover())
				if e != route.result {
					t.Errorf("%s = %q, want %q", pattern, e, route.result)
				}
			}()
			mux.Handle(pattern, route.handler)
		}()
	}
}

func TestServeMuxHandle40x(t *testing.T) {
	serveMuxTests := []struct {
		method  string
		path    string
		code    int
		pattern string
	}{
		{"GET", "/", 403, ""},
		{"GET", "/index", 200, "/index"},
		{"PUT", "/index", 403, "/index"},
	}

	mux := NewServeMux()
	mux.HandleFunc("NOTFOUND", serve(403))
	mux.HandleFunc("405", serve(403))
	mux.HandleFunc("GET /index", serve(200))

	Handle("404", NotFoundHandler())
	HandleFunc("404", NotFound)

	for _, tt := range serveMuxTests {
		r := &Request{
			Method: tt.method,
			URL: &url.URL{
				Path: tt.path,
			},
		}
		h, pattern := mux.Handler(r)
		rr := httptest.NewRecorder()
		h.ServeHTTP(rr, r)
		if pattern != tt.pattern || rr.Code != tt.code {
			t.Errorf("%s %s = %d, %q, want %d, %q", tt.method, tt.path, rr.Code, pattern, tt.code, tt.pattern)
		}
	}
}

func TestServeMuxHandleMatchEudore(t *testing.T) {
	serveMuxTests := []struct {
		method  string
		path    string
		code    int
		pattern string
	}{
		{"GET", "/method/", 200, "/method/*"},
		{"POST", "/method/", 200, "/method/*"},
		{"PUT", "/method/", 200, "/method/*"},
		{"LOCK", "/method/", 405, "/method/*"},
		{"GET", "/method/get", 200, "/method/get"},
		{"PUT", "/method/get", 405, "/method/get"},
		{"TRACE", "/method/trace", 200, "/method/trace"},
		{"GET", "/var/v1/user", 200, "/var/*v"},
		{"GET", "/var/v1", 200, "/var/v1"},
		{"GET", "/var/v2", 200, "/var/v2"},
		{"GET", "/var/v3", 200, "/var/:v"},
		{"GET", "/admin/", 200, "/admin/*"}, // first route value
		{"GET", "/admin/index", 200, "/admin/*"},
		{"PUT", "/admin/src/", 405, "/admin/*"},
		{"GET", "/admin/list", 200, "/admin/*"},
		{"GET", "/group/list", 404, "/group/*"},
		{"GET", "/redirect//v1", 301, "/redirect/v1/"},
		{"GET", "/redirect/v1", 301, "/redirect/v1/"},
		{"GET", "/space tab", 200, "/space tab"},
		{"GET", "/NotFound", 404, ""},
		{"GET", "/noroot", 200, "/noroot"},
		{"GET", "/char/", 404, ""},
		{"GET", "/char////", 301, ""},
		{"GET", "//clean/", 301, "/clean/"},
	}

	ok := serve(200)
	mux := NewServeMux()
	mux.HandleFunc("get,POST /method/*", ok)
	mux.HandleFunc("ANY /method/*", ok)
	mux.HandleFunc("GET /method/get", ok)
	mux.HandleFunc("trace /method/trace", ok)

	mux.HandleFunc("/var/*v", ok)
	mux.HandleFunc("/var/v1", ok)
	mux.HandleFunc("/var/:v", ok)
	mux.HandleFunc("/var/v2", ok)

	mux.HandleFunc("/v/:v/create", ok)
	mux.HandleFunc("/v/:v/list", ok)

	admin := NewServeMux()
	admin.HandleFunc("/*", ok)
	admin.HandleFunc("/index", ok)
	admin.HandleFunc("/pprof/", ok)
	admin.HandleFunc("GET /src/*path", ok)
	mux.Handle("/admin/*", admin)
	mux.Handle("/group/*", NewServeMux())

	mux.HandleFunc("/redirect/v1/", ok)
	mux.HandleFunc("/space tab", ok)
	mux.HandleFunc("noroot", ok)
	mux.HandleFunc("/char////", ok)
	mux.HandleFunc("/clean/", ok)

	for _, tt := range serveMuxTests {
		r := &Request{
			Method: tt.method,
			URL: &url.URL{
				Path: tt.path,
			},
		}
		h, pattern := mux.Handler(r)
		rr := httptest.NewRecorder()
		h.ServeHTTP(rr, r)
		if pattern != tt.pattern || rr.Code != tt.code {
			t.Errorf("%s %s = %d, %q, want %d, %q", tt.method, tt.path, rr.Code, pattern, tt.code, tt.pattern)
		}
	}
}

func TestServeMuxPathValue(t *testing.T) {
	serveMuxPatterns := []struct {
		pattern string
		method  string
		url     string
		want    map[string]string
	}{
		{
			"/:a/is/:b/*c",
			"GET",
			"/now/is/the/time/for/all",
			map[string]string{
				"a": "now",
				"b": "the",
				"c": "time/for/all",
				"d": "",
			},
		},
		{
			"/names/:name:/*other",
			"GET",
			"/names/john/address",
			map[string]string{
				"name":  "john",
				"other": "address",
			},
		},
		{
			"/names/:name/there/*other",
			"GET",
			"/names/john doe/there/is/more",
			map[string]string{
				"name":  "john doe",
				"other": "is/more",
			},
		},
		{
			"/api/:",
			"GET",
			"/api/v1",
			map[string]string{":": "v1"},
		},
		{
			"/api/*",
			"GET",
			"/api/v1/v3",
			map[string]string{"*": "v1/v3"},
		},
		{
			"/api/:v1/v1",
			"GET",
			"/api/v1/v1",
			map[string]string{"v1": "v1"},
		},
		{
			"/api/:v2/v2",
			"GET",
			"/api/v2/v2",
			map[string]string{"v2": "v2"},
		},
	}

	mux := NewServeMux()
	for i := range serveMuxPatterns {
		route := serveMuxPatterns[i]
		mux.HandleFunc(route.pattern, func(w ResponseWriter, r *Request) {
			if r.GetPathValue(MuxValueRoute) != route.pattern {
				t.Errorf("%q, route: got %q, want %q", route.pattern, r.GetPathValue(MuxValueRoute), route.pattern)
			}
			for name, want := range route.want {
				got := r.GetPathValue(name)
				if got != want {
					t.Errorf("%q, %q: got %q, want %q", route.pattern, name, got, want)
				}
			}
		})
	}

	for _, route := range serveMuxPatterns {
		r := &Request{
			Method: route.method,
			URL:    &url.URL{Path: route.url},
		}
		w := httptest.NewRecorder()
		mux.ServeHTTP(w, r)
	}
}

func TestServeMuxHandlePri(t *testing.T) {
	mux := NewServeMux()
	mux.ServeHTTP(httptest.NewRecorder(), &Request{
		Method:     "PRI",
		RequestURI: "*",
		ProtoMajor: 1,
		ProtoMinor: 1,
		URL:        &url.URL{Path: "/"},
	})
	mux.ServeHTTP(httptest.NewRecorder(), &Request{
		Method: "GET",
		Host:   "[golang.org:",
		URL:    &url.URL{Path: "/"},
	})
}
