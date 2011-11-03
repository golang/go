// Copyright 2011 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// Tests for CGI (the child process perspective)

package cgi

import (
	"testing"
)

func TestRequest(t *testing.T) {
	env := map[string]string{
		"SERVER_PROTOCOL": "HTTP/1.1",
		"REQUEST_METHOD":  "GET",
		"HTTP_HOST":       "example.com",
		"HTTP_REFERER":    "elsewhere",
		"HTTP_USER_AGENT": "goclient",
		"HTTP_FOO_BAR":    "baz",
		"REQUEST_URI":     "/path?a=b",
		"CONTENT_LENGTH":  "123",
		"CONTENT_TYPE":    "text/xml",
		"HTTPS":           "1",
		"REMOTE_ADDR":     "5.6.7.8",
	}
	req, err := RequestFromMap(env)
	if err != nil {
		t.Fatalf("RequestFromMap: %v", err)
	}
	if g, e := req.UserAgent(), "goclient"; e != g {
		t.Errorf("expected UserAgent %q; got %q", e, g)
	}
	if g, e := req.Method, "GET"; e != g {
		t.Errorf("expected Method %q; got %q", e, g)
	}
	if g, e := req.Header.Get("Content-Type"), "text/xml"; e != g {
		t.Errorf("expected Content-Type %q; got %q", e, g)
	}
	if g, e := req.ContentLength, int64(123); e != g {
		t.Errorf("expected ContentLength %d; got %d", e, g)
	}
	if g, e := req.Referer(), "elsewhere"; e != g {
		t.Errorf("expected Referer %q; got %q", e, g)
	}
	if req.Header == nil {
		t.Fatalf("unexpected nil Header")
	}
	if g, e := req.Header.Get("Foo-Bar"), "baz"; e != g {
		t.Errorf("expected Foo-Bar %q; got %q", e, g)
	}
	if g, e := req.URL.String(), "http://example.com/path?a=b"; e != g {
		t.Errorf("expected URL %q; got %q", e, g)
	}
	if g, e := req.FormValue("a"), "b"; e != g {
		t.Errorf("expected FormValue(a) %q; got %q", e, g)
	}
	if req.Trailer == nil {
		t.Errorf("unexpected nil Trailer")
	}
	if req.TLS == nil {
		t.Errorf("expected non-nil TLS")
	}
	if e, g := "5.6.7.8:0", req.RemoteAddr; e != g {
		t.Errorf("RemoteAddr: got %q; want %q", g, e)
	}
}

func TestRequestWithoutHost(t *testing.T) {
	env := map[string]string{
		"SERVER_PROTOCOL": "HTTP/1.1",
		"HTTP_HOST":       "",
		"REQUEST_METHOD":  "GET",
		"REQUEST_URI":     "/path?a=b",
		"CONTENT_LENGTH":  "123",
	}
	req, err := RequestFromMap(env)
	if err != nil {
		t.Fatalf("RequestFromMap: %v", err)
	}
	if req.URL == nil {
		t.Fatalf("unexpected nil URL")
	}
	if g, e := req.URL.String(), "/path?a=b"; e != g {
		t.Errorf("expected URL %q; got %q", e, g)
	}
}
