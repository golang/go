// Copyright 2011 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// Tests for CGI (the child process perspective)

package cgi

import (
	"bufio"
	"bytes"
	"net/http"
	"net/http/httptest"
	"strings"
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
		"REMOTE_ADDR":     "5.6.7.8",
		"REMOTE_PORT":     "54321",
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
	if req.TLS != nil {
		t.Errorf("expected nil TLS")
	}
	if e, g := "5.6.7.8:54321", req.RemoteAddr; e != g {
		t.Errorf("RemoteAddr: got %q; want %q", g, e)
	}
}

func TestRequestWithTLS(t *testing.T) {
	env := map[string]string{
		"SERVER_PROTOCOL": "HTTP/1.1",
		"REQUEST_METHOD":  "GET",
		"HTTP_HOST":       "example.com",
		"HTTP_REFERER":    "elsewhere",
		"REQUEST_URI":     "/path?a=b",
		"CONTENT_TYPE":    "text/xml",
		"HTTPS":           "1",
		"REMOTE_ADDR":     "5.6.7.8",
	}
	req, err := RequestFromMap(env)
	if err != nil {
		t.Fatalf("RequestFromMap: %v", err)
	}
	if g, e := req.URL.String(), "https://example.com/path?a=b"; e != g {
		t.Errorf("expected URL %q; got %q", e, g)
	}
	if req.TLS == nil {
		t.Errorf("expected non-nil TLS")
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
		t.Errorf("URL = %q; want %q", g, e)
	}
}

func TestRequestWithoutRequestURI(t *testing.T) {
	env := map[string]string{
		"SERVER_PROTOCOL": "HTTP/1.1",
		"HTTP_HOST":       "example.com",
		"REQUEST_METHOD":  "GET",
		"SCRIPT_NAME":     "/dir/scriptname",
		"PATH_INFO":       "/p1/p2",
		"QUERY_STRING":    "a=1&b=2",
		"CONTENT_LENGTH":  "123",
	}
	req, err := RequestFromMap(env)
	if err != nil {
		t.Fatalf("RequestFromMap: %v", err)
	}
	if req.URL == nil {
		t.Fatalf("unexpected nil URL")
	}
	if g, e := req.URL.String(), "http://example.com/dir/scriptname/p1/p2?a=1&b=2"; e != g {
		t.Errorf("URL = %q; want %q", g, e)
	}
}

func TestRequestWithoutRemotePort(t *testing.T) {
	env := map[string]string{
		"SERVER_PROTOCOL": "HTTP/1.1",
		"HTTP_HOST":       "example.com",
		"REQUEST_METHOD":  "GET",
		"REQUEST_URI":     "/path?a=b",
		"CONTENT_LENGTH":  "123",
		"REMOTE_ADDR":     "5.6.7.8",
	}
	req, err := RequestFromMap(env)
	if err != nil {
		t.Fatalf("RequestFromMap: %v", err)
	}
	if e, g := "5.6.7.8:0", req.RemoteAddr; e != g {
		t.Errorf("RemoteAddr: got %q; want %q", g, e)
	}
}

func TestResponse(t *testing.T) {
	var tests = []struct {
		name   string
		body   string
		wantCT string
	}{
		{
			name:   "no body",
			wantCT: "text/plain; charset=utf-8",
		},
		{
			name:   "html",
			body:   "<html><head><title>test page</title></head><body>This is a body</body></html>",
			wantCT: "text/html; charset=utf-8",
		},
		{
			name:   "text",
			body:   strings.Repeat("gopher", 86),
			wantCT: "text/plain; charset=utf-8",
		},
		{
			name:   "jpg",
			body:   "\xFF\xD8\xFF" + strings.Repeat("B", 1024),
			wantCT: "image/jpeg",
		},
	}
	for _, tt := range tests {
		t.Run(tt.name, func { t ->
			var buf bytes.Buffer
			resp := response{
				req:    httptest.NewRequest("GET", "/", nil),
				header: http.Header{},
				bufw:   bufio.NewWriter(&buf),
			}
			n, err := resp.Write([]byte(tt.body))
			if err != nil {
				t.Errorf("Write: unexpected %v", err)
			}
			if want := len(tt.body); n != want {
				t.Errorf("reported short Write: got %v want %v", n, want)
			}
			resp.writeCGIHeader(nil)
			resp.Flush()
			if got := resp.Header().Get("Content-Type"); got != tt.wantCT {
				t.Errorf("wrong content-type: got %q, want %q", got, tt.wantCT)
			}
			if !bytes.HasSuffix(buf.Bytes(), []byte(tt.body)) {
				t.Errorf("body was not correctly written")
			}
		})
	}
}
