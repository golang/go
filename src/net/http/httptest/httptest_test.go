// Copyright 2016 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package httptest

import (
	"crypto/tls"
	"io"
	"io/ioutil"
	"net/http"
	"net/url"
	"reflect"
	"strings"
	"testing"
)

func TestNewRequest(t *testing.T) {
	tests := [...]struct {
		method, uri string
		body        io.Reader

		want     *http.Request
		wantBody string
	}{
		// Empty method means GET:
		0: {
			method: "",
			uri:    "/",
			body:   nil,
			want: &http.Request{
				Method:     "GET",
				Host:       "example.com",
				URL:        &url.URL{Path: "/"},
				Header:     http.Header{},
				Proto:      "HTTP/1.1",
				ProtoMajor: 1,
				ProtoMinor: 1,
				RemoteAddr: "192.0.2.1:1234",
				RequestURI: "/",
			},
			wantBody: "",
		},

		// GET with full URL:
		1: {
			method: "GET",
			uri:    "http://foo.com/path/%2f/bar/",
			body:   nil,
			want: &http.Request{
				Method: "GET",
				Host:   "foo.com",
				URL: &url.URL{
					Scheme:  "http",
					Path:    "/path///bar/",
					RawPath: "/path/%2f/bar/",
					Host:    "foo.com",
				},
				Header:     http.Header{},
				Proto:      "HTTP/1.1",
				ProtoMajor: 1,
				ProtoMinor: 1,
				RemoteAddr: "192.0.2.1:1234",
				RequestURI: "http://foo.com/path/%2f/bar/",
			},
			wantBody: "",
		},

		// GET with full https URL:
		2: {
			method: "GET",
			uri:    "https://foo.com/path/",
			body:   nil,
			want: &http.Request{
				Method: "GET",
				Host:   "foo.com",
				URL: &url.URL{
					Scheme: "https",
					Path:   "/path/",
					Host:   "foo.com",
				},
				Header:     http.Header{},
				Proto:      "HTTP/1.1",
				ProtoMajor: 1,
				ProtoMinor: 1,
				RemoteAddr: "192.0.2.1:1234",
				RequestURI: "https://foo.com/path/",
				TLS: &tls.ConnectionState{
					Version:           tls.VersionTLS12,
					HandshakeComplete: true,
					ServerName:        "foo.com",
				},
			},
			wantBody: "",
		},

		// Post with known length
		3: {
			method: "POST",
			uri:    "/",
			body:   strings.NewReader("foo"),
			want: &http.Request{
				Method:        "POST",
				Host:          "example.com",
				URL:           &url.URL{Path: "/"},
				Header:        http.Header{},
				Proto:         "HTTP/1.1",
				ContentLength: 3,
				ProtoMajor:    1,
				ProtoMinor:    1,
				RemoteAddr:    "192.0.2.1:1234",
				RequestURI:    "/",
			},
			wantBody: "foo",
		},

		// Post with unknown length
		4: {
			method: "POST",
			uri:    "/",
			body:   struct{ io.Reader }{strings.NewReader("foo")},
			want: &http.Request{
				Method:        "POST",
				Host:          "example.com",
				URL:           &url.URL{Path: "/"},
				Header:        http.Header{},
				Proto:         "HTTP/1.1",
				ContentLength: -1,
				ProtoMajor:    1,
				ProtoMinor:    1,
				RemoteAddr:    "192.0.2.1:1234",
				RequestURI:    "/",
			},
			wantBody: "foo",
		},

		// OPTIONS *
		5: {
			method: "OPTIONS",
			uri:    "*",
			want: &http.Request{
				Method:     "OPTIONS",
				Host:       "example.com",
				URL:        &url.URL{Path: "*"},
				Header:     http.Header{},
				Proto:      "HTTP/1.1",
				ProtoMajor: 1,
				ProtoMinor: 1,
				RemoteAddr: "192.0.2.1:1234",
				RequestURI: "*",
			},
		},
	}
	for i, tt := range tests {
		got := NewRequest(tt.method, tt.uri, tt.body)
		slurp, err := ioutil.ReadAll(got.Body)
		if err != nil {
			t.Errorf("%d. ReadAll: %v", i, err)
		}
		if string(slurp) != tt.wantBody {
			t.Errorf("%d. Body = %q; want %q", i, slurp, tt.wantBody)
		}
		got.Body = nil // before DeepEqual
		if !reflect.DeepEqual(got.URL, tt.want.URL) {
			t.Errorf("%d. Request.URL mismatch:\n got: %#v\nwant: %#v", i, got.URL, tt.want.URL)
		}
		if !reflect.DeepEqual(got.Header, tt.want.Header) {
			t.Errorf("%d. Request.Header mismatch:\n got: %#v\nwant: %#v", i, got.Header, tt.want.Header)
		}
		if !reflect.DeepEqual(got.TLS, tt.want.TLS) {
			t.Errorf("%d. Request.TLS mismatch:\n got: %#v\nwant: %#v", i, got.TLS, tt.want.TLS)
		}
		if !reflect.DeepEqual(got, tt.want) {
			t.Errorf("%d. Request mismatch:\n got: %#v\nwant: %#v", i, got, tt.want)
		}
	}
}
