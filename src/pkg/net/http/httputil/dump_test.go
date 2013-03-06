// Copyright 2011 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package httputil

import (
	"bytes"
	"fmt"
	"io"
	"io/ioutil"
	"net/http"
	"net/url"
	"testing"
)

type dumpTest struct {
	Req  http.Request
	Body interface{} // optional []byte or func() io.ReadCloser to populate Req.Body

	WantDump    string
	WantDumpOut string
}

var dumpTests = []dumpTest{

	// HTTP/1.1 => chunked coding; body; empty trailer
	{
		Req: http.Request{
			Method: "GET",
			URL: &url.URL{
				Scheme: "http",
				Host:   "www.google.com",
				Path:   "/search",
			},
			ProtoMajor:       1,
			ProtoMinor:       1,
			TransferEncoding: []string{"chunked"},
		},

		Body: []byte("abcdef"),

		WantDump: "GET /search HTTP/1.1\r\n" +
			"Host: www.google.com\r\n" +
			"Transfer-Encoding: chunked\r\n\r\n" +
			chunk("abcdef") + chunk(""),
	},

	// Verify that DumpRequest preserves the HTTP version number, doesn't add a Host,
	// and doesn't add a User-Agent.
	{
		Req: http.Request{
			Method:     "GET",
			URL:        mustParseURL("/foo"),
			ProtoMajor: 1,
			ProtoMinor: 0,
			Header: http.Header{
				"X-Foo": []string{"X-Bar"},
			},
		},

		WantDump: "GET /foo HTTP/1.0\r\n" +
			"X-Foo: X-Bar\r\n\r\n",
	},

	{
		Req: *mustNewRequest("GET", "http://example.com/foo", nil),

		WantDumpOut: "GET /foo HTTP/1.1\r\n" +
			"Host: example.com\r\n" +
			"User-Agent: Go 1.1 package http\r\n" +
			"Accept-Encoding: gzip\r\n\r\n",
	},

	// Test that an https URL doesn't try to do an SSL negotiation
	// with a bytes.Buffer and hang with all goroutines not
	// runnable.
	{
		Req: *mustNewRequest("GET", "https://example.com/foo", nil),

		WantDumpOut: "GET /foo HTTP/1.1\r\n" +
			"Host: example.com\r\n" +
			"User-Agent: Go 1.1 package http\r\n" +
			"Accept-Encoding: gzip\r\n\r\n",
	},
}

func TestDumpRequest(t *testing.T) {
	for i, tt := range dumpTests {
		setBody := func() {
			if tt.Body == nil {
				return
			}
			switch b := tt.Body.(type) {
			case []byte:
				tt.Req.Body = ioutil.NopCloser(bytes.NewBuffer(b))
			case func() io.ReadCloser:
				tt.Req.Body = b()
			}
		}
		setBody()
		if tt.Req.Header == nil {
			tt.Req.Header = make(http.Header)
		}

		if tt.WantDump != "" {
			setBody()
			dump, err := DumpRequest(&tt.Req, true)
			if err != nil {
				t.Errorf("DumpRequest #%d: %s", i, err)
				continue
			}
			if string(dump) != tt.WantDump {
				t.Errorf("DumpRequest %d, expecting:\n%s\nGot:\n%s\n", i, tt.WantDump, string(dump))
				continue
			}
		}

		if tt.WantDumpOut != "" {
			setBody()
			dump, err := DumpRequestOut(&tt.Req, true)
			if err != nil {
				t.Errorf("DumpRequestOut #%d: %s", i, err)
				continue
			}
			if string(dump) != tt.WantDumpOut {
				t.Errorf("DumpRequestOut %d, expecting:\n%s\nGot:\n%s\n", i, tt.WantDumpOut, string(dump))
				continue
			}
		}
	}
}

func chunk(s string) string {
	return fmt.Sprintf("%x\r\n%s\r\n", len(s), s)
}

func mustParseURL(s string) *url.URL {
	u, err := url.Parse(s)
	if err != nil {
		panic(fmt.Sprintf("Error parsing URL %q: %v", s, err))
	}
	return u
}

func mustNewRequest(method, url string, body io.Reader) *http.Request {
	req, err := http.NewRequest(method, url, body)
	if err != nil {
		panic(fmt.Sprintf("NewRequest(%q, %q, %p) err = %v", method, url, body, err))
	}
	return req
}
