// Copyright 2010 The Go Authors.  All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package http

import (
	"bufio"
	"bytes"
	"fmt"
	"io"
	"testing"
	"url"
)

type reqTest struct {
	Raw   string
	Req   *Request
	Body  string
	Error string
}

var noError = ""
var noBody = ""

var reqTests = []reqTest{
	// Baseline test; All Request fields included for template use
	{
		"GET http://www.techcrunch.com/ HTTP/1.1\r\n" +
			"Host: www.techcrunch.com\r\n" +
			"User-Agent: Fake\r\n" +
			"Accept: text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8\r\n" +
			"Accept-Language: en-us,en;q=0.5\r\n" +
			"Accept-Encoding: gzip,deflate\r\n" +
			"Accept-Charset: ISO-8859-1,utf-8;q=0.7,*;q=0.7\r\n" +
			"Keep-Alive: 300\r\n" +
			"Content-Length: 7\r\n" +
			"Proxy-Connection: keep-alive\r\n\r\n" +
			"abcdef\n???",

		&Request{
			Method: "GET",
			URL: &url.URL{
				Raw:          "http://www.techcrunch.com/",
				Scheme:       "http",
				RawPath:      "/",
				RawAuthority: "www.techcrunch.com",
				RawUserinfo:  "",
				Host:         "www.techcrunch.com",
				Path:         "/",
				RawQuery:     "",
				Fragment:     "",
			},
			Proto:      "HTTP/1.1",
			ProtoMajor: 1,
			ProtoMinor: 1,
			Header: Header{
				"Accept":           {"text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8"},
				"Accept-Language":  {"en-us,en;q=0.5"},
				"Accept-Encoding":  {"gzip,deflate"},
				"Accept-Charset":   {"ISO-8859-1,utf-8;q=0.7,*;q=0.7"},
				"Keep-Alive":       {"300"},
				"Proxy-Connection": {"keep-alive"},
				"Content-Length":   {"7"},
				"User-Agent":       {"Fake"},
			},
			Close:         false,
			ContentLength: 7,
			Host:          "www.techcrunch.com",
			Form:          url.Values{},
		},

		"abcdef\n",

		noError,
	},

	// GET request with no body (the normal case)
	{
		"GET / HTTP/1.1\r\n" +
			"Host: foo.com\r\n\r\n",

		&Request{
			Method: "GET",
			URL: &url.URL{
				Raw:     "/",
				Path:    "/",
				RawPath: "/",
			},
			Proto:         "HTTP/1.1",
			ProtoMajor:    1,
			ProtoMinor:    1,
			Close:         false,
			ContentLength: 0,
			Host:          "foo.com",
			Form:          url.Values{},
		},

		noBody,
		noError,
	},

	// Tests that we don't parse a path that looks like a
	// scheme-relative URI as a scheme-relative URI.
	{
		"GET //user@host/is/actually/a/path/ HTTP/1.1\r\n" +
			"Host: test\r\n\r\n",

		&Request{
			Method: "GET",
			URL: &url.URL{
				Raw:          "//user@host/is/actually/a/path/",
				Scheme:       "",
				RawPath:      "//user@host/is/actually/a/path/",
				RawAuthority: "",
				RawUserinfo:  "",
				Host:         "",
				Path:         "//user@host/is/actually/a/path/",
				RawQuery:     "",
				Fragment:     "",
			},
			Proto:         "HTTP/1.1",
			ProtoMajor:    1,
			ProtoMinor:    1,
			Header:        Header{},
			Close:         false,
			ContentLength: 0,
			Host:          "test",
			Form:          url.Values{},
		},

		noBody,
		noError,
	},

	// Tests a bogus abs_path on the Request-Line (RFC 2616 section 5.1.2)
	{
		"GET ../../../../etc/passwd HTTP/1.1\r\n" +
			"Host: test\r\n\r\n",
		nil,
		noBody,
		"parse ../../../../etc/passwd: invalid URI for request",
	},

	// Tests missing URL:
	{
		"GET  HTTP/1.1\r\n" +
			"Host: test\r\n\r\n",
		nil,
		noBody,
		"parse : empty url",
	},
}

func TestReadRequest(t *testing.T) {
	for i := range reqTests {
		tt := &reqTests[i]
		var braw bytes.Buffer
		braw.WriteString(tt.Raw)
		req, err := ReadRequest(bufio.NewReader(&braw))
		if err != nil {
			if err.Error() != tt.Error {
				t.Errorf("#%d: error %q, want error %q", i, err.Error(), tt.Error)
			}
			continue
		}
		rbody := req.Body
		req.Body = nil
		diff(t, fmt.Sprintf("#%d Request", i), req, tt.Req)
		var bout bytes.Buffer
		if rbody != nil {
			io.Copy(&bout, rbody)
			rbody.Close()
		}
		body := bout.String()
		if body != tt.Body {
			t.Errorf("#%d: Body = %q want %q", i, body, tt.Body)
		}
	}
}
