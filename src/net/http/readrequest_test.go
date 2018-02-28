// Copyright 2010 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package http

import (
	"bufio"
	"bytes"
	"fmt"
	"io"
	"io/ioutil"
	"net/url"
	"reflect"
	"strings"
	"testing"
)

type reqTest struct {
	Raw     string
	Req     *Request
	Body    string
	Trailer Header
	Error   string
}

var noError = ""
var noBodyStr = ""
var noTrailer Header = nil

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
				Scheme: "http",
				Host:   "www.techcrunch.com",
				Path:   "/",
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
			RequestURI:    "http://www.techcrunch.com/",
		},

		"abcdef\n",

		noTrailer,
		noError,
	},

	// GET request with no body (the normal case)
	{
		"GET / HTTP/1.1\r\n" +
			"Host: foo.com\r\n\r\n",

		&Request{
			Method: "GET",
			URL: &url.URL{
				Path: "/",
			},
			Proto:         "HTTP/1.1",
			ProtoMajor:    1,
			ProtoMinor:    1,
			Header:        Header{},
			Close:         false,
			ContentLength: 0,
			Host:          "foo.com",
			RequestURI:    "/",
		},

		noBodyStr,
		noTrailer,
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
				Path: "//user@host/is/actually/a/path/",
			},
			Proto:         "HTTP/1.1",
			ProtoMajor:    1,
			ProtoMinor:    1,
			Header:        Header{},
			Close:         false,
			ContentLength: 0,
			Host:          "test",
			RequestURI:    "//user@host/is/actually/a/path/",
		},

		noBodyStr,
		noTrailer,
		noError,
	},

	// Tests a bogus absolute-path on the Request-Line (RFC 7230 section 5.3.1)
	{
		"GET ../../../../etc/passwd HTTP/1.1\r\n" +
			"Host: test\r\n\r\n",
		nil,
		noBodyStr,
		noTrailer,
		"parse ../../../../etc/passwd: invalid URI for request",
	},

	// Tests missing URL:
	{
		"GET  HTTP/1.1\r\n" +
			"Host: test\r\n\r\n",
		nil,
		noBodyStr,
		noTrailer,
		"parse : empty url",
	},

	// Tests chunked body with trailer:
	{
		"POST / HTTP/1.1\r\n" +
			"Host: foo.com\r\n" +
			"Transfer-Encoding: chunked\r\n\r\n" +
			"3\r\nfoo\r\n" +
			"3\r\nbar\r\n" +
			"0\r\n" +
			"Trailer-Key: Trailer-Value\r\n" +
			"\r\n",
		&Request{
			Method: "POST",
			URL: &url.URL{
				Path: "/",
			},
			TransferEncoding: []string{"chunked"},
			Proto:            "HTTP/1.1",
			ProtoMajor:       1,
			ProtoMinor:       1,
			Header:           Header{},
			ContentLength:    -1,
			Host:             "foo.com",
			RequestURI:       "/",
		},

		"foobar",
		Header{
			"Trailer-Key": {"Trailer-Value"},
		},
		noError,
	},

	// Tests chunked body and a bogus Content-Length which should be deleted.
	{
		"POST / HTTP/1.1\r\n" +
			"Host: foo.com\r\n" +
			"Transfer-Encoding: chunked\r\n" +
			"Content-Length: 9999\r\n\r\n" + // to be removed.
			"3\r\nfoo\r\n" +
			"3\r\nbar\r\n" +
			"0\r\n" +
			"\r\n",
		&Request{
			Method: "POST",
			URL: &url.URL{
				Path: "/",
			},
			TransferEncoding: []string{"chunked"},
			Proto:            "HTTP/1.1",
			ProtoMajor:       1,
			ProtoMinor:       1,
			Header:           Header{},
			ContentLength:    -1,
			Host:             "foo.com",
			RequestURI:       "/",
		},

		"foobar",
		noTrailer,
		noError,
	},

	// CONNECT request with domain name:
	{
		"CONNECT www.google.com:443 HTTP/1.1\r\n\r\n",

		&Request{
			Method: "CONNECT",
			URL: &url.URL{
				Host: "www.google.com:443",
			},
			Proto:         "HTTP/1.1",
			ProtoMajor:    1,
			ProtoMinor:    1,
			Header:        Header{},
			Close:         false,
			ContentLength: 0,
			Host:          "www.google.com:443",
			RequestURI:    "www.google.com:443",
		},

		noBodyStr,
		noTrailer,
		noError,
	},

	// CONNECT request with IP address:
	{
		"CONNECT 127.0.0.1:6060 HTTP/1.1\r\n\r\n",

		&Request{
			Method: "CONNECT",
			URL: &url.URL{
				Host: "127.0.0.1:6060",
			},
			Proto:         "HTTP/1.1",
			ProtoMajor:    1,
			ProtoMinor:    1,
			Header:        Header{},
			Close:         false,
			ContentLength: 0,
			Host:          "127.0.0.1:6060",
			RequestURI:    "127.0.0.1:6060",
		},

		noBodyStr,
		noTrailer,
		noError,
	},

	// CONNECT request for RPC:
	{
		"CONNECT /_goRPC_ HTTP/1.1\r\n\r\n",

		&Request{
			Method: "CONNECT",
			URL: &url.URL{
				Path: "/_goRPC_",
			},
			Proto:         "HTTP/1.1",
			ProtoMajor:    1,
			ProtoMinor:    1,
			Header:        Header{},
			Close:         false,
			ContentLength: 0,
			Host:          "",
			RequestURI:    "/_goRPC_",
		},

		noBodyStr,
		noTrailer,
		noError,
	},

	// SSDP Notify request. golang.org/issue/3692
	{
		"NOTIFY * HTTP/1.1\r\nServer: foo\r\n\r\n",
		&Request{
			Method: "NOTIFY",
			URL: &url.URL{
				Path: "*",
			},
			Proto:      "HTTP/1.1",
			ProtoMajor: 1,
			ProtoMinor: 1,
			Header: Header{
				"Server": []string{"foo"},
			},
			Close:         false,
			ContentLength: 0,
			RequestURI:    "*",
		},

		noBodyStr,
		noTrailer,
		noError,
	},

	// OPTIONS request. Similar to golang.org/issue/3692
	{
		"OPTIONS * HTTP/1.1\r\nServer: foo\r\n\r\n",
		&Request{
			Method: "OPTIONS",
			URL: &url.URL{
				Path: "*",
			},
			Proto:      "HTTP/1.1",
			ProtoMajor: 1,
			ProtoMinor: 1,
			Header: Header{
				"Server": []string{"foo"},
			},
			Close:         false,
			ContentLength: 0,
			RequestURI:    "*",
		},

		noBodyStr,
		noTrailer,
		noError,
	},

	// Connection: close. golang.org/issue/8261
	{
		"GET / HTTP/1.1\r\nHost: issue8261.com\r\nConnection: close\r\n\r\n",
		&Request{
			Method: "GET",
			URL: &url.URL{
				Path: "/",
			},
			Header: Header{
				// This wasn't removed from Go 1.0 to
				// Go 1.3, so locking it in that we
				// keep this:
				"Connection": []string{"close"},
			},
			Host:       "issue8261.com",
			Proto:      "HTTP/1.1",
			ProtoMajor: 1,
			ProtoMinor: 1,
			Close:      true,
			RequestURI: "/",
		},

		noBodyStr,
		noTrailer,
		noError,
	},

	// HEAD with Content-Length 0. Make sure this is permitted,
	// since I think we used to send it.
	{
		"HEAD / HTTP/1.1\r\nHost: issue8261.com\r\nConnection: close\r\nContent-Length: 0\r\n\r\n",
		&Request{
			Method: "HEAD",
			URL: &url.URL{
				Path: "/",
			},
			Header: Header{
				"Connection":     []string{"close"},
				"Content-Length": []string{"0"},
			},
			Host:       "issue8261.com",
			Proto:      "HTTP/1.1",
			ProtoMajor: 1,
			ProtoMinor: 1,
			Close:      true,
			RequestURI: "/",
		},

		noBodyStr,
		noTrailer,
		noError,
	},

	// http2 client preface:
	{
		"PRI * HTTP/2.0\r\n\r\nSM\r\n\r\n",
		&Request{
			Method: "PRI",
			URL: &url.URL{
				Path: "*",
			},
			Header:        Header{},
			Proto:         "HTTP/2.0",
			ProtoMajor:    2,
			ProtoMinor:    0,
			RequestURI:    "*",
			ContentLength: -1,
			Close:         true,
		},
		noBodyStr,
		noTrailer,
		noError,
	},
}

func TestReadRequest(t *testing.T) {
	for i := range reqTests {
		tt := &reqTests[i]
		req, err := ReadRequest(bufio.NewReader(strings.NewReader(tt.Raw)))
		if err != nil {
			if err.Error() != tt.Error {
				t.Errorf("#%d: error %q, want error %q", i, err.Error(), tt.Error)
			}
			continue
		}
		rbody := req.Body
		req.Body = nil
		testName := fmt.Sprintf("Test %d (%q)", i, tt.Raw)
		diff(t, testName, req, tt.Req)
		var bout bytes.Buffer
		if rbody != nil {
			_, err := io.Copy(&bout, rbody)
			if err != nil {
				t.Fatalf("%s: copying body: %v", testName, err)
			}
			rbody.Close()
		}
		body := bout.String()
		if body != tt.Body {
			t.Errorf("%s: Body = %q want %q", testName, body, tt.Body)
		}
		if !reflect.DeepEqual(tt.Trailer, req.Trailer) {
			t.Errorf("%s: Trailers differ.\n got: %v\nwant: %v", testName, req.Trailer, tt.Trailer)
		}
	}
}

// reqBytes treats req as a request (with \n delimiters) and returns it with \r\n delimiters,
// ending in \r\n\r\n
func reqBytes(req string) []byte {
	return []byte(strings.Replace(strings.TrimSpace(req), "\n", "\r\n", -1) + "\r\n\r\n")
}

var badRequestTests = []struct {
	name string
	req  []byte
}{
	{"bad_connect_host", reqBytes("CONNECT []%20%48%54%54%50%2f%31%2e%31%0a%4d%79%48%65%61%64%65%72%3a%20%31%32%33%0a%0a HTTP/1.0")},
	{"smuggle_two_contentlen", reqBytes(`POST / HTTP/1.1
Content-Length: 3
Content-Length: 4

abc`)},
	{"smuggle_content_len_head", reqBytes(`HEAD / HTTP/1.1
Host: foo
Content-Length: 5`)},

	// golang.org/issue/22464
	{"leading_space_in_header", reqBytes(`HEAD / HTTP/1.1
 Host: foo
Content-Length: 5`)},
	{"leading_tab_in_header", reqBytes(`HEAD / HTTP/1.1
\tHost: foo
Content-Length: 5`)},
}

func TestReadRequest_Bad(t *testing.T) {
	for _, tt := range badRequestTests {
		got, err := ReadRequest(bufio.NewReader(bytes.NewReader(tt.req)))
		if err == nil {
			all, err := ioutil.ReadAll(got.Body)
			t.Errorf("%s: got unexpected request = %#v\n  Body = %q, %v", tt.name, got, all, err)
		}
	}
}
