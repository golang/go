// Copyright 2010 The Go Authors.  All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package http

import (
	"bytes"
	"io"
	"io/ioutil"
	"os"
	"strings"
	"testing"
)

type reqWriteTest struct {
	Req      Request
	Body     []byte
	Raw      string
	RawProxy string
}

var reqWriteTests = []reqWriteTest{
	// HTTP/1.1 => chunked coding; no body; no trailer
	{
		Request{
			Method: "GET",
			RawURL: "http://www.techcrunch.com/",
			URL: &URL{
				Raw:          "http://www.techcrunch.com/",
				Scheme:       "http",
				RawPath:      "http://www.techcrunch.com/",
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
				"Accept-Charset":   {"ISO-8859-1,utf-8;q=0.7,*;q=0.7"},
				"Accept-Encoding":  {"gzip,deflate"},
				"Accept-Language":  {"en-us,en;q=0.5"},
				"Keep-Alive":       {"300"},
				"Proxy-Connection": {"keep-alive"},
				"User-Agent":       {"Fake"},
			},
			Body:  nil,
			Close: false,
			Host:  "www.techcrunch.com",
			Form:  map[string][]string{},
		},

		nil,

		"GET http://www.techcrunch.com/ HTTP/1.1\r\n" +
			"Host: www.techcrunch.com\r\n" +
			"User-Agent: Fake\r\n" +
			"Accept: text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8\r\n" +
			"Accept-Charset: ISO-8859-1,utf-8;q=0.7,*;q=0.7\r\n" +
			"Accept-Encoding: gzip,deflate\r\n" +
			"Accept-Language: en-us,en;q=0.5\r\n" +
			"Keep-Alive: 300\r\n" +
			"Proxy-Connection: keep-alive\r\n\r\n",

		"GET http://www.techcrunch.com/ HTTP/1.1\r\n" +
			"Host: www.techcrunch.com\r\n" +
			"User-Agent: Fake\r\n" +
			"Accept: text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8\r\n" +
			"Accept-Charset: ISO-8859-1,utf-8;q=0.7,*;q=0.7\r\n" +
			"Accept-Encoding: gzip,deflate\r\n" +
			"Accept-Language: en-us,en;q=0.5\r\n" +
			"Keep-Alive: 300\r\n" +
			"Proxy-Connection: keep-alive\r\n\r\n",
	},
	// HTTP/1.1 => chunked coding; body; empty trailer
	{
		Request{
			Method: "GET",
			URL: &URL{
				Scheme: "http",
				Host:   "www.google.com",
				Path:   "/search",
			},
			ProtoMajor:       1,
			ProtoMinor:       1,
			Header:           Header{},
			TransferEncoding: []string{"chunked"},
		},

		[]byte("abcdef"),

		"GET /search HTTP/1.1\r\n" +
			"Host: www.google.com\r\n" +
			"User-Agent: Go http package\r\n" +
			"Transfer-Encoding: chunked\r\n\r\n" +
			"6\r\nabcdef\r\n0\r\n\r\n",

		"GET http://www.google.com/search HTTP/1.1\r\n" +
			"Host: www.google.com\r\n" +
			"User-Agent: Go http package\r\n" +
			"Transfer-Encoding: chunked\r\n\r\n" +
			"6\r\nabcdef\r\n0\r\n\r\n",
	},
	// HTTP/1.1 POST => chunked coding; body; empty trailer
	{
		Request{
			Method: "POST",
			URL: &URL{
				Scheme: "http",
				Host:   "www.google.com",
				Path:   "/search",
			},
			ProtoMajor:       1,
			ProtoMinor:       1,
			Header:           Header{},
			Close:            true,
			TransferEncoding: []string{"chunked"},
		},

		[]byte("abcdef"),

		"POST /search HTTP/1.1\r\n" +
			"Host: www.google.com\r\n" +
			"User-Agent: Go http package\r\n" +
			"Connection: close\r\n" +
			"Transfer-Encoding: chunked\r\n\r\n" +
			"6\r\nabcdef\r\n0\r\n\r\n",

		"POST http://www.google.com/search HTTP/1.1\r\n" +
			"Host: www.google.com\r\n" +
			"User-Agent: Go http package\r\n" +
			"Connection: close\r\n" +
			"Transfer-Encoding: chunked\r\n\r\n" +
			"6\r\nabcdef\r\n0\r\n\r\n",
	},

	// HTTP/1.1 POST with Content-Length, no chunking
	{
		Request{
			Method: "POST",
			URL: &URL{
				Scheme: "http",
				Host:   "www.google.com",
				Path:   "/search",
			},
			ProtoMajor:    1,
			ProtoMinor:    1,
			Header:        Header{},
			Close:         true,
			ContentLength: 6,
		},

		[]byte("abcdef"),

		"POST /search HTTP/1.1\r\n" +
			"Host: www.google.com\r\n" +
			"User-Agent: Go http package\r\n" +
			"Connection: close\r\n" +
			"Content-Length: 6\r\n" +
			"\r\n" +
			"abcdef",

		"POST http://www.google.com/search HTTP/1.1\r\n" +
			"Host: www.google.com\r\n" +
			"User-Agent: Go http package\r\n" +
			"Connection: close\r\n" +
			"Content-Length: 6\r\n" +
			"\r\n" +
			"abcdef",
	},

	// HTTP/1.1 POST with Content-Length in headers
	{
		Request{
			Method: "POST",
			RawURL: "http://example.com/",
			Host:   "example.com",
			Header: Header{
				"Content-Length": []string{"10"}, // ignored
			},
			ContentLength: 6,
		},

		[]byte("abcdef"),

		"POST http://example.com/ HTTP/1.1\r\n" +
			"Host: example.com\r\n" +
			"User-Agent: Go http package\r\n" +
			"Content-Length: 6\r\n" +
			"\r\n" +
			"abcdef",

		"POST http://example.com/ HTTP/1.1\r\n" +
			"Host: example.com\r\n" +
			"User-Agent: Go http package\r\n" +
			"Content-Length: 6\r\n" +
			"\r\n" +
			"abcdef",
	},

	// default to HTTP/1.1
	{
		Request{
			Method: "GET",
			RawURL: "/search",
			Host:   "www.google.com",
		},

		nil,

		"GET /search HTTP/1.1\r\n" +
			"Host: www.google.com\r\n" +
			"User-Agent: Go http package\r\n" +
			"\r\n",

		// Looks weird but RawURL overrides what WriteProxy would choose.
		"GET /search HTTP/1.1\r\n" +
			"Host: www.google.com\r\n" +
			"User-Agent: Go http package\r\n" +
			"\r\n",
	},
}

func TestRequestWrite(t *testing.T) {
	for i := range reqWriteTests {
		tt := &reqWriteTests[i]
		if tt.Body != nil {
			tt.Req.Body = ioutil.NopCloser(bytes.NewBuffer(tt.Body))
		}
		if tt.Req.Header == nil {
			tt.Req.Header = make(Header)
		}
		var braw bytes.Buffer
		err := tt.Req.Write(&braw)
		if err != nil {
			t.Errorf("error writing #%d: %s", i, err)
			continue
		}
		sraw := braw.String()
		if sraw != tt.Raw {
			t.Errorf("Test %d, expecting:\n%s\nGot:\n%s\n", i, tt.Raw, sraw)
			continue
		}

		if tt.Body != nil {
			tt.Req.Body = ioutil.NopCloser(bytes.NewBuffer(tt.Body))
		}
		var praw bytes.Buffer
		err = tt.Req.WriteProxy(&praw)
		if err != nil {
			t.Errorf("error writing #%d: %s", i, err)
			continue
		}
		sraw = praw.String()
		if sraw != tt.RawProxy {
			t.Errorf("Test Proxy %d, expecting:\n%s\nGot:\n%s\n", i, tt.RawProxy, sraw)
			continue
		}
	}
}

type closeChecker struct {
	io.Reader
	closed bool
}

func (rc *closeChecker) Close() os.Error {
	rc.closed = true
	return nil
}

// TestRequestWriteClosesBody tests that Request.Write does close its request.Body.
// It also indirectly tests NewRequest and that it doesn't wrap an existing Closer
// inside a NopCloser, and that it serializes it correctly.
func TestRequestWriteClosesBody(t *testing.T) {
	rc := &closeChecker{Reader: strings.NewReader("my body")}
	req, _ := NewRequest("POST", "http://foo.com/", rc)
	if g, e := req.ContentLength, int64(-1); g != e {
		t.Errorf("got req.ContentLength %d, want %d", g, e)
	}
	buf := new(bytes.Buffer)
	req.Write(buf)
	if !rc.closed {
		t.Error("body not closed after write")
	}
	if g, e := buf.String(), "POST / HTTP/1.1\r\nHost: foo.com\r\nUser-Agent: Go http package\r\nTransfer-Encoding: chunked\r\n\r\n7\r\nmy body\r\n0\r\n\r\n"; g != e {
		t.Errorf("write:\n got: %s\nwant: %s", g, e)
	}
}

func TestZeroLengthNewRequest(t *testing.T) {
	var buf bytes.Buffer

	// Writing with default identity encoding
	req, _ := NewRequest("PUT", "http://foo.com/", strings.NewReader(""))
	if len(req.TransferEncoding) == 0 || req.TransferEncoding[0] != "identity" {
		t.Fatalf("got req.TransferEncoding of %v, want %v", req.TransferEncoding, []string{"identity"})
	}
	if g, e := req.ContentLength, int64(0); g != e {
		t.Errorf("got req.ContentLength %d, want %d", g, e)
	}
	req.Write(&buf)
	if g, e := buf.String(), "PUT / HTTP/1.1\r\nHost: foo.com\r\nUser-Agent: Go http package\r\nContent-Length: 0\r\n\r\n"; g != e {
		t.Errorf("identity write:\n got: %s\nwant: %s", g, e)
	}

	// Overriding identity encoding and forcing chunked.
	req, _ = NewRequest("PUT", "http://foo.com/", strings.NewReader(""))
	req.TransferEncoding = nil
	buf.Reset()
	req.Write(&buf)
	if g, e := buf.String(), "PUT / HTTP/1.1\r\nHost: foo.com\r\nUser-Agent: Go http package\r\nTransfer-Encoding: chunked\r\n\r\n0\r\n\r\n"; g != e {
		t.Errorf("chunked write:\n got: %s\nwant: %s", g, e)
	}
}
