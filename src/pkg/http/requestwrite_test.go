// Copyright 2010 The Go Authors.  All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package http

import (
	"bytes"
	"fmt"
	"io"
	"io/ioutil"
	"os"
	"strings"
	"testing"
	"url"
)

type reqWriteTest struct {
	Req       Request
	Body      interface{} // optional []byte or func() io.ReadCloser to populate Req.Body
	Raw       string
	RawProxy  string
	WantError os.Error
}

var reqWriteTests = []reqWriteTest{
	// HTTP/1.1 => chunked coding; no body; no trailer
	{
		Request{
			Method: "GET",
			RawURL: "http://www.techcrunch.com/",
			URL: &url.URL{
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

		nil,
	},
	// HTTP/1.1 => chunked coding; body; empty trailer
	{
		Request{
			Method: "GET",
			URL: &url.URL{
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
			chunk("abcdef") + chunk(""),

		"GET http://www.google.com/search HTTP/1.1\r\n" +
			"Host: www.google.com\r\n" +
			"User-Agent: Go http package\r\n" +
			"Transfer-Encoding: chunked\r\n\r\n" +
			chunk("abcdef") + chunk(""),

		nil,
	},
	// HTTP/1.1 POST => chunked coding; body; empty trailer
	{
		Request{
			Method: "POST",
			URL: &url.URL{
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
			chunk("abcdef") + chunk(""),

		"POST http://www.google.com/search HTTP/1.1\r\n" +
			"Host: www.google.com\r\n" +
			"User-Agent: Go http package\r\n" +
			"Connection: close\r\n" +
			"Transfer-Encoding: chunked\r\n\r\n" +
			chunk("abcdef") + chunk(""),

		nil,
	},

	// HTTP/1.1 POST with Content-Length, no chunking
	{
		Request{
			Method: "POST",
			URL: &url.URL{
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

		nil,
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

		nil,
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

		nil,
	},

	// Request with a 0 ContentLength and a 0 byte body.
	{
		Request{
			Method:        "POST",
			RawURL:        "/",
			Host:          "example.com",
			ProtoMajor:    1,
			ProtoMinor:    1,
			ContentLength: 0, // as if unset by user
		},

		func() io.ReadCloser { return ioutil.NopCloser(io.LimitReader(strings.NewReader("xx"), 0)) },

		"POST / HTTP/1.1\r\n" +
			"Host: example.com\r\n" +
			"User-Agent: Go http package\r\n" +
			"\r\n",

		"POST / HTTP/1.1\r\n" +
			"Host: example.com\r\n" +
			"User-Agent: Go http package\r\n" +
			"\r\n",

		nil,
	},

	// Request with a 0 ContentLength and a 1 byte body.
	{
		Request{
			Method:        "POST",
			RawURL:        "/",
			Host:          "example.com",
			ProtoMajor:    1,
			ProtoMinor:    1,
			ContentLength: 0, // as if unset by user
		},

		func() io.ReadCloser { return ioutil.NopCloser(io.LimitReader(strings.NewReader("xx"), 1)) },

		"POST / HTTP/1.1\r\n" +
			"Host: example.com\r\n" +
			"User-Agent: Go http package\r\n" +
			"Transfer-Encoding: chunked\r\n\r\n" +
			chunk("x") + chunk(""),

		"POST / HTTP/1.1\r\n" +
			"Host: example.com\r\n" +
			"User-Agent: Go http package\r\n" +
			"Transfer-Encoding: chunked\r\n\r\n" +
			chunk("x") + chunk(""),

		nil,
	},

	// Request with a ContentLength of 10 but a 5 byte body.
	{
		Request{
			Method:        "POST",
			RawURL:        "/",
			Host:          "example.com",
			ProtoMajor:    1,
			ProtoMinor:    1,
			ContentLength: 10, // but we're going to send only 5 bytes
		},

		[]byte("12345"),

		"", // ignored
		"", // ignored

		os.NewError("http: Request.ContentLength=10 with Body length 5"),
	},

	// Request with a ContentLength of 4 but an 8 byte body.
	{
		Request{
			Method:        "POST",
			RawURL:        "/",
			Host:          "example.com",
			ProtoMajor:    1,
			ProtoMinor:    1,
			ContentLength: 4, // but we're going to try to send 8 bytes
		},

		[]byte("12345678"),

		"", // ignored
		"", // ignored

		os.NewError("http: Request.ContentLength=4 with Body length 8"),
	},

	// Request with a 5 ContentLength and nil body.
	{
		Request{
			Method:        "POST",
			RawURL:        "/",
			Host:          "example.com",
			ProtoMajor:    1,
			ProtoMinor:    1,
			ContentLength: 5, // but we'll omit the body
		},

		nil, // missing body

		"POST / HTTP/1.1\r\n" +
			"Host: example.com\r\n" +
			"User-Agent: Go http package\r\n" +
			"Content-Length: 5\r\n\r\n" +
			"",

		"POST / HTTP/1.1\r\n" +
			"Host: example.com\r\n" +
			"User-Agent: Go http package\r\n" +
			"Content-Length: 5\r\n\r\n" +
			"",

		os.NewError("http: Request.ContentLength=5 with nil Body"),
	},
}

func TestRequestWrite(t *testing.T) {
	for i := range reqWriteTests {
		tt := &reqWriteTests[i]

		setBody := func() {
			switch b := tt.Body.(type) {
			case []byte:
				tt.Req.Body = ioutil.NopCloser(bytes.NewBuffer(b))
			case func() io.ReadCloser:
				tt.Req.Body = b()
			}
		}
		if tt.Body != nil {
			setBody()
		}
		if tt.Req.Header == nil {
			tt.Req.Header = make(Header)
		}
		var braw bytes.Buffer
		err := tt.Req.Write(&braw)
		if g, e := fmt.Sprintf("%v", err), fmt.Sprintf("%v", tt.WantError); g != e {
			t.Errorf("writing #%d, err = %q, want %q", i, g, e)
			continue
		}
		if err != nil {
			continue
		}

		sraw := braw.String()
		if sraw != tt.Raw {
			t.Errorf("Test %d, expecting:\n%s\nGot:\n%s\n", i, tt.Raw, sraw)
			continue
		}

		if tt.Body != nil {
			setBody()
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
	if req.ContentLength != 0 {
		t.Errorf("got req.ContentLength %d, want 0", req.ContentLength)
	}
	buf := new(bytes.Buffer)
	req.Write(buf)
	if !rc.closed {
		t.Error("body not closed after write")
	}
	expected := "POST / HTTP/1.1\r\n" +
		"Host: foo.com\r\n" +
		"User-Agent: Go http package\r\n" +
		"Transfer-Encoding: chunked\r\n\r\n" +
		// TODO: currently we don't buffer before chunking, so we get a
		// single "m" chunk before the other chunks, as this was the 1-byte
		// read from our MultiReader where we stiched the Body back together
		// after sniffing whether the Body was 0 bytes or not.
		chunk("m") +
		chunk("y body") +
		chunk("")
	if buf.String() != expected {
		t.Errorf("write:\n got: %s\nwant: %s", buf.String(), expected)
	}
}

func chunk(s string) string {
	return fmt.Sprintf("%x\r\n%s\r\n", len(s), s)
}
