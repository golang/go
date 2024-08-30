// Copyright 2010 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package http

import (
	"bufio"
	"bytes"
	"errors"
	"fmt"
	"io"
	"net"
	"net/url"
	"strings"
	"testing"
	"testing/iotest"
	"time"
)

type reqWriteTest struct {
	Req  Request
	Body any // optional []byte or func() io.ReadCloser to populate Req.Body

	// Any of these three may be empty to skip that test.
	WantWrite string // Request.Write
	WantProxy string // Request.WriteProxy

	WantError error // wanted error from Request.Write
}

var reqWriteTests = []reqWriteTest{
	// HTTP/1.1 => chunked coding; no body; no trailer
	0: {
		Req: Request{
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

		WantWrite: "GET / HTTP/1.1\r\n" +
			"Host: www.techcrunch.com\r\n" +
			"User-Agent: Fake\r\n" +
			"Accept: text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8\r\n" +
			"Accept-Charset: ISO-8859-1,utf-8;q=0.7,*;q=0.7\r\n" +
			"Accept-Encoding: gzip,deflate\r\n" +
			"Accept-Language: en-us,en;q=0.5\r\n" +
			"Keep-Alive: 300\r\n" +
			"Proxy-Connection: keep-alive\r\n\r\n",

		WantProxy: "GET http://www.techcrunch.com/ HTTP/1.1\r\n" +
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
	1: {
		Req: Request{
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

		Body: []byte("abcdef"),

		WantWrite: "GET /search HTTP/1.1\r\n" +
			"Host: www.google.com\r\n" +
			"User-Agent: Go-http-client/1.1\r\n" +
			"Transfer-Encoding: chunked\r\n\r\n" +
			chunk("abcdef") + chunk(""),

		WantProxy: "GET http://www.google.com/search HTTP/1.1\r\n" +
			"Host: www.google.com\r\n" +
			"User-Agent: Go-http-client/1.1\r\n" +
			"Transfer-Encoding: chunked\r\n\r\n" +
			chunk("abcdef") + chunk(""),
	},
	// HTTP/1.1 POST => chunked coding; body; empty trailer
	2: {
		Req: Request{
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

		Body: []byte("abcdef"),

		WantWrite: "POST /search HTTP/1.1\r\n" +
			"Host: www.google.com\r\n" +
			"User-Agent: Go-http-client/1.1\r\n" +
			"Connection: close\r\n" +
			"Transfer-Encoding: chunked\r\n\r\n" +
			chunk("abcdef") + chunk(""),

		WantProxy: "POST http://www.google.com/search HTTP/1.1\r\n" +
			"Host: www.google.com\r\n" +
			"User-Agent: Go-http-client/1.1\r\n" +
			"Connection: close\r\n" +
			"Transfer-Encoding: chunked\r\n\r\n" +
			chunk("abcdef") + chunk(""),
	},

	// HTTP/1.1 POST with Content-Length, no chunking
	3: {
		Req: Request{
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

		Body: []byte("abcdef"),

		WantWrite: "POST /search HTTP/1.1\r\n" +
			"Host: www.google.com\r\n" +
			"User-Agent: Go-http-client/1.1\r\n" +
			"Connection: close\r\n" +
			"Content-Length: 6\r\n" +
			"\r\n" +
			"abcdef",

		WantProxy: "POST http://www.google.com/search HTTP/1.1\r\n" +
			"Host: www.google.com\r\n" +
			"User-Agent: Go-http-client/1.1\r\n" +
			"Connection: close\r\n" +
			"Content-Length: 6\r\n" +
			"\r\n" +
			"abcdef",
	},

	// HTTP/1.1 POST with Content-Length in headers
	4: {
		Req: Request{
			Method: "POST",
			URL:    mustParseURL("http://example.com/"),
			Host:   "example.com",
			Header: Header{
				"Content-Length": []string{"10"}, // ignored
			},
			ContentLength: 6,
		},

		Body: []byte("abcdef"),

		WantWrite: "POST / HTTP/1.1\r\n" +
			"Host: example.com\r\n" +
			"User-Agent: Go-http-client/1.1\r\n" +
			"Content-Length: 6\r\n" +
			"\r\n" +
			"abcdef",

		WantProxy: "POST http://example.com/ HTTP/1.1\r\n" +
			"Host: example.com\r\n" +
			"User-Agent: Go-http-client/1.1\r\n" +
			"Content-Length: 6\r\n" +
			"\r\n" +
			"abcdef",
	},

	// default to HTTP/1.1
	5: {
		Req: Request{
			Method: "GET",
			URL:    mustParseURL("/search"),
			Host:   "www.google.com",
		},

		WantWrite: "GET /search HTTP/1.1\r\n" +
			"Host: www.google.com\r\n" +
			"User-Agent: Go-http-client/1.1\r\n" +
			"\r\n",
	},

	// Request with a 0 ContentLength and a 0 byte body.
	6: {
		Req: Request{
			Method:        "POST",
			URL:           mustParseURL("/"),
			Host:          "example.com",
			ProtoMajor:    1,
			ProtoMinor:    1,
			ContentLength: 0, // as if unset by user
		},

		Body: func() io.ReadCloser { return io.NopCloser(io.LimitReader(strings.NewReader("xx"), 0)) },

		WantWrite: "POST / HTTP/1.1\r\n" +
			"Host: example.com\r\n" +
			"User-Agent: Go-http-client/1.1\r\n" +
			"Transfer-Encoding: chunked\r\n" +
			"\r\n0\r\n\r\n",

		WantProxy: "POST / HTTP/1.1\r\n" +
			"Host: example.com\r\n" +
			"User-Agent: Go-http-client/1.1\r\n" +
			"Transfer-Encoding: chunked\r\n" +
			"\r\n0\r\n\r\n",
	},

	// Request with a 0 ContentLength and a nil body.
	7: {
		Req: Request{
			Method:        "POST",
			URL:           mustParseURL("/"),
			Host:          "example.com",
			ProtoMajor:    1,
			ProtoMinor:    1,
			ContentLength: 0, // as if unset by user
		},

		Body: func() io.ReadCloser { return nil },

		WantWrite: "POST / HTTP/1.1\r\n" +
			"Host: example.com\r\n" +
			"User-Agent: Go-http-client/1.1\r\n" +
			"Content-Length: 0\r\n" +
			"\r\n",

		WantProxy: "POST / HTTP/1.1\r\n" +
			"Host: example.com\r\n" +
			"User-Agent: Go-http-client/1.1\r\n" +
			"Content-Length: 0\r\n" +
			"\r\n",
	},

	// Request with a 0 ContentLength and a 1 byte body.
	8: {
		Req: Request{
			Method:        "POST",
			URL:           mustParseURL("/"),
			Host:          "example.com",
			ProtoMajor:    1,
			ProtoMinor:    1,
			ContentLength: 0, // as if unset by user
		},

		Body: func() io.ReadCloser { return io.NopCloser(io.LimitReader(strings.NewReader("xx"), 1)) },

		WantWrite: "POST / HTTP/1.1\r\n" +
			"Host: example.com\r\n" +
			"User-Agent: Go-http-client/1.1\r\n" +
			"Transfer-Encoding: chunked\r\n\r\n" +
			chunk("x") + chunk(""),

		WantProxy: "POST / HTTP/1.1\r\n" +
			"Host: example.com\r\n" +
			"User-Agent: Go-http-client/1.1\r\n" +
			"Transfer-Encoding: chunked\r\n\r\n" +
			chunk("x") + chunk(""),
	},

	// Request with a ContentLength of 10 but a 5 byte body.
	9: {
		Req: Request{
			Method:        "POST",
			URL:           mustParseURL("/"),
			Host:          "example.com",
			ProtoMajor:    1,
			ProtoMinor:    1,
			ContentLength: 10, // but we're going to send only 5 bytes
		},
		Body:      []byte("12345"),
		WantError: errors.New("http: ContentLength=10 with Body length 5"),
	},

	// Request with a ContentLength of 4 but an 8 byte body.
	10: {
		Req: Request{
			Method:        "POST",
			URL:           mustParseURL("/"),
			Host:          "example.com",
			ProtoMajor:    1,
			ProtoMinor:    1,
			ContentLength: 4, // but we're going to try to send 8 bytes
		},
		Body:      []byte("12345678"),
		WantError: errors.New("http: ContentLength=4 with Body length 8"),
	},

	// Request with a 5 ContentLength and nil body.
	11: {
		Req: Request{
			Method:        "POST",
			URL:           mustParseURL("/"),
			Host:          "example.com",
			ProtoMajor:    1,
			ProtoMinor:    1,
			ContentLength: 5, // but we'll omit the body
		},
		WantError: errors.New("http: Request.ContentLength=5 with nil Body"),
	},

	// Request with a 0 ContentLength and a body with 1 byte content and an error.
	12: {
		Req: Request{
			Method:        "POST",
			URL:           mustParseURL("/"),
			Host:          "example.com",
			ProtoMajor:    1,
			ProtoMinor:    1,
			ContentLength: 0, // as if unset by user
		},

		Body: func() io.ReadCloser {
			err := errors.New("Custom reader error")
			errReader := iotest.ErrReader(err)
			return io.NopCloser(io.MultiReader(strings.NewReader("x"), errReader))
		},

		WantError: errors.New("Custom reader error"),
	},

	// Request with a 0 ContentLength and a body without content and an error.
	13: {
		Req: Request{
			Method:        "POST",
			URL:           mustParseURL("/"),
			Host:          "example.com",
			ProtoMajor:    1,
			ProtoMinor:    1,
			ContentLength: 0, // as if unset by user
		},

		Body: func() io.ReadCloser {
			err := errors.New("Custom reader error")
			errReader := iotest.ErrReader(err)
			return io.NopCloser(errReader)
		},

		WantError: errors.New("Custom reader error"),
	},

	// Verify that DumpRequest preserves the HTTP version number, doesn't add a Host,
	// and doesn't add a User-Agent.
	14: {
		Req: Request{
			Method:     "GET",
			URL:        mustParseURL("/foo"),
			ProtoMajor: 1,
			ProtoMinor: 0,
			Header: Header{
				"X-Foo": []string{"X-Bar"},
			},
		},

		WantWrite: "GET /foo HTTP/1.1\r\n" +
			"Host: \r\n" +
			"User-Agent: Go-http-client/1.1\r\n" +
			"X-Foo: X-Bar\r\n\r\n",
	},

	// If no Request.Host and no Request.URL.Host, we send
	// an empty Host header, and don't use
	// Request.Header["Host"]. This is just testing that
	// we don't change Go 1.0 behavior.
	15: {
		Req: Request{
			Method: "GET",
			Host:   "",
			URL: &url.URL{
				Scheme: "http",
				Host:   "",
				Path:   "/search",
			},
			ProtoMajor: 1,
			ProtoMinor: 1,
			Header: Header{
				"Host": []string{"bad.example.com"},
			},
		},

		WantWrite: "GET /search HTTP/1.1\r\n" +
			"Host: \r\n" +
			"User-Agent: Go-http-client/1.1\r\n\r\n",
	},

	// Opaque test #1 from golang.org/issue/4860
	16: {
		Req: Request{
			Method: "GET",
			URL: &url.URL{
				Scheme: "http",
				Host:   "www.google.com",
				Opaque: "/%2F/%2F/",
			},
			ProtoMajor: 1,
			ProtoMinor: 1,
			Header:     Header{},
		},

		WantWrite: "GET /%2F/%2F/ HTTP/1.1\r\n" +
			"Host: www.google.com\r\n" +
			"User-Agent: Go-http-client/1.1\r\n\r\n",
	},

	// Opaque test #2 from golang.org/issue/4860
	17: {
		Req: Request{
			Method: "GET",
			URL: &url.URL{
				Scheme: "http",
				Host:   "x.google.com",
				Opaque: "//y.google.com/%2F/%2F/",
			},
			ProtoMajor: 1,
			ProtoMinor: 1,
			Header:     Header{},
		},

		WantWrite: "GET http://y.google.com/%2F/%2F/ HTTP/1.1\r\n" +
			"Host: x.google.com\r\n" +
			"User-Agent: Go-http-client/1.1\r\n\r\n",
	},

	// Testing custom case in header keys. Issue 5022.
	18: {
		Req: Request{
			Method: "GET",
			URL: &url.URL{
				Scheme: "http",
				Host:   "www.google.com",
				Path:   "/",
			},
			Proto:      "HTTP/1.1",
			ProtoMajor: 1,
			ProtoMinor: 1,
			Header: Header{
				"ALL-CAPS": {"x"},
			},
		},

		WantWrite: "GET / HTTP/1.1\r\n" +
			"Host: www.google.com\r\n" +
			"User-Agent: Go-http-client/1.1\r\n" +
			"ALL-CAPS: x\r\n" +
			"\r\n",
	},

	// Request with host header field; IPv6 address with zone identifier
	19: {
		Req: Request{
			Method: "GET",
			URL: &url.URL{
				Host: "[fe80::1%en0]",
			},
		},

		WantWrite: "GET / HTTP/1.1\r\n" +
			"Host: [fe80::1]\r\n" +
			"User-Agent: Go-http-client/1.1\r\n" +
			"\r\n",
	},

	// Request with optional host header field; IPv6 address with zone identifier
	20: {
		Req: Request{
			Method: "GET",
			URL: &url.URL{
				Host: "www.example.com",
			},
			Host: "[fe80::1%en0]:8080",
		},

		WantWrite: "GET / HTTP/1.1\r\n" +
			"Host: [fe80::1]:8080\r\n" +
			"User-Agent: Go-http-client/1.1\r\n" +
			"\r\n",
	},

	// CONNECT without Opaque
	21: {
		Req: Request{
			Method: "CONNECT",
			URL: &url.URL{
				Scheme: "https", // of proxy.com
				Host:   "proxy.com",
			},
		},
		// What we used to do, locking that behavior in:
		WantWrite: "CONNECT proxy.com HTTP/1.1\r\n" +
			"Host: proxy.com\r\n" +
			"User-Agent: Go-http-client/1.1\r\n" +
			"\r\n",
	},

	// CONNECT with Opaque
	22: {
		Req: Request{
			Method: "CONNECT",
			URL: &url.URL{
				Scheme: "https", // of proxy.com
				Host:   "proxy.com",
				Opaque: "backend:443",
			},
		},
		WantWrite: "CONNECT backend:443 HTTP/1.1\r\n" +
			"Host: proxy.com\r\n" +
			"User-Agent: Go-http-client/1.1\r\n" +
			"\r\n",
	},

	// Verify that a nil header value doesn't get written.
	23: {
		Req: Request{
			Method: "GET",
			URL:    mustParseURL("/foo"),
			Header: Header{
				"X-Foo":             []string{"X-Bar"},
				"X-Idempotency-Key": nil,
			},
		},

		WantWrite: "GET /foo HTTP/1.1\r\n" +
			"Host: \r\n" +
			"User-Agent: Go-http-client/1.1\r\n" +
			"X-Foo: X-Bar\r\n\r\n",
	},
	24: {
		Req: Request{
			Method: "GET",
			URL:    mustParseURL("/foo"),
			Header: Header{
				"X-Foo":             []string{"X-Bar"},
				"X-Idempotency-Key": []string{},
			},
		},

		WantWrite: "GET /foo HTTP/1.1\r\n" +
			"Host: \r\n" +
			"User-Agent: Go-http-client/1.1\r\n" +
			"X-Foo: X-Bar\r\n\r\n",
	},

	25: {
		Req: Request{
			Method: "GET",
			URL: &url.URL{
				Host:     "www.example.com",
				RawQuery: "new\nline", // or any CTL
			},
		},
		WantError: errors.New("net/http: can't write control character in Request.URL"),
	},

	26: { // Request with nil body and PATCH method. Issue #40978
		Req: Request{
			Method:        "PATCH",
			URL:           mustParseURL("/"),
			Host:          "example.com",
			ProtoMajor:    1,
			ProtoMinor:    1,
			ContentLength: 0, // as if unset by user
		},
		Body: nil,
		WantWrite: "PATCH / HTTP/1.1\r\n" +
			"Host: example.com\r\n" +
			"User-Agent: Go-http-client/1.1\r\n" +
			"Content-Length: 0\r\n\r\n",
		WantProxy: "PATCH / HTTP/1.1\r\n" +
			"Host: example.com\r\n" +
			"User-Agent: Go-http-client/1.1\r\n" +
			"Content-Length: 0\r\n\r\n",
	},
}

func TestRequestWrite(t *testing.T) {
	for i := range reqWriteTests {
		tt := &reqWriteTests[i]

		setBody := func() {
			if tt.Body == nil {
				return
			}
			switch b := tt.Body.(type) {
			case []byte:
				tt.Req.Body = io.NopCloser(bytes.NewReader(b))
			case func() io.ReadCloser:
				tt.Req.Body = b()
			}
		}
		setBody()
		if tt.Req.Header == nil {
			tt.Req.Header = make(Header)
		}

		var braw strings.Builder
		err := tt.Req.Write(&braw)
		if g, e := fmt.Sprintf("%v", err), fmt.Sprintf("%v", tt.WantError); g != e {
			t.Errorf("writing #%d, err = %q, want %q", i, g, e)
			continue
		}
		if err != nil {
			continue
		}

		if tt.WantWrite != "" {
			sraw := braw.String()
			if sraw != tt.WantWrite {
				t.Errorf("Test %d, expecting:\n%s\nGot:\n%s\n", i, tt.WantWrite, sraw)
				continue
			}
		}

		if tt.WantProxy != "" {
			setBody()
			var praw strings.Builder
			err = tt.Req.WriteProxy(&praw)
			if err != nil {
				t.Errorf("WriteProxy #%d: %s", i, err)
				continue
			}
			sraw := praw.String()
			if sraw != tt.WantProxy {
				t.Errorf("Test Proxy %d, expecting:\n%s\nGot:\n%s\n", i, tt.WantProxy, sraw)
				continue
			}
		}
	}
}

func TestRequestWriteTransport(t *testing.T) {
	t.Parallel()

	matchSubstr := func(substr string) func(string) error {
		return func(written string) error {
			if !strings.Contains(written, substr) {
				return fmt.Errorf("expected substring %q in request: %s", substr, written)
			}
			return nil
		}
	}

	noContentLengthOrTransferEncoding := func(req string) error {
		if strings.Contains(req, "Content-Length: ") {
			return fmt.Errorf("unexpected Content-Length in request: %s", req)
		}
		if strings.Contains(req, "Transfer-Encoding: ") {
			return fmt.Errorf("unexpected Transfer-Encoding in request: %s", req)
		}
		return nil
	}

	all := func(checks ...func(string) error) func(string) error {
		return func(req string) error {
			for _, c := range checks {
				if err := c(req); err != nil {
					return err
				}
			}
			return nil
		}
	}

	type testCase struct {
		method string
		clen   int64 // ContentLength
		body   io.ReadCloser
		want   func(string) error

		// optional:
		init         func(*testCase)
		afterReqRead func()
	}

	tests := []testCase{
		{
			method: "GET",
			want:   noContentLengthOrTransferEncoding,
		},
		{
			method: "GET",
			body:   io.NopCloser(strings.NewReader("")),
			want:   noContentLengthOrTransferEncoding,
		},
		{
			method: "GET",
			clen:   -1,
			body:   io.NopCloser(strings.NewReader("")),
			want:   noContentLengthOrTransferEncoding,
		},
		// A GET with a body, with explicit content length:
		{
			method: "GET",
			clen:   7,
			body:   io.NopCloser(strings.NewReader("foobody")),
			want: all(matchSubstr("Content-Length: 7"),
				matchSubstr("foobody")),
		},
		// A GET with a body, sniffing the leading "f" from "foobody".
		{
			method: "GET",
			clen:   -1,
			body:   io.NopCloser(strings.NewReader("foobody")),
			want: all(matchSubstr("Transfer-Encoding: chunked"),
				matchSubstr("\r\n1\r\nf\r\n"),
				matchSubstr("oobody")),
		},
		// But a POST request is expected to have a body, so
		// no sniffing happens:
		{
			method: "POST",
			clen:   -1,
			body:   io.NopCloser(strings.NewReader("foobody")),
			want: all(matchSubstr("Transfer-Encoding: chunked"),
				matchSubstr("foobody")),
		},
		{
			method: "POST",
			clen:   -1,
			body:   io.NopCloser(strings.NewReader("")),
			want:   all(matchSubstr("Transfer-Encoding: chunked")),
		},
		// Verify that a blocking Request.Body doesn't block forever.
		{
			method: "GET",
			clen:   -1,
			init: func(tt *testCase) {
				pr, pw := io.Pipe()
				tt.afterReqRead = func() {
					pw.Close()
				}
				tt.body = io.NopCloser(pr)
			},
			want: matchSubstr("Transfer-Encoding: chunked"),
		},
	}

	for i, tt := range tests {
		if tt.init != nil {
			tt.init(&tt)
		}
		req := &Request{
			Method: tt.method,
			URL: &url.URL{
				Scheme: "http",
				Host:   "example.com",
			},
			Header:        make(Header),
			ContentLength: tt.clen,
			Body:          tt.body,
		}
		got, err := dumpRequestOut(req, tt.afterReqRead)
		if err != nil {
			t.Errorf("test[%d]: %v", i, err)
			continue
		}
		if err := tt.want(string(got)); err != nil {
			t.Errorf("test[%d]: %v", i, err)
		}
	}
}

type closeChecker struct {
	io.Reader
	closed bool
}

func (rc *closeChecker) Close() error {
	rc.closed = true
	return nil
}

// TestRequestWriteClosesBody tests that Request.Write closes its request.Body.
// It also indirectly tests NewRequest and that it doesn't wrap an existing Closer
// inside a NopCloser, and that it serializes it correctly.
func TestRequestWriteClosesBody(t *testing.T) {
	rc := &closeChecker{Reader: strings.NewReader("my body")}
	req, err := NewRequest("POST", "http://foo.com/", rc)
	if err != nil {
		t.Fatal(err)
	}
	buf := new(strings.Builder)
	if err := req.Write(buf); err != nil {
		t.Error(err)
	}
	if !rc.closed {
		t.Error("body not closed after write")
	}
	expected := "POST / HTTP/1.1\r\n" +
		"Host: foo.com\r\n" +
		"User-Agent: Go-http-client/1.1\r\n" +
		"Transfer-Encoding: chunked\r\n\r\n" +
		chunk("my body") +
		chunk("")
	if buf.String() != expected {
		t.Errorf("write:\n got: %s\nwant: %s", buf.String(), expected)
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

type writerFunc func([]byte) (int, error)

func (f writerFunc) Write(p []byte) (int, error) { return f(p) }

// TestRequestWriteError tests the Write err != nil checks in (*Request).write.
func TestRequestWriteError(t *testing.T) {
	failAfter, writeCount := 0, 0
	errFail := errors.New("fake write failure")

	// w is the buffered io.Writer to write the request to. It
	// fails exactly once on its Nth Write call, as controlled by
	// failAfter. It also tracks the number of calls in
	// writeCount.
	w := struct {
		io.ByteWriter // to avoid being wrapped by a bufio.Writer
		io.Writer
	}{
		nil,
		writerFunc(func(p []byte) (n int, err error) {
			writeCount++
			if failAfter == 0 {
				err = errFail
			}
			failAfter--
			return len(p), err
		}),
	}

	req, _ := NewRequest("GET", "http://example.com/", nil)
	const writeCalls = 4 // number of Write calls in current implementation
	sawGood := false
	for n := 0; n <= writeCalls+2; n++ {
		failAfter = n
		writeCount = 0
		err := req.Write(w)
		var wantErr error
		if n < writeCalls {
			wantErr = errFail
		}
		if err != wantErr {
			t.Errorf("for fail-after %d Writes, err = %v; want %v", n, err, wantErr)
			continue
		}
		if err == nil {
			sawGood = true
			if writeCount != writeCalls {
				t.Fatalf("writeCalls constant is outdated in test")
			}
		}
		if writeCount > writeCalls || writeCount > n+1 {
			t.Errorf("for fail-after %d, saw unexpectedly high (%d) write calls", n, writeCount)
		}
	}
	if !sawGood {
		t.Fatalf("writeCalls constant is outdated in test")
	}
}

// dumpRequestOut is a modified copy of net/http/httputil.DumpRequestOut.
// Unlike the original, this version doesn't mutate the req.Body and
// try to restore it. It always dumps the whole body.
// And it doesn't support https.
func dumpRequestOut(req *Request, onReadHeaders func()) ([]byte, error) {

	// Use the actual Transport code to record what we would send
	// on the wire, but not using TCP.  Use a Transport with a
	// custom dialer that returns a fake net.Conn that waits
	// for the full input (and recording it), and then responds
	// with a dummy response.
	var buf bytes.Buffer // records the output
	pr, pw := io.Pipe()
	defer pr.Close()
	defer pw.Close()
	dr := &delegateReader{c: make(chan io.Reader)}

	t := &Transport{
		Dial: func(net, addr string) (net.Conn, error) {
			return &dumpConn{io.MultiWriter(&buf, pw), dr}, nil
		},
	}
	defer t.CloseIdleConnections()

	// Wait for the request before replying with a dummy response:
	go func() {
		req, err := ReadRequest(bufio.NewReader(pr))
		if err == nil {
			if onReadHeaders != nil {
				onReadHeaders()
			}
			// Ensure all the body is read; otherwise
			// we'll get a partial dump.
			io.Copy(io.Discard, req.Body)
			req.Body.Close()
		}
		dr.c <- strings.NewReader("HTTP/1.1 204 No Content\r\nConnection: close\r\n\r\n")
	}()

	_, err := t.RoundTrip(req)
	if err != nil {
		return nil, err
	}
	return buf.Bytes(), nil
}

// delegateReader is a reader that delegates to another reader,
// once it arrives on a channel.
type delegateReader struct {
	c chan io.Reader
	r io.Reader // nil until received from c
}

func (r *delegateReader) Read(p []byte) (int, error) {
	if r.r == nil {
		r.r = <-r.c
	}
	return r.r.Read(p)
}

// dumpConn is a net.Conn that writes to Writer and reads from Reader.
type dumpConn struct {
	io.Writer
	io.Reader
}

func (c *dumpConn) Close() error                       { return nil }
func (c *dumpConn) LocalAddr() net.Addr                { return nil }
func (c *dumpConn) RemoteAddr() net.Addr               { return nil }
func (c *dumpConn) SetDeadline(t time.Time) error      { return nil }
func (c *dumpConn) SetReadDeadline(t time.Time) error  { return nil }
func (c *dumpConn) SetWriteDeadline(t time.Time) error { return nil }
