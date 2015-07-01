// Copyright 2010 The Go Authors.  All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package http

import (
	"bytes"
	"errors"
	"fmt"
	"io"
	"io/ioutil"
	"net/url"
	"strings"
	"testing"
)

type reqWriteTest struct {
	Req  Request
	Body interface{} // optional []byte or func() io.ReadCloser to populate Req.Body

	// Any of these three may be empty to skip that test.
	WantWrite string // Request.Write
	WantProxy string // Request.WriteProxy

	WantError error // wanted error from Request.Write
}

var reqWriteTests = []reqWriteTest{
	// HTTP/1.1 => chunked coding; no body; no trailer
	{
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
	{
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
	{
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
	{
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
	{
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
	{
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
	{
		Req: Request{
			Method:        "POST",
			URL:           mustParseURL("/"),
			Host:          "example.com",
			ProtoMajor:    1,
			ProtoMinor:    1,
			ContentLength: 0, // as if unset by user
		},

		Body: func() io.ReadCloser { return ioutil.NopCloser(io.LimitReader(strings.NewReader("xx"), 0)) },

		// RFC 2616 Section 14.13 says Content-Length should be specified
		// unless body is prohibited by the request method.
		// Also, nginx expects it for POST and PUT.
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
	{
		Req: Request{
			Method:        "POST",
			URL:           mustParseURL("/"),
			Host:          "example.com",
			ProtoMajor:    1,
			ProtoMinor:    1,
			ContentLength: 0, // as if unset by user
		},

		Body: func() io.ReadCloser { return ioutil.NopCloser(io.LimitReader(strings.NewReader("xx"), 1)) },

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
	{
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
	{
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
	{
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
	{
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
			errReader := &errorReader{err}
			return ioutil.NopCloser(io.MultiReader(strings.NewReader("x"), errReader))
		},

		WantError: errors.New("Custom reader error"),
	},

	// Request with a 0 ContentLength and a body without content and an error.
	{
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
			errReader := &errorReader{err}
			return ioutil.NopCloser(errReader)
		},

		WantError: errors.New("Custom reader error"),
	},

	// Verify that DumpRequest preserves the HTTP version number, doesn't add a Host,
	// and doesn't add a User-Agent.
	{
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
	{
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
	{
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
	{
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
	{
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
	{
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
	{
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
				tt.Req.Body = ioutil.NopCloser(bytes.NewReader(b))
			case func() io.ReadCloser:
				tt.Req.Body = b()
			}
		}
		setBody()
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

		if tt.WantWrite != "" {
			sraw := braw.String()
			if sraw != tt.WantWrite {
				t.Errorf("Test %d, expecting:\n%s\nGot:\n%s\n", i, tt.WantWrite, sraw)
				continue
			}
		}

		if tt.WantProxy != "" {
			setBody()
			var praw bytes.Buffer
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

type closeChecker struct {
	io.Reader
	closed bool
}

func (rc *closeChecker) Close() error {
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
		"User-Agent: Go-http-client/1.1\r\n" +
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

	// w is the buffered io.Writer to write the request to.  It
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
