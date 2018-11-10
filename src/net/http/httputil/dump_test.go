// Copyright 2011 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package httputil

import (
	"bufio"
	"bytes"
	"fmt"
	"io"
	"io/ioutil"
	"net/http"
	"net/url"
	"runtime"
	"strings"
	"testing"
)

type dumpTest struct {
	Req  http.Request
	Body interface{} // optional []byte or func() io.ReadCloser to populate Req.Body

	WantDump    string
	WantDumpOut string
	NoBody      bool // if true, set DumpRequest{,Out} body to false
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
			"User-Agent: Go-http-client/1.1\r\n" +
			"Accept-Encoding: gzip\r\n\r\n",
	},

	// Test that an https URL doesn't try to do an SSL negotiation
	// with a bytes.Buffer and hang with all goroutines not
	// runnable.
	{
		Req: *mustNewRequest("GET", "https://example.com/foo", nil),

		WantDumpOut: "GET /foo HTTP/1.1\r\n" +
			"Host: example.com\r\n" +
			"User-Agent: Go-http-client/1.1\r\n" +
			"Accept-Encoding: gzip\r\n\r\n",
	},

	// Request with Body, but Dump requested without it.
	{
		Req: http.Request{
			Method: "POST",
			URL: &url.URL{
				Scheme: "http",
				Host:   "post.tld",
				Path:   "/",
			},
			ContentLength: 6,
			ProtoMajor:    1,
			ProtoMinor:    1,
		},

		Body: []byte("abcdef"),

		WantDumpOut: "POST / HTTP/1.1\r\n" +
			"Host: post.tld\r\n" +
			"User-Agent: Go-http-client/1.1\r\n" +
			"Content-Length: 6\r\n" +
			"Accept-Encoding: gzip\r\n\r\n",

		NoBody: true,
	},

	// Request with Body > 8196 (default buffer size)
	{
		Req: http.Request{
			Method: "POST",
			URL: &url.URL{
				Scheme: "http",
				Host:   "post.tld",
				Path:   "/",
			},
			Header: http.Header{
				"Content-Length": []string{"8193"},
			},

			ContentLength: 8193,
			ProtoMajor:    1,
			ProtoMinor:    1,
		},

		Body: bytes.Repeat([]byte("a"), 8193),

		WantDumpOut: "POST / HTTP/1.1\r\n" +
			"Host: post.tld\r\n" +
			"User-Agent: Go-http-client/1.1\r\n" +
			"Content-Length: 8193\r\n" +
			"Accept-Encoding: gzip\r\n\r\n" +
			strings.Repeat("a", 8193),
		WantDump: "POST / HTTP/1.1\r\n" +
			"Host: post.tld\r\n" +
			"Content-Length: 8193\r\n\r\n" +
			strings.Repeat("a", 8193),
	},

	{
		Req: *mustReadRequest("GET http://foo.com/ HTTP/1.1\r\n" +
			"User-Agent: blah\r\n\r\n"),
		NoBody: true,
		WantDump: "GET http://foo.com/ HTTP/1.1\r\n" +
			"User-Agent: blah\r\n\r\n",
	},

	// Issue #7215. DumpRequest should return the "Content-Length" when set
	{
		Req: *mustReadRequest("POST /v2/api/?login HTTP/1.1\r\n" +
			"Host: passport.myhost.com\r\n" +
			"Content-Length: 3\r\n" +
			"\r\nkey1=name1&key2=name2"),
		WantDump: "POST /v2/api/?login HTTP/1.1\r\n" +
			"Host: passport.myhost.com\r\n" +
			"Content-Length: 3\r\n" +
			"\r\nkey",
	},

	// Issue #7215. DumpRequest should return the "Content-Length" in ReadRequest
	{
		Req: *mustReadRequest("POST /v2/api/?login HTTP/1.1\r\n" +
			"Host: passport.myhost.com\r\n" +
			"Content-Length: 0\r\n" +
			"\r\nkey1=name1&key2=name2"),
		WantDump: "POST /v2/api/?login HTTP/1.1\r\n" +
			"Host: passport.myhost.com\r\n" +
			"Content-Length: 0\r\n\r\n",
	},

	// Issue #7215. DumpRequest should not return the "Content-Length" if unset
	{
		Req: *mustReadRequest("POST /v2/api/?login HTTP/1.1\r\n" +
			"Host: passport.myhost.com\r\n" +
			"\r\nkey1=name1&key2=name2"),
		WantDump: "POST /v2/api/?login HTTP/1.1\r\n" +
			"Host: passport.myhost.com\r\n\r\n",
	},

	// Issue 18506: make drainBody recognize NoBody. Otherwise
	// this was turning into a chunked request.
	{
		Req: *mustNewRequest("POST", "http://example.com/foo", http.NoBody),

		WantDumpOut: "POST /foo HTTP/1.1\r\n" +
			"Host: example.com\r\n" +
			"User-Agent: Go-http-client/1.1\r\n" +
			"Content-Length: 0\r\n" +
			"Accept-Encoding: gzip\r\n\r\n",
	},
}

func TestDumpRequest(t *testing.T) {
	numg0 := runtime.NumGoroutine()
	for i, tt := range dumpTests {
		setBody := func() {
			if tt.Body == nil {
				return
			}
			switch b := tt.Body.(type) {
			case []byte:
				tt.Req.Body = ioutil.NopCloser(bytes.NewReader(b))
			case func() io.ReadCloser:
				tt.Req.Body = b()
			default:
				t.Fatalf("Test %d: unsupported Body of %T", i, tt.Body)
			}
		}
		setBody()
		if tt.Req.Header == nil {
			tt.Req.Header = make(http.Header)
		}

		if tt.WantDump != "" {
			setBody()
			dump, err := DumpRequest(&tt.Req, !tt.NoBody)
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
			dump, err := DumpRequestOut(&tt.Req, !tt.NoBody)
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
	if dg := runtime.NumGoroutine() - numg0; dg > 4 {
		buf := make([]byte, 4096)
		buf = buf[:runtime.Stack(buf, true)]
		t.Errorf("Unexpectedly large number of new goroutines: %d new: %s", dg, buf)
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

func mustReadRequest(s string) *http.Request {
	req, err := http.ReadRequest(bufio.NewReader(strings.NewReader(s)))
	if err != nil {
		panic(err)
	}
	return req
}

var dumpResTests = []struct {
	res  *http.Response
	body bool
	want string
}{
	{
		res: &http.Response{
			Status:        "200 OK",
			StatusCode:    200,
			Proto:         "HTTP/1.1",
			ProtoMajor:    1,
			ProtoMinor:    1,
			ContentLength: 50,
			Header: http.Header{
				"Foo": []string{"Bar"},
			},
			Body: ioutil.NopCloser(strings.NewReader("foo")), // shouldn't be used
		},
		body: false, // to verify we see 50, not empty or 3.
		want: `HTTP/1.1 200 OK
Content-Length: 50
Foo: Bar`,
	},

	{
		res: &http.Response{
			Status:        "200 OK",
			StatusCode:    200,
			Proto:         "HTTP/1.1",
			ProtoMajor:    1,
			ProtoMinor:    1,
			ContentLength: 3,
			Body:          ioutil.NopCloser(strings.NewReader("foo")),
		},
		body: true,
		want: `HTTP/1.1 200 OK
Content-Length: 3

foo`,
	},

	{
		res: &http.Response{
			Status:           "200 OK",
			StatusCode:       200,
			Proto:            "HTTP/1.1",
			ProtoMajor:       1,
			ProtoMinor:       1,
			ContentLength:    -1,
			Body:             ioutil.NopCloser(strings.NewReader("foo")),
			TransferEncoding: []string{"chunked"},
		},
		body: true,
		want: `HTTP/1.1 200 OK
Transfer-Encoding: chunked

3
foo
0`,
	},
	{
		res: &http.Response{
			Status:        "200 OK",
			StatusCode:    200,
			Proto:         "HTTP/1.1",
			ProtoMajor:    1,
			ProtoMinor:    1,
			ContentLength: 0,
			Header: http.Header{
				// To verify if headers are not filtered out.
				"Foo1": []string{"Bar1"},
				"Foo2": []string{"Bar2"},
			},
			Body: nil,
		},
		body: false, // to verify we see 0, not empty.
		want: `HTTP/1.1 200 OK
Foo1: Bar1
Foo2: Bar2
Content-Length: 0`,
	},
}

func TestDumpResponse(t *testing.T) {
	for i, tt := range dumpResTests {
		gotb, err := DumpResponse(tt.res, tt.body)
		if err != nil {
			t.Errorf("%d. DumpResponse = %v", i, err)
			continue
		}
		got := string(gotb)
		got = strings.TrimSpace(got)
		got = strings.Replace(got, "\r", "", -1)

		if got != tt.want {
			t.Errorf("%d.\nDumpResponse got:\n%s\n\nWant:\n%s\n", i, got, tt.want)
		}
	}
}
