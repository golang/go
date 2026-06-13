// Copyright 2010 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package http

import (
	"io"
	"strings"
	"testing"
)

type respWriteTest struct {
	Resp      Response
	Raw       string
	WantError string // if non-empty, expect Write to return an error containing this string
}

func TestResponseWrite(t *testing.T) {
	respWriteTests := []respWriteTest{
		// HTTP/1.0, identity coding; no trailer
		{
			Resp: Response{
				StatusCode:    503,
				ProtoMajor:    1,
				ProtoMinor:    0,
				Request:       dummyReq("GET"),
				Header:        Header{},
				Body:          io.NopCloser(strings.NewReader("abcdef")),
				ContentLength: 6,
			},
			Raw: "HTTP/1.0 503 Service Unavailable\r\n" +
				"Content-Length: 6\r\n\r\n" +
				"abcdef",
		},
		// Unchunked response without Content-Length.
		{
			Resp: Response{
				StatusCode:    200,
				ProtoMajor:    1,
				ProtoMinor:    0,
				Request:       dummyReq("GET"),
				Header:        Header{},
				Body:          io.NopCloser(strings.NewReader("abcdef")),
				ContentLength: -1,
			},
			Raw: "HTTP/1.0 200 OK\r\n" +
				"\r\n" +
				"abcdef",
		},
		// HTTP/1.1 response with unknown length and Connection: close
		{
			Resp: Response{
				StatusCode:    200,
				ProtoMajor:    1,
				ProtoMinor:    1,
				Request:       dummyReq("GET"),
				Header:        Header{},
				Body:          io.NopCloser(strings.NewReader("abcdef")),
				ContentLength: -1,
				Close:         true,
			},
			Raw: "HTTP/1.1 200 OK\r\n" +
				"Connection: close\r\n" +
				"\r\n" +
				"abcdef",
		},
		// HTTP/1.1 response with unknown length and not setting connection: close
		{
			Resp: Response{
				StatusCode:    200,
				ProtoMajor:    1,
				ProtoMinor:    1,
				Request:       dummyReq11("GET"),
				Header:        Header{},
				Body:          io.NopCloser(strings.NewReader("abcdef")),
				ContentLength: -1,
				Close:         false,
			},
			Raw: "HTTP/1.1 200 OK\r\n" +
				"Connection: close\r\n" +
				"\r\n" +
				"abcdef",
		},
		// HTTP/1.1 response with unknown length and not setting connection: close, but
		// setting chunked.
		{
			Resp: Response{
				StatusCode:       200,
				ProtoMajor:       1,
				ProtoMinor:       1,
				Request:          dummyReq11("GET"),
				Header:           Header{},
				Body:             io.NopCloser(strings.NewReader("abcdef")),
				ContentLength:    -1,
				TransferEncoding: []string{"chunked"},
				Close:            false,
			},
			Raw: "HTTP/1.1 200 OK\r\n" +
				"Transfer-Encoding: chunked\r\n\r\n" +
				"6\r\nabcdef\r\n0\r\n\r\n",
		},
		// HTTP/1.1 response 0 content-length, and nil body
		{
			Resp: Response{
				StatusCode:    200,
				ProtoMajor:    1,
				ProtoMinor:    1,
				Request:       dummyReq11("GET"),
				Header:        Header{},
				Body:          nil,
				ContentLength: 0,
				Close:         false,
			},
			Raw: "HTTP/1.1 200 OK\r\n" +
				"Content-Length: 0\r\n" +
				"\r\n",
		},
		// HTTP/1.1 response 0 content-length, and non-nil empty body
		{
			Resp: Response{
				StatusCode:    200,
				ProtoMajor:    1,
				ProtoMinor:    1,
				Request:       dummyReq11("GET"),
				Header:        Header{},
				Body:          io.NopCloser(strings.NewReader("")),
				ContentLength: 0,
				Close:         false,
			},
			Raw: "HTTP/1.1 200 OK\r\n" +
				"Content-Length: 0\r\n" +
				"\r\n",
		},
		// HTTP/1.1 response 0 content-length, and non-nil non-empty body
		{
			Resp: Response{
				StatusCode:    200,
				ProtoMajor:    1,
				ProtoMinor:    1,
				Request:       dummyReq11("GET"),
				Header:        Header{},
				Body:          io.NopCloser(strings.NewReader("foo")),
				ContentLength: 0,
				Close:         false,
			},
			Raw: "HTTP/1.1 200 OK\r\n" +
				"Connection: close\r\n" +
				"\r\nfoo",
		},
		// HTTP/1.1, chunked coding; empty trailer; close
		{
			Resp: Response{
				StatusCode:       200,
				ProtoMajor:       1,
				ProtoMinor:       1,
				Request:          dummyReq("GET"),
				Header:           Header{},
				Body:             io.NopCloser(strings.NewReader("abcdef")),
				ContentLength:    6,
				TransferEncoding: []string{"chunked"},
				Close:            true,
			},
			Raw: "HTTP/1.1 200 OK\r\n" +
				"Connection: close\r\n" +
				"Transfer-Encoding: chunked\r\n\r\n" +
				"6\r\nabcdef\r\n0\r\n\r\n",
		},

		// Header value with a newline character (Issue 914).
		// Also tests removal of leading and trailing whitespace.
		{
			Resp: Response{
				StatusCode: 204,
				ProtoMajor: 1,
				ProtoMinor: 1,
				Request:    dummyReq("GET"),
				Header: Header{
					"Foo": []string{" Bar\nBaz "},
				},
				Body:             nil,
				ContentLength:    0,
				TransferEncoding: []string{"chunked"},
				Close:            true,
			},
			Raw: "HTTP/1.1 204 No Content\r\n" +
				"Connection: close\r\n" +
				"Foo: Bar Baz\r\n" +
				"\r\n",
		},

		// Want a single Content-Length header. Fixing issue 8180 where
		// there were two.
		{
			Resp: Response{
				StatusCode:       StatusOK,
				ProtoMajor:       1,
				ProtoMinor:       1,
				Request:          &Request{Method: "POST"},
				Header:           Header{},
				ContentLength:    0,
				TransferEncoding: nil,
				Body:             nil,
			},
			Raw: "HTTP/1.1 200 OK\r\nContent-Length: 0\r\n\r\n",
		},

		// When a response to a POST has Content-Length: -1, make sure we don't
		// write the Content-Length as -1.
		{
			Resp: Response{
				StatusCode:    StatusOK,
				ProtoMajor:    1,
				ProtoMinor:    1,
				Request:       &Request{Method: "POST"},
				Header:        Header{},
				ContentLength: -1,
				Body:          io.NopCloser(strings.NewReader("abcdef")),
			},
			Raw: "HTTP/1.1 200 OK\r\nConnection: close\r\n\r\nabcdef",
		},

		// Status code under 100 should be zero-padded to
		// three digits.  Still bogus, but less bogus. (be
		// consistent with generating three digits, since the
		// Transport requires it)
		{
			Resp: Response{
				StatusCode: 7,
				Status:     "license to violate specs",
				ProtoMajor: 1,
				ProtoMinor: 0,
				Request:    dummyReq("GET"),
				Header:     Header{},
				Body:       nil,
			},
			Raw: "HTTP/1.0 007 license to violate specs\r\nContent-Length: 0\r\n\r\n",
		},

		// No stutter.  Status code in 1xx range response should
		// not include a Content-Length header.  See issue #16942.
		{
			Resp: Response{
				StatusCode: 123,
				Status:     "123 Sesame Street",
				ProtoMajor: 1,
				ProtoMinor: 0,
				Request:    dummyReq("GET"),
				Header:     Header{},
				Body:       nil,
			},
			Raw: "HTTP/1.0 123 Sesame Street\r\n\r\n",
		},

		// Status code 204 (No content) response should not include a
		// Content-Length header.  See issue #16942.
		{
			Resp: Response{
				StatusCode: 204,
				Status:     "No Content",
				ProtoMajor: 1,
				ProtoMinor: 0,
				Request:    dummyReq("GET"),
				Header:     Header{},
				Body:       nil,
			},
			Raw: "HTTP/1.0 204 No Content\r\n\r\n",
		},

		// Control character in Status should cause Write to return an error.
		// See issue #78774.
		{
			Resp: Response{
				StatusCode: 200,
				Status:     "200 OK\r\nEvil-Header: injected",
				ProtoMajor: 1,
				ProtoMinor: 1,
				Request:    dummyReq("GET"),
				Header:     Header{},
				Body:       nil,
			},
			WantError: "net/http: can't write control character in Response.Status",
		},
	}

	for i := range respWriteTests {
		tt := &respWriteTests[i]
		var braw strings.Builder
		err := tt.Resp.Write(&braw)
		if tt.WantError != "" {
			if err == nil || err.Error() != tt.WantError {
				t.Errorf("Test %d, error = %v, want %q", i, err, tt.WantError)
			}
			continue
		}
		if err != nil {
			t.Errorf("error writing #%d: %s", i, err)
			continue
		}
		sraw := braw.String()
		if sraw != tt.Raw {
			t.Errorf("Test %d, expecting:\n%q\nGot:\n%q\n", i, tt.Raw, sraw)
			continue
		}
	}
}
