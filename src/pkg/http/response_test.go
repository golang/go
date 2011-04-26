// Copyright 2010 The Go Authors.  All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package http

import (
	"bufio"
	"bytes"
	"fmt"
	"io"
	"io/ioutil"
	"reflect"
	"strings"
	"testing"
)

type respTest struct {
	Raw  string
	Resp Response
	Body string
}

var respTests = []respTest{
	// Unchunked response without Content-Length.
	{
		"HTTP/1.0 200 OK\r\n" +
			"Connection: close\r\n" +
			"\r\n" +
			"Body here\n",

		Response{
			Status:        "200 OK",
			StatusCode:    200,
			Proto:         "HTTP/1.0",
			ProtoMajor:    1,
			ProtoMinor:    0,
			RequestMethod: "GET",
			Header: Header{
				"Connection": {"close"}, // TODO(rsc): Delete?
			},
			Close:         true,
			ContentLength: -1,
		},

		"Body here\n",
	},

	// Unchunked HTTP/1.1 response without Content-Length or
	// Connection headers.
	{
		"HTTP/1.1 200 OK\r\n" +
			"\r\n" +
			"Body here\n",

		Response{
			Status:        "200 OK",
			StatusCode:    200,
			Proto:         "HTTP/1.1",
			ProtoMajor:    1,
			ProtoMinor:    1,
			RequestMethod: "GET",
			Close:         true,
			ContentLength: -1,
		},

		"Body here\n",
	},

	// Unchunked HTTP/1.1 204 response without Content-Length.
	{
		"HTTP/1.1 204 No Content\r\n" +
			"\r\n" +
			"Body should not be read!\n",

		Response{
			Status:        "204 No Content",
			StatusCode:    204,
			Proto:         "HTTP/1.1",
			ProtoMajor:    1,
			ProtoMinor:    1,
			RequestMethod: "GET",
			Close:         false,
			ContentLength: 0,
		},

		"",
	},

	// Unchunked response with Content-Length.
	{
		"HTTP/1.0 200 OK\r\n" +
			"Content-Length: 10\r\n" +
			"Connection: close\r\n" +
			"\r\n" +
			"Body here\n",

		Response{
			Status:        "200 OK",
			StatusCode:    200,
			Proto:         "HTTP/1.0",
			ProtoMajor:    1,
			ProtoMinor:    0,
			RequestMethod: "GET",
			Header: Header{
				"Connection":     {"close"}, // TODO(rsc): Delete?
				"Content-Length": {"10"},    // TODO(rsc): Delete?
			},
			Close:         true,
			ContentLength: 10,
		},

		"Body here\n",
	},

	// Chunked response without Content-Length.
	{
		"HTTP/1.0 200 OK\r\n" +
			"Transfer-Encoding: chunked\r\n" +
			"\r\n" +
			"0a\r\n" +
			"Body here\n\r\n" +
			"09\r\n" +
			"continued\r\n" +
			"0\r\n" +
			"\r\n",

		Response{
			Status:           "200 OK",
			StatusCode:       200,
			Proto:            "HTTP/1.0",
			ProtoMajor:       1,
			ProtoMinor:       0,
			RequestMethod:    "GET",
			Header:           Header{},
			Close:            true,
			ContentLength:    -1,
			TransferEncoding: []string{"chunked"},
		},

		"Body here\ncontinued",
	},

	// Chunked response with Content-Length.
	{
		"HTTP/1.0 200 OK\r\n" +
			"Transfer-Encoding: chunked\r\n" +
			"Content-Length: 10\r\n" +
			"\r\n" +
			"0a\r\n" +
			"Body here\n" +
			"0\r\n" +
			"\r\n",

		Response{
			Status:           "200 OK",
			StatusCode:       200,
			Proto:            "HTTP/1.0",
			ProtoMajor:       1,
			ProtoMinor:       0,
			RequestMethod:    "GET",
			Header:           Header{},
			Close:            true,
			ContentLength:    -1, // TODO(rsc): Fix?
			TransferEncoding: []string{"chunked"},
		},

		"Body here\n",
	},

	// Chunked response in response to a HEAD request (the "chunked" should
	// be ignored, as HEAD responses never have bodies)
	{
		"HTTP/1.0 200 OK\r\n" +
			"Transfer-Encoding: chunked\r\n" +
			"\r\n",

		Response{
			Status:        "200 OK",
			StatusCode:    200,
			Proto:         "HTTP/1.0",
			ProtoMajor:    1,
			ProtoMinor:    0,
			RequestMethod: "HEAD",
			Header:        Header{},
			Close:         true,
			ContentLength: 0,
		},

		"",
	},

	// explicit Content-Length of 0.
	{
		"HTTP/1.1 200 OK\r\n" +
			"Content-Length: 0\r\n" +
			"\r\n",

		Response{
			Status:        "200 OK",
			StatusCode:    200,
			Proto:         "HTTP/1.1",
			ProtoMajor:    1,
			ProtoMinor:    1,
			RequestMethod: "GET",
			Header: Header{
				"Content-Length": {"0"},
			},
			Close:         false,
			ContentLength: 0,
		},

		"",
	},

	// Status line without a Reason-Phrase, but trailing space.
	// (permitted by RFC 2616)
	{
		"HTTP/1.0 303 \r\n\r\n",
		Response{
			Status:        "303 ",
			StatusCode:    303,
			Proto:         "HTTP/1.0",
			ProtoMajor:    1,
			ProtoMinor:    0,
			RequestMethod: "GET",
			Header:        Header{},
			Close:         true,
			ContentLength: -1,
		},

		"",
	},

	// Status line without a Reason-Phrase, and no trailing space.
	// (not permitted by RFC 2616, but we'll accept it anyway)
	{
		"HTTP/1.0 303\r\n\r\n",
		Response{
			Status:        "303 ",
			StatusCode:    303,
			Proto:         "HTTP/1.0",
			ProtoMajor:    1,
			ProtoMinor:    0,
			RequestMethod: "GET",
			Header:        Header{},
			Close:         true,
			ContentLength: -1,
		},

		"",
	},
}

func TestReadResponse(t *testing.T) {
	for i := range respTests {
		tt := &respTests[i]
		var braw bytes.Buffer
		braw.WriteString(tt.Raw)
		resp, err := ReadResponse(bufio.NewReader(&braw), tt.Resp.RequestMethod)
		if err != nil {
			t.Errorf("#%d: %s", i, err)
			continue
		}
		rbody := resp.Body
		resp.Body = nil
		diff(t, fmt.Sprintf("#%d Response", i), resp, &tt.Resp)
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

// TestReadResponseCloseInMiddle tests that for both chunked and unchunked responses,
// if we close the Body while only partway through reading, the underlying reader
// advanced to the end of the request.
func TestReadResponseCloseInMiddle(t *testing.T) {
	for _, chunked := range []bool{false, true} {
		var buf bytes.Buffer
		buf.WriteString("HTTP/1.1 200 OK\r\n")
		if chunked {
			buf.WriteString("Transfer-Encoding: chunked\r\n\r\n")
		} else {
			buf.WriteString("Content-Length: 1000000\r\n\r\n")
		}
		chunk := strings.Repeat("x", 1000)
		for i := 0; i < 1000; i++ {
			if chunked {
				buf.WriteString("03E8\r\n")
				buf.WriteString(chunk)
				buf.WriteString("\r\n")
			} else {
				buf.WriteString(chunk)
			}
		}
		if chunked {
			buf.WriteString("0\r\n\r\n")
		}
		buf.WriteString("Next Request Here")
		bufr := bufio.NewReader(&buf)
		resp, err := ReadResponse(bufr, "GET")
		if err != nil {
			t.Fatalf("parse error for chunked=%v: %v", chunked, err)
		}

		expectedLength := int64(-1)
		if !chunked {
			expectedLength = 1000000
		}
		if resp.ContentLength != expectedLength {
			t.Fatalf("chunked=%v: expected response length %d, got %d", chunked, expectedLength, resp.ContentLength)
		}
		rbuf := make([]byte, 2500)
		n, err := io.ReadFull(resp.Body, rbuf)
		if err != nil {
			t.Fatalf("ReadFull error for chunked=%v: %v", chunked, err)
		}
		if n != 2500 {
			t.Fatalf("ReadFull only read %n bytes for chunked=%v", n, chunked)
		}
		if !bytes.Equal(bytes.Repeat([]byte{'x'}, 2500), rbuf) {
			t.Fatalf("ReadFull didn't read 2500 'x' for chunked=%v; got %q", chunked, string(rbuf))
		}
		resp.Body.Close()

		rest, err := ioutil.ReadAll(bufr)
		if err != nil {
			t.Fatalf("ReadAll error on remainder for chunked=%v: %v", chunked, err)
		}
		if e, g := "Next Request Here", string(rest); e != g {
			t.Fatalf("for chunked=%v remainder = %q, expected %q", chunked, g, e)
		}
	}
}

func diff(t *testing.T, prefix string, have, want interface{}) {
	hv := reflect.ValueOf(have).Elem()
	wv := reflect.ValueOf(want).Elem()
	if hv.Type() != wv.Type() {
		t.Errorf("%s: type mismatch %v vs %v", prefix, hv.Type(), wv.Type())
	}
	for i := 0; i < hv.NumField(); i++ {
		hf := hv.Field(i).Interface()
		wf := wv.Field(i).Interface()
		if !reflect.DeepEqual(hf, wf) {
			t.Errorf("%s: %s = %v want %v", prefix, hv.Type().Field(i).Name, hf, wf)
		}
	}
}
