// Copyright 2010 The Go Authors.  All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package http

import (
	"bufio"
	"bytes"
	"compress/gzip"
	"crypto/rand"
	"fmt"
	"io"
	"io/ioutil"
	"net/url"
	"reflect"
	"strings"
	"testing"
)

type respTest struct {
	Raw  string
	Resp Response
	Body string
}

func dummyReq(method string) *Request {
	return &Request{Method: method}
}

var respTests = []respTest{
	// Unchunked response without Content-Length.
	{
		"HTTP/1.0 200 OK\r\n" +
			"Connection: close\r\n" +
			"\r\n" +
			"Body here\n",

		Response{
			Status:     "200 OK",
			StatusCode: 200,
			Proto:      "HTTP/1.0",
			ProtoMajor: 1,
			ProtoMinor: 0,
			Request:    dummyReq("GET"),
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
			Header:        Header{},
			Request:       dummyReq("GET"),
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
			Header:        Header{},
			Request:       dummyReq("GET"),
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
			Status:     "200 OK",
			StatusCode: 200,
			Proto:      "HTTP/1.0",
			ProtoMajor: 1,
			ProtoMinor: 0,
			Request:    dummyReq("GET"),
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
			Request:          dummyReq("GET"),
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
			Request:          dummyReq("GET"),
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
			Request:       dummyReq("HEAD"),
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
			Status:     "200 OK",
			StatusCode: 200,
			Proto:      "HTTP/1.1",
			ProtoMajor: 1,
			ProtoMinor: 1,
			Request:    dummyReq("GET"),
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
			Request:       dummyReq("GET"),
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
			Request:       dummyReq("GET"),
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
		resp, err := ReadResponse(bufio.NewReader(&braw), tt.Resp.Request)
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

var readResponseCloseInMiddleTests = []struct {
	chunked, compressed bool
}{
	{false, false},
	{true, false},
	{true, true},
}

// TestReadResponseCloseInMiddle tests that closing a body after
// reading only part of its contents advances the read to the end of
// the request, right up until the next request.
func TestReadResponseCloseInMiddle(t *testing.T) {
	for _, test := range readResponseCloseInMiddleTests {
		fatalf := func(format string, args ...interface{}) {
			args = append([]interface{}{test.chunked, test.compressed}, args...)
			t.Fatalf("on test chunked=%v, compressed=%v: "+format, args...)
		}
		checkErr := func(err error, msg string) {
			if err == nil {
				return
			}
			fatalf(msg+": %v", err)
		}
		var buf bytes.Buffer
		buf.WriteString("HTTP/1.1 200 OK\r\n")
		if test.chunked {
			buf.WriteString("Transfer-Encoding: chunked\r\n")
		} else {
			buf.WriteString("Content-Length: 1000000\r\n")
		}
		var wr io.Writer = &buf
		if test.chunked {
			wr = newChunkedWriter(wr)
		}
		if test.compressed {
			buf.WriteString("Content-Encoding: gzip\r\n")
			wr = gzip.NewWriter(wr)
		}
		buf.WriteString("\r\n")

		chunk := bytes.Repeat([]byte{'x'}, 1000)
		for i := 0; i < 1000; i++ {
			if test.compressed {
				// Otherwise this compresses too well.
				_, err := io.ReadFull(rand.Reader, chunk)
				checkErr(err, "rand.Reader ReadFull")
			}
			wr.Write(chunk)
		}
		if test.compressed {
			err := wr.(*gzip.Writer).Close()
			checkErr(err, "compressor close")
		}
		if test.chunked {
			buf.WriteString("0\r\n\r\n")
		}
		buf.WriteString("Next Request Here")

		bufr := bufio.NewReader(&buf)
		resp, err := ReadResponse(bufr, dummyReq("GET"))
		checkErr(err, "ReadResponse")
		expectedLength := int64(-1)
		if !test.chunked {
			expectedLength = 1000000
		}
		if resp.ContentLength != expectedLength {
			fatalf("expected response length %d, got %d", expectedLength, resp.ContentLength)
		}
		if resp.Body == nil {
			fatalf("nil body")
		}
		if test.compressed {
			gzReader, err := gzip.NewReader(resp.Body)
			checkErr(err, "gzip.NewReader")
			resp.Body = &readFirstCloseBoth{gzReader, resp.Body}
		}

		rbuf := make([]byte, 2500)
		n, err := io.ReadFull(resp.Body, rbuf)
		checkErr(err, "2500 byte ReadFull")
		if n != 2500 {
			fatalf("ReadFull only read %d bytes", n)
		}
		if test.compressed == false && !bytes.Equal(bytes.Repeat([]byte{'x'}, 2500), rbuf) {
			fatalf("ReadFull didn't read 2500 'x'; got %q", string(rbuf))
		}
		resp.Body.Close()

		rest, err := ioutil.ReadAll(bufr)
		checkErr(err, "ReadAll on remainder")
		if e, g := "Next Request Here", string(rest); e != g {
			fatalf("remainder = %q, expected %q", g, e)
		}
	}
}

func diff(t *testing.T, prefix string, have, want interface{}) {
	hv := reflect.ValueOf(have).Elem()
	wv := reflect.ValueOf(want).Elem()
	if hv.Type() != wv.Type() {
		t.Errorf("%s: type mismatch %v want %v", prefix, hv.Type(), wv.Type())
	}
	for i := 0; i < hv.NumField(); i++ {
		hf := hv.Field(i).Interface()
		wf := wv.Field(i).Interface()
		if !reflect.DeepEqual(hf, wf) {
			t.Errorf("%s: %s = %v want %v", prefix, hv.Type().Field(i).Name, hf, wf)
		}
	}
}

type responseLocationTest struct {
	location string // Response's Location header or ""
	requrl   string // Response.Request.URL or ""
	want     string
	wantErr  error
}

var responseLocationTests = []responseLocationTest{
	{"/foo", "http://bar.com/baz", "http://bar.com/foo", nil},
	{"http://foo.com/", "http://bar.com/baz", "http://foo.com/", nil},
	{"", "http://bar.com/baz", "", ErrNoLocation},
}

func TestLocationResponse(t *testing.T) {
	for i, tt := range responseLocationTests {
		res := new(Response)
		res.Header = make(Header)
		res.Header.Set("Location", tt.location)
		if tt.requrl != "" {
			res.Request = &Request{}
			var err error
			res.Request.URL, err = url.Parse(tt.requrl)
			if err != nil {
				t.Fatalf("bad test URL %q: %v", tt.requrl, err)
			}
		}

		got, err := res.Location()
		if tt.wantErr != nil {
			if err == nil {
				t.Errorf("%d. err=nil; want %q", i, tt.wantErr)
				continue
			}
			if g, e := err.Error(), tt.wantErr.Error(); g != e {
				t.Errorf("%d. err=%q; want %q", i, g, e)
				continue
			}
			continue
		}
		if err != nil {
			t.Errorf("%d. err=%q", i, err)
			continue
		}
		if g, e := got.String(), tt.want; g != e {
			t.Errorf("%d. Location=%q; want %q", i, g, e)
		}
	}
}

func TestResponseStatusStutter(t *testing.T) {
	r := &Response{
		Status:     "123 some status",
		StatusCode: 123,
		ProtoMajor: 1,
		ProtoMinor: 3,
	}
	var buf bytes.Buffer
	r.Write(&buf)
	if strings.Contains(buf.String(), "123 123") {
		t.Errorf("stutter in status: %s", buf.String())
	}
}
