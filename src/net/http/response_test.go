// Copyright 2010 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package http

import (
	"bufio"
	"bytes"
	"compress/gzip"
	"crypto/rand"
	"fmt"
	"go/ast"
	"io"
	"io/ioutil"
	"net/http/internal"
	"net/url"
	"reflect"
	"regexp"
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

func dummyReq11(method string) *Request {
	return &Request{Method: method, Proto: "HTTP/1.1", ProtoMajor: 1, ProtoMinor: 1}
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
				"Connection":     {"close"},
				"Content-Length": {"10"},
			},
			Close:         true,
			ContentLength: 10,
		},

		"Body here\n",
	},

	// Chunked response without Content-Length.
	{
		"HTTP/1.1 200 OK\r\n" +
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
			Proto:            "HTTP/1.1",
			ProtoMajor:       1,
			ProtoMinor:       1,
			Request:          dummyReq("GET"),
			Header:           Header{},
			Close:            false,
			ContentLength:    -1,
			TransferEncoding: []string{"chunked"},
		},

		"Body here\ncontinued",
	},

	// Chunked response with Content-Length.
	{
		"HTTP/1.1 200 OK\r\n" +
			"Transfer-Encoding: chunked\r\n" +
			"Content-Length: 10\r\n" +
			"\r\n" +
			"0a\r\n" +
			"Body here\n\r\n" +
			"0\r\n" +
			"\r\n",

		Response{
			Status:           "200 OK",
			StatusCode:       200,
			Proto:            "HTTP/1.1",
			ProtoMajor:       1,
			ProtoMinor:       1,
			Request:          dummyReq("GET"),
			Header:           Header{},
			Close:            false,
			ContentLength:    -1,
			TransferEncoding: []string{"chunked"},
		},

		"Body here\n",
	},

	// Chunked response in response to a HEAD request
	{
		"HTTP/1.1 200 OK\r\n" +
			"Transfer-Encoding: chunked\r\n" +
			"\r\n",

		Response{
			Status:           "200 OK",
			StatusCode:       200,
			Proto:            "HTTP/1.1",
			ProtoMajor:       1,
			ProtoMinor:       1,
			Request:          dummyReq("HEAD"),
			Header:           Header{},
			TransferEncoding: []string{"chunked"},
			Close:            false,
			ContentLength:    -1,
		},

		"",
	},

	// Content-Length in response to a HEAD request
	{
		"HTTP/1.0 200 OK\r\n" +
			"Content-Length: 256\r\n" +
			"\r\n",

		Response{
			Status:           "200 OK",
			StatusCode:       200,
			Proto:            "HTTP/1.0",
			ProtoMajor:       1,
			ProtoMinor:       0,
			Request:          dummyReq("HEAD"),
			Header:           Header{"Content-Length": {"256"}},
			TransferEncoding: nil,
			Close:            true,
			ContentLength:    256,
		},

		"",
	},

	// Content-Length in response to a HEAD request with HTTP/1.1
	{
		"HTTP/1.1 200 OK\r\n" +
			"Content-Length: 256\r\n" +
			"\r\n",

		Response{
			Status:           "200 OK",
			StatusCode:       200,
			Proto:            "HTTP/1.1",
			ProtoMajor:       1,
			ProtoMinor:       1,
			Request:          dummyReq("HEAD"),
			Header:           Header{"Content-Length": {"256"}},
			TransferEncoding: nil,
			Close:            false,
			ContentLength:    256,
		},

		"",
	},

	// No Content-Length or Chunked in response to a HEAD request
	{
		"HTTP/1.0 200 OK\r\n" +
			"\r\n",

		Response{
			Status:           "200 OK",
			StatusCode:       200,
			Proto:            "HTTP/1.0",
			ProtoMajor:       1,
			ProtoMinor:       0,
			Request:          dummyReq("HEAD"),
			Header:           Header{},
			TransferEncoding: nil,
			Close:            true,
			ContentLength:    -1,
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
			Status:        "303",
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

	// golang.org/issue/4767: don't special-case multipart/byteranges responses
	{
		`HTTP/1.1 206 Partial Content
Connection: close
Content-Type: multipart/byteranges; boundary=18a75608c8f47cef

some body`,
		Response{
			Status:     "206 Partial Content",
			StatusCode: 206,
			Proto:      "HTTP/1.1",
			ProtoMajor: 1,
			ProtoMinor: 1,
			Request:    dummyReq("GET"),
			Header: Header{
				"Content-Type": []string{"multipart/byteranges; boundary=18a75608c8f47cef"},
			},
			Close:         true,
			ContentLength: -1,
		},

		"some body",
	},

	// Unchunked response without Content-Length, Request is nil
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
			Header: Header{
				"Connection": {"close"}, // TODO(rsc): Delete?
			},
			Close:         true,
			ContentLength: -1,
		},

		"Body here\n",
	},

	// 206 Partial Content. golang.org/issue/8923
	{
		"HTTP/1.1 206 Partial Content\r\n" +
			"Content-Type: text/plain; charset=utf-8\r\n" +
			"Accept-Ranges: bytes\r\n" +
			"Content-Range: bytes 0-5/1862\r\n" +
			"Content-Length: 6\r\n\r\n" +
			"foobar",

		Response{
			Status:     "206 Partial Content",
			StatusCode: 206,
			Proto:      "HTTP/1.1",
			ProtoMajor: 1,
			ProtoMinor: 1,
			Request:    dummyReq("GET"),
			Header: Header{
				"Accept-Ranges":  []string{"bytes"},
				"Content-Length": []string{"6"},
				"Content-Type":   []string{"text/plain; charset=utf-8"},
				"Content-Range":  []string{"bytes 0-5/1862"},
			},
			ContentLength: 6,
		},

		"foobar",
	},

	// Both keep-alive and close, on the same Connection line. (Issue 8840)
	{
		"HTTP/1.1 200 OK\r\n" +
			"Content-Length: 256\r\n" +
			"Connection: keep-alive, close\r\n" +
			"\r\n",

		Response{
			Status:     "200 OK",
			StatusCode: 200,
			Proto:      "HTTP/1.1",
			ProtoMajor: 1,
			ProtoMinor: 1,
			Request:    dummyReq("HEAD"),
			Header: Header{
				"Content-Length": {"256"},
			},
			TransferEncoding: nil,
			Close:            true,
			ContentLength:    256,
		},

		"",
	},

	// Both keep-alive and close, on different Connection lines. (Issue 8840)
	{
		"HTTP/1.1 200 OK\r\n" +
			"Content-Length: 256\r\n" +
			"Connection: keep-alive\r\n" +
			"Connection: close\r\n" +
			"\r\n",

		Response{
			Status:     "200 OK",
			StatusCode: 200,
			Proto:      "HTTP/1.1",
			ProtoMajor: 1,
			ProtoMinor: 1,
			Request:    dummyReq("HEAD"),
			Header: Header{
				"Content-Length": {"256"},
			},
			TransferEncoding: nil,
			Close:            true,
			ContentLength:    256,
		},

		"",
	},

	// Issue 12785: HTTP/1.0 response with bogus (to be ignored) Transfer-Encoding.
	// Without a Content-Length.
	{
		"HTTP/1.0 200 OK\r\n" +
			"Transfer-Encoding: bogus\r\n" +
			"\r\n" +
			"Body here\n",

		Response{
			Status:        "200 OK",
			StatusCode:    200,
			Proto:         "HTTP/1.0",
			ProtoMajor:    1,
			ProtoMinor:    0,
			Request:       dummyReq("GET"),
			Header:        Header{},
			Close:         true,
			ContentLength: -1,
		},

		"Body here\n",
	},

	// Issue 12785: HTTP/1.0 response with bogus (to be ignored) Transfer-Encoding.
	// With a Content-Length.
	{
		"HTTP/1.0 200 OK\r\n" +
			"Transfer-Encoding: bogus\r\n" +
			"Content-Length: 10\r\n" +
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
				"Content-Length": {"10"},
			},
			Close:         true,
			ContentLength: 10,
		},

		"Body here\n",
	},

	{
		"HTTP/1.1 200 OK\r\n" +
			"Content-Encoding: gzip\r\n" +
			"Content-Length: 23\r\n" +
			"Connection: keep-alive\r\n" +
			"Keep-Alive: timeout=7200\r\n\r\n" +
			"\x1f\x8b\b\x00\x00\x00\x00\x00\x00\x00s\xf3\xf7\a\x00\xab'\xd4\x1a\x03\x00\x00\x00",
		Response{
			Status:     "200 OK",
			StatusCode: 200,
			Proto:      "HTTP/1.1",
			ProtoMajor: 1,
			ProtoMinor: 1,
			Request:    dummyReq("GET"),
			Header: Header{
				"Content-Length":   {"23"},
				"Content-Encoding": {"gzip"},
				"Connection":       {"keep-alive"},
				"Keep-Alive":       {"timeout=7200"},
			},
			Close:         false,
			ContentLength: 23,
		},
		"\x1f\x8b\b\x00\x00\x00\x00\x00\x00\x00s\xf3\xf7\a\x00\xab'\xd4\x1a\x03\x00\x00\x00",
	},

	// Issue 19989: two spaces between HTTP version and status.
	{
		"HTTP/1.0  401 Unauthorized\r\n" +
			"Content-type: text/html\r\n" +
			"WWW-Authenticate: Basic realm=\"\"\r\n\r\n" +
			"Your Authentication failed.\r\n",
		Response{
			Status:     "401 Unauthorized",
			StatusCode: 401,
			Proto:      "HTTP/1.0",
			ProtoMajor: 1,
			ProtoMinor: 0,
			Request:    dummyReq("GET"),
			Header: Header{
				"Content-Type":     {"text/html"},
				"Www-Authenticate": {`Basic realm=""`},
			},
			Close:         true,
			ContentLength: -1,
		},
		"Your Authentication failed.\r\n",
	},
}

// tests successful calls to ReadResponse, and inspects the returned Response.
// For error cases, see TestReadResponseErrors below.
func TestReadResponse(t *testing.T) {
	for i, tt := range respTests {
		resp, err := ReadResponse(bufio.NewReader(strings.NewReader(tt.Raw)), tt.Resp.Request)
		if err != nil {
			t.Errorf("#%d: %v", i, err)
			continue
		}
		rbody := resp.Body
		resp.Body = nil
		diff(t, fmt.Sprintf("#%d Response", i), resp, &tt.Resp)
		var bout bytes.Buffer
		if rbody != nil {
			_, err = io.Copy(&bout, rbody)
			if err != nil {
				t.Errorf("#%d: %v", i, err)
				continue
			}
			rbody.Close()
		}
		body := bout.String()
		if body != tt.Body {
			t.Errorf("#%d: Body = %q want %q", i, body, tt.Body)
		}
	}
}

func TestWriteResponse(t *testing.T) {
	for i, tt := range respTests {
		resp, err := ReadResponse(bufio.NewReader(strings.NewReader(tt.Raw)), tt.Resp.Request)
		if err != nil {
			t.Errorf("#%d: %v", i, err)
			continue
		}
		err = resp.Write(ioutil.Discard)
		if err != nil {
			t.Errorf("#%d: %v", i, err)
			continue
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
	t.Parallel()
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
			wr = internal.NewChunkedWriter(wr)
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
			resp.Body = &readerAndCloser{gzReader, resp.Body}
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
			g = regexp.MustCompile(`(xx+)`).ReplaceAllStringFunc(g, func(match string) string {
				return fmt.Sprintf("x(repeated x%d)", len(match))
			})
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
		name := hv.Type().Field(i).Name
		if !ast.IsExported(name) {
			continue
		}
		hf := hv.Field(i).Interface()
		wf := wv.Field(i).Interface()
		if !reflect.DeepEqual(hf, wf) {
			t.Errorf("%s: %s = %v want %v", prefix, name, hf, wf)
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
	{"/bar", "", "/bar", nil},
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

func TestResponseContentLengthShortBody(t *testing.T) {
	const shortBody = "Short body, not 123 bytes."
	br := bufio.NewReader(strings.NewReader("HTTP/1.1 200 OK\r\n" +
		"Content-Length: 123\r\n" +
		"\r\n" +
		shortBody))
	res, err := ReadResponse(br, &Request{Method: "GET"})
	if err != nil {
		t.Fatal(err)
	}
	if res.ContentLength != 123 {
		t.Fatalf("Content-Length = %d; want 123", res.ContentLength)
	}
	var buf bytes.Buffer
	n, err := io.Copy(&buf, res.Body)
	if n != int64(len(shortBody)) {
		t.Errorf("Copied %d bytes; want %d, len(%q)", n, len(shortBody), shortBody)
	}
	if buf.String() != shortBody {
		t.Errorf("Read body %q; want %q", buf.String(), shortBody)
	}
	if err != io.ErrUnexpectedEOF {
		t.Errorf("io.Copy error = %#v; want io.ErrUnexpectedEOF", err)
	}
}

// Test various ReadResponse error cases. (also tests success cases, but mostly
// it's about errors).  This does not test anything involving the bodies. Only
// the return value from ReadResponse itself.
func TestReadResponseErrors(t *testing.T) {
	type testCase struct {
		name    string // optional, defaults to in
		in      string
		header  Header
		wantErr interface{} // nil, err value, or string substring
	}

	status := func(s string, wantErr interface{}) testCase {
		if wantErr == true {
			wantErr = "malformed HTTP status code"
		}
		return testCase{
			name:    fmt.Sprintf("status %q", s),
			in:      "HTTP/1.1 " + s + "\r\nFoo: bar\r\n\r\n",
			wantErr: wantErr,
		}
	}

	version := func(s string, wantErr interface{}) testCase {
		if wantErr == true {
			wantErr = "malformed HTTP version"
		}
		return testCase{
			name:    fmt.Sprintf("version %q", s),
			in:      s + " 200 OK\r\n\r\n",
			wantErr: wantErr,
		}
	}

	contentLength := func(status, body string, wantErr interface{}, header Header) testCase {
		return testCase{
			name:    fmt.Sprintf("status %q %q", status, body),
			in:      fmt.Sprintf("HTTP/1.1 %s\r\n%s", status, body),
			wantErr: wantErr,
			header:  header,
		}
	}

	errMultiCL := "message cannot contain multiple Content-Length headers"

	tests := []testCase{
		{"", "", nil, io.ErrUnexpectedEOF},
		{"", "HTTP/1.1 301 Moved Permanently\r\nFoo: bar", nil, io.ErrUnexpectedEOF},
		{"", "HTTP/1.1", nil, "malformed HTTP response"},
		{"", "HTTP/2.0", nil, "malformed HTTP response"},
		status("20X Unknown", true),
		status("abcd Unknown", true),
		status("二百/两百 OK", true),
		status(" Unknown", true),
		status("c8 OK", true),
		status("0x12d Moved Permanently", true),
		status("200 OK", nil),
		status("000 OK", nil),
		status("001 OK", nil),
		status("404 NOTFOUND", nil),
		status("20 OK", true),
		status("00 OK", true),
		status("-10 OK", true),
		status("1000 OK", true),
		status("999 Done", nil),
		status("-1 OK", true),
		status("-200 OK", true),
		version("HTTP/1.2", nil),
		version("HTTP/2.0", nil),
		version("HTTP/1.100000000002", true),
		version("HTTP/1.-1", true),
		version("HTTP/A.B", true),
		version("HTTP/1", true),
		version("http/1.1", true),

		contentLength("200 OK", "Content-Length: 10\r\nContent-Length: 7\r\n\r\nGopher hey\r\n", errMultiCL, nil),
		contentLength("200 OK", "Content-Length: 7\r\nContent-Length: 7\r\n\r\nGophers\r\n", nil, Header{"Content-Length": {"7"}}),
		contentLength("201 OK", "Content-Length: 0\r\nContent-Length: 7\r\n\r\nGophers\r\n", errMultiCL, nil),
		contentLength("300 OK", "Content-Length: 0\r\nContent-Length: 0 \r\n\r\nGophers\r\n", nil, Header{"Content-Length": {"0"}}),
		contentLength("200 OK", "Content-Length:\r\nContent-Length:\r\n\r\nGophers\r\n", nil, nil),
		contentLength("206 OK", "Content-Length:\r\nContent-Length: 0 \r\nConnection: close\r\n\r\nGophers\r\n", errMultiCL, nil),

		// multiple content-length headers for 204 and 304 should still be checked
		contentLength("204 OK", "Content-Length: 7\r\nContent-Length: 8\r\n\r\n", errMultiCL, nil),
		contentLength("204 OK", "Content-Length: 3\r\nContent-Length: 3\r\n\r\n", nil, nil),
		contentLength("304 OK", "Content-Length: 880\r\nContent-Length: 1\r\n\r\n", errMultiCL, nil),
		contentLength("304 OK", "Content-Length: 961\r\nContent-Length: 961\r\n\r\n", nil, nil),
	}

	for i, tt := range tests {
		br := bufio.NewReader(strings.NewReader(tt.in))
		_, rerr := ReadResponse(br, nil)
		if err := matchErr(rerr, tt.wantErr); err != nil {
			name := tt.name
			if name == "" {
				name = fmt.Sprintf("%d. input %q", i, tt.in)
			}
			t.Errorf("%s: %v", name, err)
		}
	}
}

// wantErr can be nil, an error value to match exactly, or type string to
// match a substring.
func matchErr(err error, wantErr interface{}) error {
	if err == nil {
		if wantErr == nil {
			return nil
		}
		if sub, ok := wantErr.(string); ok {
			return fmt.Errorf("unexpected success; want error with substring %q", sub)
		}
		return fmt.Errorf("unexpected success; want error %v", wantErr)
	}
	if wantErr == nil {
		return fmt.Errorf("%v; want success", err)
	}
	if sub, ok := wantErr.(string); ok {
		if strings.Contains(err.Error(), sub) {
			return nil
		}
		return fmt.Errorf("error = %v; want an error with substring %q", err, sub)
	}
	if err == wantErr {
		return nil
	}
	return fmt.Errorf("%v; want %v", err, wantErr)
}

func TestNeedsSniff(t *testing.T) {
	// needsSniff returns true with an empty response.
	r := &response{}
	if got, want := r.needsSniff(), true; got != want {
		t.Errorf("needsSniff = %t; want %t", got, want)
	}
	// needsSniff returns false when Content-Type = nil.
	r.handlerHeader = Header{"Content-Type": nil}
	if got, want := r.needsSniff(), false; got != want {
		t.Errorf("needsSniff empty Content-Type = %t; want %t", got, want)
	}
}

// A response should only write out single Connection: close header. Tests #19499.
func TestResponseWritesOnlySingleConnectionClose(t *testing.T) {
	const connectionCloseHeader = "Connection: close"

	res, err := ReadResponse(bufio.NewReader(strings.NewReader("HTTP/1.0 200 OK\r\n\r\nAAAA")), nil)
	if err != nil {
		t.Fatalf("ReadResponse failed %v", err)
	}

	var buf1 bytes.Buffer
	if err = res.Write(&buf1); err != nil {
		t.Fatalf("Write failed %v", err)
	}
	if res, err = ReadResponse(bufio.NewReader(&buf1), nil); err != nil {
		t.Fatalf("ReadResponse failed %v", err)
	}

	var buf2 bytes.Buffer
	if err = res.Write(&buf2); err != nil {
		t.Fatalf("Write failed %v", err)
	}
	if count := strings.Count(buf2.String(), connectionCloseHeader); count != 1 {
		t.Errorf("Found %d %q header", count, connectionCloseHeader)
	}
}
