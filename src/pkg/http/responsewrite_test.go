// Copyright 2010 The Go Authors.  All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package http

import (
	"bytes"
	"testing"
)

type respWriteTest struct {
	Resp Response
	Raw  string
}

var respWriteTests = []respWriteTest{
	// HTTP/1.0, identity coding; no trailer
	{
		Response{
			StatusCode:    503,
			ProtoMajor:    1,
			ProtoMinor:    0,
			RequestMethod: "GET",
			Header:        map[string]string{},
			Body:          nopCloser{bytes.NewBufferString("abcdef")},
			ContentLength: 6,
		},

		"HTTP/1.0 503 Service Unavailable\r\n" +
			"Content-Length: 6\r\n\r\n" +
			"abcdef",
	},
	// Unchunked response without Content-Length.
	{
		Response{
			StatusCode:    200,
			ProtoMajor:    1,
			ProtoMinor:    0,
			RequestMethod: "GET",
			Header:        map[string]string{},
			Body:          nopCloser{bytes.NewBufferString("abcdef")},
			ContentLength: -1,
		},
		"HTTP/1.0 200 OK\r\n" +
			"\r\n" +
			"abcdef",
	},
	// HTTP/1.1, chunked coding; empty trailer; close
	{
		Response{
			StatusCode:       200,
			ProtoMajor:       1,
			ProtoMinor:       1,
			RequestMethod:    "GET",
			Header:           map[string]string{},
			Body:             nopCloser{bytes.NewBufferString("abcdef")},
			ContentLength:    6,
			TransferEncoding: []string{"chunked"},
			Close:            true,
		},

		"HTTP/1.1 200 OK\r\n" +
			"Connection: close\r\n" +
			"Transfer-Encoding: chunked\r\n\r\n" +
			"6\r\nabcdef\r\n0\r\n\r\n",
	},
}

func TestResponseWrite(t *testing.T) {
	for i := range respWriteTests {
		tt := &respWriteTests[i]
		var braw bytes.Buffer
		err := tt.Resp.Write(&braw)
		if err != nil {
			t.Errorf("error writing #%d: %s", i, err)
			continue
		}
		sraw := braw.String()
		if sraw != tt.Raw {
			t.Errorf("Test %d, expecting:\n%s\nGot:\n%s\n", i, tt.Raw, sraw)
			continue
		}
	}
}
