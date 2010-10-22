// Copyright 2010 The Go Authors.  All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package http

import (
	"bytes"
	"testing"
)

type reqWriteTest struct {
	Req Request
	Raw string
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
			Header: map[string]string{
				"Accept":           "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
				"Accept-Charset":   "ISO-8859-1,utf-8;q=0.7,*;q=0.7",
				"Accept-Encoding":  "gzip,deflate",
				"Accept-Language":  "en-us,en;q=0.5",
				"Keep-Alive":       "300",
				"Proxy-Connection": "keep-alive",
			},
			Body:      nil,
			Close:     false,
			Host:      "www.techcrunch.com",
			Referer:   "",
			UserAgent: "Fake",
			Form:      map[string][]string{},
		},

		"GET http://www.techcrunch.com/ HTTP/1.1\r\n" +
			"Host: www.techcrunch.com\r\n" +
			"User-Agent: Fake\r\n" +
			"Accept-Charset: ISO-8859-1,utf-8;q=0.7,*;q=0.7\r\n" +
			"Accept-Encoding: gzip,deflate\r\n" +
			"Accept-Language: en-us,en;q=0.5\r\n" +
			"Accept: text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8\r\n" +
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
			Header:           map[string]string{},
			Body:             nopCloser{bytes.NewBufferString("abcdef")},
			TransferEncoding: []string{"chunked"},
		},

		"GET /search HTTP/1.1\r\n" +
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
			Header:           map[string]string{},
			Close:            true,
			Body:             nopCloser{bytes.NewBufferString("abcdef")},
			TransferEncoding: []string{"chunked"},
		},

		"POST /search HTTP/1.1\r\n" +
			"Host: www.google.com\r\n" +
			"User-Agent: Go http package\r\n" +
			"Connection: close\r\n" +
			"Transfer-Encoding: chunked\r\n\r\n" +
			"6\r\nabcdef\r\n0\r\n\r\n",
	},
	// default to HTTP/1.1
	{
		Request{
			Method: "GET",
			RawURL: "/search",
			Host:   "www.google.com",
		},

		"GET /search HTTP/1.1\r\n" +
			"Host: www.google.com\r\n" +
			"User-Agent: Go http package\r\n" +
			"\r\n",
	},
}

func TestRequestWrite(t *testing.T) {
	for i := range reqWriteTests {
		tt := &reqWriteTests[i]
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
	}
}
