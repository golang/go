// Copyright 2010 The Go Authors.  All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package http

import (
	"bufio"
	"bytes"
	"fmt"
	"io"
	"reflect"
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
			Header: map[string]string{
				"Connection": "close", // TODO(rsc): Delete?
			},
			Close:         true,
			ContentLength: -1,
		},

		"Body here\n",
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
			Header: map[string]string{
				"Connection":     "close", // TODO(rsc): Delete?
				"Content-Length": "10",    // TODO(rsc): Delete?
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
			Header:           map[string]string{},
			Close:            true,
			ContentLength:    -1,
			TransferEncoding: []string{"chunked"},
		},

		"Body here\n",
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
			Header:           map[string]string{},
			Close:            true,
			ContentLength:    -1, // TODO(rsc): Fix?
			TransferEncoding: []string{"chunked"},
		},

		"Body here\n",
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
			Header:        map[string]string{},
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
			Header:        map[string]string{},
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

func diff(t *testing.T, prefix string, have, want interface{}) {
	hv := reflect.NewValue(have).(*reflect.PtrValue).Elem().(*reflect.StructValue)
	wv := reflect.NewValue(want).(*reflect.PtrValue).Elem().(*reflect.StructValue)
	if hv.Type() != wv.Type() {
		t.Errorf("%s: type mismatch %v vs %v", prefix, hv.Type(), wv.Type())
	}
	for i := 0; i < hv.NumField(); i++ {
		hf := hv.Field(i).Interface()
		wf := wv.Field(i).Interface()
		if !reflect.DeepEqual(hf, wf) {
			t.Errorf("%s: %s = %v want %v", prefix, hv.Type().(*reflect.StructType).Field(i).Name, hf, wf)
		}
	}
}
