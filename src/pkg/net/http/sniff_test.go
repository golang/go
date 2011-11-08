// Copyright 2011 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package http_test

import (
	"bytes"
	"io/ioutil"
	"log"
	. "net/http"
	"net/http/httptest"
	"strconv"
	"testing"
)

var sniffTests = []struct {
	desc        string
	data        []byte
	contentType string
}{
	// Some nonsense.
	{"Empty", []byte{}, "text/plain; charset=utf-8"},
	{"Binary", []byte{1, 2, 3}, "application/octet-stream"},

	{"HTML document #1", []byte(`<HtMl><bOdY>blah blah blah</body></html>`), "text/html; charset=utf-8"},
	{"HTML document #2", []byte(`<HTML></HTML>`), "text/html; charset=utf-8"},
	{"HTML document #3 (leading whitespace)", []byte(`   <!DOCTYPE HTML>...`), "text/html; charset=utf-8"},
	{"HTML document #4 (leading CRLF)", []byte("\r\n<html>..."), "text/html; charset=utf-8"},

	{"Plain text", []byte(`This is not HTML. It has ☃ though.`), "text/plain; charset=utf-8"},

	{"XML", []byte("\n<?xml!"), "text/xml; charset=utf-8"},

	// Image types.
	{"GIF 87a", []byte(`GIF87a`), "image/gif"},
	{"GIF 89a", []byte(`GIF89a...`), "image/gif"},

	// TODO(dsymonds): Re-enable this when the spec is sorted w.r.t. MP4.
	//{"MP4 video", []byte("\x00\x00\x00\x18ftypmp42\x00\x00\x00\x00mp42isom<\x06t\xbfmdat"), "video/mp4"},
	//{"MP4 audio", []byte("\x00\x00\x00\x20ftypM4A \x00\x00\x00\x00M4A mp42isom\x00\x00\x00\x00"), "audio/mp4"},
}

func TestDetectContentType(t *testing.T) {
	for _, tt := range sniffTests {
		ct := DetectContentType(tt.data)
		if ct != tt.contentType {
			t.Errorf("%v: DetectContentType = %q, want %q", tt.desc, ct, tt.contentType)
		}
	}
}

func TestServerContentType(t *testing.T) {
	ts := httptest.NewServer(HandlerFunc(func(w ResponseWriter, r *Request) {
		i, _ := strconv.Atoi(r.FormValue("i"))
		tt := sniffTests[i]
		n, err := w.Write(tt.data)
		if n != len(tt.data) || err != nil {
			log.Fatalf("%v: Write(%q) = %v, %v want %d, nil", tt.desc, tt.data, n, err, len(tt.data))
		}
	}))
	defer ts.Close()

	for i, tt := range sniffTests {
		resp, err := Get(ts.URL + "/?i=" + strconv.Itoa(i))
		if err != nil {
			t.Errorf("%v: %v", tt.desc, err)
			continue
		}
		if ct := resp.Header.Get("Content-Type"); ct != tt.contentType {
			t.Errorf("%v: Content-Type = %q, want %q", tt.desc, ct, tt.contentType)
		}
		data, err := ioutil.ReadAll(resp.Body)
		if err != nil {
			t.Errorf("%v: reading body: %v", tt.desc, err)
		} else if !bytes.Equal(data, tt.data) {
			t.Errorf("%v: data is %q, want %q", tt.desc, data, tt.data)
		}
		resp.Body.Close()
	}
}
