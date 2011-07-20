// Copyright 2011 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package http

import (
	"testing"
)

var sniffTests = []struct {
	desc string
	data []byte
	exp  string
}{
	// Some nonsense.
	{"Empty", []byte{}, "text/plain; charset=utf-8"},
	{"Binary", []byte{1, 2, 3}, "application/octet-stream"},

	{"HTML document #1", []byte(`<HtMl><bOdY>blah blah blah</body></html>`), "text/html; charset=utf-8"},
	{"HTML document #2", []byte(`<HTML></HTML>`), "text/html; charset=utf-8"},
	{"HTML document #3 (leading whitespace)", []byte(`   <!DOCTYPE HTML>...`), "text/html; charset=utf-8"},

	{"Plain text", []byte(`This is not HTML. It has ☃ though.`), "text/plain; charset=utf-8"},

	{"XML", []byte("\n<?xml!"), "text/xml; charset=utf-8"},

	// Image types.
	{"GIF 87a", []byte(`GIF87a`), "image/gif"},
	{"GIF 89a", []byte(`GIF89a...`), "image/gif"},
}

func TestSniffing(t *testing.T) {
	for _, st := range sniffTests {
		got := DetectContentType(st.data)
		if got != st.exp {
			t.Errorf("%v: sniffed as %v, want %v", st.desc, got, st.exp)
		}
	}
}
