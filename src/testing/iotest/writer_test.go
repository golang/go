// Copyright 2019 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package iotest

import (
	"strings"
	"testing"
)

var truncateWriterTests = []struct {
	in    string
	want  string
	trunc int64
	n     int
}{
	{"hello", "", -1, 5},
	{"world", "", 0, 5},
	{"abcde", "abc", 3, 5},
	{"edcba", "edcba", 7, 5},
}

func TestTruncateWriter(t *testing.T) {
	for _, tt := range truncateWriterTests {
		buf := new(strings.Builder)
		tw := TruncateWriter(buf, tt.trunc)
		n, err := tw.Write([]byte(tt.in))
		if err != nil {
			t.Errorf("Unexpected error %v for\n\t%+v", err, tt)
		}
		if g, w := buf.String(), tt.want; g != w {
			t.Errorf("got %q, expected %q", g, w)
		}
		if g, w := n, tt.n; g != w {
			t.Errorf("read %d bytes, but expected to have read %d bytes for\n\t%+v", g, w, tt)
		}
	}
}
