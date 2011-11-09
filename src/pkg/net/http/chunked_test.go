// Copyright 2011 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// This code is duplicated in httputil/chunked_test.go.
// Please make any changes in both files.

package http

import (
	"bytes"
	"io/ioutil"
	"testing"
)

func TestChunk(t *testing.T) {
	var b bytes.Buffer

	w := newChunkedWriter(&b)
	const chunk1 = "hello, "
	const chunk2 = "world! 0123456789abcdef"
	w.Write([]byte(chunk1))
	w.Write([]byte(chunk2))
	w.Close()

	if g, e := b.String(), "7\r\nhello, \r\n17\r\nworld! 0123456789abcdef\r\n0\r\n"; g != e {
		t.Fatalf("chunk writer wrote %q; want %q", g, e)
	}

	r := newChunkedReader(&b)
	data, err := ioutil.ReadAll(r)
	if err != nil {
		t.Logf(`data: "%s"`, data)
		t.Fatalf("ReadAll from reader: %v", err)
	}
	if g, e := string(data), chunk1+chunk2; g != e {
		t.Errorf("chunk reader read %q; want %q", g, e)
	}
}
