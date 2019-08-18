// Copyright 2009 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package iotest

import (
	"bytes"
	"testing"
)

var truncateWriterTests = []struct {
	in  string
	out string
	twn int64
	n   int
}{
	{"hello", "", -1, 5},
	{"hello", "", 0, 5},
	{"hello", "hel", 3, 5},
	{"hello", "hello", 7, 5},
}

func TestTruncateWriter(t *testing.T) {
	for _, tt := range truncateWriterTests {
		rb := new(bytes.Buffer)
		trb := TruncateWriter(rb, tt.twn)
		n, err := trb.Write([]byte(tt.in))
		if err != nil {
			t.Error(err)
		}
		if rb.String() != tt.out {
			t.Errorf("expected %s, got %s", tt.out, rb.String())
		}
		if n != tt.n {
			t.Errorf("expected to have read %d bytes, but read %d", tt.n, n)
		}
	}
}
