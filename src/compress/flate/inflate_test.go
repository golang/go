// Copyright 2014 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package flate

import (
	"bytes"
	"io"
	"testing"
)

func TestReset(t *testing.T) {
	ss := []string{
		"lorem ipsum izzle fo rizzle",
		"the quick brown fox jumped over",
	}

	deflated := make([]bytes.Buffer, 2)
	for i, s := range ss {
		w, _ := NewWriter(&deflated[i], 1)
		w.Write([]byte(s))
		w.Close()
	}

	inflated := make([]bytes.Buffer, 2)

	f := NewReader(&deflated[0])
	io.Copy(&inflated[0], f)
	f.(Resetter).Reset(&deflated[1], nil)
	io.Copy(&inflated[1], f)
	f.Close()

	for i, s := range ss {
		if s != inflated[i].String() {
			t.Errorf("inflated[%d]:\ngot  %q\nwant %q", i, inflated[i], s)
		}
	}
}

func TestResetDict(t *testing.T) {
	dict := []byte("the lorem fox")
	ss := []string{
		"lorem ipsum izzle fo rizzle",
		"the quick brown fox jumped over",
	}

	deflated := make([]bytes.Buffer, len(ss))
	for i, s := range ss {
		w, _ := NewWriterDict(&deflated[i], DefaultCompression, dict)
		w.Write([]byte(s))
		w.Close()
	}

	inflated := make([]bytes.Buffer, len(ss))

	f := NewReader(nil)
	for i := range inflated {
		f.(Resetter).Reset(&deflated[i], dict)
		io.Copy(&inflated[i], f)
	}
	f.Close()

	for i, s := range ss {
		if s != inflated[i].String() {
			t.Errorf("inflated[%d]:\ngot  %q\nwant %q", i, inflated[i], s)
		}
	}
}
