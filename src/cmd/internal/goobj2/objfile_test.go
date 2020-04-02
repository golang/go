// Copyright 2020 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package goobj2

import (
	"bufio"
	"bytes"
	"cmd/internal/bio"
	"testing"
)

func dummyWriter() *Writer {
	var buf bytes.Buffer
	wr := &bio.Writer{Writer: bufio.NewWriter(&buf)} // hacky: no file, so cannot seek
	return NewWriter(wr)
}

func TestSize(t *testing.T) {
	// This test checks that hard-coded sizes match the actual sizes
	// in the object file format.
	tests := []struct {
		x    interface{ Write(*Writer) }
		want uint32
	}{
		{&Reloc{}, RelocSize},
		{&Aux{}, AuxSize},
	}
	w := dummyWriter()
	for _, test := range tests {
		off0 := w.off
		test.x.Write(w)
		got := w.off - off0
		if got != test.want {
			t.Errorf("size(%T) mismatch: %d bytes written, but size=%d", test.x, got, test.want)
		}
	}
}
