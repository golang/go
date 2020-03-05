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
	w := dummyWriter()
	(&Reloc{}).Write(w)
	off := w.off
	if sz := uint32(RelocSize); off != sz {
		t.Errorf("size mismatch: %d bytes written, but size=%d", off, sz)
	}
}
