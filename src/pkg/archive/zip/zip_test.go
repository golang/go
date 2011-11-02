// Copyright 2011 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// Tests that involve both reading and writing.

package zip

import (
	"bytes"
	"fmt"
	"io"
	"testing"
)

type stringReaderAt string

func (s stringReaderAt) ReadAt(p []byte, off int64) (n int, err error) {
	if off >= int64(len(s)) {
		return 0, io.EOF
	}
	n = copy(p, s[off:])
	return
}

func TestOver65kFiles(t *testing.T) {
	if testing.Short() {
		t.Logf("slow test; skipping")
		return
	}
	buf := new(bytes.Buffer)
	w := NewWriter(buf)
	const nFiles = (1 << 16) + 42
	for i := 0; i < nFiles; i++ {
		_, err := w.Create(fmt.Sprintf("%d.dat", i))
		if err != nil {
			t.Fatalf("creating file %d: %v", i, err)
		}
	}
	if err := w.Close(); err != nil {
		t.Fatalf("Writer.Close: %v", err)
	}
	rat := stringReaderAt(buf.String())
	zr, err := NewReader(rat, int64(len(rat)))
	if err != nil {
		t.Fatalf("NewReader: %v", err)
	}
	if got := len(zr.File); got != nFiles {
		t.Fatalf("File contains %d files, want %d", got, nFiles)
	}
	for i := 0; i < nFiles; i++ {
		want := fmt.Sprintf("%d.dat", i)
		if zr.File[i].Name != want {
			t.Fatalf("File(%d) = %q, want %q", i, zr.File[i].Name, want)
		}
	}
}
