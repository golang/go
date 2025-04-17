// Copyright 2022 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package slicewriter

import (
	"io"
	"testing"
)

func TestSliceWriter(t *testing.T) {

	sleq := func(t *testing.T, got []byte, want []byte) {
		t.Helper()
		if len(got) != len(want) {
			t.Fatalf("bad length got %d want %d", len(got), len(want))
		}
		for i := range got {
			if got[i] != want[i] {
				t.Fatalf("bad read at %d got %d want %d", i, got[i], want[i])
			}
		}
	}

	wf := func(t *testing.T, ws *WriteSeeker, p []byte) {
		t.Helper()
		nw, werr := ws.Write(p)
		if werr != nil {
			t.Fatalf("unexpected write error: %v", werr)
		}
		if nw != len(p) {
			t.Fatalf("wrong amount written want %d got %d", len(p), nw)
		}
	}

	rf := func(t *testing.T, ws *WriteSeeker, p []byte) {
		t.Helper()
		b := make([]byte, len(p))
		nr, rerr := ws.Read(b)
		if rerr != nil {
			t.Fatalf("unexpected read error: %v", rerr)
		}
		if nr != len(p) {
			t.Fatalf("wrong amount read want %d got %d", len(p), nr)
		}
		sleq(t, b, p)
	}

	sk := func(t *testing.T, ws *WriteSeeker, offset int64, whence int) int64 {
		t.Helper()
		off, err := ws.Seek(offset, whence)
		if err != nil {
			t.Fatalf("unexpected seek error: %v", err)
		}
		return off
	}

	wp1 := []byte{1, 2}
	ws := &WriteSeeker{}

	// write some stuff
	wf(t, ws, wp1)
	// check that BytesWritten returns what we wrote.
	sleq(t, ws.BytesWritten(), wp1)
	// offset is at end of slice, so reading should return zero bytes.
	rf(t, ws, []byte{})

	// write some more stuff
	wp2 := []byte{7, 8, 9}
	wf(t, ws, wp2)
	// check that BytesWritten returns what we expect.
	wpex := []byte{1, 2, 7, 8, 9}
	sleq(t, ws.BytesWritten(), wpex)
	rf(t, ws, []byte{})

	// seeks and reads.
	sk(t, ws, 1, io.SeekStart)
	rf(t, ws, []byte{2, 7})
	sk(t, ws, -2, io.SeekCurrent)
	rf(t, ws, []byte{2, 7})
	sk(t, ws, -4, io.SeekEnd)
	rf(t, ws, []byte{2, 7})
	off := sk(t, ws, 0, io.SeekEnd)
	sk(t, ws, off, io.SeekStart)

	// seek back and overwrite
	sk(t, ws, 1, io.SeekStart)
	wf(t, ws, []byte{9, 11})
	wpex = []byte{1, 9, 11, 8, 9}
	sleq(t, ws.BytesWritten(), wpex)

	// seeks on empty writer.
	ws2 := &WriteSeeker{}
	sk(t, ws2, 0, io.SeekStart)
	sk(t, ws2, 0, io.SeekCurrent)
	sk(t, ws2, 0, io.SeekEnd)

	// check for seek errors.
	_, err := ws.Seek(-1, io.SeekStart)
	if err == nil {
		t.Fatalf("expected error on invalid -1 seek")
	}
	_, err = ws.Seek(int64(len(ws.BytesWritten())+1), io.SeekStart)
	if err == nil {
		t.Fatalf("expected error on invalid %d seek", len(ws.BytesWritten()))
	}

	ws.Seek(0, io.SeekStart)
	_, err = ws.Seek(-1, io.SeekCurrent)
	if err == nil {
		t.Fatalf("expected error on invalid -1 seek")
	}
	_, err = ws.Seek(int64(len(ws.BytesWritten())+1), io.SeekCurrent)
	if err == nil {
		t.Fatalf("expected error on invalid %d seek", len(ws.BytesWritten()))
	}

	_, err = ws.Seek(1, io.SeekEnd)
	if err == nil {
		t.Fatalf("expected error on invalid 1 seek")
	}
	bsamt := int64(-1*len(ws.BytesWritten()) - 1)
	_, err = ws.Seek(bsamt, io.SeekEnd)
	if err == nil {
		t.Fatalf("expected error on invalid %d seek", bsamt)
	}

	// bad seek mode
	_, err = ws.Seek(-1, io.SeekStart+9)
	if err == nil {
		t.Fatalf("expected error on invalid seek mode")
	}
}
