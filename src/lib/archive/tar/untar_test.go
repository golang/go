// Copyright 2009 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package tar

import (
	"archive/tar";
	"bytes";
	"fmt";
	"io";
	"os";
	"testing";
)

func TestUntar(t *testing.T) {
	f, err := os.Open("testdata/test.tar", os.O_RDONLY, 0444);
	if err != nil {
		t.Fatalf("Unexpected error: %v", err);
	}
	defer f.Close();

	tr := NewReader(f);

	// First file
	hdr, err := tr.Next();
	if err != nil || hdr == nil {
		t.Fatalf("Didn't get first file: %v", err);
	}
	if hdr.Name != "small.txt" {
		t.Errorf(`hdr.Name = %q, want "small.txt"`, hdr.Name);
	}
	if hdr.Mode != 0640 {
		t.Errorf("hdr.Mode = %v, want 0640", hdr.Mode);
	}
	if hdr.Size != 5 {
		t.Errorf("hdr.Size = %v, want 5", hdr.Size);
	}

	// Read the first four bytes; Next() should skip the last one.
	buf := make([]byte, 4);
	if n, err := io.FullRead(tr, buf); err != nil {
		t.Fatalf("Unexpected error: %v", err);
	}
	if expected := io.StringBytes("Kilt"); !bytes.Equal(buf, expected) {
		t.Errorf("Contents = %v, want %v", buf, expected);
	}

	// Second file
	hdr, err = tr.Next();
	if err != nil {
		t.Fatalf("Didn't get second file: %v", err);
	}
	if hdr.Name != "small2.txt" {
		t.Errorf(`hdr.Name = %q, want "small2.txt"`, hdr.Name);
	}
	if hdr.Mode != 0640 {
		t.Errorf("hdr.Mode = %v, want 0640", hdr.Mode);
	}
	if hdr.Size != 11 {
		t.Errorf("hdr.Size = %v, want 11", hdr.Size);
	}


	hdr, err = tr.Next();
	if hdr != nil || err != nil {
		t.Fatalf("Unexpected third file or error: %v", err);
	}
}
