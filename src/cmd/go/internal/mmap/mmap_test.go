// Copyright 2025 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package mmap

import (
	"bytes"
	"testing"
)

// TestMmap does a round trip to make sure the slice returned by
// mmap contains the same data as was written to the file. It's
// a test on one of the issues in #71059: on Windows we were
// returning a slice containing all the data in the mmapped pages,
// which could be longer than the file.
func TestMmap(t *testing.T) {
	// Use an already existing file as our test data. Avoid creating
	// a temporary file so that we don't have to close the mapping on
	// Windows before deleting the file during test cleanup.
	f := "testdata/small_file.txt"

	want := []byte("This file is shorter than 4096 bytes.\n")

	data, _, err := Mmap(f)
	if err != nil {
		t.Fatalf("calling Mmap: %v", err)
	}
	if !bytes.Equal(data.Data, want) {
		t.Fatalf("mmapped data slice: got %q; want %q", data.Data, want)
	}
}
