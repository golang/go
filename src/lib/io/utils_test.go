// Copyright 2009 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package io

import (
	"io";
	"os";
	"testing";
)

func checkSize(t *testing.T, path string, size uint64) {
	dir, err := os.Stat(path);
	if err != nil {
		t.Fatalf("Stat %q (looking for size %d): %s", path, size, err);
	}
	if dir.Size != size {
		t.Errorf("Stat %q: size %d want %d", path, dir.Size, size);
	}
}

func TestReadFile(t *testing.T) {
	filename := "rumpelstilzchen";
	contents, err := ReadFile(filename);
	if err == nil {
		t.Fatalf("ReadFile %s: error expected, none found", filename);
	}

	filename = "utils_test.go";
	contents, err = ReadFile(filename);
	if err != nil {
		t.Fatalf("ReadFile %s: %v", filename, err);
	}

	checkSize(t, filename, uint64(len(contents)));
}
