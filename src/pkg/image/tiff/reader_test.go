// Copyright 2011 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package tiff

import (
	"os"
	"testing"
)

// TestNoRPS tries to decode an image that has no RowsPerStrip tag.
// The tag is mandatory according to the spec but some software omits
// it in the case of a single strip.
func TestNoRPS(t *testing.T) {
	f, err := os.Open("testdata/no_rps.tiff")
	if err != nil {
		t.Fatal(err)
	}
	defer f.Close()
	_, err = Decode(f)
	if err != nil {
		t.Fatal(err)
	}
}
