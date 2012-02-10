// Copyright 2010 The Go Authors.  All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package rand

import (
	"bytes"
	"compress/flate"
	"testing"
)

func TestRead(t *testing.T) {
	var n int = 4e6
	if testing.Short() {
		n = 1e5
	}
	b := make([]byte, n)
	n, err := Read(b)
	if n != len(b) || err != nil {
		t.Fatalf("Read(buf) = %d, %s", n, err)
	}

	var z bytes.Buffer
	f, _ := flate.NewWriter(&z, 5)
	f.Write(b)
	f.Close()
	if z.Len() < len(b)*99/100 {
		t.Fatalf("Compressed %d -> %d", len(b), z.Len())
	}
}
