// Copyright 2009 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// This test tests some internals of the flate package.
// The tests in package compress/gzip serve as the
// end-to-end test of the decompressor.

package flate

import (
	"bytes"
	"testing"
)

func TestUncompressedSource(t *testing.T) {
	decoder := NewReader(bytes.NewBuffer([]byte{0x01, 0x01, 0x00, 0xfe, 0xff, 0x11}))
	output := make([]byte, 1)
	n, error := decoder.Read(output)
	if n != 1 || error != nil {
		t.Fatalf("decoder.Read() = %d, %v, want 1, nil", n, error)
	}
	if output[0] != 0x11 {
		t.Errorf("output[0] = %x, want 0x11", output[0])
	}
}
