// Copyright 2026 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package profile

import (
	"bytes"
	"testing"
)

func TestParseDanglingLocationID(t *testing.T) {
	// Profile containing a sample referencing non-existent location ID 99.
	raw := []byte{
		0x32, 0x00, 0x32, 0x07, 0x73, 0x61, 0x6d, 0x70, 0x6c, 0x65, 0x73, 0x32, 0x05, 0x63, 0x6f,
		0x75, 0x6e, 0x74, 0x0a, 0x04, 0x08, 0x01, 0x10, 0x02, 0x12, 0x05, 0x0a, 0x01, 0x63, 0x10, 0x01,
	}
	if _, err := Parse(bytes.NewReader(raw)); err == nil {
		t.Fatal("Parse succeeded, want error for dangling location ID")
	}
}
