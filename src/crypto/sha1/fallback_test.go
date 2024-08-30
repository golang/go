// Copyright 2016 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

//go:build s390x && !purego

package sha1

import (
	"fmt"
	"io"
	"testing"
)

// Tests the fallback code path in case the optimized asm
// implementation cannot be used.
// See also TestBlockGeneric.
func TestGenericPath(t *testing.T) {
	if !useAsm {
		t.Skipf("assembly implementation unavailable")
	}
	useAsm = false
	defer func() { useAsm = true }()
	c := New()
	in := "ΑΒΓΔΕϜΖΗΘΙΚΛΜΝΞΟΠϺϘΡΣΤΥΦΧΨΩ"
	gold := "0f58c2bb130f8182375f325c18342215255387e5"
	if _, err := io.WriteString(c, in); err != nil {
		t.Fatalf("could not write to c: %v", err)
	}
	out := fmt.Sprintf("%x", c.Sum(nil))
	if out != gold {
		t.Fatalf("mismatch: got %s, wanted %s", out, gold)
	}
}
