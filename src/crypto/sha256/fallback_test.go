// Copyright 2016 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// +build s390x

package sha256

import (
	"fmt"
	"io"
	"testing"
)

// Tests the fallback code path in case the optimized asm
// implementation cannot be used.
// See also TestBlockGeneric.
func TestGenericPath(t *testing.T) {
	if useAsm == false {
		t.Skipf("assembly implementation unavailable")
	}
	useAsm = false
	defer func() { useAsm = true }()
	c := New()
	in := "ΑΒΓΔΕϜΖΗΘΙΚΛΜΝΞΟΠϺϘΡΣΤΥΦΧΨΩ"
	gold := "e93d84ec2b22383123be9f713697fb25" +
		"338c86e2f7d8d1ddc2d89d332dd9d76c"
	if _, err := io.WriteString(c, in); err != nil {
		t.Fatalf("could not write to c: %v", err)
	}
	out := fmt.Sprintf("%x", c.Sum(nil))
	if out != gold {
		t.Fatalf("mismatch: got %s, wanted %s", out, gold)
	}
}
