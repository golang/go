// Copyright 2012 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package curve25519

import (
	"fmt"
	"testing"
)

const expectedHex = "89161fde887b2b53de549af483940106ecc114d6982daa98256de23bdf77661a"

func TestBaseScalarMult(t *testing.T) {
	var a, b [32]byte
	in := &a
	out := &b
	a[0] = 1

	for i := 0; i < 200; i++ {
		ScalarBaseMult(out, in)
		in, out = out, in
	}

	result := fmt.Sprintf("%x", in[:])
	if result != expectedHex {
		t.Errorf("incorrect result: got %s, want %s", result, expectedHex)
	}
}

func BenchmarkScalarBaseMult(b *testing.B) {
	var in, out [32]byte
	in[0] = 1

	b.SetBytes(32)
	for i := 0; i < b.N; i++ {
		ScalarBaseMult(&out, &in)
	}
}
