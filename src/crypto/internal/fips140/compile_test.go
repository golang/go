// Copyright 2026 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package fips140_test

import (
	"crypto/internal/fips140"
	"slices"
	"testing"
)

// This test checks that we can compile and link specific
// code patterns in a FIPS package, where there are restrictions
// on what relocations are allowed to use.
// Also checks that the code inside and outside of FIPS mode
// produce same result. The code in fips140.Literals and
// wantLiterals are the same. Keep them in sync.

func wantLiterals() (a []fips140.TestType, b []fips140.TestType2) {
	a = append(a, fips140.TestType{
		A: "a",
		B: "",
		C: "",
	})
	a = append(a, fips140.TestType{
		A: "a",
		B: fips140.DynamicString(),
		C: "",
	})
	b = append(b, fips140.TestType2{
		A: "a",
		B: "",
		C: "",
		D: "",
		E: "",
	})
	b = append(b, fips140.TestType2{
		A: "a",
		B: fips140.DynamicString(),
		C: "",
		D: "d",
		E: fips140.DynamicString(),
	})
	return a, b
}

func TestCompile(t *testing.T) {
	wantA, wantB := wantLiterals()
	gotA, gotB := fips140.Literals()
	if !slices.Equal(gotA, wantA) {
		t.Errorf("FIPS and non-FIPS mode produce different results: want %q, got %q", wantA, gotA)
	}
	if !slices.Equal(gotB, wantB) {
		t.Errorf("FIPS and non-FIPS mode produce different results: want %q, got %q", wantB, gotB)
	}
}
