// Copyright 2026 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package fips140

import "testing"

// This test checks that we can compile and link specific
// code patterns in a FIPS package, where there are restrictions
// on what relocations are allowed to use.

type testType struct {
	A, B, C string
}

func TestCompile(t *testing.T) {
	var a []testType
	a = append(a, testType{
		A: "",
		B: "",
		C: "",
	})
}
