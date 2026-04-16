// Copyright 2026 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package fips140

type TestType struct {
	A, B, C string
}

type TestType2 struct {
	A, B, C, D, E string
}

//go:noinline
func DynamicString() string { return "dyn" }

// This code is in the fips140 package, so it is compiled in
// FIPS mode. In the external test compile_test.go, wantLiterals
// is the same code. Keep them in sync. See TestCompile.

//go:noinline
func Literals() (a []TestType, b []TestType2) {
	a = append(a, TestType{
		A: "a",
		B: "",
		C: "",
	})
	a = append(a, TestType{
		A: "a",
		B: DynamicString(),
		C: "",
	})
	b = append(b, TestType2{
		A: "a",
		B: "",
		C: "",
		D: "",
		E: "",
	})
	b = append(b, TestType2{
		A: "a",
		B: DynamicString(),
		C: "",
		D: "d",
		E: DynamicString(),
	})
	return a, b
}
