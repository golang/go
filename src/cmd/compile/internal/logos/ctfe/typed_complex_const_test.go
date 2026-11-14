// Copyright 2026 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package ctfe

import "testing"

func TestTypedComplexConstantsRoundOncePerOperation(t *testing.T) {
	const (
		mul128 = complex128(1+0x1p-52+1i) * complex128(1-0x1p-52+1i)
		div128 = complex128(1+1i) / complex128(3+0i)
		sub128 = complex128(9007199254740992+1i) - complex128(9007199254740991+0.5i)
		mul64  = complex64(1+0x1p-23+1i) * complex64(1-0x1p-23+1i)
	)
	for _, tt := range []struct {
		name string
		typ  string
		expr string
		want complex128
	}{
		{"mul128", "complex128", "complex128(1+0x1p-52+1i) * complex128(1-0x1p-52+1i)", mul128},
		{"div128", "complex128", "complex128(1+1i) / complex128(3+0i)", div128},
		{"sub128", "complex128", "complex128(9007199254740992+1i) - complex128(9007199254740991+0.5i)", sub128},
		{"mul64", "complex64", "complex64(1+0x1p-23+1i) * complex64(1-0x1p-23+1i)", complex128(mul64)},
	} {
		t.Run(tt.name, func(t *testing.T) {
			v, _, err := evalConstFunc(t, Config{}, constSpec{
				src:    "package p\nfunc F() " + tt.typ + " { return " + tt.expr + " }",
				rtypes: []*Type{BasicType(tt.typ)},
				name:   "F",
				call:   "F()",
			})
			if err != nil {
				t.Fatal(err)
			}
			if got := v.Complex(); got != tt.want {
				t.Fatalf("%s = %v, want %v", tt.expr, got, tt.want)
			}
		})
	}
}

func TestComplexIncrementAndDecrementStepTheRealPart(t *testing.T) {
	for _, tt := range []struct {
		name string
		typ  string
		step string
		want complex128
	}{
		{"inc128", "complex128", "x++", 2 + 2i},
		{"dec128", "complex128", "x--", 0 + 2i},
		{"inc64", "complex64", "x++", 2 + 2i},
		{"dec64", "complex64", "x--", 0 + 2i},
	} {
		t.Run(tt.name, func(t *testing.T) {
			v, _, err := evalConstFunc(t, Config{}, constSpec{
				src:    "package p\nfunc F() " + tt.typ + " { x := " + tt.typ + "(1+2i); " + tt.step + "; return x }",
				rtypes: []*Type{BasicType(tt.typ)},
				name:   "F",
				call:   "F()",
			})
			if err != nil {
				t.Fatal(err)
			}
			if got := v.Complex(); got != tt.want {
				t.Fatalf("%s on 1+2i = %v, want %v", tt.step, got, tt.want)
			}
		})
	}
}
