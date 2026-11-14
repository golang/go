// Copyright 2026 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package ctfe

import (
	"strings"
	"testing"
)

func TestConstFloatOverflowReports(t *testing.T) {
	cfg := Config{MaxSteps: 100000, MaxValueBytes: 1 << 20, WordSize: 8}
	cases := []struct {
		name string
		typ  string
		src  string
	}{
		{"Float64", "float64", "package p\nfunc OverflowFloat64() float64 { return float64(1e400) }"},
		{"Float32", "float32", "package p\nfunc OverflowFloat32() float32 { return float32(1e400) }"},
	}
	for _, tc := range cases {
		_, _, err := evalConstFunc(t, cfg, constSpec{
			src:    tc.src,
			rtypes: []*Type{BasicType(tc.typ)},
			name:   tc.name,
			call:   tc.name + `()`,
		})
		if err == nil {
			t.Fatalf("%s: Eval succeeded, want const_eval overflow error", tc.name)
		}
		cerr, ok := err.(*Error)
		if !ok {
			t.Fatalf("%s: want *Error, got %T (%v)", tc.name, err, err)
		}
		if cerr.Code != CodeEval {
			t.Fatalf("%s: code = %d, want %d", tc.name, cerr.Code, CodeEval)
		}
		if !strings.Contains(cerr.Detail, tc.typ) {
			t.Fatalf("%s: detail = %q, want [correctness::const_eval] overflow naming %s", tc.name, cerr.Detail, tc.typ)
		}
	}
}

func TestConstStringConversionRange(t *testing.T) {
	cfg := Config{MaxSteps: 100000, MaxValueBytes: 1 << 20, WordSize: 8}
	cases := []struct {
		name string
		expr string
		want string
	}{
		{"Ascii", `string(65)`, "A"},
		{"Huge", `string(1<<64 + 65)`, "\uFFFD"},
		{"HugeBits", `string(1 << 100)`, "\uFFFD"},
		{"Negative", `string(-1)`, "\uFFFD"},
		{"HugeNegative", `string(-(1 << 100))`, "\uFFFD"},
		{"SurrogateLow", `string(0xD800)`, "\uFFFD"},
		{"SurrogateHigh", `string(0xDFFF)`, "\uFFFD"},
		{"MaxRune", `string(0x10FFFF)`, "\U0010FFFF"},
		{"AboveMaxRune", `string(0x110000)`, "\uFFFD"},
	}
	for _, tc := range cases {
		src := "package p\nfunc Str" + tc.name + "() string { return " + tc.expr + " }"
		v, _, err := evalConstFunc(t, cfg, constSpec{
			src:    src,
			rtypes: []*Type{BasicType("string")},
			name:   "Str" + tc.name,
			call:   "Str" + tc.name + `()`,
		})
		if err != nil {
			t.Fatalf("%s: %v", tc.name, err)
		}
		if v.Str() != tc.want {
			t.Fatalf("%s = %q, want %q", tc.name, v.Str(), tc.want)
		}
	}
}

func TestConstComplexOverflowReports(t *testing.T) {
	cfg := Config{MaxSteps: 100000, MaxValueBytes: 1 << 20, WordSize: 8}
	cases := []struct {
		name string
		typ  string
		want string
		src  string
	}{
		{"CxFloat128", "complex128", "complex128", "package p\nfunc CxFloat128() complex128 { return complex(1e400, 0) }"},
		{"CxFloat64", "complex64", "complex64", "package p\nfunc CxFloat64() complex64 { return complex64(complex(1e400, 0)) }"},
		{"CxMul128", "complex128", "complex128", "package p\nfunc CxMul128() complex128 { return complex(1e308, 1e308) * complex(1e308, 0) }"},
		{"CxMul64", "complex64", "complex64", "package p\nfunc CxMul64() complex64 { return complex64(complex(1e38, 0) * complex(1e38, 0)) }"},
	}
	for _, tc := range cases {
		_, _, err := evalConstFunc(t, cfg, constSpec{
			src:    tc.src,
			rtypes: []*Type{BasicType(tc.typ)},
			name:   tc.name,
			call:   tc.name + `()`,
		})
		if err == nil {
			t.Fatalf("%s: Eval succeeded, want const_eval overflow error", tc.name)
		}
		cerr, ok := err.(*Error)
		if !ok {
			t.Fatalf("%s: want *Error, got %T (%v)", tc.name, err, err)
		}
		if cerr.Code != CodeEval {
			t.Fatalf("%s: code = %d, want %d", tc.name, cerr.Code, CodeEval)
		}
		if !strings.Contains(cerr.Detail, tc.want) {
			t.Fatalf("%s: detail = %q, want [correctness::const_eval] overflow naming %s", tc.name, cerr.Detail, tc.want)
		}
	}
}
