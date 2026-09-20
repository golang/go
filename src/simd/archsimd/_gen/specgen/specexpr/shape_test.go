// Copyright 2026 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package specexpr

import (
	"reflect"
	"testing"
)

func TestParseTypeName(t *testing.T) {
	tests := []struct {
		name string
		want Type
	}{
		// Basic types
		{"int", Basic{Base: "int", Bits: 0}},
		{"int8", Basic{Base: "int", Bits: 8}},
		{"int16", Basic{Base: "int", Bits: 16}},
		{"int32", Basic{Base: "int", Bits: 32}},
		{"int64", Basic{Base: "int", Bits: 64}},
		{"uint", Basic{Base: "uint", Bits: 0}},
		{"uint8", Basic{Base: "uint", Bits: 8}},
		{"uint16", Basic{Base: "uint", Bits: 16}},
		{"uint32", Basic{Base: "uint", Bits: 32}},
		{"uint64", Basic{Base: "uint", Bits: 64}},
		{"uintptr", Basic{Base: "uintptr", Bits: 0}},
		{"byte", Basic{Base: "uint", Bits: 8}},
		{"rune", Basic{Base: "int", Bits: 32}},
		{"float32", Basic{Base: "float", Bits: 32}},
		{"float64", Basic{Base: "float", Bits: 64}},
		{"complex64", Basic{Base: "complex", Bits: 64}},
		{"complex128", Basic{Base: "complex", Bits: 128}},
		{"bool", Basic{Base: "bool", Bits: 0}},
		{"string", Basic{Base: "string", Bits: 0}},

		// Fixed vectors (x<lanes>)
		{"Int8x16", Vector{Elem: Basic{Base: "int", Bits: 8}, Width: Int(128)}},
		{"Int16x8", Vector{Elem: Basic{Base: "int", Bits: 16}, Width: Int(128)}},
		{"Int32x4", Vector{Elem: Basic{Base: "int", Bits: 32}, Width: Int(128)}},
		{"Int64x2", Vector{Elem: Basic{Base: "int", Bits: 64}, Width: Int(128)}},
		{"Int32x8", Vector{Elem: Basic{Base: "int", Bits: 32}, Width: Int(256)}},
		{"Int32x16", Vector{Elem: Basic{Base: "int", Bits: 32}, Width: Int(512)}},
		{"Uint8x16", Vector{Elem: Basic{Base: "uint", Bits: 8}, Width: Int(128)}},
		{"Uint64x4", Vector{Elem: Basic{Base: "uint", Bits: 64}, Width: Int(256)}},
		{"Float32x4", Vector{Elem: Basic{Base: "float", Bits: 32}, Width: Int(128)}},
		{"Float64x2", Vector{Elem: Basic{Base: "float", Bits: 64}, Width: Int(128)}},
		{"Float64x4", Vector{Elem: Basic{Base: "float", Bits: 64}, Width: Int(256)}},
		{"Mask8x16", Vector{Elem: Basic{Base: "Mask", Bits: 8}, Width: Int(128)}},
		{"Mask16x16", Vector{Elem: Basic{Base: "Mask", Bits: 16}, Width: Int(256)}},
		{"Mask64x8", Vector{Elem: Basic{Base: "Mask", Bits: 64}, Width: Int(512)}},

		// Scalable vectors (s)
		{"Int8s", Vector{Elem: Basic{Base: "int", Bits: 8}, Width: VW()}},
		{"Int32s", Vector{Elem: Basic{Base: "int", Bits: 32}, Width: VW()}},
		{"Uint64s", Vector{Elem: Basic{Base: "uint", Bits: 64}, Width: VW()}},
		{"Float32s", Vector{Elem: Basic{Base: "float", Bits: 32}, Width: VW()}},
		{"Float64s", Vector{Elem: Basic{Base: "float", Bits: 64}, Width: VW()}},
		{"Mask8s", Vector{Elem: Basic{Base: "Mask", Bits: 8}, Width: VW()}},
		{"Mask32s", Vector{Elem: Basic{Base: "Mask", Bits: 32}, Width: VW()}},
	}

	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			got, err := ParseTypeName(tc.name)
			if err != nil {
				t.Fatalf("ParseTypeName(%q) unexpected error: %v", tc.name, err)
			}
			if !reflect.DeepEqual(got, tc.want) {
				t.Errorf("ParseTypeName(%q) = %+v; want %+v", tc.name, got, tc.want)
			}
		})
	}
}

func TestParseTypeNameErrors(t *testing.T) {
	bad := []string{
		"",
		"   ",
		"int48",
		"any",
		"Mask",
		"Mask8",
		"Int",
		"Int9999999999999999999999x1", // Overflow
		"Int1x9999999999999999999999",
		"Int32x-1",
		"Int32x0",
		"Int32x1",   // width = 32, invalid
		"Int32x3",   // width = 96, invalid
		"Int32w128", // w forms rejected
	}

	for _, s := range bad {
		t.Run(s, func(t *testing.T) {
			got, err := ParseTypeName(s)
			if err == nil {
				t.Errorf("ParseTypeName(%q) succeeded (%+v), want error", s, got)
			}
		})
	}
}
