// Copyright 2026 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package main

import (
	"simd/archsimd/_gen/simdgen/types"
	"slices"
	"testing"
)

func TestCombineCPUFeatures(t *testing.T) {
	cases := []struct {
		a, b []string
		want []string
	}{
		{[]string{"AVX"}, []string{"AVX512"}, []string{"AVX512"}},
		{[]string{"AVX2"}, []string{"AVX512"}, []string{"AVX512"}},
		{[]string{"AVX512"}, []string{"AVX512"}, []string{"AVX512"}},
		{[]string{"AVX512VBMI"}, []string{"AVX512"}, []string{"AVX512VBMI"}},
		{[]string{"AVX512"}, []string{"AVX512VBMI"}, []string{"AVX512VBMI"}},
		{[]string{"FMA"}, []string{"AVX512"}, []string{"FMA", "AVX512"}},
		{[]string{"FMA", "AVX"}, []string{"AVX512"}, []string{"FMA", "AVX512"}},
		{nil, []string{"AVX512"}, []string{"AVX512"}},
	}
	for _, tc := range cases {
		if got := combineCPUFeatures("amd64", tc.a, tc.b); !slices.Equal(got, tc.want) {
			t.Errorf("combineCPUFeatures(%v, %v) = %v, want %v", tc.a, tc.b, got, tc.want)
		}
	}
}

func TestAddOperandCPUFeatures(t *testing.T) {
	vreg := func(bits, goBits int) types.Operand {
		return types.Operand{
			Class:  "vreg",
			Bits:   types.VectorSize{NRaw: bits},
			GoBits: &goBits,
		}
	}
	ops := []Operation{
		{
			rawOperation: types.RawOperation{
				GOARCH:     "amd64",
				CPUFeature: "AVX",
				Out:        []types.Operand{vreg(128, 256)},
			},
			In:         []types.Operand{vreg(128, 128)},
			CPUFeature: "AVX",
		},
		{
			rawOperation: types.RawOperation{
				GOARCH:     "amd64",
				CPUFeature: "AVX",
				Out:        []types.Operand{vreg(256, 512)},
			},
			In:         []types.Operand{vreg(256, 256)},
			CPUFeature: "AVX",
		},
	}
	addOperandCPUFeatures(ops)

	if got := ops[0].CPUFeature; got != "AVX" {
		t.Errorf("128->256 CPUFeature = %q, want AVX", got)
	}
	if got := ops[1].CPUFeature; got != "AVX512" {
		t.Errorf("256->512 CPUFeature = %q, want AVX512", got)
	}
	if got := ops[1].rawOperation.CPUFeature; got != "AVX" {
		t.Errorf("256->512 rawOperation.CPUFeature = %q, want AVX", got)
	}
}
