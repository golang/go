// Copyright 2026 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

//go:build goexperiment.simd

package testdata_test

import (
	"simd"
	"testing"
)

func TestLoadPartCount(t *testing.T) {
	tests := []struct {
		name string
		load func() int
		want int
	}{
		{
			"Int64s",
			func() int {
				_, n := simd.LoadInt64sPart(make([]int64, simd.Int64s{}.Len()+1))
				return n
			},
			simd.Int64s{}.Len(),
		},
		{
			"Uint64s",
			func() int {
				_, n := simd.LoadUint64sPart(make([]uint64, simd.Uint64s{}.Len()+1))
				return n
			},
			simd.Uint64s{}.Len(),
		},
		{
			"Float64s",
			func() int {
				_, n := simd.LoadFloat64sPart(make([]float64, simd.Float64s{}.Len()+1))
				return n
			},
			simd.Float64s{}.Len(),
		},
	}

	for _, test := range tests {
		t.Run(test.name, func(t *testing.T) {
			if got := test.load(); got != test.want {
				t.Errorf("Load%sPart returned count %d, want %d", test.name, got, test.want)
			}
		})
	}
}
