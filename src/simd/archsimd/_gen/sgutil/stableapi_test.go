// Copyright 2026 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package sgutil

import "testing"

func TestNonSpecAPI(t *testing.T) {
	tests := []struct {
		pkg, recv, name string
		want            bool
	}{
		{"archsimd", "", "X86Features", true},
		{"archsimd", "", "X86", true},
		{"archsimd", "", "ARM64", true},
		{"archsimd", "X86Features", "AVX512", true},
		{"archsimd", "ARM64Features", "SVE", true},
		{"simd", "", "VectorBits", true},
		{"simd", "", "Emulated", true},
		{"simd", "Int8s", "ToArch", true},
		{"simd", "Mask64s", "ToArch", true},
		{"simd", "", "Float32sFromArch", true},

		{"simd", "Int8x16", "ToArch", false}, // not a portable type
		{"archsimd", "Int8s", "ToArch", false},
		{"simd", "", "Int8x16FromArch", false},
		{"simd", "", "FromArch", false},
		{"simd", "Int8s", "FromArch", false},
		{"simd", "", "X86", false},
		{"archsimd", "", "VectorBits", false},
		{"archsimd", "Int32x8", "Merge", false}, // deprecated, not GA
		{"simd", "", "VectorBitSize", false},    // deprecated, not GA
		{"archsimd", "", "ClearAVXUpperBits", false},
	}
	for _, tt := range tests {
		if got := NonSpecAPI(tt.pkg, tt.recv, tt.name); got != tt.want {
			t.Errorf("NonSpecAPI(%q, %q, %q) = %v, want %v", tt.pkg, tt.recv, tt.name, got, tt.want)
		}
	}
}
