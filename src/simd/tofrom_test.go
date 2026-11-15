// Copyright 2026 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

//go:build goexperiment.simd

package simd_test

import (
	"internal/testenv"
	"simd"
	"testing"
)

type namedToArch struct {
	name string
	val  any
}

func allZeroToArch() []namedToArch {
	return []namedToArch{
		{"Int8s", (simd.Int8s{}).ToArch()},
		{"Int16s", (simd.Int16s{}).ToArch()},
		{"Int32s", (simd.Int32s{}).ToArch()},
		{"Int64s", (simd.Int64s{}).ToArch()},
		{"Uint8s", (simd.Uint8s{}).ToArch()},
		{"Uint16s", (simd.Uint16s{}).ToArch()},
		{"Uint32s", (simd.Uint32s{}).ToArch()},
		{"Uint64s", (simd.Uint64s{}).ToArch()},
		{"Float32s", (simd.Float32s{}).ToArch()},
		{"Float64s", (simd.Float64s{}).ToArch()},
		{"Mask8s", (simd.Mask8s{}).ToArch()},
		{"Mask16s", (simd.Mask16s{}).ToArch()},
		{"Mask32s", (simd.Mask32s{}).ToArch()},
		{"Mask64s", (simd.Mask64s{}).ToArch()},
	}
}

func TestToArchEmulated(t *testing.T) {
	if !simd.Emulated() {
		for _, tc := range allZeroToArch() {
			if tc.val == nil {
				t.Errorf("(%s{}).ToArch() = nil when Emulated() is false", tc.name)
			}
		}

		// Self-invoke in emulated mode.
		cmd := testenv.Command(t, testenv.Executable(t), "-test.run=^TestToArchEmulated$")
		cmd.Env = append(cmd.Environ(), "GODEBUG=simd=0")
		if out, err := cmd.CombinedOutput(); err != nil {
			t.Fatalf("GODEBUG=simd=0 subprocess failed: %v\n%s", err, out)
		}
		return
	}

	for _, tc := range allZeroToArch() {
		if tc.val != nil {
			t.Errorf("(%s{}).ToArch() = %T(%v), want nil when Emulated() is true", tc.name, tc.val, tc.val)
		}
	}
}
