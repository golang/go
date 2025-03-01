// Copyright 2025 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package test

import (
	"internal/testenv"
	"testing"
	"unsafe"
)

// Stack allocation size for variable-sized allocations.
// Matches constant of the same name in ../walk/builtin.go:walkMakeSlice.
const maxStackSize = 32

//go:noinline
func genericUse[T any](s []T) {
	// Doesn't escape s.
}

func TestStackAllocation(t *testing.T) {
	testenv.SkipIfOptimizationOff(t)

	type testCase struct {
		f        func(int)
		elemSize uintptr
	}

	for _, tc := range []testCase{
		{
			f: func(n int) {
				genericUse(make([]int, n))
			},
			elemSize: unsafe.Sizeof(int(0)),
		},
	} {
		max := maxStackSize / int(tc.elemSize)
		if n := testing.AllocsPerRun(10, func() {
			tc.f(max)
		}); n != 0 {
			t.Fatalf("unexpected allocation: %f", n)
		}
		if n := testing.AllocsPerRun(10, func() {
			tc.f(max + 1)
		}); n != 1 {
			t.Fatalf("unexpected allocation: %f", n)
		}
	}
}
