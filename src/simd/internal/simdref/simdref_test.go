// Copyright 2026 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package simdref_test

import (
	"fmt"
	"simd/internal/simdref"
	"simd/internal/spec"
	"testing"
)

func TestScalableExecution(t *testing.T) {
	// Calling scalable operation before setting width must panic
	func() {
		defer func() {
			if r := recover(); r == nil {
				t.Fatalf("expected panic on LoadInt8s when scalable width unset, but did not panic")
			}
		}()
		sUnset := make([]int8, 16)
		simdref.LoadInt8s(sUnset)
	}()

	// Test at two different scalable widths
	for _, width := range []int{128, 256} {
		t.Run(fmt.Sprint(width), func(t *testing.T) {
			s1 := make([]int8, width/8)
			s2 := make([]int8, width/8)
			for i := range s1 {
				s1[i] = int8(i)
				s2[i] = 10
			}
			out := make([]int8, width/8)

			defer spec.SetScalableWidth(width)()
			x := simdref.LoadInt8s(s1)
			y := simdref.LoadInt8s(s2)
			z := x.Add(y)
			z.Store(out)
			for i := range out {
				if out[i] != int8(i)+10 {
					t.Fatalf("out[%d] = %d, want %d", i, out[i], int8(i)+10)
				}
			}
		})
	}
}
