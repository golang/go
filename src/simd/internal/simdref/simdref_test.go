// Copyright 2026 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package simdref_test

import (
	"fmt"
	"math/bits"
	"simd/internal/simdref"
	"simd/internal/spec"
	"slices"
	"testing"
)

func TestScalableMaskToBits(t *testing.T) {
	for _, width := range []int{128, 256, 512, 1024, 2048} {
		t.Run(fmt.Sprint(width), func(t *testing.T) {
			defer spec.SetScalableWidth(width)()
			s := make([]int8, width/8)
			for i := range s {
				if i%3 == 0 || i == len(s)-1 {
					s[i] = -1
				}
			}
			mask := simdref.LoadInt8s(s).ToMask()
			n := (len(s) + bits.UintSize - 1) / bits.UintSize
			dst := make([]uint, n+1)
			for i := range dst {
				dst[i] = ^uint(0)
			}
			if got := mask.ToBits(dst); got != n {
				t.Fatalf("ToBits wrote %d words, want %d", got, n)
			}
			for i := 0; i < n*bits.UintSize; i++ {
				want := i < len(s) && s[i] != 0
				if got := dst[i/bits.UintSize]&(1<<(i%bits.UintSize)) != 0; got != want {
					t.Errorf("bit %d = %v, want %v", i, got, want)
				}
			}
			if dst[n] != ^uint(0) {
				t.Error("modified suffix")
			}
			before := slices.Clone(dst)
			func() {
				defer func() {
					if recover() == nil {
						t.Error("short destination did not panic")
					}
				}()
				// The capacity is sufficient, but the length is not.
				mask.ToBits(dst[:n-1])
			}()
			if !slices.Equal(dst, before) {
				t.Error("short destination modified the bitmap")
			}
		})
	}
}

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
