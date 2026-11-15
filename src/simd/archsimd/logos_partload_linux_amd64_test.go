// Copyright 2026 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

//go:build goexperiment.simd && linux && amd64

package archsimd_test

import (
	"simd/archsimd"
	"testing"
)

func TestLoadIntPartPageEnd(t *testing.T) {
	t.Run("Uint8x16", func(t *testing.T) {
		for n := 0; n <= 16; n++ {
			s, release := pageEndSlice[uint8](t, n)
			for i := range s {
				s[i] = uint8(i + 1)
			}
			v, got := archsimd.LoadUint8x16Part(s)
			if got != n {
				t.Fatalf("n=%d: count %d, want %d", n, got, n)
			}
			for i := range 16 {
				want := uint8(0)
				if i < n {
					want = uint8(i + 1)
				}
				if v.GetElem(uint8(i)) != want {
					t.Errorf("n=%d: lane %d is %v, want %v", n, i, v.GetElem(uint8(i)), want)
				}
			}
			release()
		}
	})
	t.Run("Uint16x8", func(t *testing.T) {
		for n := 0; n <= 8; n++ {
			s, release := pageEndSlice[uint16](t, n)
			for i := range s {
				s[i] = uint16(i + 1)
			}
			v, got := archsimd.LoadUint16x8Part(s)
			if got != n {
				t.Fatalf("n=%d: count %d, want %d", n, got, n)
			}
			for i := range 8 {
				want := uint16(0)
				if i < n {
					want = uint16(i + 1)
				}
				if v.GetElem(uint8(i)) != want {
					t.Errorf("n=%d: lane %d is %v, want %v", n, i, v.GetElem(uint8(i)), want)
				}
			}
			release()
		}
	})
	t.Run("Int8x16", func(t *testing.T) {
		for n := 0; n <= 16; n++ {
			s, release := pageEndSlice[int8](t, n)
			for i := range s {
				s[i] = int8(i + 1)
			}
			v, got := archsimd.LoadInt8x16Part(s)
			if got != n {
				t.Fatalf("n=%d: count %d, want %d", n, got, n)
			}
			for i := range 16 {
				want := int8(0)
				if i < n {
					want = int8(i + 1)
				}
				if v.GetElem(uint8(i)) != want {
					t.Errorf("n=%d: lane %d is %v, want %v", n, i, v.GetElem(uint8(i)), want)
				}
			}
			release()
		}
	})
	t.Run("Int16x8", func(t *testing.T) {
		for n := 0; n <= 8; n++ {
			s, release := pageEndSlice[int16](t, n)
			for i := range s {
				s[i] = int16(i + 1)
			}
			v, got := archsimd.LoadInt16x8Part(s)
			if got != n {
				t.Fatalf("n=%d: count %d, want %d", n, got, n)
			}
			for i := range 8 {
				want := int16(0)
				if i < n {
					want = int16(i + 1)
				}
				if v.GetElem(uint8(i)) != want {
					t.Errorf("n=%d: lane %d is %v, want %v", n, i, v.GetElem(uint8(i)), want)
				}
			}
			release()
		}
	})
}
