// Copyright 2025 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

//go:build goexperiment.simd && (amd64 || arm64 || wasm)

package simd_test

import (
	"simd/archsimd"
	"testing"
)

func TestPartInt8x16(t *testing.T) {
	Do(t, 16, func(a, c []int8) {
		u, _ := archsimd.LoadInt8x16Part(a)
		u.Store(c)
	})
}

func TestPartUint8x16(t *testing.T) {
	a := []uint8{1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16}
	b := []uint8{1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16}
	for i := 16; i >= 0; i-- {
		u, _ := archsimd.LoadUint8x16Part(a[:i])
		c := make([]uint8, 32, 32)
		u.Store(c)
		checkSlices(t, c, b)
		if i > 0 {
			b[i-1] = 0
		}
	}
}

func TestPartInt16x8(t *testing.T) {
	a := []int16{1, 2, 3, 4, 5, 6, 7, 8}
	b := []int16{1, 2, 3, 4, 5, 6, 7, 8}
	for i := 8; i >= 0; i-- {
		u, _ := archsimd.LoadInt16x8Part(a[:i])
		c := make([]int16, 16, 16)
		u.Store(c)
		checkSlices(t, c, b)
		if i > 0 {
			b[i-1] = 0
		}
	}
}

func TestSlicesPartStoreInt8x16(t *testing.T) {
	a := []int8{1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16}
	b := []int8{1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16}
	for i := 16; i >= 0; i-- {
		v := archsimd.LoadInt8x16(a)
		c := make([]int8, 32, 32)
		v.StorePart(c[:i])
		checkSlices(t, c, b)
		if i > 0 {
			b[i-1] = 0
		}
	}
}

func TestSlicesPartStoreInt16x8(t *testing.T) {
	a := []int16{1, 2, 3, 4, 5, 6, 7, 8}
	b := []int16{1, 2, 3, 4, 5, 6, 7, 8}
	for i := 8; i >= 0; i-- {
		v := archsimd.LoadInt16x8(a)
		c := make([]int16, 32, 32)
		v.StorePart(c[:i])
		checkSlices(t, c, b)
		if i > 0 {
			b[i-1] = 0
		}
	}
}

func TestSlicesPartStoreUint8x16(t *testing.T) {
	a := []uint8{1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16}
	b := []uint8{1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16}
	for i := 16; i >= 0; i-- {
		v := archsimd.LoadUint8x16(a)
		c := make([]uint8, 32, 32)
		v.StorePart(c[:i])
		checkSlices(t, c, b)
		if i > 0 {
			b[i-1] = 0
		}
	}
}

func TestPartInt32(t *testing.T) {
	// 32x4
	L := 4
	c := []int32{1, 2, 3, 4, 5, -1, -1, -1, -1}
	a := c[:L+1]
	for i := range a {
		// Test the load first
		// e is a partial slice.
		e := a[i:]
		v, _ := archsimd.LoadInt32x4Part(e)
		// d contains what a ought to contain
		d := make([]int32, L)
		for j := 0; j < len(e) && j < len(d); j++ {
			d[j] = e[j]
		}

		b := make([]int32, L)
		v.Store(b)
		// test the load
		checkSlices(t, d, b)

		// Test the store
		f := make([]int32, L+1)
		for i := range f {
			f[i] = 99
		}

		v.StorePart(f[:len(e)])
		if len(e) < len(b) {
			checkSlices(t, f, b[:len(e)])
		} else {
			checkSlices(t, f, b)
		}
		for i := len(e); i < len(f); i++ {
			if f[i] != 99 {
				t.Errorf("StorePart altered f[%d], expected 99, saw %d", i, f[i])
			}
		}
	}
}

func TestPartFloat64(t *testing.T) {
	// 64x2
	L := 2
	c := []float64{1, 2, 3, 86, 86, 86, 86}
	a := c[:L+1]
	for i := range a {
		// Test the load first
		// e is a partial slice.
		e := a[i:]
		v, _ := archsimd.LoadFloat64x2Part(e)
		// d contains what a ought to contain
		d := make([]float64, L)
		for j := 0; j < len(e) && j < len(d); j++ {
			d[j] = e[j]
		}

		b := make([]float64, L)
		v.Store(b)
		// test the load
		checkSlices(t, d, b)

		// Test the store
		f := make([]float64, L+1)
		for i := range f {
			f[i] = 99
		}

		v.StorePart(f[:len(e)])
		if len(e) < len(b) {
			checkSlices(t, f, b[:len(e)])
		} else {
			checkSlices(t, f, b)
		}
		for i := len(e); i < len(f); i++ {
			if f[i] != 99 {
				t.Errorf("StorePart altered f[%d], expected 99, saw %v", i, f[i])
			}
		}
	}
}
