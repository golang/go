// Copyright 2025 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

//go:build goexperiment.simd && amd64

package simd_test

import (
	"simd/archsimd"
	"testing"
)

func TestPartInt8x32(t *testing.T) {
	a := []int8{1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16,
		17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32}
	b := []int8{1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16,
		17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32}
	for i := 32; i >= 0; i-- {
		u, _ := archsimd.LoadInt8x32Part(a[:i])
		c := make([]int8, 32, 32)
		u.Store(c)
		checkSlices(t, c, b)
		if i > 0 {
			b[i-1] = 0
		}
	}
}

func xTestPartUint8x16(t *testing.T) {
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

func TestPartUint8x32(t *testing.T) {
	a := []uint8{1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16,
		17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32}
	b := []uint8{1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16,
		17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32}
	for i := 32; i >= 0; i-- {
		u, _ := archsimd.LoadUint8x32Part(a[:i])
		c := make([]uint8, 32, 32)
		u.Store(c)
		checkSlices(t, c, b)
		if i > 0 {
			b[i-1] = 0
		}
	}
}

func xTestPartInt16x8(t *testing.T) {
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

func TestPartInt16x16(t *testing.T) {
	a := []int16{1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16}
	b := []int16{1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16}
	for i := 16; i >= 0; i-- {
		u, _ := archsimd.LoadInt16x16Part(a[:i])
		c := make([]int16, 16, 16)
		u.Store(c)
		checkSlices(t, c, b)
		if i > 0 {
			b[i-1] = 0
		}
	}
}

func xTestSlicesPartStoreInt8x16(t *testing.T) {
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

func xTestSlicesPartStoreInt16x8(t *testing.T) {
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

func TestSlicesPartStoreInt16x16(t *testing.T) {
	a := []int16{1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16}
	b := []int16{1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16}
	for i := 16; i >= 0; i-- {
		v := archsimd.LoadInt16x16(a)
		c := make([]int16, 32, 32)
		v.StorePart(c[:i])
		checkSlices(t, c, b)
		if i > 0 {
			b[i-1] = 0
		}
	}
}

func xTestSlicesPartStoreUint8x16(t *testing.T) {
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

func TestSlicesPartStoreUint16x16(t *testing.T) {
	a := []uint16{1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16}
	b := []uint16{1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16}
	for i := 16; i >= 0; i-- {
		v := archsimd.LoadUint16x16(a)
		c := make([]uint16, 32, 32)
		v.StorePart(c[:i])
		checkSlices(t, c, b)
		if i > 0 {
			b[i-1] = 0
		}
	}
}

func TestSlicesPartStoreUint8x32(t *testing.T) {
	a := []uint8{1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16,
		17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32}
	b := []uint8{1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16,
		17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32}
	for i := 32; i >= 0; i-- {
		v := archsimd.LoadUint8x32(a)
		c := make([]uint8, 32, 32)
		v.StorePart(c[:i])
		checkSlices(t, c, b)
		if i > 0 {
			b[i-1] = 0
		}
	}
}

func TestPartUint64(t *testing.T) {
	// 64x4
	L := 4
	c := []uint64{1, 2, 3, 4, 5, 86, 86, 86, 86}
	a := c[:L+1]
	for i := range a {
		// Test the load first
		// e is a partial slice.
		e := a[i:]
		v, _ := archsimd.LoadUint64x4Part(e)
		// d contains what a ought to contain
		d := make([]uint64, L)
		for j := 0; j < len(e) && j < len(d); j++ {
			d[j] = e[j]
		}

		b := make([]uint64, L)
		v.Store(b)
		// test the load
		checkSlices(t, d, b)

		// Test the store
		f := make([]uint64, L+1)
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

func TestPartFloat32(t *testing.T) {
	// 32x8
	L := 8
	c := []float32{1, 2, 3, 4, 5, 6, 7, 8, 86, 86, 86, 86}
	a := c[:L+1]
	for i := range a {
		// Test the load first
		// e is a partial slice.
		e := a[i:]
		v, _ := archsimd.LoadFloat32x8Part(e)
		// d contains what a ought to contain
		d := make([]float32, L)
		for j := 0; j < len(e) && j < len(d); j++ {
			d[j] = e[j]
		}

		b := make([]float32, L)
		v.Store(b)
		// test the load
		checkSlices(t, d, b)

		// Test the store
		f := make([]float32, L+1)
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

// 512-bit load

func TestPartInt64(t *testing.T) {
	if !archsimd.X86.AVX512() {
		t.Skip("Test requires X86.AVX512, not available on this hardware")
		return
	}

	L := 8
	c := []int64{1, 2, 3, 4, 5, 6, 7, 8, 86, 86, 86, 86}
	a := c[:L+1]
	for i := range a {
		// Test the load first
		// e is a partial slice.
		e := a[i:]
		v, _ := archsimd.LoadInt64x8Part(e)
		// d contains what a ought to contain
		d := make([]int64, L)
		for j := 0; j < len(e) && j < len(d); j++ {
			d[j] = e[j]
		}

		b := make([]int64, L)
		v.Store(b)
		// test the load
		checkSlicesLogInput(t, b, d, 0.0, func() { t.Helper(); t.Logf("Len(e)=%d", len(e)) })

		// Test the store
		f := make([]int64, L+1)
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
