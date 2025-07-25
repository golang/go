// Copyright 2025 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

//go:build goexperiment.simd && amd64

package simd_test

import (
	"simd"
	"testing"
)

func TestSlicePartInt8x16(t *testing.T) {
	Do(t, 16, func(a, c []int8) {
		u := simd.LoadInt8x16SlicePart(a)
		u.StoreSlice(c)
	})
}

func TestSlicePartInt8x32(t *testing.T) {
	a := []int8{1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16,
		17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32}
	b := []int8{1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16,
		17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32}
	for i := 32; i >= 0; i-- {
		u := simd.LoadInt8x32SlicePart(a[:i])
		c := make([]int8, 32, 32)
		u.StoreSlice(c)
		checkSlices(t, c, b)
		if i > 0 {
			b[i-1] = 0
		}
	}
}

func TestSlicePartUint8x16(t *testing.T) {
	a := []uint8{1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16}
	b := []uint8{1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16}
	for i := 16; i >= 0; i-- {
		u := simd.LoadUint8x16SlicePart(a[:i])
		c := make([]uint8, 32, 32)
		u.StoreSlice(c)
		checkSlices(t, c, b)
		if i > 0 {
			b[i-1] = 0
		}
	}
}

func TestSlicePartUint8x32(t *testing.T) {
	a := []uint8{1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16,
		17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32}
	b := []uint8{1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16,
		17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32}
	for i := 32; i >= 0; i-- {
		u := simd.LoadUint8x32SlicePart(a[:i])
		c := make([]uint8, 32, 32)
		u.StoreSlice(c)
		checkSlices(t, c, b)
		if i > 0 {
			b[i-1] = 0
		}
	}
}

func TestSlicePartInt16x8(t *testing.T) {
	a := []int16{1, 2, 3, 4, 5, 6, 7, 8}
	b := []int16{1, 2, 3, 4, 5, 6, 7, 8}
	for i := 8; i >= 0; i-- {
		u := simd.LoadInt16x8SlicePart(a[:i])
		c := make([]int16, 16, 16)
		u.StoreSlice(c)
		checkSlices(t, c, b)
		if i > 0 {
			b[i-1] = 0
		}
	}
}

func TestSlicePartInt16x16(t *testing.T) {
	a := []int16{1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16}
	b := []int16{1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16}
	for i := 16; i >= 0; i-- {
		u := simd.LoadInt16x16SlicePart(a[:i])
		c := make([]int16, 16, 16)
		u.StoreSlice(c)
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
		v := simd.LoadInt8x16Slice(a)
		c := make([]int8, 32, 32)
		v.StoreSlicePart(c[:i])
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
		v := simd.LoadInt16x8Slice(a)
		c := make([]int16, 32, 32)
		v.StoreSlicePart(c[:i])
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
		v := simd.LoadInt16x16Slice(a)
		c := make([]int16, 32, 32)
		v.StoreSlicePart(c[:i])
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
		v := simd.LoadUint8x16Slice(a)
		c := make([]uint8, 32, 32)
		v.StoreSlicePart(c[:i])
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
		v := simd.LoadUint16x16Slice(a)
		c := make([]uint16, 32, 32)
		v.StoreSlicePart(c[:i])
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
		v := simd.LoadUint8x32Slice(a)
		c := make([]uint8, 32, 32)
		v.StoreSlicePart(c[:i])
		checkSlices(t, c, b)
		if i > 0 {
			b[i-1] = 0
		}
	}
}

func TestSlicePartInt32(t *testing.T) {
	// 32x4
	L := 4
	c := []int32{1, 2, 3, 4, 5, -1, -1, -1, -1}
	a := c[:L+1]
	for i := range a {
		// Test the load first
		// e is a partial slice.
		e := a[i:]
		v := simd.LoadInt32x4SlicePart(e)
		// d contains what a ought to contain
		d := make([]int32, L)
		for j := 0; j < len(e) && j < len(d); j++ {
			d[j] = e[j]
		}

		b := make([]int32, L)
		v.StoreSlice(b)
		// test the load
		checkSlices(t, d, b)

		// Test the store
		f := make([]int32, L+1)
		for i := range f {
			f[i] = 99
		}

		v.StoreSlicePart(f[:len(e)])
		if len(e) < len(b) {
			checkSlices(t, f, b[:len(e)])
		} else {
			checkSlices(t, f, b)
		}
		for i := len(e); i < len(f); i++ {
			if f[i] != 99 {
				t.Errorf("StoreSlicePart altered f[%d], expected 99, saw %d", i, f[i])
			}
		}
	}
}

func TestSlicePartUint64(t *testing.T) {
	// 64x4
	L := 4
	c := []uint64{1, 2, 3, 4, 5, 86, 86, 86, 86}
	a := c[:L+1]
	for i := range a {
		// Test the load first
		// e is a partial slice.
		e := a[i:]
		v := simd.LoadUint64x4SlicePart(e)
		// d contains what a ought to contain
		d := make([]uint64, L)
		for j := 0; j < len(e) && j < len(d); j++ {
			d[j] = e[j]
		}

		b := make([]uint64, L)
		v.StoreSlice(b)
		// test the load
		checkSlices(t, d, b)

		// Test the store
		f := make([]uint64, L+1)
		for i := range f {
			f[i] = 99
		}

		v.StoreSlicePart(f[:len(e)])
		if len(e) < len(b) {
			checkSlices(t, f, b[:len(e)])
		} else {
			checkSlices(t, f, b)
		}
		for i := len(e); i < len(f); i++ {
			if f[i] != 99 {
				t.Errorf("StoreSlicePart altered f[%d], expected 99, saw %d", i, f[i])
			}
		}
	}
}

func TestSlicePartFloat64(t *testing.T) {
	// 64x2
	L := 2
	c := []float64{1, 2, 3, 86, 86, 86, 86}
	a := c[:L+1]
	for i := range a {
		// Test the load first
		// e is a partial slice.
		e := a[i:]
		v := simd.LoadFloat64x2SlicePart(e)
		// d contains what a ought to contain
		d := make([]float64, L)
		for j := 0; j < len(e) && j < len(d); j++ {
			d[j] = e[j]
		}

		b := make([]float64, L)
		v.StoreSlice(b)
		// test the load
		checkSlices(t, d, b)

		// Test the store
		f := make([]float64, L+1)
		for i := range f {
			f[i] = 99
		}

		v.StoreSlicePart(f[:len(e)])
		if len(e) < len(b) {
			checkSlices(t, f, b[:len(e)])
		} else {
			checkSlices(t, f, b)
		}
		for i := len(e); i < len(f); i++ {
			if f[i] != 99 {
				t.Errorf("StoreSlicePart altered f[%d], expected 99, saw %v", i, f[i])
			}
		}
	}
}

func TestSlicePartFloat32(t *testing.T) {
	// 32x8
	L := 8
	c := []float32{1, 2, 3, 4, 5, 6, 7, 8, 86, 86, 86, 86}
	a := c[:L+1]
	for i := range a {
		// Test the load first
		// e is a partial slice.
		e := a[i:]
		v := simd.LoadFloat32x8SlicePart(e)
		// d contains what a ought to contain
		d := make([]float32, L)
		for j := 0; j < len(e) && j < len(d); j++ {
			d[j] = e[j]
		}

		b := make([]float32, L)
		v.StoreSlice(b)
		// test the load
		checkSlices(t, d, b)

		// Test the store
		f := make([]float32, L+1)
		for i := range f {
			f[i] = 99
		}

		v.StoreSlicePart(f[:len(e)])
		if len(e) < len(b) {
			checkSlices(t, f, b[:len(e)])
		} else {
			checkSlices(t, f, b)
		}
		for i := len(e); i < len(f); i++ {
			if f[i] != 99 {
				t.Errorf("StoreSlicePart altered f[%d], expected 99, saw %v", i, f[i])
			}
		}
	}
}

// 512-bit load

func TestSlicePartInt64(t *testing.T) {
	if !simd.HasAVX512() {
		t.Skip("Test requires HasAVX512, not available on this hardware")
		return
	}

	L := 8
	c := []int64{1, 2, 3, 4, 5, 6, 7, 8, 86, 86, 86, 86}
	a := c[:L+1]
	for i := range a {
		// Test the load first
		// e is a partial slice.
		e := a[i:]
		v := simd.LoadInt64x8SlicePart(e)
		// d contains what a ought to contain
		d := make([]int64, L)
		for j := 0; j < len(e) && j < len(d); j++ {
			d[j] = e[j]
		}

		b := make([]int64, L)
		v.StoreSlice(b)
		// test the load
		checkSlicesLogInput(t, b, d, 0.0, func() { t.Helper(); t.Logf("Len(e)=%d", len(e)) })

		// Test the store
		f := make([]int64, L+1)
		for i := range f {
			f[i] = 99
		}

		v.StoreSlicePart(f[:len(e)])
		if len(e) < len(b) {
			checkSlices(t, f, b[:len(e)])
		} else {
			checkSlices(t, f, b)
		}
		for i := len(e); i < len(f); i++ {
			if f[i] != 99 {
				t.Errorf("StoreSlicePart altered f[%d], expected 99, saw %v", i, f[i])
			}
		}
	}
}
