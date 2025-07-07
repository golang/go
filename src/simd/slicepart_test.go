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
