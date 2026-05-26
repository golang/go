// Copyright 2025 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

//go:build goexperiment.simd && amd64

package archsimd

// Implementation of all the {Int,Uint}{8,16} load and store slice part
// functions and methods for 256-bit vectors.

/* These two masks are used by generated code */

var vecMask64 = [16]int64{
	-1, -1, -1, -1,
	-1, -1, -1, -1,
	0, 0, 0, 0,
	0, 0, 0, 0,
}

var vecMask32 = [32]int32{
	-1, -1, -1, -1,
	-1, -1, -1, -1,
	-1, -1, -1, -1,
	-1, -1, -1, -1,
	0, 0, 0, 0,
	0, 0, 0, 0,
	0, 0, 0, 0,
	0, 0, 0, 0,
}

/* 256-bit int vector loads and stores made from 128-bit parts */

// LoadInt8x32Part loads a Int8x32 from the slice s.
// If s has fewer than 32 elements, the remaining elements of the vector are filled with zeroes.
// If s has 32 or more elements, the function is equivalent to LoadInt8x32Slice.
func LoadInt8x32Part(s []int8) (Int8x32, int) {
	l := len(s)
	if l >= 32 {
		return LoadInt8x32(s), 32
	}
	var x Int8x32
	if l == 0 {
		return x, 0
	}
	if l > 16 {
		v, _ := LoadInt8x16Part(s[16:])
		return x.SetLo(LoadInt8x16(s)).SetHi(v), l
	} else {
		v, _ := LoadInt8x16Part(s)
		return x.SetLo(v), l
	}
}

// LoadInt16x16Part loads a Int16x16 from the slice s.
// If s has fewer than 16 elements, the remaining elements of the vector are filled with zeroes.
// If s has 16 or more elements, the function is equivalent to LoadInt16x16Slice.
func LoadInt16x16Part(s []int16) (Int16x16, int) {
	l := len(s)
	if l >= 16 {
		return LoadInt16x16(s), 16
	}
	var x Int16x16
	if l == 0 {
		return x, 0
	}
	if l > 8 {
		v, _ := LoadInt16x8Part(s[8:])
		return x.SetLo(LoadInt16x8(s)).SetHi(v), l
	} else {
		v, _ := LoadInt16x8Part(s)
		return x.SetLo(v), l
	}
}

// StorePart stores the elements of x into the slice s.
// It stores as many elements as will fit in s.
// If s has 32 or more elements, the method is equivalent to x.StoreSlice.
func (x Int8x32) StorePart(s []int8) {
	l := len(s)
	if l >= 32 {
		x.Store(s)
		return
	}
	if l == 0 {
		return
	}
	if l > 16 {
		x.GetLo().Store(s)
		x.GetHi().StorePart(s[16:])
	} else { // fits in one
		x.GetLo().StorePart(s)
	}
}

// StorePart stores the elements of x into the slice s.
// It stores as many elements as will fit in s.
// If s has 16 or more elements, the method is equivalent to x.StoreSlice.
func (x Int16x16) StorePart(s []int16) {
	l := len(s)
	if l >= 16 {
		x.Store(s)
		return
	}
	if l == 0 {
		return
	}
	if l > 8 {
		x.GetLo().Store(s)
		x.GetHi().StorePart(s[8:])
	} else { // fits in one
		x.GetLo().StorePart(s)
	}
}
