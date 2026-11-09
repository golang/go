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

// LoadInt8x32Part loads n=min(len(s), 32) elements from slice s as a
// vector and returns the vector and n. If len(s) < 32, the
// remaining vector elements will be zero.
func LoadInt8x32Part(s []int8) (z Int8x32, n int) {
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

// LoadInt16x16Part loads n=min(len(s), 16) elements from slice s as a
// vector and returns the vector and n. If len(s) < 16, the
// remaining vector elements will be zero.
func LoadInt16x16Part(s []int16) (z Int16x16, n int) {
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

// StorePart stores n=min(len(s), 32) elements of x into s and returns n.
func (x Int8x32) StorePart(s []int8) (n int) {
	l := len(s)
	if l >= 32 {
		x.Store(s)
		return 32
	}
	if l == 0 {
		return 0
	}
	if l > 16 {
		x.GetLo().Store(s)
		x.GetHi().StorePart(s[16:])
	} else { // fits in one
		x.GetLo().StorePart(s)
	}
	return l
}

// StorePart stores n=min(len(s), 16) elements of x into s and returns n.
func (x Int16x16) StorePart(s []int16) (n int) {
	l := len(s)
	if l >= 16 {
		x.Store(s)
		return 16
	}
	if l == 0 {
		return 0
	}
	if l > 8 {
		x.GetLo().Store(s)
		x.GetHi().StorePart(s[8:])
	} else { // fits in one
		x.GetLo().StorePart(s)
	}
	return l
}
