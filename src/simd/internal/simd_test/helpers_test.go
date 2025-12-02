// Copyright 2025 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

//go:build goexperiment.simd && amd64

package simd_test

import (
	"math"
	"simd/internal/test_helpers"
	"testing"
)

type signed interface {
	~int | ~int8 | ~int16 | ~int32 | ~int64
}

type integer interface {
	~int | ~int8 | ~int16 | ~int32 | ~int64 | ~uint | ~uint8 | ~uint16 | ~uint32 | ~uint64 | ~uintptr
}

type float interface {
	~float32 | ~float64
}

type number interface {
	~int | ~int8 | ~int16 | ~int32 | ~int64 | ~uint | ~uint8 | ~uint16 | ~uint32 | ~uint64 | ~uintptr | ~float32 | ~float64
}

func checkSlices[T number](t *testing.T, got, want []T) bool {
	t.Helper()
	return test_helpers.CheckSlicesLogInput[T](t, got, want, 0.0, nil)
}

func checkSlicesLogInput[T number](t *testing.T, got, want []T, flakiness float64, logInput func()) bool {
	t.Helper()
	return test_helpers.CheckSlicesLogInput[T](t, got, want, flakiness, logInput)
}

// sliceOf returns a slice n T's, with each
// element of the slice initialized to its
// index + 1.
func sliceOf[T number](n int) []T {
	s := make([]T, n)
	for i := 0; i < n; i++ {
		s[i] = T(i + 1)
	}
	return s
}

func toVect[T signed](b []bool) []T {
	s := make([]T, len(b))
	for i := range b {
		if b[i] {
			s[i] = -1
		}
	}
	return s
}

// s64 converts a slice of some integer type into a slice of int64
func s64[T number](s []T) []int64 {
	var is any = s
	if r, ok := is.([]int64); ok {
		return r
	}
	r := make([]int64, len(s))
	for i := range s {
		r[i] = int64(s[i])
	}
	return r
}

// Do implements slice part testing.  It repeatedly calls
// body on smaller and smaller slices and an output slice
// for the result, then compares the result to its own
// calculation of what the result should be.
func Do[T number](t *testing.T, n int, body func(a, c []T)) {
	a := sliceOf[T](n)
	b := sliceOf[T](n)

	for i := n; i >= 0; i-- {
		c := make([]T, n, n)
		body(a[:i], c)
		checkSlices(t, c, b)
		if i > 0 {
			b[i-1] = T(0)
		}
	}
}

// map3 returns a function that returns the slice of the results of applying
// input parameter elem to the respective elements of its 3 slice inputs.
func map3[T, U any](elem func(x, y, z T) U) func(x, y, z []T) []U {
	return func(x, y, z []T) []U {
		s := make([]U, len(x))
		for i := range s {
			s[i] = elem(x[i], y[i], z[i])
		}
		return s
	}
}

// map2 returns a function that returns the slice of the results of applying
// input parameter elem to the respective elements of its 2 slice inputs.
func map2[T, U any](elem func(x, y T) U) func(x, y []T) []U {
	return func(x, y []T) []U {
		s := make([]U, len(x))
		for i := range s {
			s[i] = elem(x[i], y[i])
		}
		return s
	}
}

// map1 returns a function that returns the slice of the results of applying
// input parameter elem to the respective elements of its single slice input.
func map1[T, U any](elem func(x T) U) func(x []T) []U {
	return func(x []T) []U {
		s := make([]U, len(x))
		for i := range s {
			s[i] = elem(x[i])
		}
		return s
	}
}

// map1 returns a function that returns the slice of the results of applying
// comparison function elem to the respective elements of its two slice inputs.
func mapCompare[T number](elem func(x, y T) bool) func(x, y []T) []int64 {
	return func(x, y []T) []int64 {
		s := make([]int64, len(x))
		for i := range s {
			if elem(x[i], y[i]) {
				s[i] = -1
			}
		}
		return s
	}
}

// nOf returns a slice of length n whose elements are taken
// from input slice s.
func nOf[T any](n int, s []T) []T {
	if len(s) >= n {
		return s
	}
	r := make([]T, n)
	for i := range r {
		r[i] = s[i%len(s)]
	}
	return r
}

const (
	PN22  = 1.0 / 1024 / 1024 / 4
	PN24  = 1.0 / 1024 / 1024 / 16
	PN53  = PN24 * PN24 / 32
	F0    = float32(1.0 + 513*PN22/2)
	F1    = float32(1.0 + 511*PN22*8)
	Aeasy = float32(2046 * PN53)
	Ahard = float32(2047 * PN53) // 2047 provokes a 2-rounding in 64-bit FMA rounded to 32-bit
)

var zero = 0.0
var nzero = -zero
var inf = 1 / zero
var ninf = -1 / zero
var nan = math.NaN()

// N controls how large the test vectors are
const N = 144

var float32s = nOf(N, []float32{float32(inf), float32(ninf), 1, float32(nan), float32(zero), 2, float32(nan), float32(zero), 3, float32(-zero), float32(1.0 / zero), float32(-1.0 / zero), 1.0 / 2, 1.0 / 4, 1.0 / 8, 1.0 / 1000, 1.0 / 1000000, 1, -1, 0, 2, -2, 3, -3, math.MaxFloat32, 1 / math.MaxFloat32, 10, -10, 100, 20, -20, 300, -300, -4000, -80, -160, -3200, -64, -4, -8, -16, -32, -64})
var float64s = nOf(N, []float64{inf, ninf, nan, zero, -zero, 1 / zero, -1 / zero, 0.0001, 0.0000001, 1, -1, 0, 2, -2, 3, -3, math.MaxFloat64, 1.0 / math.MaxFloat64, 10, -10, 100, 20, -20, 300, -300, -4000, -80, -16, -32, -64})

var int32s = nOf(N, []int32{1, -1, 0, 2, 4, 8, 1024, 0xffffff, -0xffffff, 0x55555, 0x77777, 0xccccc, -0x55555, -0x77777, -0xccccc, -4, -8, -16, -32, -64})
var uint32s = nOf(N, []uint32{1, 0, 2, 4, 8, 1024, 0xffffff, ^uint32(0xffffff), 0x55555, 0x77777, 0xccccc, ^uint32(0x55555), ^uint32(0x77777), ^uint32(0xccccc)})

var int64s = nOf(N, []int64{1, -1, 0, 2, 4, 8, 1024, 0xffffff, -0xffffff, 0x55555, 0x77777, 0xccccc, -0x55555, -0x77777, -0xccccc, -4, -8, -16, -32, -64})
var uint64s = nOf(N, []uint64{1, 0, 2, 4, 8, 1024, 0xffffff, ^uint64(0xffffff), 0x55555, 0x77777, 0xccccc, ^uint64(0x55555), ^uint64(0x77777), ^uint64(0xccccc)})

var int16s = nOf(N, []int16{1, -1, 0, 2, 4, 8, 1024, 3, 5, 7, 11, 13, 3000, 5555, 7777, 11111, 32767, 32766, -32767, -32768, -11111, -4, -8, -16, -32, -64})
var uint16s = nOf(N, []uint16{1, 0, 2, 4, 8, 1024, 3, 5, 7, 11, 13, 3000, 5555, 7777, 11111, 32767, 32766, 32768, 65535, 45678, 56789})

var int8s = nOf(N, []int8{0, 1, 2, 3, 5, 7, 11, 22, 33, 55, 77, 121, 127, -1, -2, -3, -5, -7, -11, -77, -121, -127, -128, 4, 8, 16, 32, 64, -4, -8, -16, -32, -64})
var uint8s = nOf(N, []uint8{0, 1, 2, 3, 5, 7, 11, 22, 33, 55, 77, 121, 127, 128, 255, 233, 211, 177, 144, 4, 8, 16, 32, 64})

var bools = nOf(N, []bool{
	true, false, true, true, false, false, true, true, true, false, false, false, true, true, true, true, false, false, false, false})

func forSlice[T number](t *testing.T, s []T, n int, f func(a []T) bool) {
	t.Helper()
	for i := 0; i < len(s)-n; i++ {
		if !f(s[i : i+n]) {
			return
		}
	}
}

func forSlicePair[T number](t *testing.T, s []T, n int, f func(a, b []T) bool) {
	t.Helper()
	for i := 0; i < len(s)-n; i++ {
		for j := 0; j < len(s)-n; j++ {
			if !f(s[i:i+n], s[j:j+n]) {
				return
			}
		}
	}
}

func forSliceTriple[T number](t *testing.T, s []T, n int, f func(a, b, c []T) bool) {
	t.Helper()
	for i := 0; i < len(s)-n; i += 3 {
		for j := 0; j < len(s)-n; j += 3 {
			for k := 0; k < len(s)-n; k += 3 {
				if !f(s[i:i+n], s[j:j+n], s[k:k+n]) {
					return
				}
			}
		}
	}
}

func forSlicePairMasked[T number](t *testing.T, s []T, n int, f func(a, b []T, m []bool) bool) {
	t.Helper()
	m := bools
	// Step slice pair masked forward much more quickly, otherwise it is slooooow
	for i := 0; i < len(s)-n; i += 3 {
		for j := 0; j < len(s)-n; j += 3 {
			for k := 0; k < len(m)-n; k += 3 {
				if !f(s[i:i+n], s[j:j+n], m[k:k+n]) {
					return
				}
			}
		}
	}
}
