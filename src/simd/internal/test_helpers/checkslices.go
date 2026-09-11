// Copyright 2025 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

//go:build goexperiment.simd

package test_helpers

import (
	"math"
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

func CheckSlices[T number](t *testing.T, got, want []T) bool {
	t.Helper()
	return CheckSlicesLogInput[T](t, got, want, 0.0, nil)
}

func isSameFloat(ia, ib any, flakiness float64) bool {
	switch x := ia.(type) {
	case float32:
		y := ib.(float32)
		if math.IsNaN(float64(x)) && math.IsNaN(float64(y)) {
			return true
		}
		if flakiness > 0 {
			if y == 0 {
				if math.Abs(float64(x)) < flakiness {
					return true
				}
			} else {
				if math.Abs(float64((x-y)/y)) < flakiness {
					return true
				}
			}
		}
	case float64:
		y := ib.(float64)
		if math.IsNaN(x) && math.IsNaN(y) {
			return true
		}
		if flakiness > 0 {
			if y == 0 {
				if math.Abs(x) < flakiness {
					return true
				}
			} else if math.Abs((x-y)/y) < flakiness {
				return true
			}
		}

	default:
	}
	return false
}

func isDifferentSignZero(ia, ib any) bool {
	switch x := ia.(type) {
	case float32:
		y := ib.(float32)
		if math.Float32bits(x) != math.Float32bits(y) {
			return true
		}
	case float64:
		y := ib.(float64)
		if math.Float64bits(x) != math.Float64bits(y) {
			return true
		}
	default:
	}
	return false
}

// CheckSlicesLogInput compares two slices for equality,
// reporting a test error if there is a problem,
// and also consumes the two slices so that a
// test/benchmark won't be dead-code eliminated.
func CheckSlicesLogInput[T number](t *testing.T, got, want []T, flakiness float64, logInput func()) bool {
	t.Helper()
	var z T
	for i := range want {
		if got[i] != want[i] {
			var ia any = got[i]
			var ib any = want[i]
			if isSameFloat(ia, ib, flakiness) {
				continue
			}

			t.Logf("For %T vector elements:", z)
			t.Logf("got =%v", got)
			t.Logf("want=%v", want)
			if logInput != nil {
				logInput()
			}
			t.Errorf("at index %d, got=%v, want=%v", i, got[i], want[i])
			return false
		} else if got[i] == 0 { // for floating point, 0.0 == -0.0 but a bitwise check can see the difference
			var ia any = got[i]
			var ib any = want[i]
			if isDifferentSignZero(ia, ib) {
				t.Logf("For %T vector elements:", z)
				t.Logf("got =%v", got)
				t.Logf("want=%v", want)
				if logInput != nil {
					logInput()
				}
				t.Errorf("at index %d, different signs of zero", i)
				return false
			}
		}
	}
	return true
}

// CheckScalarsLogInput compares two values for equality,
// reporting a test error if there is a problem,
// and also consumes the two values so that a
// test/benchmark won't be dead-code eliminated.
// The goal is to match the behavior and output for
// CheckSlicesLogInput.
func CheckScalarsLogInput[T number](t *testing.T, got, want T, flakiness float64, logInput func()) bool {
	t.Helper()
	var z T
	if got != want {
		var ia any = got
		var ib any = want
		if isSameFloat(ia, ib, flakiness) {
			return true
		}

		t.Logf("For %T scalar:", z)
		t.Logf("got =%v", got)
		t.Logf("want=%v", want)
		if logInput != nil {
			logInput()
		}
		t.Errorf("got=%v, want=%v", got, want)
		return false

	} else if got == 0 { // for floating point, 0.0 == -0.0 but a bitwise check can see the difference
		var ia any = got
		var ib any = want
		if isDifferentSignZero(ia, ib) {
			t.Logf("For %T vector elements:", z)
			t.Logf("got =%v", got)
			t.Logf("want=%v", want)
			if logInput != nil {
				logInput()
			}
			t.Errorf("different signs of zero")
			return false
		}
	}
	return true
}

// CheckStringsLogInput compares two strings for equality,
// reporting a test error if there is a problem,
// and also consumes the two values so that a
// test/benchmark won't be dead-code eliminated.
// The goal is to match the behavior and output for
// CheckSlicesLogInput.
func CheckStringsLogInput(t *testing.T, got, want string, flakiness float64, logInput func()) bool {
	t.Helper()
	if got != want {
		t.Logf("For string:")
		t.Logf("got =%v", got)
		t.Logf("want=%v", want)
		if logInput != nil {
			logInput()
		}
		t.Errorf("got=%v, want=%v", got, want)
		return false
	}
	return true
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
var snan32 = math.Float32frombits(0x7f800001)
var snan64 = math.Float64frombits(0x7ff0000000000001)

// N controls how large the test vectors are
const N = 144

var float32s = nOf(N, []float32{float32(inf), float32(ninf), 1, float32(nan), snan32, -float32(nan), -snan32, float32(zero), 2, float32(nan), float32(zero), 3, float32(-zero), float32(1.0 / zero), float32(-1.0 / zero), 1.0 / 2, 1.0 / 4, 1.0 / 8, 1.0 / 1000, 1.0 / 1000000, 1, -1, 0, 2, -2, 3, -3, math.MaxFloat32, 1 / math.MaxFloat32, 10, -10, 100, 20, -20, 300, -300, -4000, -80, -160, -3200, -64, -4, -8, -16, -32, -64})
var float64s = nOf(N, []float64{inf, ninf, nan, snan64, -nan, -snan64, zero, -zero, 1 / zero, -1 / zero, 0.0001, 0.0000001, 1, -1, 0, 2, -2, 3, -3, math.MaxFloat64, 1.0 / math.MaxFloat64, 10, -10, 100, 20, -20, 300, -300, -4000, -80, -16, -32, -64})

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

func Float64s() []float64 {
	return float64s
}
func Int64s() []int64 {
	return int64s
}
func Uint64s() []uint64 {
	return uint64s
}

func Float32s() []float32 {
	return float32s
}
func Int32s() []int32 {
	return int32s
}
func Uint32s() []uint32 {
	return uint32s
}

func Int16s() []int16 {
	return int16s
}
func Uint16s() []uint16 {
	return uint16s
}

func Int8s() []int8 {
	return int8s
}
func Uint8s() []uint8 {
	return uint8s
}
func Bools() []bool {
	return bools
}

var s8s = [...]uint64{0, 1, 3, 7, 8, 9, 16, 9999999999}
var s16s = [...]uint64{0, 1, 3, 15, 16, 17, 32, 9999999999}
var s32s = [...]uint64{0, 1, 3, 31, 32, 33, 64, 9999999999}
var s64s = [...]uint64{0, 1, 3, 63, 64, 65, 128, 9999999999}

func Shift8s() []uint64 {
	return s8s[:]
}
func Shift16s() []uint64 {
	return s16s[:]
}
func Shift32s() []uint64 {
	return s32s[:]
}
func Shift64s() []uint64 {
	return s64s[:]
}

// ImpliedDo returns a slice T with length n and each element i
// initialized to val(i).
func ImpliedDo[T number](n int, val func(i int) T) []T {
	s := make([]T, n)
	for i := range s {
		s[i] = val(i)
	}
	return s
}

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
