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

// CheckSlicesLogInput compares two slices for equality,
// reporting a test error if there is a problem,
// and also consumes the two slices so that a
// test/benchmark won't be dead-code eliminated.
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
