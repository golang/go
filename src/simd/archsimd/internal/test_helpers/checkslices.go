// Copyright 2025 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

//go:build goexperiment.simd && amd64

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

// CheckSlices compares two slices for equality,
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
			switch x := ia.(type) {
			case float32:
				y := ib.(float32)
				if math.IsNaN(float64(x)) && math.IsNaN(float64(y)) {
					continue
				}
				if flakiness > 0 {
					if y == 0 {
						if math.Abs(float64(x)) < flakiness {
							continue
						}
					} else {
						if math.Abs(float64((x-y)/y)) < flakiness {
							continue
						}
					}
				}
			case float64:
				y := ib.(float64)
				if math.IsNaN(x) && math.IsNaN(y) {
					continue
				}
				if flakiness > 0 {
					if y == 0 {
						if math.Abs(x) < flakiness {
							continue
						}
					} else if math.Abs((x-y)/y) < flakiness {
						continue
					}
				}

			default:
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
			switch x := ia.(type) {
			case float32:
				y := ib.(float32)
				if math.Float32bits(x) != math.Float32bits(y) {
					t.Logf("For %T vector elements:", z)
					t.Logf("got =%v", got)
					t.Logf("want=%v", want)
					if logInput != nil {
						logInput()
					}
					t.Errorf("at index %d, different signs of zero", i)
					return false
				}
			case float64:
				y := ib.(float64)
				if math.Float64bits(x) != math.Float64bits(y) {
					t.Logf("For %T vector elements:", z)
					t.Logf("got =%v", got)
					t.Logf("want=%v", want)
					if logInput != nil {
						logInput()
					}
					t.Errorf("at index %d, different signs of zero", i)
					return false
				}
			default:
			}

		}
	}
	return true
}
