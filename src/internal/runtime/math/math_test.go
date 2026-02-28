// Copyright 2018 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package math_test

import (
	. "internal/runtime/math"
	"testing"
)

const (
	UintptrSize = 32 << (^uintptr(0) >> 63)
)

type mulUintptrTest struct {
	a        uintptr
	b        uintptr
	overflow bool
}

var mulUintptrTests = []mulUintptrTest{
	{0, 0, false},
	{1000, 1000, false},
	{MaxUintptr, 0, false},
	{MaxUintptr, 1, false},
	{MaxUintptr / 2, 2, false},
	{MaxUintptr / 2, 3, true},
	{MaxUintptr, 10, true},
	{MaxUintptr, 100, true},
	{MaxUintptr / 100, 100, false},
	{MaxUintptr / 1000, 1001, true},
	{1<<(UintptrSize/2) - 1, 1<<(UintptrSize/2) - 1, false},
	{1 << (UintptrSize / 2), 1 << (UintptrSize / 2), true},
	{MaxUintptr >> 32, MaxUintptr >> 32, false},
	{MaxUintptr, MaxUintptr, true},
}

func TestMulUintptr(t *testing.T) {
	for _, test := range mulUintptrTests {
		a, b := test.a, test.b
		for i := 0; i < 2; i++ {
			mul, overflow := MulUintptr(a, b)
			if mul != a*b || overflow != test.overflow {
				t.Errorf("MulUintptr(%v, %v) = %v, %v want %v, %v",
					a, b, mul, overflow, a*b, test.overflow)
			}
			a, b = b, a
		}
	}
}

var SinkUintptr uintptr
var SinkBool bool

var x, y uintptr

func BenchmarkMulUintptr(b *testing.B) {
	x, y = 1, 2
	b.Run("small", func(b *testing.B) {
		for i := 0; i < b.N; i++ {
			var overflow bool
			SinkUintptr, overflow = MulUintptr(x, y)
			if overflow {
				SinkUintptr = 0
			}
		}
	})
	x, y = MaxUintptr, MaxUintptr-1
	b.Run("large", func(b *testing.B) {
		for i := 0; i < b.N; i++ {
			var overflow bool
			SinkUintptr, overflow = MulUintptr(x, y)
			if overflow {
				SinkUintptr = 0
			}
		}
	})
}
