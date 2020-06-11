// Copyright 2020 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package ssa

import (
	"math"
	"math/rand"
	"testing"
)

var (
	x64   int64  = math.MaxInt64 - 2
	x64b  int64  = math.MaxInt64 - 2
	x64c  int64  = math.MaxInt64 - 2
	y64   int64  = math.MinInt64 + 1
	x32   int32  = math.MaxInt32 - 2
	x32b  int32  = math.MaxInt32 - 2
	x32c  int32  = math.MaxInt32 - 2
	y32   int32  = math.MinInt32 + 1
	one64 int64  = 1
	one32 int32  = 1
	v64   int64  = 11 // ensure it's not 2**n +/- 1
	v64_n int64  = -11
	v32   int32  = 11
	v32_n int32  = -11
	uv32  uint32 = 19
	uz    uint8  = 1 // for lowering to SLL/SRL/SRA
)

var crTests = []struct {
	name string
	tf   func(t *testing.T)
}{
	{"AddConst64", testAddConst64},
	{"AddConst32", testAddConst32},
	{"AddVar64", testAddVar64},
	{"AddVar32", testAddVar32},
	{"MAddVar64", testMAddVar64},
	{"MAddVar32", testMAddVar32},
	{"MSubVar64", testMSubVar64},
	{"MSubVar32", testMSubVar32},
	{"AddShift32", testAddShift32},
	{"SubShift32", testSubShift32},
}

var crBenches = []struct {
	name string
	bf   func(b *testing.B)
}{
	{"SoloJump", benchSoloJump},
	{"CombJump", benchCombJump},
}

// Test int32/int64's add/sub/madd/msub operations with boundary values to
// ensure the optimization to 'comparing to zero' expressions of if-statements
// yield expected results.
// 32 rewriting rules are covered. At least two scenarios for "Canonicalize
// the order of arguments to comparisons", which helps with CSE, are covered.
// The tedious if-else structures are necessary to ensure all concerned rules
// and machine code sequences are covered.
// It's for arm64 initially, please see https://github.com/golang/go/issues/38740
func TestCondRewrite(t *testing.T) {
	for _, test := range crTests {
		t.Run(test.name, test.tf)
	}
}

// Profile the aforementioned optimization from two angles:
//   SoloJump: generated branching code has one 'jump', for '<' and '>='
//   CombJump: generated branching code has two consecutive 'jump', for '<=' and '>'
// We expect that 'CombJump' is generally on par with the non-optimized code, and
// 'SoloJump' demonstrates some improvement.
// It's for arm64 initially, please see https://github.com/golang/go/issues/38740
func BenchmarkCondRewrite(b *testing.B) {
	for _, bench := range crBenches {
		b.Run(bench.name, bench.bf)
	}
}

// var +/- const
func testAddConst64(t *testing.T) {
	if x64+11 < 0 {
	} else {
		t.Errorf("'%#x + 11 < 0' failed", x64)
	}

	if x64+13 <= 0 {
	} else {
		t.Errorf("'%#x + 13 <= 0' failed", x64)
	}

	if y64-11 > 0 {
	} else {
		t.Errorf("'%#x - 11 > 0' failed", y64)
	}

	if y64-13 >= 0 {
	} else {
		t.Errorf("'%#x - 13 >= 0' failed", y64)
	}

	if x64+19 > 0 {
		t.Errorf("'%#x + 19 > 0' failed", x64)
	}

	if x64+23 >= 0 {
		t.Errorf("'%#x + 23 >= 0' failed", x64)
	}

	if y64-19 < 0 {
		t.Errorf("'%#x - 19 < 0' failed", y64)
	}

	if y64-23 <= 0 {
		t.Errorf("'%#x - 23 <= 0' failed", y64)
	}
}

// 32-bit var +/- const
func testAddConst32(t *testing.T) {
	if x32+11 < 0 {
	} else {
		t.Errorf("'%#x + 11 < 0' failed", x32)
	}

	if x32+13 <= 0 {
	} else {
		t.Errorf("'%#x + 13 <= 0' failed", x32)
	}

	if y32-11 > 0 {
	} else {
		t.Errorf("'%#x - 11 > 0' failed", y32)
	}

	if y32-13 >= 0 {
	} else {
		t.Errorf("'%#x - 13 >= 0' failed", y32)
	}

	if x32+19 > 0 {
		t.Errorf("'%#x + 19 > 0' failed", x32)
	}

	if x32+23 >= 0 {
		t.Errorf("'%#x + 23 >= 0' failed", x32)
	}

	if y32-19 < 0 {
		t.Errorf("'%#x - 19 < 0' failed", y32)
	}

	if y32-23 <= 0 {
		t.Errorf("'%#x - 23 <= 0' failed", y32)
	}
}

// var + var
func testAddVar64(t *testing.T) {
	if x64+v64 < 0 {
	} else {
		t.Errorf("'%#x + %#x < 0' failed", x64, v64)
	}

	if x64+v64 <= 0 {
	} else {
		t.Errorf("'%#x + %#x <= 0' failed", x64, v64)
	}

	if y64+v64_n > 0 {
	} else {
		t.Errorf("'%#x + %#x > 0' failed", y64, v64_n)
	}

	if y64+v64_n >= 0 {
	} else {
		t.Errorf("'%#x + %#x >= 0' failed", y64, v64_n)
	}

	if x64+v64 > 0 {
		t.Errorf("'%#x + %#x > 0' failed", x64, v64)
	}

	if x64+v64 >= 0 {
		t.Errorf("'%#x + %#x >= 0' failed", x64, v64)
	}

	if y64+v64_n < 0 {
		t.Errorf("'%#x + %#x < 0' failed", y64, v64_n)
	}

	if y64+v64_n <= 0 {
		t.Errorf("'%#x + %#x <= 0' failed", y64, v64_n)
	}
}

// 32-bit var+var
func testAddVar32(t *testing.T) {
	if x32+v32 < 0 {
	} else {
		t.Errorf("'%#x + %#x < 0' failed", x32, v32)
	}

	if x32+v32 <= 0 {
	} else {
		t.Errorf("'%#x + %#x <= 0' failed", x32, v32)
	}

	if y32+v32_n > 0 {
	} else {
		t.Errorf("'%#x + %#x > 0' failed", y32, v32_n)
	}

	if y32+v32_n >= 0 {
	} else {
		t.Errorf("'%#x + %#x >= 0' failed", y32, v32_n)
	}

	if x32+v32 > 0 {
		t.Errorf("'%#x + %#x > 0' failed", x32, v32)
	}

	if x32+v32 >= 0 {
		t.Errorf("'%#x + %#x >= 0' failed", x32, v32)
	}

	if y32+v32_n < 0 {
		t.Errorf("'%#x + %#x < 0' failed", y32, v32_n)
	}

	if y32+v32_n <= 0 {
		t.Errorf("'%#x + %#x <= 0' failed", y32, v32_n)
	}
}

// multiply-add
func testMAddVar64(t *testing.T) {
	if x64+v64*one64 < 0 {
	} else {
		t.Errorf("'%#x + %#x*1 < 0' failed", x64, v64)
	}

	if x64+v64*one64 <= 0 {
	} else {
		t.Errorf("'%#x + %#x*1 <= 0' failed", x64, v64)
	}

	if y64+v64_n*one64 > 0 {
	} else {
		t.Errorf("'%#x + %#x*1 > 0' failed", y64, v64_n)
	}

	if y64+v64_n*one64 >= 0 {
	} else {
		t.Errorf("'%#x + %#x*1 >= 0' failed", y64, v64_n)
	}

	if x64+v64*one64 > 0 {
		t.Errorf("'%#x + %#x*1 > 0' failed", x64, v64)
	}

	if x64+v64*one64 >= 0 {
		t.Errorf("'%#x + %#x*1 >= 0' failed", x64, v64)
	}

	if y64+v64_n*one64 < 0 {
		t.Errorf("'%#x + %#x*1 < 0' failed", y64, v64_n)
	}

	if y64+v64_n*one64 <= 0 {
		t.Errorf("'%#x + %#x*1 <= 0' failed", y64, v64_n)
	}
}

// 32-bit multiply-add
func testMAddVar32(t *testing.T) {
	if x32+v32*one32 < 0 {
	} else {
		t.Errorf("'%#x + %#x*1 < 0' failed", x32, v32)
	}

	if x32+v32*one32 <= 0 {
	} else {
		t.Errorf("'%#x + %#x*1 <= 0' failed", x32, v32)
	}

	if y32+v32_n*one32 > 0 {
	} else {
		t.Errorf("'%#x + %#x*1 > 0' failed", y32, v32_n)
	}

	if y32+v32_n*one32 >= 0 {
	} else {
		t.Errorf("'%#x + %#x*1 >= 0' failed", y32, v32_n)
	}

	if x32+v32*one32 > 0 {
		t.Errorf("'%#x + %#x*1 > 0' failed", x32, v32)
	}

	if x32+v32*one32 >= 0 {
		t.Errorf("'%#x + %#x*1 >= 0' failed", x32, v32)
	}

	if y32+v32_n*one32 < 0 {
		t.Errorf("'%#x + %#x*1 < 0' failed", y32, v32_n)
	}

	if y32+v32_n*one32 <= 0 {
		t.Errorf("'%#x + %#x*1 <= 0' failed", y32, v32_n)
	}
}

// multiply-sub
func testMSubVar64(t *testing.T) {
	if x64-v64_n*one64 < 0 {
	} else {
		t.Errorf("'%#x - %#x*1 < 0' failed", x64, v64_n)
	}

	if x64-v64_n*one64 <= 0 {
	} else {
		t.Errorf("'%#x - %#x*1 <= 0' failed", x64, v64_n)
	}

	if y64-v64*one64 > 0 {
	} else {
		t.Errorf("'%#x - %#x*1 > 0' failed", y64, v64)
	}

	if y64-v64*one64 >= 0 {
	} else {
		t.Errorf("'%#x - %#x*1 >= 0' failed", y64, v64)
	}

	if x64-v64_n*one64 > 0 {
		t.Errorf("'%#x - %#x*1 > 0' failed", x64, v64_n)
	}

	if x64-v64_n*one64 >= 0 {
		t.Errorf("'%#x - %#x*1 >= 0' failed", x64, v64_n)
	}

	if y64-v64*one64 < 0 {
		t.Errorf("'%#x - %#x*1 < 0' failed", y64, v64)
	}

	if y64-v64*one64 <= 0 {
		t.Errorf("'%#x - %#x*1 <= 0' failed", y64, v64)
	}

	if x64-x64b*one64 < 0 {
		t.Errorf("'%#x - %#x*1 < 0' failed", x64, x64b)
	}

	if x64-x64b*one64 >= 0 {
	} else {
		t.Errorf("'%#x - %#x*1 >= 0' failed", x64, x64b)
	}
}

// 32-bit multiply-sub
func testMSubVar32(t *testing.T) {
	if x32-v32_n*one32 < 0 {
	} else {
		t.Errorf("'%#x - %#x*1 < 0' failed", x32, v32_n)
	}

	if x32-v32_n*one32 <= 0 {
	} else {
		t.Errorf("'%#x - %#x*1 <= 0' failed", x32, v32_n)
	}

	if y32-v32*one32 > 0 {
	} else {
		t.Errorf("'%#x - %#x*1 > 0' failed", y32, v32)
	}

	if y32-v32*one32 >= 0 {
	} else {
		t.Errorf("'%#x - %#x*1 >= 0' failed", y32, v32)
	}

	if x32-v32_n*one32 > 0 {
		t.Errorf("'%#x - %#x*1 > 0' failed", x32, v32_n)
	}

	if x32-v32_n*one32 >= 0 {
		t.Errorf("'%#x - %#x*1 >= 0' failed", x32, v32_n)
	}

	if y32-v32*one32 < 0 {
		t.Errorf("'%#x - %#x*1 < 0' failed", y32, v32)
	}

	if y32-v32*one32 <= 0 {
		t.Errorf("'%#x - %#x*1 <= 0' failed", y32, v32)
	}

	if x32-x32b*one32 < 0 {
		t.Errorf("'%#x - %#x*1 < 0' failed", x32, x32b)
	}

	if x32-x32b*one32 >= 0 {
	} else {
		t.Errorf("'%#x - %#x*1 >= 0' failed", x32, x32b)
	}
}

// 32-bit ADDshift, pick up 1~2 scenarios randomly for each condition
func testAddShift32(t *testing.T) {
	if x32+v32<<1 < 0 {
	} else {
		t.Errorf("'%#x + %#x<<%#x < 0' failed", x32, v32, 1)
	}

	if x32+v32>>1 <= 0 {
	} else {
		t.Errorf("'%#x + %#x>>%#x <= 0' failed", x32, v32, 1)
	}

	if x32+int32(uv32>>1) > 0 {
		t.Errorf("'%#x + int32(%#x>>%#x) > 0' failed", x32, uv32, 1)
	}

	if x32+v32<<uz >= 0 {
		t.Errorf("'%#x + %#x<<%#x >= 0' failed", x32, v32, uz)
	}

	if x32+v32>>uz > 0 {
		t.Errorf("'%#x + %#x>>%#x > 0' failed", x32, v32, uz)
	}

	if x32+int32(uv32>>uz) < 0 {
	} else {
		t.Errorf("'%#x + int32(%#x>>%#x) < 0' failed", x32, uv32, uz)
	}
}

// 32-bit SUBshift, pick up 1~2 scenarios randomly for each condition
func testSubShift32(t *testing.T) {
	if y32-v32<<1 > 0 {
	} else {
		t.Errorf("'%#x - %#x<<%#x > 0' failed", y32, v32, 1)
	}

	if y32-v32>>1 < 0 {
		t.Errorf("'%#x - %#x>>%#x < 0' failed", y32, v32, 1)
	}

	if y32-int32(uv32>>1) >= 0 {
	} else {
		t.Errorf("'%#x - int32(%#x>>%#x) >= 0' failed", y32, uv32, 1)
	}

	if y32-v32<<uz < 0 {
		t.Errorf("'%#x - %#x<<%#x < 0' failed", y32, v32, uz)
	}

	if y32-v32>>uz >= 0 {
	} else {
		t.Errorf("'%#x - %#x>>%#x >= 0' failed", y32, v32, uz)
	}

	if y32-int32(uv32>>uz) <= 0 {
		t.Errorf("'%#x - int32(%#x>>%#x) <= 0' failed", y32, uv32, uz)
	}
}

var rnd = rand.New(rand.NewSource(0))
var sink int64

func benchSoloJump(b *testing.B) {
	r1 := x64
	r2 := x64b
	r3 := x64c
	r4 := y64
	d := rnd.Int63n(10)

	// 6 out 10 conditions evaluate to true
	for i := 0; i < b.N; i++ {
		if r1+r2 < 0 {
			d *= 2
			d /= 2
		}

		if r1+r3 >= 0 {
			d *= 2
			d /= 2
		}

		if r1+r2*one64 < 0 {
			d *= 2
			d /= 2
		}

		if r2+r3*one64 >= 0 {
			d *= 2
			d /= 2
		}

		if r1-r2*v64 >= 0 {
			d *= 2
			d /= 2
		}

		if r3-r4*v64 < 0 {
			d *= 2
			d /= 2
		}

		if r1+11 < 0 {
			d *= 2
			d /= 2
		}

		if r1+13 >= 0 {
			d *= 2
			d /= 2
		}

		if r4-17 < 0 {
			d *= 2
			d /= 2
		}

		if r4-19 >= 0 {
			d *= 2
			d /= 2
		}
	}
	sink = d
}

func benchCombJump(b *testing.B) {
	r1 := x64
	r2 := x64b
	r3 := x64c
	r4 := y64
	d := rnd.Int63n(10)

	// 6 out 10 conditions evaluate to true
	for i := 0; i < b.N; i++ {
		if r1+r2 <= 0 {
			d *= 2
			d /= 2
		}

		if r1+r3 > 0 {
			d *= 2
			d /= 2
		}

		if r1+r2*one64 <= 0 {
			d *= 2
			d /= 2
		}

		if r2+r3*one64 > 0 {
			d *= 2
			d /= 2
		}

		if r1-r2*v64 > 0 {
			d *= 2
			d /= 2
		}

		if r3-r4*v64 <= 0 {
			d *= 2
			d /= 2
		}

		if r1+11 <= 0 {
			d *= 2
			d /= 2
		}

		if r1+13 > 0 {
			d *= 2
			d /= 2
		}

		if r4-17 <= 0 {
			d *= 2
			d /= 2
		}

		if r4-19 > 0 {
			d *= 2
			d /= 2
		}
	}
	sink = d
}
