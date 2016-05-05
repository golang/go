// run

// Check conversion of constant to float32/float64 near min/max boundaries.

// Copyright 2014 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package main

import (
	"fmt"
	"math"
)

// The largest exact float32 is f₁ = (1+(1-2²³))×2¹²⁷ = (1-2²⁴)×2¹²⁸ = 2¹²⁸ - 2¹⁰⁴.
// The next float32 would be f₂ = (1+1)×2¹²⁷ = 1×2¹²⁸, except that exponent is out of range.
// Float32 conversion rounds to the nearest float32, rounding to even mantissa:
// between f₁ and f₂, values closer to f₁ round to f₁ and values closer to f₂ are rejected as out of range.
// f₁ is an odd mantissa, so the halfway point (f₁+f₂)/2 rounds to f₂ and is rejected.
// The halfway point is (f₁+f₂)/2 = 2¹²⁸ - 2¹⁰⁵.
//
// The same is true of float64, with different constants: s/24/53/ and s/128/1024/.

const (
	two24   = 1.0 * (1 << 24)
	two53   = 1.0 * (1 << 53)
	two64   = 1.0 * (1 << 64)
	two128  = two64 * two64
	two256  = two128 * two128
	two512  = two256 * two256
	two768  = two512 * two256
	two1024 = two512 * two512

	ulp32 = two128 / two24
	max32 = two128 - ulp32

	ulp64 = two1024 / two53
	max64 = two1024 - ulp64
)

var cvt = []struct {
	bits   uint64 // keep us honest
	exact  interface{}
	approx interface{}
	text   string
}{
	// 0
	{0x7f7ffffe, float32(max32 - ulp32), float32(max32 - ulp32 - ulp32/2), "max32 - ulp32 - ulp32/2"},
	{0x7f7ffffe, float32(max32 - ulp32), float32(max32 - ulp32), "max32 - ulp32"},
	{0x7f7ffffe, float32(max32 - ulp32), float32(max32 - ulp32/2), "max32 - ulp32/2"},
	{0x7f7ffffe, float32(max32 - ulp32), float32(max32 - ulp32 + ulp32/2), "max32 - ulp32 + ulp32/2"},
	{0x7f7fffff, float32(max32), float32(max32 - ulp32 + ulp32/2 + ulp32/two64), "max32 - ulp32 + ulp32/2 + ulp32/two64"},
	{0x7f7fffff, float32(max32), float32(max32 - ulp32/2 + ulp32/two64), "max32 - ulp32/2 + ulp32/two64"},
	{0x7f7fffff, float32(max32), float32(max32), "max32"},
	{0x7f7fffff, float32(max32), float32(max32 + ulp32/2 - ulp32/two64), "max32 + ulp32/2 - ulp32/two64"},

	{0xff7ffffe, float32(-(max32 - ulp32)), float32(-(max32 - ulp32 - ulp32/2)), "-(max32 - ulp32 - ulp32/2)"},
	{0xff7ffffe, float32(-(max32 - ulp32)), float32(-(max32 - ulp32)), "-(max32 - ulp32)"},
	{0xff7ffffe, float32(-(max32 - ulp32)), float32(-(max32 - ulp32/2)), "-(max32 - ulp32/2)"},
	{0xff7ffffe, float32(-(max32 - ulp32)), float32(-(max32 - ulp32 + ulp32/2)), "-(max32 - ulp32 + ulp32/2)"},
	{0xff7fffff, float32(-(max32)), float32(-(max32 - ulp32 + ulp32/2 + ulp32/two64)), "-(max32 - ulp32 + ulp32/2 + ulp32/two64)"},
	{0xff7fffff, float32(-(max32)), float32(-(max32 - ulp32/2 + ulp32/two64)), "-(max32 - ulp32/2 + ulp32/two64)"},
	{0xff7fffff, float32(-(max32)), float32(-(max32)), "-(max32)"},
	{0xff7fffff, float32(-(max32)), float32(-(max32 + ulp32/2 - ulp32/two64)), "-(max32 + ulp32/2 - ulp32/two64)"},

	// These are required to work: according to the Go spec, the internal float mantissa must be at least 256 bits,
	// and these expressions can be represented exactly with a 256-bit mantissa.
	{0x7f7fffff, float32(max32), float32(max32 - ulp32 + ulp32/2 + 1), "max32 - ulp32 + ulp32/2 + 1"},
	{0x7f7fffff, float32(max32), float32(max32 - ulp32/2 + 1), "max32 - ulp32/2 + 1"},
	{0x7f7fffff, float32(max32), float32(max32 + ulp32/2 - 1), "max32 + ulp32/2 - 1"},
	{0xff7fffff, float32(-(max32)), float32(-(max32 - ulp32 + ulp32/2 + 1)), "-(max32 - ulp32 + ulp32/2 + 1)"},
	{0xff7fffff, float32(-(max32)), float32(-(max32 - ulp32/2 + 1)), "-(max32 - ulp32/2 + 1)"},
	{0xff7fffff, float32(-(max32)), float32(-(max32 + ulp32/2 - 1)), "-(max32 + ulp32/2 - 1)"},

	{0x7f7fffff, float32(max32), float32(max32 - ulp32 + ulp32/2 + 1/two128), "max32 - ulp32 + ulp32/2 + 1/two128"},
	{0x7f7fffff, float32(max32), float32(max32 - ulp32/2 + 1/two128), "max32 - ulp32/2 + 1/two128"},
	{0x7f7fffff, float32(max32), float32(max32 + ulp32/2 - 1/two128), "max32 + ulp32/2 - 1/two128"},
	{0xff7fffff, float32(-(max32)), float32(-(max32 - ulp32 + ulp32/2 + 1/two128)), "-(max32 - ulp32 + ulp32/2 + 1/two128)"},
	{0xff7fffff, float32(-(max32)), float32(-(max32 - ulp32/2 + 1/two128)), "-(max32 - ulp32/2 + 1/two128)"},
	{0xff7fffff, float32(-(max32)), float32(-(max32 + ulp32/2 - 1/two128)), "-(max32 + ulp32/2 - 1/two128)"},

	{0x7feffffffffffffe, float64(max64 - ulp64), float64(max64 - ulp64 - ulp64/2), "max64 - ulp64 - ulp64/2"},
	{0x7feffffffffffffe, float64(max64 - ulp64), float64(max64 - ulp64), "max64 - ulp64"},
	{0x7feffffffffffffe, float64(max64 - ulp64), float64(max64 - ulp64/2), "max64 - ulp64/2"},
	{0x7feffffffffffffe, float64(max64 - ulp64), float64(max64 - ulp64 + ulp64/2), "max64 - ulp64 + ulp64/2"},
	{0x7fefffffffffffff, float64(max64), float64(max64 - ulp64 + ulp64/2 + ulp64/two64), "max64 - ulp64 + ulp64/2 + ulp64/two64"},
	{0x7fefffffffffffff, float64(max64), float64(max64 - ulp64/2 + ulp64/two64), "max64 - ulp64/2 + ulp64/two64"},
	{0x7fefffffffffffff, float64(max64), float64(max64), "max64"},
	{0x7fefffffffffffff, float64(max64), float64(max64 + ulp64/2 - ulp64/two64), "max64 + ulp64/2 - ulp64/two64"},

	{0xffeffffffffffffe, float64(-(max64 - ulp64)), float64(-(max64 - ulp64 - ulp64/2)), "-(max64 - ulp64 - ulp64/2)"},
	{0xffeffffffffffffe, float64(-(max64 - ulp64)), float64(-(max64 - ulp64)), "-(max64 - ulp64)"},
	{0xffeffffffffffffe, float64(-(max64 - ulp64)), float64(-(max64 - ulp64/2)), "-(max64 - ulp64/2)"},
	{0xffeffffffffffffe, float64(-(max64 - ulp64)), float64(-(max64 - ulp64 + ulp64/2)), "-(max64 - ulp64 + ulp64/2)"},
	{0xffefffffffffffff, float64(-(max64)), float64(-(max64 - ulp64 + ulp64/2 + ulp64/two64)), "-(max64 - ulp64 + ulp64/2 + ulp64/two64)"},
	{0xffefffffffffffff, float64(-(max64)), float64(-(max64 - ulp64/2 + ulp64/two64)), "-(max64 - ulp64/2 + ulp64/two64)"},
	{0xffefffffffffffff, float64(-(max64)), float64(-(max64)), "-(max64)"},
	{0xffefffffffffffff, float64(-(max64)), float64(-(max64 + ulp64/2 - ulp64/two64)), "-(max64 + ulp64/2 - ulp64/two64)"},

	// These are required to work.
	// The mantissas are exactly 256 bits.
	// max64 is just below 2¹⁰²⁴ so the bottom bit we can use is 2⁷⁶⁸.
	{0x7fefffffffffffff, float64(max64), float64(max64 - ulp64 + ulp64/2 + two768), "max64 - ulp64 + ulp64/2 + two768"},
	{0x7fefffffffffffff, float64(max64), float64(max64 - ulp64/2 + two768), "max64 - ulp64/2 + two768"},
	{0x7fefffffffffffff, float64(max64), float64(max64 + ulp64/2 - two768), "max64 + ulp64/2 - two768"},
	{0xffefffffffffffff, float64(-(max64)), float64(-(max64 - ulp64 + ulp64/2 + two768)), "-(max64 - ulp64 + ulp64/2 + two768)"},
	{0xffefffffffffffff, float64(-(max64)), float64(-(max64 - ulp64/2 + two768)), "-(max64 - ulp64/2 + two768)"},
	{0xffefffffffffffff, float64(-(max64)), float64(-(max64 + ulp64/2 - two768)), "-(max64 + ulp64/2 - two768)"},
}

var bugged = false

func bug() {
	if !bugged {
		bugged = true
		fmt.Println("BUG")
	}
}

func main() {
	u64 := math.Float64frombits(0x7fefffffffffffff) - math.Float64frombits(0x7feffffffffffffe)
	if ulp64 != u64 {
		bug()
		fmt.Printf("ulp64=%g, want %g", ulp64, u64)
	}

	u32 := math.Float32frombits(0x7f7fffff) - math.Float32frombits(0x7f7ffffe)
	if ulp32 != u32 {
		bug()
		fmt.Printf("ulp32=%g, want %g", ulp32, u32)
	}

	for _, c := range cvt {
		if bits(c.exact) != c.bits {
			bug()
			fmt.Printf("%s: inconsistent table: bits=%#x (%g) but exact=%g (%#x)\n", c.text, c.bits, fromBits(c.bits, c.exact), c.exact, bits(c.exact))
		}
		if c.approx != c.exact || bits(c.approx) != c.bits {
			bug()
			fmt.Printf("%s: have %g (%#x) want %g (%#x)\n", c.text, c.approx, bits(c.approx), c.exact, c.bits)
		}
	}
}

func bits(x interface{}) interface{} {
	switch x := x.(type) {
	case float32:
		return uint64(math.Float32bits(x))
	case float64:
		return math.Float64bits(x)
	}
	return 0
}

func fromBits(b uint64, x interface{}) interface{} {
	switch x.(type) {
	case float32:
		return math.Float32frombits(uint32(b))
	case float64:
		return math.Float64frombits(b)
	}
	return "?"
}
