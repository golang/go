// $G $D/$F.go && $L $F.$A && ./$A.out >tmp.go &&
// $G tmp.go && $L tmp.$A && ./$A.out || echo BUG: 64bit
// rm -f tmp.go

// Copyright 2009 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// Generate test of 64-bit arithmetic.
// Most synthesized routines have different cases for
// constants vs variables and even the generated code has
// different cases for large and small constants,
// so try a good range of inputs.

package main

import (
	"bufio"
	"fmt"
	"os"
)

var bout *bufio.Writer

// 64-bit math without using 64-bit numbers,
// so that we can generate the test program even
// if the compiler has buggy or missing 64-bit support.

type Uint64 struct {
	hi	uint32
	lo	uint32
}

type Int64 struct {
	hi	int32
	lo	uint32
}

func (a Uint64) Int64() (c Int64) {
	c.hi = int32(a.hi)
	c.lo = a.lo
	return
}

func (a Uint64) Cmp(b Uint64) int {
	switch {
	case a.hi < b.hi:
		return -1
	case a.hi > b.hi:
		return 1
	case a.lo < b.lo:
		return -1
	case a.lo > b.lo:
		return 1
	}
	return 0
}

func (a Uint64) LeftShift(b uint) (c Uint64) {
	switch {
	case b >= 64:
		c.hi = 0
		c.lo = 0
	case b >= 32:
		c.hi = a.lo << (b - 32)
		c.lo = 0
	default:
		c.hi = a.hi<<b | a.lo>>(32-b)
		c.lo = a.lo << b
	}
	return
}

func (a Uint64) RightShift(b uint) (c Uint64) {
	switch {
	case b >= 64:
		c.hi = 0
		c.lo = a.hi
	case b >= 32:
		c.hi = 0
		c.lo = a.hi >> (b - 32)
	default:
		c.hi = a.hi >> b
		c.lo = a.hi<<(32-b) | a.lo>>b
	}
	return
}

func (a Uint64) LeftShift64(b Uint64) (c Uint64) {
	if b.hi != 0 || b.lo >= 64 {
		return
	}
	return a.LeftShift(uint(b.lo))
}

func (a Uint64) RightShift64(b Uint64) (c Uint64) {
	if b.hi != 0 || b.lo >= 64 {
		return
	}
	return a.RightShift(uint(b.lo))
}

func (a Uint64) Plus(b Uint64) (c Uint64) {
	var carry uint32
	if c.lo = a.lo + b.lo; c.lo < a.lo {
		carry = 1
	}
	c.hi = a.hi + b.hi + carry
	return
}

func (a Uint64) Minus(b Uint64) (c Uint64) {
	var borrow uint32
	if c.lo = a.lo - b.lo; c.lo > a.lo {
		borrow = 1
	}
	c.hi = a.hi - b.hi - borrow
	return
}

func (a Uint64) Neg() (c Uint64) {
	var zero Uint64
	return zero.Minus(a)
}

func (a Uint64) Com() (c Uint64) {
	c.hi = ^a.hi
	c.lo = ^a.lo
	return
}

func (a Uint64) Len() int {
	switch {
	case a.hi != 0:
		for i := 31; i >= 0; i-- {
			if a.hi&(1<<uint(i)) != 0 {
				return i + 1 + 32
			}
		}
	case a.lo != 0:
		for i := 31; i >= 0; i-- {
			if a.lo&(1<<uint(i)) != 0 {
				return i + 1
			}
		}
	}
	return 0
}

func (a Uint64) HasBit(b uint) bool {
	switch {
	case b >= 64:
		return false
	case b >= 32:
		return a.hi&(1<<(b-32)) != 0
	}
	return a.lo&(1<<b) != 0
}

func (a Uint64) Times(b Uint64) (c Uint64) {
	for i := uint(0); i < 64; i++ {
		if b.HasBit(i) {
			c = c.Plus(a.LeftShift(i))
		}
	}
	return
}

func (a Uint64) DivMod(b Uint64) (quo, rem Uint64) {
	n := a.Len() - b.Len()
	if n >= 0 {
		b = b.LeftShift(uint(n))
		for i := 0; i <= n; i++ {
			quo = quo.LeftShift(1)
			if b.Cmp(a) <= 0 {	// b <= a
				quo.lo |= 1
				a = a.Minus(b)
			}
			b = b.RightShift(1)
		}
	}
	rem = a
	return
}

func (a Uint64) And(b Uint64) (c Uint64) {
	c.hi = a.hi & b.hi
	c.lo = a.lo & b.lo
	return
}

func (a Uint64) AndNot(b Uint64) (c Uint64) {
	c.hi = a.hi &^ b.hi
	c.lo = a.lo &^ b.lo
	return
}

func (a Uint64) Or(b Uint64) (c Uint64) {
	c.hi = a.hi | b.hi
	c.lo = a.lo | b.lo
	return
}

func (a Uint64) Xor(b Uint64) (c Uint64) {
	c.hi = a.hi ^ b.hi
	c.lo = a.lo ^ b.lo
	return
}

func (a Uint64) String() string	{ return fmt.Sprintf("%#x%08x", a.hi, a.lo) }

func (a Int64) Uint64() (c Uint64) {
	c.hi = uint32(a.hi)
	c.lo = a.lo
	return
}

func (a Int64) Cmp(b Int64) int {
	// Same body as Uint64.Cmp,
	// but behaves differently
	// because hi is uint32 not int32.
	switch {
	case a.hi < b.hi:
		return -1
	case a.hi > b.hi:
		return 1
	case a.lo < b.lo:
		return -1
	case a.lo > b.lo:
		return 1
	}
	return 0
}

func (a Int64) LeftShift(b uint) (c Int64)	{ return a.Uint64().LeftShift(b).Int64() }

func (a Int64) RightShift(b uint) (c Int64) {
	switch {
	case b >= 64:
		c.hi = a.hi >> 31	// sign extend
		c.lo = uint32(c.hi)
	case b >= 32:
		c.hi = a.hi >> 31	// sign extend
		c.lo = uint32(a.hi >> (b - 32))
	default:
		c.hi = a.hi >> b
		c.lo = uint32(a.hi<<(32-b)) | a.lo>>b
	}
	return
}

func (a Int64) LeftShift64(b Uint64) (c Int64) {
	if b.hi != 0 || b.lo >= 64 {
		return
	}
	return a.LeftShift(uint(b.lo))
}

func (a Int64) RightShift64(b Uint64) (c Int64) {
	if b.hi != 0 || b.lo >= 64 {
		return a.RightShift(64)
	}
	return a.RightShift(uint(b.lo))
}

func (a Int64) Plus(b Int64) (c Int64)	{ return a.Uint64().Plus(b.Uint64()).Int64() }

func (a Int64) Minus(b Int64) (c Int64)	{ return a.Uint64().Minus(b.Uint64()).Int64() }

func (a Int64) Neg() (c Int64)	{ return a.Uint64().Neg().Int64() }

func (a Int64) Com() (c Int64)	{ return a.Uint64().Com().Int64() }

func (a Int64) Times(b Int64) (c Int64)	{ return a.Uint64().Times(b.Uint64()).Int64() }

func (a Int64) DivMod(b Int64) (quo Int64, rem Int64) {
	var zero Int64

	quoSign := +1
	remSign := +1
	if a.Cmp(zero) < 0 {
		quoSign = -1
		remSign = -1
		a = a.Neg()
	}
	if b.Cmp(zero) < 0 {
		quoSign = -quoSign
		b = b.Neg()
	}

	q, r := a.Uint64().DivMod(b.Uint64())
	quo = q.Int64()
	rem = r.Int64()

	if quoSign < 0 {
		quo = quo.Neg()
	}
	if remSign < 0 {
		rem = rem.Neg()
	}
	return
}

func (a Int64) And(b Int64) (c Int64)	{ return a.Uint64().And(b.Uint64()).Int64() }

func (a Int64) AndNot(b Int64) (c Int64)	{ return a.Uint64().AndNot(b.Uint64()).Int64() }

func (a Int64) Or(b Int64) (c Int64)	{ return a.Uint64().Or(b.Uint64()).Int64() }

func (a Int64) Xor(b Int64) (c Int64)	{ return a.Uint64().Xor(b.Uint64()).Int64() }

func (a Int64) String() string {
	if a.hi < 0 {
		return fmt.Sprintf("-%s", a.Neg().Uint64())
	}
	return a.Uint64().String()
}

var int64Values = []Int64{
	Int64{0, 0},
	Int64{0, 1},
	Int64{0, 2},
	Int64{0, 3},
	Int64{0, 100},
	Int64{0, 10001},
	Int64{0, 1<<31 - 1},
	Int64{0, 1 << 31},
	Int64{0, 1<<31 + 1},
	Int64{0, 1<<32 - 1<<30},
	Int64{0, 1<<32 - 1},
	Int64{1, 0},
	Int64{1, 1},
	Int64{2, 0},
	Int64{1<<31 - 1, 1<<32 - 10000},
	Int64{1<<31 - 1, 1<<32 - 1},
	Int64{0x789abcde, 0xf0123456},

	Int64{-1, 1<<32 - 1},
	Int64{-1, 1<<32 - 2},
	Int64{-1, 1<<32 - 3},
	Int64{-1, 1<<32 - 100},
	Int64{-1, 1<<32 - 10001},
	Int64{-1, 1<<32 - (1<<31 - 1)},
	Int64{-1, 1<<32 - 1<<31},
	Int64{-1, 1<<32 - (1<<31 + 1)},
	Int64{-1, 1<<32 - (1<<32 - 1<<30)},
	Int64{-1, 0},
	Int64{-1, 1},
	Int64{-2, 0},
	Int64{-(1 << 31), 10000},
	Int64{-(1 << 31), 1},
	Int64{-(1 << 31), 0},
	Int64{-0x789abcde, 0xf0123456},
}

var uint64Values = []Uint64{
	Uint64{0, 0},
	Uint64{0, 1},
	Uint64{0, 2},
	Uint64{0, 3},
	Uint64{0, 100},
	Uint64{0, 10001},
	Uint64{0, 1<<31 - 1},
	Uint64{0, 1 << 31},
	Uint64{0, 1<<31 + 1},
	Uint64{0, 1<<32 - 1<<30},
	Uint64{0, 1<<32 - 1},
	Uint64{1, 0},
	Uint64{1, 1},
	Uint64{2, 0},
	Uint64{1<<31 - 1, 1<<32 - 10000},
	Uint64{1<<31 - 1, 1<<32 - 1},
	Uint64{1<<32 - 1<<30, 0},
	Uint64{1<<32 - 1, 0},
	Uint64{1<<32 - 1, 1<<32 - 100},
	Uint64{1<<32 - 1, 1<<32 - 1},
	Uint64{0x789abcde, 0xf0123456},
	Uint64{0xfedcba98, 0x76543210},
}

var shiftValues = []Uint64{
	Uint64{0, 0},
	Uint64{0, 1},
	Uint64{0, 2},
	Uint64{0, 3},
	Uint64{0, 15},
	Uint64{0, 16},
	Uint64{0, 17},
	Uint64{0, 31},
	Uint64{0, 32},
	Uint64{0, 33},
	Uint64{0, 61},
	Uint64{0, 62},
	Uint64{0, 63},
	Uint64{0, 64},
	Uint64{0, 65},
	Uint64{0, 1<<32 - 1},
	Uint64{1, 0},
	Uint64{1, 1},
	Uint64{1 << 28, 0},
	Uint64{1 << 31, 0},
	Uint64{1<<32 - 1, 0},
	Uint64{1<<32 - 1, 1<<32 - 1},
}

var ntest = 0

// Part 1 is tests of variable operations; generic functions
// called by repetitive code.  Could make a table but not worth it.

const prolog = "\n" +
	"package main\n" +
	"\n" +
	"import \"os\"\n" +
	"\n" +
	"var ok = true\n" +
	"\n" +
	"func testInt64Unary(a, plus, xor, minus int64) {\n" +
	"	if n, op, want := +a, `+`, plus; n != want { ok=false; println(`int64`, op, a, `=`, n, `should be`, want); }\n" +
	"	if n, op, want := ^a, `^`, xor; n != want { ok=false; println(`int64`, op, a, `=`, n, `should be`, want); }\n" +
	"	if n, op, want := -a, `-`, minus; n != want { ok=false; println(`int64`, op, a, `=`, n, `should be`, want); }\n" +
	"}\n" +
	"\n" +
	"func testInt64Binary(a, b, add, sub, mul, div, mod, and, or, xor, andnot int64, dodiv bool) {\n" +
	"	if n, op, want := a + b, `+`, add; n != want { ok=false; println(`int64`, a, op, b, `=`, n, `should be`, want); }\n" +
	"	if n, op, want := a - b, `-`, sub; n != want { ok=false; println(`int64`, a, op, b, `=`, n, `should be`, want); }\n" +
	"	if n, op, want := a * b, `*`, mul; n != want { ok=false; println(`int64`, a, op, b, `=`, n, `should be`, want); }\n" +
	"	if dodiv {\n" +
	"		if n, op, want := a / b, `/`, div; n != want { ok=false; println(`int64`, a, op, b, `=`, n, `should be`, want); }\n" +
	"		if n, op, want := a % b, `%`, mod; n != want { ok=false; println(`int64`, a, op, b, `=`, n, `should be`, want); }\n" +
	"	}\n" +
	"	if n, op, want := a & b, `&`, and; n != want { ok=false; println(`int64`, a, op, b, `=`, n, `should be`, want); }\n" +
	"	if n, op, want := a | b, `|`, or; n != want { ok=false; println(`int64`, a, op, b, `=`, n, `should be`, want); }\n" +
	"	if n, op, want := a ^ b, `^`, xor; n != want { ok=false; println(`int64`, a, op, b, `=`, n, `should be`, want); }\n" +
	"	if n, op, want := a &^ b, `&^`, andnot; n != want { ok=false; println(`int64`, a, op, b, `=`, n, `should be`, want); }\n" +
	"}\n" +
	"\n" +
	"func testInt64Shift(a int64, b uint64, left, right int64) {\n" +
	"	if n, op, s, want := a << b, `<<`, b, left; n != want { ok=false; println(`int64`, a, op, `uint64`, s, `=`, n, `should be`, want); }\n" +
	"	if n, op, s, want := a >> b, `>>`, b, right; n != want { ok=false; println(`int64`, a, op, `uint64`, s, `=`, n, `should be`, want); }\n" +
	"	if uint64(uint(b)) == b {\n" +
	"		b := uint(b);\n" +
	"		if n, op, s, want := a << b, `<<`, b, left; n != want { ok=false; println(`int64`, a, op, `uint`, s, `=`, n, `should be`, want); }\n" +
	"		if n, op, s, want := a >> b, `>>`, b, right; n != want { ok=false; println(`int64`, a, op, `uint`, s, `=`, n, `should be`, want); }\n" +
	"	}\n" +
	"	if uint64(uint32(b)) == b {\n" +
	"		b := uint32(b);\n" +
	"		if n, op, s, want := a << b, `<<`, b, left; n != want { ok=false; println(`int64`, a, op, `uint32`, s, `=`, n, `should be`, want); }\n" +
	"		if n, op, s, want := a >> b, `>>`, b, right; n != want { ok=false; println(`int64`, a, op, `uint32`, s, `=`, n, `should be`, want); }\n" +
	"	}\n" +
	"	if uint64(uint16(b)) == b {\n" +
	"		b := uint16(b);\n" +
	"		if n, op, s, want := a << b, `<<`, b, left; n != want { ok=false; println(`int64`, a, op, `uint16`, s, `=`, n, `should be`, want); }\n" +
	"		if n, op, s, want := a >> b, `>>`, b, right; n != want { ok=false; println(`int64`, a, op, `uint16`, s, `=`, n, `should be`, want); }\n" +
	"	}\n" +
	"	if uint64(uint8(b)) == b {\n" +
	"		b := uint8(b);\n" +
	"		if n, op, s, want := a << b, `<<`, b, left; n != want { ok=false; println(`int64`, a, op, `uint8`, s, `=`, n, `should be`, want); }\n" +
	"		if n, op, s, want := a >> b, `>>`, b, right; n != want { ok=false; println(`int64`, a, op, `uint8`, s, `=`, n, `should be`, want); }\n" +
	"	}\n" +
	"}\n" +
	"\n" +
	"func testUint64Unary(a, plus, xor, minus uint64) {\n" +
	"	if n, op, want := +a, `+`, plus; n != want { ok=false; println(`uint64`, op, a, `=`, n, `should be`, want); }\n" +
	"	if n, op, want := ^a, `^`, xor; n != want { ok=false; println(`uint64`, op, a, `=`, n, `should be`, want); }\n" +
	"	if n, op, want := -a, `-`, minus; n != want { ok=false; println(`uint64`, op, a, `=`, n, `should be`, want); }\n" +
	"}\n" +
	"\n" +
	"func testUint64Binary(a, b, add, sub, mul, div, mod, and, or, xor, andnot uint64, dodiv bool) {\n" +
	"	if n, op, want := a + b, `+`, add; n != want { ok=false; println(`uint64`, a, op, b, `=`, n, `should be`, want); }\n" +
	"	if n, op, want := a - b, `-`, sub; n != want { ok=false; println(`uint64`, a, op, b, `=`, n, `should be`, want); }\n" +
	"	if n, op, want := a * b, `*`, mul; n != want { ok=false; println(`uint64`, a, op, b, `=`, n, `should be`, want); }\n" +
	"	if dodiv {\n" +
	"		if n, op, want := a / b, `/`, div; n != want { ok=false; println(`uint64`, a, op, b, `=`, n, `should be`, want); }\n" +
	"		if n, op, want := a % b, `%`, mod; n != want { ok=false; println(`uint64`, a, op, b, `=`, n, `should be`, want); }\n" +
	"	}\n" +
	"	if n, op, want := a & b, `&`, and; n != want { ok=false; println(`uint64`, a, op, b, `=`, n, `should be`, want); }\n" +
	"	if n, op, want := a | b, `|`, or; n != want { ok=false; println(`uint64`, a, op, b, `=`, n, `should be`, want); }\n" +
	"	if n, op, want := a ^ b, `^`, xor; n != want { ok=false; println(`uint64`, a, op, b, `=`, n, `should be`, want); }\n" +
	"	if n, op, want := a &^ b, `&^`, andnot; n != want { ok=false; println(`uint64`, a, op, b, `=`, n, `should be`, want); }\n" +
	"}\n" +
	"\n" +
	"func testUint64Shift(a, b, left, right uint64) {\n" +
	"	if n, op, s, want := a << b, `<<`, b, left; n != want { ok=false; println(`uint64`, a, op, `uint64`, s, `=`, n, `should be`, want); }\n" +
	"	if n, op, s, want := a >> b, `>>`, b, right; n != want { ok=false; println(`uint64`, a, op, `uint64`, s, `=`, n, `should be`, want); }\n" +
	"	if uint64(uint(b)) == b {\n" +
	"		b := uint(b);\n" +
	"		if n, op, s, want := a << b, `<<`, b, left; n != want { ok=false; println(`uint64`, a, op, `uint`, s, `=`, n, `should be`, want); }\n" +
	"		if n, op, s, want := a >> b, `>>`, b, right; n != want { ok=false; println(`uint64`, a, op, `uint`, s, `=`, n, `should be`, want); }\n" +
	"	}\n" +
	"	if uint64(uint32(b)) == b {\n" +
	"		b := uint32(b);\n" +
	"		if n, op, s, want := a << b, `<<`, b, left; n != want { ok=false; println(`uint64`, a, op, `uint32`, s, `=`, n, `should be`, want); }\n" +
	"		if n, op, s, want := a >> b, `>>`, b, right; n != want { ok=false; println(`uint64`, a, op, `uint32`, s, `=`, n, `should be`, want); }\n" +
	"	}\n" +
	"	if uint64(uint16(b)) == b {\n" +
	"		b := uint16(b);\n" +
	"		if n, op, s, want := a << b, `<<`, b, left; n != want { ok=false; println(`uint64`, a, op, `uint16`, s, `=`, n, `should be`, want); }\n" +
	"		if n, op, s, want := a >> b, `>>`, b, right; n != want { ok=false; println(`uint64`, a, op, `uint16`, s, `=`, n, `should be`, want); }\n" +
	"	}\n" +
	"	if uint64(uint8(b)) == b {\n" +
	"		b := uint8(b);\n" +
	"		if n, op, s, want := a << b, `<<`, b, left; n != want { ok=false; println(`uint64`, a, op, `uint8`, s, `=`, n, `should be`, want); }\n" +
	"		if n, op, s, want := a >> b, `>>`, b, right; n != want { ok=false; println(`uint64`, a, op, `uint8`, s, `=`, n, `should be`, want); }\n" +
	"	}\n" +
	"}\n" +
	"\n"

func varTests() {
	fmt.Fprint(bout, prolog)
	for _, a := range int64Values {
		fmt.Fprintf(bout, "func test%v() {\n", ntest)
		ntest++
		fmt.Fprintf(bout, "\ttestInt64Unary(%v, %v, %v, %v);\n", a, a, a.Com(), a.Neg())
		for _, b := range int64Values {
			var div, mod Int64
			dodiv := false
			var zero Int64
			if b.Cmp(zero) != 0 {	// b != 0
				// Can't divide by zero but also can't divide -0x8000...000 by -1.
				var bigneg = Int64{-0x80000000, 0}
				var minus1 = Int64{-1, ^uint32(0)}
				if a.Cmp(bigneg) != 0 || b.Cmp(minus1) != 0 {	// a != -1<<63 || b != -1
					div, mod = a.DivMod(b)
					dodiv = true
				}
			}
			fmt.Fprintf(bout, "\ttestInt64Binary(%v, %v, %v, %v, %v, %v, %v, %v, %v, %v, %v, %v);\n",
				a, b, a.Plus(b), a.Minus(b), a.Times(b), div, mod,
				a.And(b), a.Or(b), a.Xor(b), a.AndNot(b), dodiv)
		}
		for _, b := range shiftValues {
			fmt.Fprintf(bout, "\ttestInt64Shift(%v, %v, %v, %v);\n",
				a, b, a.LeftShift64(b), a.RightShift64(b))
		}
		fmt.Fprintf(bout, "}\n")
	}

	for _, a := range uint64Values {
		fmt.Fprintf(bout, "func test%v() {\n", ntest)
		ntest++
		fmt.Fprintf(bout, "\ttestUint64Unary(%v, %v, %v, %v);\n", a, a, a.Com(), a.Neg())
		for _, b := range uint64Values {
			var div, mod Uint64
			dodiv := false
			var zero Uint64
			if b.Cmp(zero) != 0 {	// b != 0
				div, mod = a.DivMod(b)
				dodiv = true
			}
			fmt.Fprintf(bout, "\ttestUint64Binary(%v, %v, %v, %v, %v, %v, %v, %v, %v, %v, %v, %v);\n",
				a, b, a.Plus(b), a.Minus(b), a.Times(b), div, mod,
				a.And(b), a.Or(b), a.Xor(b), a.AndNot(b), dodiv)
		}
		for _, b := range shiftValues {
			fmt.Fprintf(bout, "\ttestUint64Shift(%v, %v, %v, %v);\n",
				a, b, a.LeftShift64(b), a.RightShift64(b))
		}
		fmt.Fprintf(bout, "}\n")
	}
}

// Part 2 is tests of operations involving one variable and one constant.

const binaryConstL = "func test%vBinaryL%v(b, add, sub, mul, div, mod, and, or, xor, andnot %v, dodiv bool) {\n" +
	"	const a %v = %v;\n" +
	"	const typ = `%s`;\n" +
	"	if n, op, want := a + b, `+`, add; n != want { ok=false; println(typ, `const`, a, op, `var`, b, `=`, n, `should be`, want); }\n" +
	"	if n, op, want := a - b, `-`, sub; n != want { ok=false; println(typ, `const`, a, op, `var`, b, `=`, n, `should be`, want); }\n" +
	"	if n, op, want := a * b, `*`, mul; n != want { ok=false; println(typ, `const`, a, op, `var`, b, `=`, n, `should be`, want); }\n" +
	"	if dodiv {\n" +
	"		if n, op, want := a / b, `/`, div; n != want { ok=false; println(typ, `const`, a, op, `var`, b, `=`, n, `should be`, want); }\n" +
	"		if n, op, want := a %% b, `%%`, mod; n != want { ok=false; println(typ, `const`, a, op, `var`, b, `=`, n, `should be`, want); }\n" +
	"	}\n" +
	"	if n, op, want := a & b, `&`, and; n != want { ok=false; println(typ, `const`, a, op, `var`, b, `=`, n, `should be`, want); }\n" +
	"	if n, op, want := a | b, `|`, or; n != want { ok=false; println(typ, `const`, a, op, `var`, b, `=`, n, `should be`, want); }\n" +
	"	if n, op, want := a ^ b, `^`, xor; n != want { ok=false; println(typ, `const`, a, op, `var`, b, `=`, n, `should be`, want); }\n" +
	"	if n, op, want := a &^ b, `&^`, andnot; n != want { ok=false; println(typ, `const`, a, op, `var`, b, `=`, n, `should be`, want); }\n" +
	"}\n" +
	"\n"

const binaryConstR = "func test%vBinaryR%v(a, add, sub, mul, div, mod, and, or, xor, andnot %v, dodiv bool) {\n" +
	"	const b %v = %v;\n" +
	"	const typ = `%s`;\n" +
	"	if n, op, want := a + b, `+`, add; n != want { ok=false; println(typ, `var`, a, op, `const`, b, `=`, n, `should be`, want); }\n" +
	"	if n, op, want := a - b, `-`, sub; n != want { ok=false; println(typ, `var`, a, op, `const`, b, `=`, n, `should be`, want); }\n" +
	"	if n, op, want := a * b, `*`, mul; n != want { ok=false; println(typ, `var`, a, op, `const`, b, `=`, n, `should be`, want); }\n" +
	"	if dodiv {\n" +
	"		if n, op, want := a / b, `/`, div; n != want { ok=false; println(typ, `var`, a, op, `const`, b, `=`, n, `should be`, want); }\n" +
	"		if n, op, want := a %% b, `%%`, mod; n != want { ok=false; println(typ, `var`, a, op, `const`, b, `=`, n, `should be`, want); }\n" +
	"	}\n" +
	"	if n, op, want := a & b, `&`, and; n != want { ok=false; println(typ, `var`, a, op, `const`, b, `=`, n, `should be`, want); }\n" +
	"	if n, op, want := a | b, `|`, or; n != want { ok=false; println(typ, `var`, a, op, `const`, b, `=`, n, `should be`, want); }\n" +
	"	if n, op, want := a ^ b, `^`, xor; n != want { ok=false; println(typ, `var`, a, op, `const`, b, `=`, n, `should be`, want); }\n" +
	"	if n, op, want := a &^ b, `&^`, andnot; n != want { ok=false; println(typ, `var`, a, op, `const`, b, `=`, n, `should be`, want); }\n" +
	"}\n" +
	"\n"

const shiftConstL = "func test%vShiftL%v(b uint64, left, right %v) {\n" +
	"	const a %v = %v;\n" +
	"	const typ = `%s`;\n" +
	"	if n, op, s, want := a << b, `<<`, b, left; n != want { ok=false; println(typ, `const`, a, op, `var`, s, `=`, n, `should be`, want); }\n" +
	"	if n, op, s, want := a >> b, `>>`, b, right; n != want { ok=false; println(typ, `const`, a, op, `var`, s, `=`, n, `should be`, want); }\n" +
	"	if uint64(uint32(b)) == b {\n" +
	"		b := uint32(b);\n" +
	"		if n, op, s, want := a << b, `<<`, b, left; n != want { ok=false; println(typ, `const`, a, op, `var`, s, `=`, n, `should be`, want); }\n" +
	"		if n, op, s, want := a >> b, `>>`, b, right; n != want { ok=false; println(typ, `const`, a, op, `var`, s, `=`, n, `should be`, want); }\n" +
	"	}\n" +
	"}\n"

const shiftConstR = "func test%vShiftR%v(a, left, right %v) {\n" +
	"	const b uint64 = %v;\n" +
	"	const typ = `%s`;\n" +
	"	if n, op, s, want := a << b, `<<`, b, left; n != want { ok=false; println(typ, `var`, a, op, `const`, s, `=`, n, `should be`, want); }\n" +
	"	if n, op, s, want := a >> b, `>>`, b, right; n != want { ok=false; println(typ, `var`, a, op, `const`, s, `=`, n, `should be`, want); }\n" +
	"	if b & 0xffffffff == b {\n" +
	"		const b = uint32(b & 0xffffffff);\n" +
	"		if n, op, s, want := a << b, `<<`, b, left; n != want { ok=false; println(typ, `var`, a, op, `const`, s, `=`, n, `should be`, want); }\n" +
	"		if n, op, s, want := a >> b, `>>`, b, right; n != want { ok=false; println(typ, `var`, a, op, `const`, s, `=`, n, `should be`, want); }\n" +
	"	}\n" +
	"}\n"

func constTests() {
	for i, a := range int64Values {
		fmt.Fprintf(bout, binaryConstL, "Int64", i, "int64", "int64", a, "int64")
		fmt.Fprintf(bout, binaryConstR, "Int64", i, "int64", "int64", a, "int64")
		fmt.Fprintf(bout, shiftConstL, "Int64", i, "int64", "int64", a, "int64")
	}
	for i, a := range uint64Values {
		fmt.Fprintf(bout, binaryConstL, "Uint64", i, "uint64", "uint64", a, "uint64")
		fmt.Fprintf(bout, binaryConstR, "Uint64", i, "uint64", "uint64", a, "uint64")
		fmt.Fprintf(bout, shiftConstL, "Uint64", i, "uint64", "uint64", a, "uint64")
	}
	for i, a := range shiftValues {
		fmt.Fprintf(bout, shiftConstR, "Int64", i, "int64", a, "int64")
		fmt.Fprintf(bout, shiftConstR, "Uint64", i, "uint64", a, "uint64")
	}
	for i, a := range int64Values {
		fmt.Fprintf(bout, "func test%v() {\n", ntest)
		ntest++
		for j, b := range int64Values {
			var div, mod Int64
			dodiv := false
			var zero Int64
			if b.Cmp(zero) != 0 {	// b != 0
				// Can't divide by zero but also can't divide -0x8000...000 by -1.
				var bigneg = Int64{-0x80000000, 0}
				var minus1 = Int64{-1, ^uint32(0)}
				if a.Cmp(bigneg) != 0 || b.Cmp(minus1) != 0 {	// a != -1<<63 || b != -1
					div, mod = a.DivMod(b)
					dodiv = true
				}
			}
			fmt.Fprintf(bout, "\ttestInt64BinaryL%v(%v, %v, %v, %v, %v, %v, %v, %v, %v, %v, %v);\n",
				i, b, a.Plus(b), a.Minus(b), a.Times(b), div, mod,
				a.And(b), a.Or(b), a.Xor(b), a.AndNot(b), dodiv)
			fmt.Fprintf(bout, "\ttestInt64BinaryR%v(%v, %v, %v, %v, %v, %v, %v, %v, %v, %v, %v);\n",
				j, a, a.Plus(b), a.Minus(b), a.Times(b), div, mod,
				a.And(b), a.Or(b), a.Xor(b), a.AndNot(b), dodiv)
		}
		for j, b := range shiftValues {
			fmt.Fprintf(bout, "\ttestInt64ShiftL%v(%v, %v, %v);\n",
				i, b, a.LeftShift64(b), a.RightShift64(b))
			fmt.Fprintf(bout, "\ttestInt64ShiftR%v(%v, %v, %v);\n",
				j, a, a.LeftShift64(b), a.RightShift64(b))
		}
		fmt.Fprintf(bout, "}\n")
	}
	for i, a := range uint64Values {
		fmt.Fprintf(bout, "func test%v() {\n", ntest)
		ntest++
		for j, b := range uint64Values {
			var div, mod Uint64
			dodiv := false
			var zero Uint64
			if b.Cmp(zero) != 0 {	// b != 0
				div, mod = a.DivMod(b)
				dodiv = true
			}
			fmt.Fprintf(bout, "\ttestUint64BinaryL%v(%v, %v, %v, %v, %v, %v, %v, %v, %v, %v, %v);\n",
				i, b, a.Plus(b), a.Minus(b), a.Times(b), div, mod,
				a.And(b), a.Or(b), a.Xor(b), a.AndNot(b), dodiv)
			fmt.Fprintf(bout, "\ttestUint64BinaryR%v(%v, %v, %v, %v, %v, %v, %v, %v, %v, %v, %v);\n",
				j, a, a.Plus(b), a.Minus(b), a.Times(b), div, mod,
				a.And(b), a.Or(b), a.Xor(b), a.AndNot(b), dodiv)
		}
		for j, b := range shiftValues {
			fmt.Fprintf(bout, "\ttestUint64ShiftL%v(%v, %v, %v);\n",
				i, b, a.LeftShift64(b), a.RightShift64(b))
			fmt.Fprintf(bout, "\ttestUint64ShiftR%v(%v, %v, %v);\n",
				j, a, a.LeftShift64(b), a.RightShift64(b))
		}
		fmt.Fprintf(bout, "}\n")
	}
}

func main() {
	bout = bufio.NewWriter(os.Stdout)
	varTests()
	constTests()

	fmt.Fprintf(bout, "func main() {\n")
	for i := 0; i < ntest; i++ {
		fmt.Fprintf(bout, "\ttest%v();\n", i)
	}
	fmt.Fprintf(bout, "\tif !ok { os.Exit(1) }\n")
	fmt.Fprintf(bout, "}\n")
	bout.Flush()
}
