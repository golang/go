// Copyright (c) 2021 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package main

import (
	"fmt"

	. "github.com/mmcloughlin/avo/build"
	. "github.com/mmcloughlin/avo/gotypes"
	. "github.com/mmcloughlin/avo/operand"
	. "github.com/mmcloughlin/avo/reg"
)

//go:generate go run . -out ../fe_amd64.s -stubs ../fe_amd64.go -pkg field

func main() {
	Package("crypto/internal/fips140/edwards25519/field")
	ConstraintExpr("!purego")
	feMul()
	feSquare()
	Generate()
}

type namedComponent struct {
	Component
	name string
}

func (c namedComponent) String() string { return c.name }

type uint128 struct {
	name   string
	hi, lo GPVirtual
}

func (c uint128) String() string { return c.name }

func feSquare() {
	TEXT("feSquare", NOSPLIT, "func(out, a *Element)")
	Doc("feSquare sets out = a * a. It works like feSquareGeneric.")
	Pragma("noescape")

	a := Dereference(Param("a"))
	l0 := namedComponent{a.Field("l0"), "l0"}
	l1 := namedComponent{a.Field("l1"), "l1"}
	l2 := namedComponent{a.Field("l2"), "l2"}
	l3 := namedComponent{a.Field("l3"), "l3"}
	l4 := namedComponent{a.Field("l4"), "l4"}

	// r0 = l0×l0 + 19×2×(l1×l4 + l2×l3)
	r0 := uint128{"r0", GP64(), GP64()}
	mul64(r0, 1, l0, l0)
	addMul64(r0, 38, l1, l4)
	addMul64(r0, 38, l2, l3)

	// r1 = 2×l0×l1 + 19×2×l2×l4 + 19×l3×l3
	r1 := uint128{"r1", GP64(), GP64()}
	mul64(r1, 2, l0, l1)
	addMul64(r1, 38, l2, l4)
	addMul64(r1, 19, l3, l3)

	// r2 = = 2×l0×l2 + l1×l1 + 19×2×l3×l4
	r2 := uint128{"r2", GP64(), GP64()}
	mul64(r2, 2, l0, l2)
	addMul64(r2, 1, l1, l1)
	addMul64(r2, 38, l3, l4)

	// r3 = = 2×l0×l3 + 2×l1×l2 + 19×l4×l4
	r3 := uint128{"r3", GP64(), GP64()}
	mul64(r3, 2, l0, l3)
	addMul64(r3, 2, l1, l2)
	addMul64(r3, 19, l4, l4)

	// r4 = = 2×l0×l4 + 2×l1×l3 + l2×l2
	r4 := uint128{"r4", GP64(), GP64()}
	mul64(r4, 2, l0, l4)
	addMul64(r4, 2, l1, l3)
	addMul64(r4, 1, l2, l2)

	Comment("First reduction chain")
	maskLow51Bits := GP64()
	MOVQ(Imm((1<<51)-1), maskLow51Bits)
	c0, r0lo := shiftRightBy51(&r0)
	c1, r1lo := shiftRightBy51(&r1)
	c2, r2lo := shiftRightBy51(&r2)
	c3, r3lo := shiftRightBy51(&r3)
	c4, r4lo := shiftRightBy51(&r4)
	maskAndAdd(r0lo, maskLow51Bits, c4, 19)
	maskAndAdd(r1lo, maskLow51Bits, c0, 1)
	maskAndAdd(r2lo, maskLow51Bits, c1, 1)
	maskAndAdd(r3lo, maskLow51Bits, c2, 1)
	maskAndAdd(r4lo, maskLow51Bits, c3, 1)

	Comment("Second reduction chain (carryPropagate)")
	// c0 = r0 >> 51
	MOVQ(r0lo, c0)
	SHRQ(Imm(51), c0)
	// c1 = r1 >> 51
	MOVQ(r1lo, c1)
	SHRQ(Imm(51), c1)
	// c2 = r2 >> 51
	MOVQ(r2lo, c2)
	SHRQ(Imm(51), c2)
	// c3 = r3 >> 51
	MOVQ(r3lo, c3)
	SHRQ(Imm(51), c3)
	// c4 = r4 >> 51
	MOVQ(r4lo, c4)
	SHRQ(Imm(51), c4)
	maskAndAdd(r0lo, maskLow51Bits, c4, 19)
	maskAndAdd(r1lo, maskLow51Bits, c0, 1)
	maskAndAdd(r2lo, maskLow51Bits, c1, 1)
	maskAndAdd(r3lo, maskLow51Bits, c2, 1)
	maskAndAdd(r4lo, maskLow51Bits, c3, 1)

	Comment("Store output")
	out := Dereference(Param("out"))
	Store(r0lo, out.Field("l0"))
	Store(r1lo, out.Field("l1"))
	Store(r2lo, out.Field("l2"))
	Store(r3lo, out.Field("l3"))
	Store(r4lo, out.Field("l4"))

	RET()
}

func feMul() {
	TEXT("feMul", NOSPLIT, "func(out, a, b *Element)")
	Doc("feMul sets out = a * b. It works like feMulGeneric.")
	Pragma("noescape")

	a := Dereference(Param("a"))
	a0 := namedComponent{a.Field("l0"), "a0"}
	a1 := namedComponent{a.Field("l1"), "a1"}
	a2 := namedComponent{a.Field("l2"), "a2"}
	a3 := namedComponent{a.Field("l3"), "a3"}
	a4 := namedComponent{a.Field("l4"), "a4"}

	b := Dereference(Param("b"))
	b0 := namedComponent{b.Field("l0"), "b0"}
	b1 := namedComponent{b.Field("l1"), "b1"}
	b2 := namedComponent{b.Field("l2"), "b2"}
	b3 := namedComponent{b.Field("l3"), "b3"}
	b4 := namedComponent{b.Field("l4"), "b4"}

	// r0 = a0×b0 + 19×(a1×b4 + a2×b3 + a3×b2 + a4×b1)
	r0 := uint128{"r0", GP64(), GP64()}
	mul64(r0, 1, a0, b0)
	addMul64(r0, 19, a1, b4)
	addMul64(r0, 19, a2, b3)
	addMul64(r0, 19, a3, b2)
	addMul64(r0, 19, a4, b1)

	// r1 = a0×b1 + a1×b0 + 19×(a2×b4 + a3×b3 + a4×b2)
	r1 := uint128{"r1", GP64(), GP64()}
	mul64(r1, 1, a0, b1)
	addMul64(r1, 1, a1, b0)
	addMul64(r1, 19, a2, b4)
	addMul64(r1, 19, a3, b3)
	addMul64(r1, 19, a4, b2)

	// r2 = a0×b2 + a1×b1 + a2×b0 + 19×(a3×b4 + a4×b3)
	r2 := uint128{"r2", GP64(), GP64()}
	mul64(r2, 1, a0, b2)
	addMul64(r2, 1, a1, b1)
	addMul64(r2, 1, a2, b0)
	addMul64(r2, 19, a3, b4)
	addMul64(r2, 19, a4, b3)

	// r3 = a0×b3 + a1×b2 + a2×b1 + a3×b0 + 19×a4×b4
	r3 := uint128{"r3", GP64(), GP64()}
	mul64(r3, 1, a0, b3)
	addMul64(r3, 1, a1, b2)
	addMul64(r3, 1, a2, b1)
	addMul64(r3, 1, a3, b0)
	addMul64(r3, 19, a4, b4)

	// r4 = a0×b4 + a1×b3 + a2×b2 + a3×b1 + a4×b0
	r4 := uint128{"r4", GP64(), GP64()}
	mul64(r4, 1, a0, b4)
	addMul64(r4, 1, a1, b3)
	addMul64(r4, 1, a2, b2)
	addMul64(r4, 1, a3, b1)
	addMul64(r4, 1, a4, b0)

	Comment("First reduction chain")
	maskLow51Bits := GP64()
	MOVQ(Imm((1<<51)-1), maskLow51Bits)
	c0, r0lo := shiftRightBy51(&r0)
	c1, r1lo := shiftRightBy51(&r1)
	c2, r2lo := shiftRightBy51(&r2)
	c3, r3lo := shiftRightBy51(&r3)
	c4, r4lo := shiftRightBy51(&r4)
	maskAndAdd(r0lo, maskLow51Bits, c4, 19)
	maskAndAdd(r1lo, maskLow51Bits, c0, 1)
	maskAndAdd(r2lo, maskLow51Bits, c1, 1)
	maskAndAdd(r3lo, maskLow51Bits, c2, 1)
	maskAndAdd(r4lo, maskLow51Bits, c3, 1)

	Comment("Second reduction chain (carryPropagate)")
	// c0 = r0 >> 51
	MOVQ(r0lo, c0)
	SHRQ(Imm(51), c0)
	// c1 = r1 >> 51
	MOVQ(r1lo, c1)
	SHRQ(Imm(51), c1)
	// c2 = r2 >> 51
	MOVQ(r2lo, c2)
	SHRQ(Imm(51), c2)
	// c3 = r3 >> 51
	MOVQ(r3lo, c3)
	SHRQ(Imm(51), c3)
	// c4 = r4 >> 51
	MOVQ(r4lo, c4)
	SHRQ(Imm(51), c4)
	maskAndAdd(r0lo, maskLow51Bits, c4, 19)
	maskAndAdd(r1lo, maskLow51Bits, c0, 1)
	maskAndAdd(r2lo, maskLow51Bits, c1, 1)
	maskAndAdd(r3lo, maskLow51Bits, c2, 1)
	maskAndAdd(r4lo, maskLow51Bits, c3, 1)

	Comment("Store output")
	out := Dereference(Param("out"))
	Store(r0lo, out.Field("l0"))
	Store(r1lo, out.Field("l1"))
	Store(r2lo, out.Field("l2"))
	Store(r3lo, out.Field("l3"))
	Store(r4lo, out.Field("l4"))

	RET()
}

// mul64 sets r to i * aX * bX.
func mul64(r uint128, i int, aX, bX namedComponent) {
	switch i {
	case 1:
		Comment(fmt.Sprintf("%s = %s×%s", r, aX, bX))
		Load(aX, RAX)
	case 2:
		Comment(fmt.Sprintf("%s = 2×%s×%s", r, aX, bX))
		Load(aX, RAX)
		SHLQ(Imm(1), RAX)
	default:
		panic("unsupported i value")
	}
	MULQ(mustAddr(bX)) // RDX, RAX = RAX * bX
	MOVQ(RAX, r.lo)
	MOVQ(RDX, r.hi)
}

// addMul64 sets r to r + i * aX * bX.
func addMul64(r uint128, i uint64, aX, bX namedComponent) {
	switch i {
	case 1:
		Comment(fmt.Sprintf("%s += %s×%s", r, aX, bX))
		Load(aX, RAX)
	case 2:
		Comment(fmt.Sprintf("%s += %d×%s×%s", r, i, aX, bX))
		Load(aX, RAX)
		SHLQ(U8(1), RAX)
	case 19:
		Comment(fmt.Sprintf("%s += %d×%s×%s", r, i, aX, bX))
		// 19 * v ==> v + (v+v*8)*2
		tmp := Load(aX, GP64())
		LEAQ(Mem{Base: tmp, Index: tmp, Scale: 8}, RAX)
		LEAQ(Mem{Base: tmp, Index: RAX, Scale: 2}, RAX)
	case 38:
		Comment(fmt.Sprintf("%s += %d×%s×%s", r, i, aX, bX))
		// 38 * v ==> (v + (v+v*8)*2) * 2
		tmp := Load(aX, GP64())
		LEAQ(Mem{Base: tmp, Index: tmp, Scale: 8}, RAX)
		LEAQ(Mem{Base: tmp, Index: RAX, Scale: 2}, RAX)
		SHLQ(U8(1), RAX)
	default:
		Comment(fmt.Sprintf("%s += %d×%s×%s", r, i, aX, bX))
		IMUL3Q(Imm(i), Load(aX, GP64()), RAX)
	}
	MULQ(mustAddr(bX)) // RDX, RAX = RAX * bX
	ADDQ(RAX, r.lo)
	ADCQ(RDX, r.hi)
}

// shiftRightBy51 returns r >> 51 and r.lo.
//
// After this function is called, the uint128 may not be used anymore.
func shiftRightBy51(r *uint128) (out, lo GPVirtual) {
	out = r.hi
	lo = r.lo
	SHLQ(Imm(64-51), r.lo, r.hi)
	r.lo, r.hi = nil, nil // make sure the uint128 is unusable
	return
}

// maskAndAdd sets r = r&mask + c*i.
func maskAndAdd(r, mask, c GPVirtual, i uint64) {
	ANDQ(mask, r)
	if i != 1 {
		IMUL3Q(Imm(i), c, c)
	}
	ADDQ(c, r)
}

func mustAddr(c Component) Op {
	b, err := c.Resolve()
	if err != nil {
		panic(err)
	}
	return b.Addr
}
