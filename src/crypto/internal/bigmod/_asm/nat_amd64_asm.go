// Copyright 2022 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package main

import (
	. "github.com/mmcloughlin/avo/build"
	. "github.com/mmcloughlin/avo/operand"
	. "github.com/mmcloughlin/avo/reg"
)

//go:generate go run . -out ../nat_amd64.s -stubs ../nat_amd64.go -pkg bigmod

func main() {
	Package("crypto/internal/bigmod")
	ConstraintExpr("amd64,gc,!purego")

	Implement("montgomeryLoop")
	Pragma("noescape")

	size := Load(Param("d").Len(), GP64())
	d := Mem{Base: Load(Param("d").Base(), GP64())}
	b := Mem{Base: Load(Param("b").Base(), GP64())}
	m := Mem{Base: Load(Param("m").Base(), GP64())}
	m0inv := Load(Param("m0inv"), GP64())

	overflow := zero()
	i := zero()
	Label("outerLoop")

	ai := Load(Param("a").Base(), GP64())
	MOVQ(Mem{Base: ai}.Idx(i, 8), ai)

	z := uint128{GP64(), GP64()}
	mul64(z, b, ai)
	add64(z, d)
	f := GP64()
	MOVQ(m0inv, f)
	IMULQ(z.lo, f)
	_MASK(f)
	addMul64(z, m, f)
	carry := shiftBy63(z)

	j := zero()
	INCQ(j)
	JMP(LabelRef("innerLoopCondition"))
	Label("innerLoop")

	// z = d[j] + a[i] * b[j] + f * m[j] + carry
	z = uint128{GP64(), GP64()}
	mul64(z, b.Idx(j, 8), ai)
	addMul64(z, m.Idx(j, 8), f)
	add64(z, d.Idx(j, 8))
	add64(z, carry)
	// d[j-1] = z_lo & _MASK
	storeMasked(z.lo, d.Idx(j, 8).Offset(-8))
	// carry = z_hi<<1 | z_lo>>_W
	MOVQ(shiftBy63(z), carry)

	INCQ(j)
	Label("innerLoopCondition")
	CMPQ(size, j)
	JGT(LabelRef("innerLoop"))

	ADDQ(carry, overflow)
	storeMasked(overflow, d.Idx(size, 8).Offset(-8))
	SHRQ(Imm(63), overflow)

	INCQ(i)
	CMPQ(size, i)
	JGT(LabelRef("outerLoop"))

	Store(overflow, ReturnIndex(0))
	RET()
	Generate()
}

// zero zeroes a new register and returns it.
func zero() Register {
	r := GP64()
	XORQ(r, r)
	return r
}

// _MASK masks out the top bit of r.
func _MASK(r Register) {
	BTRQ(Imm(63), r)
}

type uint128 struct {
	hi, lo GPVirtual
}

// storeMasked stores _MASK(src) in dst. It doesn't modify src.
func storeMasked(src, dst Op) {
	out := GP64()
	MOVQ(src, out)
	_MASK(out)
	MOVQ(out, dst)
}

// shiftBy63 returns z >> 63. It reuses z.lo.
func shiftBy63(z uint128) Register {
	SHRQ(Imm(63), z.hi, z.lo)
	result := z.lo
	z.hi, z.lo = nil, nil
	return result
}

// add64 sets r to r + a.
func add64(r uint128, a Op) {
	ADDQ(a, r.lo)
	ADCQ(Imm(0), r.hi)
}

// mul64 sets r to a * b.
func mul64(r uint128, a, b Op) {
	MOVQ(a, RAX)
	MULQ(b) // RDX, RAX = RAX * b
	MOVQ(RAX, r.lo)
	MOVQ(RDX, r.hi)
}

// addMul64 sets r to r + a * b.
func addMul64(r uint128, a, b Op) {
	MOVQ(a, RAX)
	MULQ(b) // RDX, RAX = RAX * b
	ADDQ(RAX, r.lo)
	ADCQ(RDX, r.hi)
}
