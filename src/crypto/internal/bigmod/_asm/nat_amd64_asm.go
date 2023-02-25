// Copyright 2023 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package main

import (
	"strconv"

	. "github.com/mmcloughlin/avo/build"
	. "github.com/mmcloughlin/avo/operand"
	. "github.com/mmcloughlin/avo/reg"
)

//go:generate go run . -out ../nat_amd64.s -pkg bigmod

func main() {
	Package("crypto/internal/bigmod")
	ConstraintExpr("!purego")

	addMulVVW(1024)
	addMulVVW(1536)
	addMulVVW(2048)

	Generate()
}

func addMulVVW(bits int) {
	if bits%64 != 0 {
		panic("bit size unsupported")
	}

	Implement("addMulVVW" + strconv.Itoa(bits))

	CMPB(Mem{Symbol: Symbol{Name: "·supportADX"}, Base: StaticBase}, Imm(1))
	JEQ(LabelRef("adx"))

	z := Mem{Base: Load(Param("z"), GP64())}
	x := Mem{Base: Load(Param("x"), GP64())}
	y := Load(Param("y"), GP64())

	carry := GP64()
	XORQ(carry, carry) // zero out carry

	for i := 0; i < bits/64; i++ {
		Comment("Iteration " + strconv.Itoa(i))
		hi, lo := RDX, RAX // implicit MULQ inputs and outputs
		MOVQ(x.Offset(i*8), lo)
		MULQ(y)
		ADDQ(z.Offset(i*8), lo)
		ADCQ(Imm(0), hi)
		ADDQ(carry, lo)
		ADCQ(Imm(0), hi)
		MOVQ(hi, carry)
		MOVQ(lo, z.Offset(i*8))
	}

	Store(carry, ReturnIndex(0))
	RET()

	Label("adx")

	// The ADX strategy implements the following function, where c1 and c2 are
	// the overflow and the carry flag respectively.
	//
	//    func addMulVVW(z, x []uint, y uint) (carry uint) {
	//        var c1, c2 uint
	//        for i := range z {
	//            hi, lo := bits.Mul(x[i], y)
	//            lo, c1 = bits.Add(lo, z[i], c1)
	//            z[i], c2 = bits.Add(lo, carry, c2)
	//            carry = hi
	//        }
	//        return carry + c1 + c2
	//    }
	//
	// The loop is fully unrolled and the hi / carry registers are alternated
	// instead of introducing a MOV.

	z = Mem{Base: Load(Param("z"), GP64())}
	x = Mem{Base: Load(Param("x"), GP64())}
	Load(Param("y"), RDX) // implicit source of MULXQ

	carry = GP64()
	XORQ(carry, carry) // zero out carry
	z0 := GP64()
	XORQ(z0, z0) // unset flags and zero out z0

	for i := 0; i < bits/64; i++ {
		hi, lo := GP64(), GP64()

		Comment("Iteration " + strconv.Itoa(i))
		MULXQ(x.Offset(i*8), lo, hi)
		ADCXQ(carry, lo)
		ADOXQ(z.Offset(i*8), lo)
		MOVQ(lo, z.Offset(i*8))

		i++

		Comment("Iteration " + strconv.Itoa(i))
		MULXQ(x.Offset(i*8), lo, carry)
		ADCXQ(hi, lo)
		ADOXQ(z.Offset(i*8), lo)
		MOVQ(lo, z.Offset(i*8))
	}

	Comment("Add back carry flags and return")
	ADCXQ(z0, carry)
	ADOXQ(z0, carry)

	Store(carry, ReturnIndex(0))
	RET()
}
