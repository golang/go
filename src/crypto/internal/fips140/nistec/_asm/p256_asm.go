// Copyright 2024 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// This file contains constant-time, 64-bit assembly implementation of
// P256. The optimizations performed here are described in detail in:
// S.Gueron and V.Krasnov, "Fast prime field elliptic-curve cryptography with
//                          256-bit primes"
// https://link.springer.com/article/10.1007%2Fs13389-014-0090-x
// https://eprint.iacr.org/2013/816.pdf

package main

import (
	"os"
	"strings"

	. "github.com/mmcloughlin/avo/build"
	"github.com/mmcloughlin/avo/ir"
	. "github.com/mmcloughlin/avo/operand"
	. "github.com/mmcloughlin/avo/reg"
)

//go:generate go run . -out ../p256_asm_amd64.s

var (
	res_ptr GPPhysical = RDI
	x_ptr              = RSI
	y_ptr              = RCX
)

// These variables have been versioned as they get redfined in the reference implementation.
// This is done to produce a minimal semantic diff.
var (
	acc0_v1 GPPhysical = R8
	acc1_v1            = R9
	acc2_v1            = R10
	acc3_v1            = R11
	acc4_v1            = R12
	acc5_v1            = R13
	t0_v1              = R14
	t1_v1              = R15
)

func main() {
	Package("crypto/internal/fips140/nistec")
	ConstraintExpr("!purego")
	p256MovCond()
	p256NegCond()
	p256Sqr()
	p256Mul()
	p256FromMont()
	p256Select()
	p256SelectAffine()
	p256OrdMul()
	p256OrdSqr()
	p256SubInternal()
	p256MulInternal()
	p256SqrInternal()
	p256PointAddAffineAsm()
	p256IsZero()
	p256PointAddAsm()
	p256PointDoubleAsm()
	Generate()

	internalFunctions := []string{
		"·p256SubInternal",
		"·p256MulInternal",
		"·p256SqrInternal",
		"·p256IsZero",
	}
	removePeskyUnicodeDot(internalFunctions, "../p256_asm_amd64.s")
}

// Implements:
//
//	func p256MovCond(res, a, b *P256Point, cond int)
func p256MovCond() {
	Implement("p256MovCond")
	Attributes(NOSPLIT)

	Load(Param("res"), res_ptr)
	Load(Param("a"), x_ptr)
	Load(Param("b"), y_ptr)
	Load(Param("cond"), X12)

	PXOR(X13, X13)
	PSHUFD(Imm(0), X12, X12)
	PCMPEQL(X13, X12)

	MOVOU(X12, X0)
	MOVOU(Mem{Base: x_ptr}.Offset(16*0), X6)
	PANDN(X6, X0)
	MOVOU(X12, X1)
	MOVOU(Mem{Base: x_ptr}.Offset(16*1), X7)
	PANDN(X7, X1)
	MOVOU(X12, X2)
	MOVOU(Mem{Base: x_ptr}.Offset(16*2), X8)
	PANDN(X8, X2)
	MOVOU(X12, X3)
	MOVOU(Mem{Base: x_ptr}.Offset(16*3), X9)
	PANDN(X9, X3)
	MOVOU(X12, X4)
	MOVOU(Mem{Base: x_ptr}.Offset(16*4), X10)
	PANDN(X10, X4)
	MOVOU(X12, X5)
	MOVOU(Mem{Base: x_ptr}.Offset(16*5), X11)
	PANDN(X11, X5)

	MOVOU(Mem{Base: y_ptr}.Offset(16*0), X6)
	MOVOU(Mem{Base: y_ptr}.Offset(16*1), X7)
	MOVOU(Mem{Base: y_ptr}.Offset(16*2), X8)
	MOVOU(Mem{Base: y_ptr}.Offset(16*3), X9)
	MOVOU(Mem{Base: y_ptr}.Offset(16*4), X10)
	MOVOU(Mem{Base: y_ptr}.Offset(16*5), X11)

	PAND(X12, X6)
	PAND(X12, X7)
	PAND(X12, X8)
	PAND(X12, X9)
	PAND(X12, X10)
	PAND(X12, X11)

	PXOR(X6, X0)
	PXOR(X7, X1)
	PXOR(X8, X2)
	PXOR(X9, X3)
	PXOR(X10, X4)
	PXOR(X11, X5)

	MOVOU(X0, Mem{Base: res_ptr}.Offset(16*0))
	MOVOU(X1, Mem{Base: res_ptr}.Offset(16*1))
	MOVOU(X2, Mem{Base: res_ptr}.Offset(16*2))
	MOVOU(X3, Mem{Base: res_ptr}.Offset(16*3))
	MOVOU(X4, Mem{Base: res_ptr}.Offset(16*4))
	MOVOU(X5, Mem{Base: res_ptr}.Offset(16*5))

	RET()
}

// Implements:
//
//	func p256NegCond(val *p256Element, cond int)
func p256NegCond() {
	Implement("p256NegCond")
	Attributes(NOSPLIT)

	Load(Param("val"), res_ptr)
	Load(Param("cond"), t0_v1)

	Comment("acc = poly")
	MOVQ(I32(-1), acc0_v1)
	p256const0 := p256const0_DATA()
	MOVQ(p256const0, acc1_v1)
	MOVQ(I32(0), acc2_v1)
	p256const1 := p256const1_DATA()
	MOVQ(p256const1, acc3_v1)

	Comment("Load the original value")
	MOVQ(Mem{Base: res_ptr}.Offset(8*0), acc5_v1)
	MOVQ(Mem{Base: res_ptr}.Offset(8*1), x_ptr)
	MOVQ(Mem{Base: res_ptr}.Offset(8*2), y_ptr)
	MOVQ(Mem{Base: res_ptr}.Offset(8*3), t1_v1)

	Comment("Speculatively subtract")
	SUBQ(acc5_v1, acc0_v1)
	SBBQ(x_ptr, acc1_v1)
	SBBQ(y_ptr, acc2_v1)
	SBBQ(t1_v1, acc3_v1)

	Comment("If condition is 0, keep original value")
	TESTQ(t0_v1, t0_v1)
	CMOVQEQ(acc5_v1, acc0_v1)
	CMOVQEQ(x_ptr, acc1_v1)
	CMOVQEQ(y_ptr, acc2_v1)
	CMOVQEQ(t1_v1, acc3_v1)

	Comment("Store result")
	MOVQ(acc0_v1, Mem{Base: res_ptr}.Offset(8*0))
	MOVQ(acc1_v1, Mem{Base: res_ptr}.Offset(8*1))
	MOVQ(acc2_v1, Mem{Base: res_ptr}.Offset(8*2))
	MOVQ(acc3_v1, Mem{Base: res_ptr}.Offset(8*3))

	RET()
}

// Implements:
//
//	func p256Sqr(res, in *p256Element, n int)
func p256Sqr() {
	Implement("p256Sqr")
	Attributes(NOSPLIT)

	Load(Param("res"), res_ptr)
	Load(Param("in"), x_ptr)
	Load(Param("n"), RBX)

	Label("sqrLoop")

	Comment("y[1:] * y[0]")
	MOVQ(Mem{Base: x_ptr}.Offset(8*0), t0_v1)

	MOVQ(Mem{Base: x_ptr}.Offset(8*1), RAX)
	MULQ(t0_v1)
	MOVQ(RAX, acc1_v1)
	MOVQ(RDX, acc2_v1)

	MOVQ(Mem{Base: x_ptr}.Offset(8*2), RAX)
	MULQ(t0_v1)
	ADDQ(RAX, acc2_v1)
	ADCQ(Imm(0), RDX)
	MOVQ(RDX, acc3_v1)

	MOVQ(Mem{Base: x_ptr}.Offset(8*3), RAX)
	MULQ(t0_v1)
	ADDQ(RAX, acc3_v1)
	ADCQ(Imm(0), RDX)
	MOVQ(RDX, acc4_v1)

	Comment("y[2:] * y[1]")
	MOVQ(Mem{Base: x_ptr}.Offset(8*1), t0_v1)

	MOVQ(Mem{Base: x_ptr}.Offset(8*2), RAX)
	MULQ(t0_v1)
	ADDQ(RAX, acc3_v1)
	ADCQ(Imm(0), RDX)
	MOVQ(RDX, t1_v1)

	MOVQ(Mem{Base: x_ptr}.Offset(8*3), RAX)
	MULQ(t0_v1)
	ADDQ(t1_v1, acc4_v1)
	ADCQ(Imm(0), RDX)
	ADDQ(RAX, acc4_v1)
	ADCQ(Imm(0), RDX)
	MOVQ(RDX, acc5_v1)

	Comment("y[3] * y[2]")
	MOVQ(Mem{Base: x_ptr}.Offset(8*2), t0_v1)

	MOVQ(Mem{Base: x_ptr}.Offset(8*3), RAX)
	MULQ(t0_v1)
	ADDQ(RAX, acc5_v1)
	ADCQ(Imm(0), RDX)
	MOVQ(RDX, y_ptr)
	XORQ(t1_v1, t1_v1)

	Comment("*2")
	ADDQ(acc1_v1, acc1_v1)
	ADCQ(acc2_v1, acc2_v1)
	ADCQ(acc3_v1, acc3_v1)
	ADCQ(acc4_v1, acc4_v1)
	ADCQ(acc5_v1, acc5_v1)
	ADCQ(y_ptr, y_ptr)
	ADCQ(Imm(0), t1_v1)

	Comment("Missing products")
	MOVQ(Mem{Base: x_ptr}.Offset(8*0), RAX)
	MULQ(RAX)
	MOVQ(RAX, acc0_v1)
	MOVQ(RDX, t0_v1)

	MOVQ(Mem{Base: x_ptr}.Offset(8*1), RAX)
	MULQ(RAX)
	ADDQ(t0_v1, acc1_v1)
	ADCQ(RAX, acc2_v1)
	ADCQ(Imm(0), RDX)
	MOVQ(RDX, t0_v1)

	MOVQ(Mem{Base: x_ptr}.Offset(8*2), RAX)
	MULQ(RAX)
	ADDQ(t0_v1, acc3_v1)
	ADCQ(RAX, acc4_v1)
	ADCQ(Imm(0), RDX)
	MOVQ(RDX, t0_v1)

	MOVQ(Mem{Base: x_ptr}.Offset(8*3), RAX)
	MULQ(RAX)
	ADDQ(t0_v1, acc5_v1)
	ADCQ(RAX, y_ptr)
	ADCQ(RDX, t1_v1)
	MOVQ(t1_v1, x_ptr)

	Comment("First reduction step")
	MOVQ(acc0_v1, RAX)
	MOVQ(acc0_v1, t1_v1)
	SHLQ(Imm(32), acc0_v1)

	p256const1 := p256const1_DATA()
	MULQ(p256const1)

	SHRQ(Imm(32), t1_v1)
	ADDQ(acc0_v1, acc1_v1)
	ADCQ(t1_v1, acc2_v1)
	ADCQ(RAX, acc3_v1)
	ADCQ(Imm(0), RDX)
	MOVQ(RDX, acc0_v1)

	Comment("Second reduction step")
	MOVQ(acc1_v1, RAX)
	MOVQ(acc1_v1, t1_v1)
	SHLQ(Imm(32), acc1_v1)
	MULQ(p256const1)
	SHRQ(Imm(32), t1_v1)
	ADDQ(acc1_v1, acc2_v1)
	ADCQ(t1_v1, acc3_v1)
	ADCQ(RAX, acc0_v1)
	ADCQ(Imm(0), RDX)
	MOVQ(RDX, acc1_v1)

	Comment("Third reduction step")
	MOVQ(acc2_v1, RAX)
	MOVQ(acc2_v1, t1_v1)
	SHLQ(Imm(32), acc2_v1)
	MULQ(p256const1)
	SHRQ(Imm(32), t1_v1)
	ADDQ(acc2_v1, acc3_v1)
	ADCQ(t1_v1, acc0_v1)
	ADCQ(RAX, acc1_v1)
	ADCQ(Imm(0), RDX)
	MOVQ(RDX, acc2_v1)

	Comment("Last reduction step")
	XORQ(t0_v1, t0_v1)
	MOVQ(acc3_v1, RAX)
	MOVQ(acc3_v1, t1_v1)
	SHLQ(Imm(32), acc3_v1)
	MULQ(p256const1)
	SHRQ(Imm(32), t1_v1)
	ADDQ(acc3_v1, acc0_v1)
	ADCQ(t1_v1, acc1_v1)
	ADCQ(RAX, acc2_v1)
	ADCQ(Imm(0), RDX)
	MOVQ(RDX, acc3_v1)

	Comment("Add bits [511:256] of the sqr result")
	ADCQ(acc4_v1, acc0_v1)
	ADCQ(acc5_v1, acc1_v1)
	ADCQ(y_ptr, acc2_v1)
	ADCQ(x_ptr, acc3_v1)
	ADCQ(Imm(0), t0_v1)

	MOVQ(acc0_v1, acc4_v1)
	MOVQ(acc1_v1, acc5_v1)
	MOVQ(acc2_v1, y_ptr)
	MOVQ(acc3_v1, t1_v1)

	Comment("Subtract p256")
	SUBQ(I8(-1), acc0_v1)

	p256const0 := p256const0_DATA()
	SBBQ(p256const0, acc1_v1)
	SBBQ(Imm(0), acc2_v1)
	SBBQ(p256const1, acc3_v1)
	SBBQ(Imm(0), t0_v1)

	CMOVQCS(acc4_v1, acc0_v1)
	CMOVQCS(acc5_v1, acc1_v1)
	CMOVQCS(y_ptr, acc2_v1)
	CMOVQCS(t1_v1, acc3_v1)

	MOVQ(acc0_v1, Mem{Base: res_ptr}.Offset(8*0))
	MOVQ(acc1_v1, Mem{Base: res_ptr}.Offset(8*1))
	MOVQ(acc2_v1, Mem{Base: res_ptr}.Offset(8*2))
	MOVQ(acc3_v1, Mem{Base: res_ptr}.Offset(8*3))
	MOVQ(res_ptr, x_ptr)
	DECQ(RBX)
	JNE(LabelRef("sqrLoop"))

	RET()
}

// Implements:
//
//	func p256Mul(res, in1, in2 *p256Element)
func p256Mul() {
	Implement("p256Mul")
	Attributes(NOSPLIT)

	Load(Param("res"), res_ptr)
	Load(Param("in1"), x_ptr)
	Load(Param("in2"), y_ptr)

	Comment("x * y[0]")
	MOVQ(Mem{Base: y_ptr}.Offset(8*0), t0_v1)

	MOVQ(Mem{Base: x_ptr}.Offset(8*0), RAX)
	MULQ(t0_v1)
	MOVQ(RAX, acc0_v1)
	MOVQ(RDX, acc1_v1)

	MOVQ(Mem{Base: x_ptr}.Offset(8*1), RAX)
	MULQ(t0_v1)
	ADDQ(RAX, acc1_v1)
	ADCQ(Imm(0), RDX)
	MOVQ(RDX, acc2_v1)

	MOVQ(Mem{Base: x_ptr}.Offset(8*2), RAX)
	MULQ(t0_v1)
	ADDQ(RAX, acc2_v1)
	ADCQ(Imm(0), RDX)
	MOVQ(RDX, acc3_v1)

	MOVQ(Mem{Base: x_ptr}.Offset(8*3), RAX)
	MULQ(t0_v1)
	ADDQ(RAX, acc3_v1)
	ADCQ(Imm(0), RDX)
	MOVQ(RDX, acc4_v1)
	XORQ(acc5_v1, acc5_v1)

	Comment("First reduction step")
	MOVQ(acc0_v1, RAX)
	MOVQ(acc0_v1, t1_v1)
	SHLQ(Imm(32), acc0_v1)
	p256const1 := p256const1_DATA()
	MULQ(p256const1)
	SHRQ(Imm(32), t1_v1)
	ADDQ(acc0_v1, acc1_v1)
	ADCQ(t1_v1, acc2_v1)
	ADCQ(RAX, acc3_v1)
	ADCQ(RDX, acc4_v1)
	ADCQ(Imm(0), acc5_v1)
	XORQ(acc0_v1, acc0_v1)

	Comment("x * y[1]")
	MOVQ(Mem{Base: y_ptr}.Offset(8*1), t0_v1)

	MOVQ(Mem{Base: x_ptr}.Offset(8*0), RAX)
	MULQ(t0_v1)
	ADDQ(RAX, acc1_v1)
	ADCQ(Imm(0), RDX)
	MOVQ(RDX, t1_v1)

	MOVQ(Mem{Base: x_ptr}.Offset(8*1), RAX)
	MULQ(t0_v1)
	ADDQ(t1_v1, acc2_v1)
	ADCQ(Imm(0), RDX)
	ADDQ(RAX, acc2_v1)
	ADCQ(Imm(0), RDX)
	MOVQ(RDX, t1_v1)

	MOVQ(Mem{Base: x_ptr}.Offset(8*2), RAX)
	MULQ(t0_v1)
	ADDQ(t1_v1, acc3_v1)
	ADCQ(Imm(0), RDX)
	ADDQ(RAX, acc3_v1)
	ADCQ(Imm(0), RDX)
	MOVQ(RDX, t1_v1)

	MOVQ(Mem{Base: x_ptr}.Offset(8*3), RAX)
	MULQ(t0_v1)
	ADDQ(t1_v1, acc4_v1)
	ADCQ(Imm(0), RDX)
	ADDQ(RAX, acc4_v1)
	ADCQ(RDX, acc5_v1)
	ADCQ(Imm(0), acc0_v1)

	Comment("Second reduction step")
	MOVQ(acc1_v1, RAX)
	MOVQ(acc1_v1, t1_v1)
	SHLQ(Imm(32), acc1_v1)
	MULQ(p256const1)
	SHRQ(Imm(32), t1_v1)
	ADDQ(acc1_v1, acc2_v1)
	ADCQ(t1_v1, acc3_v1)
	ADCQ(RAX, acc4_v1)
	ADCQ(RDX, acc5_v1)
	ADCQ(Imm(0), acc0_v1)
	XORQ(acc1_v1, acc1_v1)

	Comment("x * y[2]")
	MOVQ(Mem{Base: y_ptr}.Offset(8*2), t0_v1)

	MOVQ(Mem{Base: x_ptr}.Offset(8*0), RAX)
	MULQ(t0_v1)
	ADDQ(RAX, acc2_v1)
	ADCQ(Imm(0), RDX)
	MOVQ(RDX, t1_v1)

	MOVQ(Mem{Base: x_ptr}.Offset(8*1), RAX)
	MULQ(t0_v1)
	ADDQ(t1_v1, acc3_v1)
	ADCQ(Imm(0), RDX)
	ADDQ(RAX, acc3_v1)
	ADCQ(Imm(0), RDX)
	MOVQ(RDX, t1_v1)

	MOVQ(Mem{Base: x_ptr}.Offset(8*2), RAX)
	MULQ(t0_v1)
	ADDQ(t1_v1, acc4_v1)
	ADCQ(Imm(0), RDX)
	ADDQ(RAX, acc4_v1)
	ADCQ(Imm(0), RDX)
	MOVQ(RDX, t1_v1)

	MOVQ(Mem{Base: x_ptr}.Offset(8*3), RAX)
	MULQ(t0_v1)
	ADDQ(t1_v1, acc5_v1)
	ADCQ(Imm(0), RDX)
	ADDQ(RAX, acc5_v1)
	ADCQ(RDX, acc0_v1)
	ADCQ(Imm(0), acc1_v1)

	Comment("Third reduction step")
	MOVQ(acc2_v1, RAX)
	MOVQ(acc2_v1, t1_v1)
	SHLQ(Imm(32), acc2_v1)
	MULQ(p256const1)
	SHRQ(Imm(32), t1_v1)
	ADDQ(acc2_v1, acc3_v1)
	ADCQ(t1_v1, acc4_v1)
	ADCQ(RAX, acc5_v1)
	ADCQ(RDX, acc0_v1)
	ADCQ(Imm(0), acc1_v1)
	XORQ(acc2_v1, acc2_v1)
	Comment("x * y[3]")

	MOVQ(Mem{Base: y_ptr}.Offset(8*3), t0_v1)

	MOVQ(Mem{Base: x_ptr}.Offset(8*0), RAX)
	MULQ(t0_v1)
	ADDQ(RAX, acc3_v1)
	ADCQ(Imm(0), RDX)
	MOVQ(RDX, t1_v1)

	MOVQ(Mem{Base: x_ptr}.Offset(8*1), RAX)
	MULQ(t0_v1)
	ADDQ(t1_v1, acc4_v1)
	ADCQ(Imm(0), RDX)
	ADDQ(RAX, acc4_v1)
	ADCQ(Imm(0), RDX)
	MOVQ(RDX, t1_v1)

	MOVQ(Mem{Base: x_ptr}.Offset(8*2), RAX)
	MULQ(t0_v1)
	ADDQ(t1_v1, acc5_v1)
	ADCQ(Imm(0), RDX)
	ADDQ(RAX, acc5_v1)
	ADCQ(Imm(0), RDX)
	MOVQ(RDX, t1_v1)

	MOVQ(Mem{Base: x_ptr}.Offset(8*3), RAX)
	MULQ(t0_v1)
	ADDQ(t1_v1, acc0_v1)
	ADCQ(Imm(0), RDX)
	ADDQ(RAX, acc0_v1)
	ADCQ(RDX, acc1_v1)
	ADCQ(Imm(0), acc2_v1)

	Comment("Last reduction step")
	MOVQ(acc3_v1, RAX)
	MOVQ(acc3_v1, t1_v1)
	SHLQ(Imm(32), acc3_v1)
	MULQ(p256const1)
	SHRQ(Imm(32), t1_v1)
	ADDQ(acc3_v1, acc4_v1)
	ADCQ(t1_v1, acc5_v1)
	ADCQ(RAX, acc0_v1)
	ADCQ(RDX, acc1_v1)
	ADCQ(Imm(0), acc2_v1)

	Comment("Copy result [255:0]")
	MOVQ(acc4_v1, x_ptr)
	MOVQ(acc5_v1, acc3_v1)
	MOVQ(acc0_v1, t0_v1)
	MOVQ(acc1_v1, t1_v1)

	Comment("Subtract p256")
	SUBQ(I8(-1), acc4_v1)
	p256const0 := p256const0_DATA()
	SBBQ(p256const0, acc5_v1)
	SBBQ(Imm(0), acc0_v1)
	// SBBQ p256const1<>(SB), acc1_v1
	SBBQ(p256const1, acc1_v1)
	SBBQ(Imm(0), acc2_v1)

	CMOVQCS(x_ptr, acc4_v1)
	CMOVQCS(acc3_v1, acc5_v1)
	CMOVQCS(t0_v1, acc0_v1)
	CMOVQCS(t1_v1, acc1_v1)

	MOVQ(acc4_v1, Mem{Base: res_ptr}.Offset(8*0))
	MOVQ(acc5_v1, Mem{Base: res_ptr}.Offset(8*1))
	MOVQ(acc0_v1, Mem{Base: res_ptr}.Offset(8*2))
	MOVQ(acc1_v1, Mem{Base: res_ptr}.Offset(8*3))

	RET()
}

// Implements:
//
//	func p256FromMont(res, in *p256Element)
func p256FromMont() {
	Implement("p256FromMont")
	Attributes(NOSPLIT)

	Load(Param("res"), res_ptr)
	Load(Param("in"), x_ptr)

	MOVQ(Mem{Base: x_ptr}.Offset(8*0), acc0_v1)
	MOVQ(Mem{Base: x_ptr}.Offset(8*1), acc1_v1)
	MOVQ(Mem{Base: x_ptr}.Offset(8*2), acc2_v1)
	MOVQ(Mem{Base: x_ptr}.Offset(8*3), acc3_v1)
	XORQ(acc4_v1, acc4_v1)

	Comment("Only reduce, no multiplications are needed")
	Comment("First stage")
	MOVQ(acc0_v1, RAX)
	MOVQ(acc0_v1, t1_v1)
	SHLQ(Imm(32), acc0_v1)
	p256const1 := p256const1_DATA()
	MULQ(p256const1)
	SHRQ(Imm(32), t1_v1)
	ADDQ(acc0_v1, acc1_v1)
	ADCQ(t1_v1, acc2_v1)
	ADCQ(RAX, acc3_v1)
	ADCQ(RDX, acc4_v1)
	XORQ(acc5_v1, acc5_v1)

	Comment("Second stage")
	MOVQ(acc1_v1, RAX)
	MOVQ(acc1_v1, t1_v1)
	SHLQ(Imm(32), acc1_v1)
	MULQ(p256const1)
	SHRQ(Imm(32), t1_v1)
	ADDQ(acc1_v1, acc2_v1)
	ADCQ(t1_v1, acc3_v1)
	ADCQ(RAX, acc4_v1)
	ADCQ(RDX, acc5_v1)
	XORQ(acc0_v1, acc0_v1)

	Comment("Third stage")
	MOVQ(acc2_v1, RAX)
	MOVQ(acc2_v1, t1_v1)
	SHLQ(Imm(32), acc2_v1)
	MULQ(p256const1)
	SHRQ(Imm(32), t1_v1)
	ADDQ(acc2_v1, acc3_v1)
	ADCQ(t1_v1, acc4_v1)
	ADCQ(RAX, acc5_v1)
	ADCQ(RDX, acc0_v1)
	XORQ(acc1_v1, acc1_v1)

	Comment("Last stage")
	MOVQ(acc3_v1, RAX)
	MOVQ(acc3_v1, t1_v1)
	SHLQ(Imm(32), acc3_v1)
	MULQ(p256const1)
	SHRQ(Imm(32), t1_v1)
	ADDQ(acc3_v1, acc4_v1)
	ADCQ(t1_v1, acc5_v1)
	ADCQ(RAX, acc0_v1)
	ADCQ(RDX, acc1_v1)

	MOVQ(acc4_v1, x_ptr)
	MOVQ(acc5_v1, acc3_v1)
	MOVQ(acc0_v1, t0_v1)
	MOVQ(acc1_v1, t1_v1)

	SUBQ(I8(-1), acc4_v1)
	p256const0 := p256const0_DATA()
	SBBQ(p256const0, acc5_v1)
	SBBQ(Imm(0), acc0_v1)
	SBBQ(p256const1, acc1_v1)

	CMOVQCS(x_ptr, acc4_v1)
	CMOVQCS(acc3_v1, acc5_v1)
	CMOVQCS(t0_v1, acc0_v1)
	CMOVQCS(t1_v1, acc1_v1)

	MOVQ(acc4_v1, Mem{Base: res_ptr}.Offset(8*0))
	MOVQ(acc5_v1, Mem{Base: res_ptr}.Offset(8*1))
	MOVQ(acc0_v1, Mem{Base: res_ptr}.Offset(8*2))
	MOVQ(acc1_v1, Mem{Base: res_ptr}.Offset(8*3))

	RET()
}

// Implements:
//
//	func p256Select(res *P256Point, table *p256Table, idx int)
func p256Select() {
	Implement("p256Select")
	Attributes(NOSPLIT)

	Load(Param("idx"), RAX)
	Load(Param("table"), RDI)
	Load(Param("res"), RDX)

	PXOR(X15, X15)    // X15 =  0
	PCMPEQL(X14, X14) // X14 = -1
	PSUBL(X14, X15)   // X15 =  1
	// Force Avo to emit:
	// 	MOVL AX, X14
	Instruction(&ir.Instruction{
		Opcode: "MOVL",
		Operands: []Op{
			EAX, X14,
		},
	})
	PSHUFD(Imm(0), X14, X14)

	PXOR(X0, X0)
	PXOR(X1, X1)
	PXOR(X2, X2)
	PXOR(X3, X3)
	PXOR(X4, X4)
	PXOR(X5, X5)
	MOVQ(U32(16), RAX)

	MOVOU(X15, X13)

	Label("loop_select")

	MOVOU(X13, X12)
	PADDL(X15, X13)
	PCMPEQL(X14, X12)

	MOVOU(Mem{Base: DI}.Offset(16*0), X6)
	MOVOU(Mem{Base: DI}.Offset(16*1), X7)
	MOVOU(Mem{Base: DI}.Offset(16*2), X8)
	MOVOU(Mem{Base: DI}.Offset(16*3), X9)
	MOVOU(Mem{Base: DI}.Offset(16*4), X10)
	MOVOU(Mem{Base: DI}.Offset(16*5), X11)
	ADDQ(U8(16*6), RDI)

	PAND(X12, X6)
	PAND(X12, X7)
	PAND(X12, X8)
	PAND(X12, X9)
	PAND(X12, X10)
	PAND(X12, X11)

	PXOR(X6, X0)
	PXOR(X7, X1)
	PXOR(X8, X2)
	PXOR(X9, X3)
	PXOR(X10, X4)
	PXOR(X11, X5)

	DECQ(RAX)
	JNE(LabelRef("loop_select"))

	MOVOU(X0, Mem{Base: DX}.Offset(16*0))
	MOVOU(X1, Mem{Base: DX}.Offset(16*1))
	MOVOU(X2, Mem{Base: DX}.Offset(16*2))
	MOVOU(X3, Mem{Base: DX}.Offset(16*3))
	MOVOU(X4, Mem{Base: DX}.Offset(16*4))
	MOVOU(X5, Mem{Base: DX}.Offset(16*5))

	RET()
}

// Implements:
//
//	func p256SelectAffine(res *p256AffinePoint, table *p256AffineTable, idx int)
func p256SelectAffine() {
	Implement("p256SelectAffine")
	Attributes(NOSPLIT)

	Load(Param("idx"), RAX)
	Load(Param("table"), RDI)
	Load(Param("res"), RDX)

	PXOR(X15, X15)    // X15 =  0
	PCMPEQL(X14, X14) // X14 = -1
	PSUBL(X14, X15)   // X15 =  1

	// Hack to get Avo to emit:
	// 	MOVL AX, X14
	Instruction(&ir.Instruction{Opcode: "MOVL", Operands: []Op{RAX, X14}})

	PSHUFD(Imm(0), X14, X14)

	PXOR(X0, X0)
	PXOR(X1, X1)
	PXOR(X2, X2)
	PXOR(X3, X3)
	MOVQ(U32(16), RAX)

	MOVOU(X15, X13)

	Label("loop_select_base")

	MOVOU(X13, X12)
	PADDL(X15, X13)
	PCMPEQL(X14, X12)

	MOVOU(Mem{Base: DI}.Offset(16*0), X4)
	MOVOU(Mem{Base: DI}.Offset(16*1), X5)
	MOVOU(Mem{Base: DI}.Offset(16*2), X6)
	MOVOU(Mem{Base: DI}.Offset(16*3), X7)

	MOVOU(Mem{Base: DI}.Offset(16*4), X8)
	MOVOU(Mem{Base: DI}.Offset(16*5), X9)
	MOVOU(Mem{Base: DI}.Offset(16*6), X10)
	MOVOU(Mem{Base: DI}.Offset(16*7), X11)

	ADDQ(Imm(16*8), RDI)

	PAND(X12, X4)
	PAND(X12, X5)
	PAND(X12, X6)
	PAND(X12, X7)

	MOVOU(X13, X12)
	PADDL(X15, X13)
	PCMPEQL(X14, X12)

	PAND(X12, X8)
	PAND(X12, X9)
	PAND(X12, X10)
	PAND(X12, X11)

	PXOR(X4, X0)
	PXOR(X5, X1)
	PXOR(X6, X2)
	PXOR(X7, X3)

	PXOR(X8, X0)
	PXOR(X9, X1)
	PXOR(X10, X2)
	PXOR(X11, X3)

	DECQ(RAX)
	JNE(LabelRef("loop_select_base"))

	MOVOU(X0, Mem{Base: DX}.Offset(16*0))
	MOVOU(X1, Mem{Base: DX}.Offset(16*1))
	MOVOU(X2, Mem{Base: DX}.Offset(16*2))
	MOVOU(X3, Mem{Base: DX}.Offset(16*3))

	RET()
}

// Implements:
//
//	func p256OrdMul(res, in1, in2 *p256OrdElement)
func p256OrdMul() {
	Implement("p256OrdMul")
	Attributes(NOSPLIT)

	Load(Param("res"), res_ptr)
	Load(Param("in1"), x_ptr)
	Load(Param("in2"), y_ptr)

	Comment("x * y[0]")
	MOVQ(Mem{Base: y_ptr}.Offset(8*0), t0_v1)

	MOVQ(Mem{Base: x_ptr}.Offset(8*0), RAX)
	MULQ(t0_v1)
	MOVQ(RAX, acc0_v1)
	MOVQ(RDX, acc1_v1)

	MOVQ(Mem{Base: x_ptr}.Offset(8*1), RAX)
	MULQ(t0_v1)
	ADDQ(RAX, acc1_v1)
	ADCQ(Imm(0), RDX)
	MOVQ(RDX, acc2_v1)

	MOVQ(Mem{Base: x_ptr}.Offset(8*2), RAX)
	MULQ(t0_v1)
	ADDQ(RAX, acc2_v1)
	ADCQ(Imm(0), RDX)
	MOVQ(RDX, acc3_v1)

	MOVQ(Mem{Base: x_ptr}.Offset(8*3), RAX)
	MULQ(t0_v1)
	ADDQ(RAX, acc3_v1)
	ADCQ(Imm(0), RDX)
	MOVQ(RDX, acc4_v1)
	XORQ(acc5_v1, acc5_v1)

	Comment("First reduction step")
	MOVQ(acc0_v1, RAX)
	p256ordK0 := p256ordK0_DATA()
	MULQ(p256ordK0)
	MOVQ(RAX, t0_v1)

	p256ord := p256ord_DATA()
	MOVQ(p256ord.Offset(0x00), RAX)
	MULQ(t0_v1)
	ADDQ(RAX, acc0_v1)
	ADCQ(Imm(0), RDX)
	MOVQ(RDX, t1_v1)

	MOVQ(p256ord.Offset(0x08), RAX)
	MULQ(t0_v1)
	ADDQ(t1_v1, acc1_v1)
	ADCQ(Imm(0), RDX)
	ADDQ(RAX, acc1_v1)
	ADCQ(Imm(0), RDX)
	MOVQ(RDX, t1_v1)

	MOVQ(p256ord.Offset(0x10), RAX)
	MULQ(t0_v1)
	ADDQ(t1_v1, acc2_v1)
	ADCQ(Imm(0), RDX)
	ADDQ(RAX, acc2_v1)
	ADCQ(Imm(0), RDX)
	MOVQ(RDX, t1_v1)

	MOVQ(p256ord.Offset(0x18), RAX)
	MULQ(t0_v1)
	ADDQ(t1_v1, acc3_v1)
	ADCQ(Imm(0), RDX)
	ADDQ(RAX, acc3_v1)
	ADCQ(RDX, acc4_v1)
	ADCQ(Imm(0), acc5_v1)

	Comment("x * y[1]")
	MOVQ(Mem{Base: y_ptr}.Offset(8*1), t0_v1)

	MOVQ(Mem{Base: x_ptr}.Offset(8*0), RAX)
	MULQ(t0_v1)
	ADDQ(RAX, acc1_v1)
	ADCQ(Imm(0), RDX)
	MOVQ(RDX, t1_v1)

	MOVQ(Mem{Base: x_ptr}.Offset(8*1), RAX)
	MULQ(t0_v1)
	ADDQ(t1_v1, acc2_v1)
	ADCQ(Imm(0), RDX)
	ADDQ(RAX, acc2_v1)
	ADCQ(Imm(0), RDX)
	MOVQ(RDX, t1_v1)

	MOVQ(Mem{Base: x_ptr}.Offset(8*2), RAX)
	MULQ(t0_v1)
	ADDQ(t1_v1, acc3_v1)
	ADCQ(Imm(0), RDX)
	ADDQ(RAX, acc3_v1)
	ADCQ(Imm(0), RDX)
	MOVQ(RDX, t1_v1)

	MOVQ(Mem{Base: x_ptr}.Offset(8*3), RAX)
	MULQ(t0_v1)
	ADDQ(t1_v1, acc4_v1)
	ADCQ(Imm(0), RDX)
	ADDQ(RAX, acc4_v1)
	ADCQ(RDX, acc5_v1)
	ADCQ(Imm(0), acc0_v1)

	Comment("Second reduction step")
	MOVQ(acc1_v1, RAX)
	MULQ(p256ordK0)
	MOVQ(RAX, t0_v1)

	MOVQ(p256ord.Offset(0x00), RAX)
	MULQ(t0_v1)
	ADDQ(RAX, acc1_v1)
	ADCQ(Imm(0), RDX)
	MOVQ(RDX, t1_v1)

	MOVQ(p256ord.Offset(0x08), RAX)
	MULQ(t0_v1)
	ADDQ(t1_v1, acc2_v1)
	ADCQ(Imm(0), RDX)
	ADDQ(RAX, acc2_v1)
	ADCQ(Imm(0), RDX)
	MOVQ(RDX, t1_v1)

	MOVQ(p256ord.Offset(0x10), RAX)
	MULQ(t0_v1)
	ADDQ(t1_v1, acc3_v1)
	ADCQ(Imm(0), RDX)
	ADDQ(RAX, acc3_v1)
	ADCQ(Imm(0), RDX)
	MOVQ(RDX, t1_v1)

	MOVQ(p256ord.Offset(0x18), RAX)
	MULQ(t0_v1)
	ADDQ(t1_v1, acc4_v1)
	ADCQ(Imm(0), RDX)
	ADDQ(RAX, acc4_v1)
	ADCQ(RDX, acc5_v1)
	ADCQ(Imm(0), acc0_v1)

	Comment("x * y[2]")
	MOVQ(Mem{Base: y_ptr}.Offset(8*2), t0_v1)

	MOVQ(Mem{Base: x_ptr}.Offset(8*0), RAX)
	MULQ(t0_v1)
	ADDQ(RAX, acc2_v1)
	ADCQ(Imm(0), RDX)
	MOVQ(RDX, t1_v1)

	MOVQ(Mem{Base: x_ptr}.Offset(8*1), RAX)
	MULQ(t0_v1)
	ADDQ(t1_v1, acc3_v1)
	ADCQ(Imm(0), RDX)
	ADDQ(RAX, acc3_v1)
	ADCQ(Imm(0), RDX)
	MOVQ(RDX, t1_v1)

	MOVQ(Mem{Base: x_ptr}.Offset(8*2), RAX)
	MULQ(t0_v1)
	ADDQ(t1_v1, acc4_v1)
	ADCQ(Imm(0), RDX)
	ADDQ(RAX, acc4_v1)
	ADCQ(Imm(0), RDX)
	MOVQ(RDX, t1_v1)

	MOVQ(Mem{Base: x_ptr}.Offset(8*3), RAX)
	MULQ(t0_v1)
	ADDQ(t1_v1, acc5_v1)
	ADCQ(Imm(0), RDX)
	ADDQ(RAX, acc5_v1)
	ADCQ(RDX, acc0_v1)
	ADCQ(Imm(0), acc1_v1)

	Comment("Third reduction step")
	MOVQ(acc2_v1, RAX)
	MULQ(p256ordK0)
	MOVQ(RAX, t0_v1)

	MOVQ(p256ord.Offset(0x00), RAX)
	MULQ(t0_v1)
	ADDQ(RAX, acc2_v1)
	ADCQ(Imm(0), RDX)
	MOVQ(RDX, t1_v1)

	MOVQ(p256ord.Offset(0x08), RAX)
	MULQ(t0_v1)
	ADDQ(t1_v1, acc3_v1)
	ADCQ(Imm(0), RDX)
	ADDQ(RAX, acc3_v1)
	ADCQ(Imm(0), RDX)
	MOVQ(RDX, t1_v1)

	MOVQ(p256ord.Offset(0x10), RAX)
	MULQ(t0_v1)
	ADDQ(t1_v1, acc4_v1)
	ADCQ(Imm(0), RDX)
	ADDQ(RAX, acc4_v1)
	ADCQ(Imm(0), RDX)
	MOVQ(RDX, t1_v1)

	MOVQ(p256ord.Offset(0x18), RAX)
	MULQ(t0_v1)
	ADDQ(t1_v1, acc5_v1)
	ADCQ(Imm(0), RDX)
	ADDQ(RAX, acc5_v1)
	ADCQ(RDX, acc0_v1)
	ADCQ(Imm(0), acc1_v1)

	Comment("x * y[3]")
	MOVQ(Mem{Base: y_ptr}.Offset(8*3), t0_v1)

	MOVQ(Mem{Base: x_ptr}.Offset(8*0), RAX)
	MULQ(t0_v1)
	ADDQ(RAX, acc3_v1)
	ADCQ(Imm(0), RDX)
	MOVQ(RDX, t1_v1)

	MOVQ(Mem{Base: x_ptr}.Offset(8*1), RAX)
	MULQ(t0_v1)
	ADDQ(t1_v1, acc4_v1)
	ADCQ(Imm(0), RDX)
	ADDQ(RAX, acc4_v1)
	ADCQ(Imm(0), RDX)
	MOVQ(RDX, t1_v1)

	MOVQ(Mem{Base: x_ptr}.Offset(8*2), RAX)
	MULQ(t0_v1)
	ADDQ(t1_v1, acc5_v1)
	ADCQ(Imm(0), RDX)
	ADDQ(RAX, acc5_v1)
	ADCQ(Imm(0), RDX)
	MOVQ(RDX, t1_v1)

	MOVQ(Mem{Base: x_ptr}.Offset(8*3), RAX)
	MULQ(t0_v1)
	ADDQ(t1_v1, acc0_v1)
	ADCQ(Imm(0), RDX)
	ADDQ(RAX, acc0_v1)
	ADCQ(RDX, acc1_v1)
	ADCQ(Imm(0), acc2_v1)

	Comment("Last reduction step")
	MOVQ(acc3_v1, RAX)
	MULQ(p256ordK0)
	MOVQ(RAX, t0_v1)

	MOVQ(p256ord.Offset(0x00), RAX)
	MULQ(t0_v1)
	ADDQ(RAX, acc3_v1)
	ADCQ(Imm(0), RDX)
	MOVQ(RDX, t1_v1)

	MOVQ(p256ord.Offset(0x08), RAX)
	MULQ(t0_v1)
	ADDQ(t1_v1, acc4_v1)
	ADCQ(Imm(0), RDX)
	ADDQ(RAX, acc4_v1)
	ADCQ(Imm(0), RDX)
	MOVQ(RDX, t1_v1)

	MOVQ(p256ord.Offset(0x10), RAX)
	MULQ(t0_v1)
	ADDQ(t1_v1, acc5_v1)
	ADCQ(Imm(0), RDX)
	ADDQ(RAX, acc5_v1)
	ADCQ(Imm(0), RDX)
	MOVQ(RDX, t1_v1)

	MOVQ(p256ord.Offset(0x18), RAX)
	MULQ(t0_v1)
	ADDQ(t1_v1, acc0_v1)
	ADCQ(Imm(0), RDX)
	ADDQ(RAX, acc0_v1)
	ADCQ(RDX, acc1_v1)
	ADCQ(Imm(0), acc2_v1)

	Comment("Copy result [255:0]")
	MOVQ(acc4_v1, x_ptr)
	MOVQ(acc5_v1, acc3_v1)
	MOVQ(acc0_v1, t0_v1)
	MOVQ(acc1_v1, t1_v1)

	Comment("Subtract p256")
	SUBQ(p256ord.Offset(0x00), acc4_v1)
	SBBQ(p256ord.Offset(0x08), acc5_v1)
	SBBQ(p256ord.Offset(0x10), acc0_v1)
	SBBQ(p256ord.Offset(0x18), acc1_v1)
	SBBQ(Imm(0), acc2_v1)

	CMOVQCS(x_ptr, acc4_v1)
	CMOVQCS(acc3_v1, acc5_v1)
	CMOVQCS(t0_v1, acc0_v1)
	CMOVQCS(t1_v1, acc1_v1)

	MOVQ(acc4_v1, Mem{Base: res_ptr}.Offset(8*0))
	MOVQ(acc5_v1, Mem{Base: res_ptr}.Offset(8*1))
	MOVQ(acc0_v1, Mem{Base: res_ptr}.Offset(8*2))
	MOVQ(acc1_v1, Mem{Base: res_ptr}.Offset(8*3))

	RET()
}

// Implements:
//
//	func p256OrdSqr(res, in *p256OrdElement, n int)
func p256OrdSqr() {
	Implement("p256OrdSqr")
	Attributes(NOSPLIT)

	Load(Param("res"), res_ptr)
	Load(Param("in"), x_ptr)
	Load(Param("n"), RBX)

	Label("ordSqrLoop")

	Comment("y[1:] * y[0]")
	MOVQ(Mem{Base: x_ptr}.Offset(8*0), t0_v1)

	MOVQ(Mem{Base: x_ptr}.Offset(8*1), RAX)
	MULQ(t0_v1)
	MOVQ(RAX, acc1_v1)
	MOVQ(RDX, acc2_v1)

	MOVQ(Mem{Base: x_ptr}.Offset(8*2), RAX)
	MULQ(t0_v1)
	ADDQ(RAX, acc2_v1)
	ADCQ(Imm(0), RDX)
	MOVQ(RDX, acc3_v1)

	MOVQ(Mem{Base: x_ptr}.Offset(8*3), RAX)
	MULQ(t0_v1)
	ADDQ(RAX, acc3_v1)
	ADCQ(Imm(0), RDX)
	MOVQ(RDX, acc4_v1)

	Comment("y[2:] * y[1]")
	MOVQ(Mem{Base: x_ptr}.Offset(8*1), t0_v1)

	MOVQ(Mem{Base: x_ptr}.Offset(8*2), RAX)
	MULQ(t0_v1)
	ADDQ(RAX, acc3_v1)
	ADCQ(Imm(0), RDX)
	MOVQ(RDX, t1_v1)

	MOVQ(Mem{Base: x_ptr}.Offset(8*3), RAX)
	MULQ(t0_v1)
	ADDQ(t1_v1, acc4_v1)
	ADCQ(Imm(0), RDX)
	ADDQ(RAX, acc4_v1)
	ADCQ(Imm(0), RDX)
	MOVQ(RDX, acc5_v1)

	Comment("y[3] * y[2]")
	MOVQ(Mem{Base: x_ptr}.Offset(8*2), t0_v1)

	MOVQ(Mem{Base: x_ptr}.Offset(8*3), RAX)
	MULQ(t0_v1)
	ADDQ(RAX, acc5_v1)
	ADCQ(Imm(0), RDX)
	MOVQ(RDX, y_ptr)
	XORQ(t1_v1, t1_v1)

	Comment("*2")
	ADDQ(acc1_v1, acc1_v1)
	ADCQ(acc2_v1, acc2_v1)
	ADCQ(acc3_v1, acc3_v1)
	ADCQ(acc4_v1, acc4_v1)
	ADCQ(acc5_v1, acc5_v1)
	ADCQ(y_ptr, y_ptr)
	ADCQ(Imm(0), t1_v1)

	Comment("Missing products")
	MOVQ(Mem{Base: x_ptr}.Offset(8*0), RAX)
	MULQ(RAX)
	MOVQ(RAX, acc0_v1)
	MOVQ(RDX, t0_v1)

	MOVQ(Mem{Base: x_ptr}.Offset(8*1), RAX)
	MULQ(RAX)
	ADDQ(t0_v1, acc1_v1)
	ADCQ(RAX, acc2_v1)
	ADCQ(Imm(0), RDX)
	MOVQ(RDX, t0_v1)

	MOVQ(Mem{Base: x_ptr}.Offset(8*2), RAX)
	MULQ(RAX)
	ADDQ(t0_v1, acc3_v1)
	ADCQ(RAX, acc4_v1)
	ADCQ(Imm(0), RDX)
	MOVQ(RDX, t0_v1)

	MOVQ(Mem{Base: x_ptr}.Offset(8*3), RAX)
	MULQ(RAX)
	ADDQ(t0_v1, acc5_v1)
	ADCQ(RAX, y_ptr)
	ADCQ(RDX, t1_v1)
	MOVQ(t1_v1, x_ptr)

	Comment("First reduction step")
	MOVQ(acc0_v1, RAX)
	p256ordK0 := p256ordK0_DATA()
	MULQ(p256ordK0)
	MOVQ(RAX, t0_v1)

	p256ord := p256ord_DATA()
	MOVQ(p256ord.Offset(0x00), RAX)
	MULQ(t0_v1)
	ADDQ(RAX, acc0_v1)
	ADCQ(Imm(0), RDX)
	MOVQ(RDX, t1_v1)

	MOVQ(p256ord.Offset(0x08), RAX)
	MULQ(t0_v1)
	ADDQ(t1_v1, acc1_v1)
	ADCQ(Imm(0), RDX)
	ADDQ(RAX, acc1_v1)

	MOVQ(t0_v1, t1_v1)
	ADCQ(RDX, acc2_v1)
	ADCQ(Imm(0), t1_v1)
	SUBQ(t0_v1, acc2_v1)
	SBBQ(Imm(0), t1_v1)

	MOVQ(t0_v1, RAX)
	MOVQ(t0_v1, RDX)
	MOVQ(t0_v1, acc0_v1)
	SHLQ(Imm(32), RAX)
	SHRQ(Imm(32), RDX)

	ADDQ(t1_v1, acc3_v1)
	ADCQ(Imm(0), acc0_v1)
	SUBQ(RAX, acc3_v1)
	SBBQ(RDX, acc0_v1)

	Comment("Second reduction step")
	MOVQ(acc1_v1, RAX)
	MULQ(p256ordK0)
	MOVQ(RAX, t0_v1)

	MOVQ(p256ord.Offset(0x00), RAX)
	MULQ(t0_v1)
	ADDQ(RAX, acc1_v1)
	ADCQ(Imm(0), RDX)
	MOVQ(RDX, t1_v1)

	MOVQ(p256ord.Offset(0x08), RAX)
	MULQ(t0_v1)
	ADDQ(t1_v1, acc2_v1)
	ADCQ(Imm(0), RDX)
	ADDQ(RAX, acc2_v1)

	MOVQ(t0_v1, t1_v1)
	ADCQ(RDX, acc3_v1)
	ADCQ(Imm(0), t1_v1)
	SUBQ(t0_v1, acc3_v1)
	SBBQ(Imm(0), t1_v1)

	MOVQ(t0_v1, RAX)
	MOVQ(t0_v1, RDX)
	MOVQ(t0_v1, acc1_v1)
	SHLQ(Imm(32), RAX)
	SHRQ(Imm(32), RDX)

	ADDQ(t1_v1, acc0_v1)
	ADCQ(Imm(0), acc1_v1)
	SUBQ(RAX, acc0_v1)
	SBBQ(RDX, acc1_v1)

	Comment("Third reduction step")
	MOVQ(acc2_v1, RAX)
	MULQ(p256ordK0)
	MOVQ(RAX, t0_v1)

	MOVQ(p256ord.Offset(0x00), RAX)
	MULQ(t0_v1)
	ADDQ(RAX, acc2_v1)
	ADCQ(Imm(0), RDX)
	MOVQ(RDX, t1_v1)

	MOVQ(p256ord.Offset(0x08), RAX)
	MULQ(t0_v1)
	ADDQ(t1_v1, acc3_v1)
	ADCQ(Imm(0), RDX)
	ADDQ(RAX, acc3_v1)

	MOVQ(t0_v1, t1_v1)
	ADCQ(RDX, acc0_v1)
	ADCQ(Imm(0), t1_v1)
	SUBQ(t0_v1, acc0_v1)
	SBBQ(Imm(0), t1_v1)

	MOVQ(t0_v1, RAX)
	MOVQ(t0_v1, RDX)
	MOVQ(t0_v1, acc2_v1)
	SHLQ(Imm(32), RAX)
	SHRQ(Imm(32), RDX)

	ADDQ(t1_v1, acc1_v1)
	ADCQ(Imm(0), acc2_v1)
	SUBQ(RAX, acc1_v1)
	SBBQ(RDX, acc2_v1)

	Comment("Last reduction step")
	MOVQ(acc3_v1, RAX)
	MULQ(p256ordK0)
	MOVQ(RAX, t0_v1)

	MOVQ(p256ord.Offset(0x00), RAX)
	MULQ(t0_v1)
	ADDQ(RAX, acc3_v1)
	ADCQ(Imm(0), RDX)
	MOVQ(RDX, t1_v1)

	MOVQ(p256ord.Offset(0x08), RAX)
	MULQ(t0_v1)
	ADDQ(t1_v1, acc0_v1)
	ADCQ(Imm(0), RDX)
	ADDQ(RAX, acc0_v1)
	ADCQ(Imm(0), RDX)
	MOVQ(RDX, t1_v1)

	MOVQ(t0_v1, t1_v1)
	ADCQ(RDX, acc1_v1)
	ADCQ(Imm(0), t1_v1)
	SUBQ(t0_v1, acc1_v1)
	SBBQ(Imm(0), t1_v1)

	MOVQ(t0_v1, RAX)
	MOVQ(t0_v1, RDX)
	MOVQ(t0_v1, acc3_v1)
	SHLQ(Imm(32), RAX)
	SHRQ(Imm(32), RDX)

	ADDQ(t1_v1, acc2_v1)
	ADCQ(Imm(0), acc3_v1)
	SUBQ(RAX, acc2_v1)
	SBBQ(RDX, acc3_v1)
	XORQ(t0_v1, t0_v1)

	Comment("Add bits [511:256] of the sqr result")
	ADCQ(acc4_v1, acc0_v1)
	ADCQ(acc5_v1, acc1_v1)
	ADCQ(y_ptr, acc2_v1)
	ADCQ(x_ptr, acc3_v1)
	ADCQ(Imm(0), t0_v1)

	MOVQ(acc0_v1, acc4_v1)
	MOVQ(acc1_v1, acc5_v1)
	MOVQ(acc2_v1, y_ptr)
	MOVQ(acc3_v1, t1_v1)

	Comment("Subtract p256")
	SUBQ(p256ord.Offset(0x00), acc0_v1)
	SBBQ(p256ord.Offset(0x08), acc1_v1)
	SBBQ(p256ord.Offset(0x10), acc2_v1)
	SBBQ(p256ord.Offset(0x18), acc3_v1)
	SBBQ(Imm(0), t0_v1)

	CMOVQCS(acc4_v1, acc0_v1)
	CMOVQCS(acc5_v1, acc1_v1)
	CMOVQCS(y_ptr, acc2_v1)
	CMOVQCS(t1_v1, acc3_v1)

	MOVQ(acc0_v1, Mem{Base: res_ptr}.Offset(8*0))
	MOVQ(acc1_v1, Mem{Base: res_ptr}.Offset(8*1))
	MOVQ(acc2_v1, Mem{Base: res_ptr}.Offset(8*2))
	MOVQ(acc3_v1, Mem{Base: res_ptr}.Offset(8*3))
	MOVQ(res_ptr, x_ptr)
	DECQ(RBX)
	JNE(LabelRef("ordSqrLoop"))

	RET()
}

// These variables have been versioned as they get redfined in the reference implementation.
// This is done to produce a minimal semantic diff.
var (
	mul0_v2 = RAX
	mul1_v2 = RDX
	acc0_v2 = RBX
	acc1_v2 = RCX
	acc2_v2 = R8
	acc3_v2 = R9
	acc4_v2 = R10
	acc5_v2 = R11
	acc6_v2 = R12
	acc7_v2 = R13
	t0_v2   = R14
	t1_v2   = R15
	t2_v2   = RDI
	t3_v2   = RSI
	hlp_v2  = RBP
)

func p256SubInternal() {
	Function("p256SubInternal")
	Attributes(NOSPLIT)

	XORQ(mul0_v2, mul0_v2)
	SUBQ(t0_v2, acc4_v2)
	SBBQ(t1_v2, acc5_v2)
	SBBQ(t2_v2, acc6_v2)
	SBBQ(t3_v2, acc7_v2)
	SBBQ(Imm(0), mul0_v2)

	MOVQ(acc4_v2, acc0_v2)
	MOVQ(acc5_v2, acc1_v2)
	MOVQ(acc6_v2, acc2_v2)
	MOVQ(acc7_v2, acc3_v2)

	ADDQ(I8(-1), acc4_v2)
	p256const0 := p256const0_DATA()
	ADCQ(p256const0, acc5_v2)
	ADCQ(Imm(0), acc6_v2)
	p256const1 := p256const1_DATA()
	ADCQ(p256const1, acc7_v2)
	ANDQ(Imm(1), mul0_v2)

	CMOVQEQ(acc0_v2, acc4_v2)
	CMOVQEQ(acc1_v2, acc5_v2)
	CMOVQEQ(acc2_v2, acc6_v2)
	CMOVQEQ(acc3_v2, acc7_v2)

	RET()
}

func p256MulInternal() {
	Function("p256MulInternal")
	Attributes(NOSPLIT)

	MOVQ(acc4_v2, mul0_v2)
	MULQ(t0_v2)
	MOVQ(mul0_v2, acc0_v2)
	MOVQ(mul1_v2, acc1_v2)

	MOVQ(acc4_v2, mul0_v2)
	MULQ(t1_v2)
	ADDQ(mul0_v2, acc1_v2)
	ADCQ(Imm(0), mul1_v2)
	MOVQ(mul1_v2, acc2_v2)

	MOVQ(acc4_v2, mul0_v2)
	MULQ(t2_v2)
	ADDQ(mul0_v2, acc2_v2)
	ADCQ(Imm(0), mul1_v2)
	MOVQ(mul1_v2, acc3_v2)

	MOVQ(acc4_v2, mul0_v2)
	MULQ(t3_v2)
	ADDQ(mul0_v2, acc3_v2)
	ADCQ(Imm(0), mul1_v2)
	MOVQ(mul1_v2, acc4_v2)

	MOVQ(acc5_v2, mul0_v2)
	MULQ(t0_v2)
	ADDQ(mul0_v2, acc1_v2)
	ADCQ(Imm(0), mul1_v2)
	MOVQ(mul1_v2, hlp_v2)

	MOVQ(acc5_v2, mul0_v2)
	MULQ(t1_v2)
	ADDQ(hlp_v2, acc2_v2)
	ADCQ(Imm(0), mul1_v2)
	ADDQ(mul0_v2, acc2_v2)
	ADCQ(Imm(0), mul1_v2)
	MOVQ(mul1_v2, hlp_v2)

	MOVQ(acc5_v2, mul0_v2)
	MULQ(t2_v2)
	ADDQ(hlp_v2, acc3_v2)
	ADCQ(Imm(0), mul1_v2)
	ADDQ(mul0_v2, acc3_v2)
	ADCQ(Imm(0), mul1_v2)
	MOVQ(mul1_v2, hlp_v2)

	MOVQ(acc5_v2, mul0_v2)
	MULQ(t3_v2)
	ADDQ(hlp_v2, acc4_v2)
	ADCQ(Imm(0), mul1_v2)
	ADDQ(mul0_v2, acc4_v2)
	ADCQ(Imm(0), mul1_v2)
	MOVQ(mul1_v2, acc5_v2)

	MOVQ(acc6_v2, mul0_v2)
	MULQ(t0_v2)
	ADDQ(mul0_v2, acc2_v2)
	ADCQ(Imm(0), mul1_v2)
	MOVQ(mul1_v2, hlp_v2)

	MOVQ(acc6_v2, mul0_v2)
	MULQ(t1_v2)
	ADDQ(hlp_v2, acc3_v2)
	ADCQ(Imm(0), mul1_v2)
	ADDQ(mul0_v2, acc3_v2)
	ADCQ(Imm(0), mul1_v2)
	MOVQ(mul1_v2, hlp_v2)

	MOVQ(acc6_v2, mul0_v2)
	MULQ(t2_v2)
	ADDQ(hlp_v2, acc4_v2)
	ADCQ(Imm(0), mul1_v2)
	ADDQ(mul0_v2, acc4_v2)
	ADCQ(Imm(0), mul1_v2)
	MOVQ(mul1_v2, hlp_v2)

	MOVQ(acc6_v2, mul0_v2)
	MULQ(t3_v2)
	ADDQ(hlp_v2, acc5_v2)
	ADCQ(Imm(0), mul1_v2)
	ADDQ(mul0_v2, acc5_v2)
	ADCQ(Imm(0), mul1_v2)
	MOVQ(mul1_v2, acc6_v2)

	MOVQ(acc7_v2, mul0_v2)
	MULQ(t0_v2)
	ADDQ(mul0_v2, acc3_v2)
	ADCQ(Imm(0), mul1_v2)
	MOVQ(mul1_v2, hlp_v2)

	MOVQ(acc7_v2, mul0_v2)
	MULQ(t1_v2)
	ADDQ(hlp_v2, acc4_v2)
	ADCQ(Imm(0), mul1_v2)
	ADDQ(mul0_v2, acc4_v2)
	ADCQ(Imm(0), mul1_v2)
	MOVQ(mul1_v2, hlp_v2)

	MOVQ(acc7_v2, mul0_v2)
	MULQ(t2_v2)
	ADDQ(hlp_v2, acc5_v2)
	ADCQ(Imm(0), mul1_v2)
	ADDQ(mul0_v2, acc5_v2)
	ADCQ(Imm(0), mul1_v2)
	MOVQ(mul1_v2, hlp_v2)

	MOVQ(acc7_v2, mul0_v2)
	MULQ(t3_v2)
	ADDQ(hlp_v2, acc6_v2)
	ADCQ(Imm(0), mul1_v2)
	ADDQ(mul0_v2, acc6_v2)
	ADCQ(Imm(0), mul1_v2)
	MOVQ(mul1_v2, acc7_v2)

	Comment("First reduction step")
	MOVQ(acc0_v2, mul0_v2)
	MOVQ(acc0_v2, hlp_v2)
	SHLQ(Imm(32), acc0_v2)
	p256const1 := p256const1_DATA()
	MULQ(p256const1)
	SHRQ(Imm(32), hlp_v2)
	ADDQ(acc0_v2, acc1_v2)
	ADCQ(hlp_v2, acc2_v2)
	ADCQ(mul0_v2, acc3_v2)
	ADCQ(Imm(0), mul1_v2)
	MOVQ(mul1_v2, acc0_v2)

	Comment("Second reduction step")
	MOVQ(acc1_v2, mul0_v2)
	MOVQ(acc1_v2, hlp_v2)
	SHLQ(Imm(32), acc1_v2)
	MULQ(p256const1)
	SHRQ(Imm(32), hlp_v2)
	ADDQ(acc1_v2, acc2_v2)
	ADCQ(hlp_v2, acc3_v2)
	ADCQ(mul0_v2, acc0_v2)
	ADCQ(Imm(0), mul1_v2)
	MOVQ(mul1_v2, acc1_v2)

	Comment("Third reduction step")
	MOVQ(acc2_v2, mul0_v2)
	MOVQ(acc2_v2, hlp_v2)
	SHLQ(Imm(32), acc2_v2)
	MULQ(p256const1)
	SHRQ(Imm(32), hlp_v2)
	ADDQ(acc2_v2, acc3_v2)
	ADCQ(hlp_v2, acc0_v2)
	ADCQ(mul0_v2, acc1_v2)
	ADCQ(Imm(0), mul1_v2)
	MOVQ(mul1_v2, acc2_v2)

	Comment("Last reduction step")
	MOVQ(acc3_v2, mul0_v2)
	MOVQ(acc3_v2, hlp_v2)
	SHLQ(Imm(32), acc3_v2)
	MULQ(p256const1)
	SHRQ(Imm(32), hlp_v2)
	ADDQ(acc3_v2, acc0_v2)
	ADCQ(hlp_v2, acc1_v2)
	ADCQ(mul0_v2, acc2_v2)
	ADCQ(Imm(0), mul1_v2)
	MOVQ(mul1_v2, acc3_v2)
	MOVQ(U32(0), RBP)

	Comment("Add bits [511:256] of the result")
	ADCQ(acc0_v2, acc4_v2)
	ADCQ(acc1_v2, acc5_v2)
	ADCQ(acc2_v2, acc6_v2)
	ADCQ(acc3_v2, acc7_v2)
	ADCQ(Imm(0), hlp_v2)

	Comment("Copy result")
	MOVQ(acc4_v2, acc0_v2)
	MOVQ(acc5_v2, acc1_v2)
	MOVQ(acc6_v2, acc2_v2)
	MOVQ(acc7_v2, acc3_v2)

	Comment("Subtract p256")
	SUBQ(I8(-1), acc4_v2)
	p256const0 := p256const0_DATA()
	SBBQ(p256const0, acc5_v2)
	SBBQ(Imm(0), acc6_v2)
	SBBQ(p256const1, acc7_v2)
	SBBQ(Imm(0), hlp_v2)

	Comment("If the result of the subtraction is negative, restore the previous result")
	CMOVQCS(acc0_v2, acc4_v2)
	CMOVQCS(acc1_v2, acc5_v2)
	CMOVQCS(acc2_v2, acc6_v2)
	CMOVQCS(acc3_v2, acc7_v2)

	RET()
}

func p256SqrInternal() {
	Function("p256SqrInternal")
	Attributes(NOSPLIT)

	MOVQ(acc4_v2, mul0_v2)
	MULQ(acc5_v2)
	MOVQ(mul0_v2, acc1_v2)
	MOVQ(mul1_v2, acc2_v2)

	MOVQ(acc4_v2, mul0_v2)
	MULQ(acc6_v2)
	ADDQ(mul0_v2, acc2_v2)
	ADCQ(Imm(0), mul1_v2)
	MOVQ(mul1_v2, acc3_v2)

	MOVQ(acc4_v2, mul0_v2)
	MULQ(acc7_v2)
	ADDQ(mul0_v2, acc3_v2)
	ADCQ(Imm(0), mul1_v2)
	MOVQ(mul1_v2, t0_v2)

	MOVQ(acc5_v2, mul0_v2)
	MULQ(acc6_v2)
	ADDQ(mul0_v2, acc3_v2)
	ADCQ(Imm(0), mul1_v2)
	MOVQ(mul1_v2, hlp_v2)

	MOVQ(acc5_v2, mul0_v2)
	MULQ(acc7_v2)
	ADDQ(hlp_v2, t0_v2)
	ADCQ(Imm(0), mul1_v2)
	ADDQ(mul0_v2, t0_v2)
	ADCQ(Imm(0), mul1_v2)
	MOVQ(mul1_v2, t1_v2)

	MOVQ(acc6_v2, mul0_v2)
	MULQ(acc7_v2)
	ADDQ(mul0_v2, t1_v2)
	ADCQ(Imm(0), mul1_v2)
	MOVQ(mul1_v2, t2_v2)
	XORQ(t3_v2, t3_v2)

	Comment("*2")
	ADDQ(acc1_v2, acc1_v2)
	ADCQ(acc2_v2, acc2_v2)
	ADCQ(acc3_v2, acc3_v2)
	ADCQ(t0_v2, t0_v2)
	ADCQ(t1_v2, t1_v2)
	ADCQ(t2_v2, t2_v2)
	ADCQ(Imm(0), t3_v2)

	Comment("Missing products")
	MOVQ(acc4_v2, mul0_v2)
	MULQ(mul0_v2)
	MOVQ(mul0_v2, acc0_v2)
	MOVQ(RDX, acc4_v2)

	MOVQ(acc5_v2, mul0_v2)
	MULQ(mul0_v2)
	ADDQ(acc4_v2, acc1_v2)
	ADCQ(mul0_v2, acc2_v2)
	ADCQ(Imm(0), RDX)
	MOVQ(RDX, acc4_v2)

	MOVQ(acc6_v2, mul0_v2)
	MULQ(mul0_v2)
	ADDQ(acc4_v2, acc3_v2)
	ADCQ(mul0_v2, t0_v2)
	ADCQ(Imm(0), RDX)
	MOVQ(RDX, acc4_v2)

	MOVQ(acc7_v2, mul0_v2)
	MULQ(mul0_v2)
	ADDQ(acc4_v2, t1_v2)
	ADCQ(mul0_v2, t2_v2)
	ADCQ(RDX, t3_v2)

	Comment("First reduction step")
	MOVQ(acc0_v2, mul0_v2)
	MOVQ(acc0_v2, hlp_v2)
	SHLQ(Imm(32), acc0_v2)
	p256const1 := p256const1_DATA()
	MULQ(p256const1)
	SHRQ(Imm(32), hlp_v2)
	ADDQ(acc0_v2, acc1_v2)
	ADCQ(hlp_v2, acc2_v2)
	ADCQ(mul0_v2, acc3_v2)
	ADCQ(Imm(0), mul1_v2)
	MOVQ(mul1_v2, acc0_v2)

	Comment("Second reduction step")
	MOVQ(acc1_v2, mul0_v2)
	MOVQ(acc1_v2, hlp_v2)
	SHLQ(Imm(32), acc1_v2)
	MULQ(p256const1)
	SHRQ(Imm(32), hlp_v2)
	ADDQ(acc1_v2, acc2_v2)
	ADCQ(hlp_v2, acc3_v2)
	ADCQ(mul0_v2, acc0_v2)
	ADCQ(Imm(0), mul1_v2)
	MOVQ(mul1_v2, acc1_v2)

	Comment("Third reduction step")
	MOVQ(acc2_v2, mul0_v2)
	MOVQ(acc2_v2, hlp_v2)
	SHLQ(Imm(32), acc2_v2)
	MULQ(p256const1)
	SHRQ(Imm(32), hlp_v2)
	ADDQ(acc2_v2, acc3_v2)
	ADCQ(hlp_v2, acc0_v2)
	ADCQ(mul0_v2, acc1_v2)
	ADCQ(Imm(0), mul1_v2)
	MOVQ(mul1_v2, acc2_v2)

	Comment("Last reduction step")
	MOVQ(acc3_v2, mul0_v2)
	MOVQ(acc3_v2, hlp_v2)
	SHLQ(Imm(32), acc3_v2)
	MULQ(p256const1)
	SHRQ(Imm(32), hlp_v2)
	ADDQ(acc3_v2, acc0_v2)
	ADCQ(hlp_v2, acc1_v2)
	ADCQ(mul0_v2, acc2_v2)
	ADCQ(Imm(0), mul1_v2)
	MOVQ(mul1_v2, acc3_v2)
	MOVQ(U32(0), RBP)

	Comment("Add bits [511:256] of the result")
	ADCQ(acc0_v2, t0_v2)
	ADCQ(acc1_v2, t1_v2)
	ADCQ(acc2_v2, t2_v2)
	ADCQ(acc3_v2, t3_v2)
	ADCQ(Imm(0), hlp_v2)

	Comment("Copy result")
	MOVQ(t0_v2, acc4_v2)
	MOVQ(t1_v2, acc5_v2)
	MOVQ(t2_v2, acc6_v2)
	MOVQ(t3_v2, acc7_v2)

	Comment("Subtract p256")
	SUBQ(I8(-1), acc4_v2)
	p256const0 := p256const0_DATA()
	SBBQ(p256const0, acc5_v2)
	SBBQ(Imm(0), acc6_v2)
	SBBQ(p256const1, acc7_v2)
	SBBQ(Imm(0), hlp_v2)

	Comment("If the result of the subtraction is negative, restore the previous result")
	CMOVQCS(t0_v2, acc4_v2)
	CMOVQCS(t1_v2, acc5_v2)
	CMOVQCS(t2_v2, acc6_v2)
	CMOVQCS(t3_v2, acc7_v2)

	RET()
}

func p256MulBy2Inline() {
	XORQ(mul0_v2, mul0_v2)
	ADDQ(acc4_v2, acc4_v2)
	ADCQ(acc5_v2, acc5_v2)
	ADCQ(acc6_v2, acc6_v2)
	ADCQ(acc7_v2, acc7_v2)
	ADCQ(I8(0), mul0_v2)
	MOVQ(acc4_v2, t0_v2)
	MOVQ(acc5_v2, t1_v2)
	MOVQ(acc6_v2, t2_v2)
	MOVQ(acc7_v2, t3_v2)
	SUBQ(I8(-1), t0_v2)
	p256const0 := p256const0_DATA()
	SBBQ(p256const0, t1_v2)
	SBBQ(I8(0), t2_v2)
	p256const1 := p256const1_DATA()
	SBBQ(p256const1, t3_v2)
	SBBQ(I8(0), mul0_v2)
	CMOVQCS(acc4_v2, t0_v2)
	CMOVQCS(acc5_v2, t1_v2)
	CMOVQCS(acc6_v2, t2_v2)
	CMOVQCS(acc7_v2, t3_v2)
}

func p256AddInline() {
	XORQ(mul0_v2, mul0_v2)
	ADDQ(t0_v2, acc4_v2)
	ADCQ(t1_v2, acc5_v2)
	ADCQ(t2_v2, acc6_v2)
	ADCQ(t3_v2, acc7_v2)
	ADCQ(I8(0), mul0_v2)
	MOVQ(acc4_v2, t0_v2)
	MOVQ(acc5_v2, t1_v2)
	MOVQ(acc6_v2, t2_v2)
	MOVQ(acc7_v2, t3_v2)
	SUBQ(I8(-1), t0_v2)
	p256const0 := p256const0_DATA()
	SBBQ(p256const0, t1_v2)
	SBBQ(I8(0), t2_v2)
	p256const1 := p256const1_DATA()
	SBBQ(p256const1, t3_v2)
	SBBQ(I8(0), mul0_v2)
	CMOVQCS(acc4_v2, t0_v2)
	CMOVQCS(acc5_v2, t1_v2)
	CMOVQCS(acc6_v2, t2_v2)
	CMOVQCS(acc7_v2, t3_v2)
}

/* ---------------------------------------*/

type MemFunc func(off int) Mem

func LDacc(src MemFunc) {
	MOVQ(src(8*0), acc4_v2)
	MOVQ(src(8*1), acc5_v2)
	MOVQ(src(8*2), acc6_v2)
	MOVQ(src(8*3), acc7_v2)
}

func LDt(src MemFunc) {
	MOVQ(src(8*0), t0_v2)
	MOVQ(src(8*1), t1_v2)
	MOVQ(src(8*2), t2_v2)
	MOVQ(src(8*3), t3_v2)
}

func ST(dst MemFunc) {
	MOVQ(acc4_v2, dst(8*0))
	MOVQ(acc5_v2, dst(8*1))
	MOVQ(acc6_v2, dst(8*2))
	MOVQ(acc7_v2, dst(8*3))
}

func STt(dst MemFunc) {
	MOVQ(t0_v2, dst(8*0))
	MOVQ(t1_v2, dst(8*1))
	MOVQ(t2_v2, dst(8*2))
	MOVQ(t3_v2, dst(8*3))
}

func acc2t() {
	MOVQ(acc4_v2, t0_v2)
	MOVQ(acc5_v2, t1_v2)
	MOVQ(acc6_v2, t2_v2)
	MOVQ(acc7_v2, t3_v2)
}

func t2acc() {
	MOVQ(t0_v2, acc4_v2)
	MOVQ(t1_v2, acc5_v2)
	MOVQ(t2_v2, acc6_v2)
	MOVQ(t3_v2, acc7_v2)
}

/* ---------------------------------------*/

// These functions exist as #define macros in the reference implementation.
//
// In the reference assembly, these macros are later undefined and redefined.
// They are implemented here as versioned functions.

func x1in_v1(off int) Mem  { return Mem{Base: SP}.Offset(32*0 + off) }
func y1in_v1(off int) Mem  { return Mem{Base: SP}.Offset(32*1 + off) }
func z1in_v1(off int) Mem  { return Mem{Base: SP}.Offset(32*2 + off) }
func x2in_v1(off int) Mem  { return Mem{Base: SP}.Offset(32*3 + off) }
func y2in_v1(off int) Mem  { return Mem{Base: SP}.Offset(32*4 + off) }
func xout_v1(off int) Mem  { return Mem{Base: SP}.Offset(32*5 + off) }
func yout_v1(off int) Mem  { return Mem{Base: SP}.Offset(32*6 + off) }
func zout_v1(off int) Mem  { return Mem{Base: SP}.Offset(32*7 + off) }
func s2_v1(off int) Mem    { return Mem{Base: SP}.Offset(32*8 + off) }
func z1sqr_v1(off int) Mem { return Mem{Base: SP}.Offset(32*9 + off) }
func h_v1(off int) Mem     { return Mem{Base: SP}.Offset(32*10 + off) }
func r_v1(off int) Mem     { return Mem{Base: SP}.Offset(32*11 + off) }
func hsqr_v1(off int) Mem  { return Mem{Base: SP}.Offset(32*12 + off) }
func rsqr_v1(off int) Mem  { return Mem{Base: SP}.Offset(32*13 + off) }
func hcub_v1(off int) Mem  { return Mem{Base: SP}.Offset(32*14 + off) }

var (
	rptr_v1      Mem = Mem{Base: SP}.Offset(32*15 + 0)
	sel_save_v1      = Mem{Base: SP}.Offset(32*15 + 8)
	zero_save_v1     = Mem{Base: SP}.Offset(32*15 + 8 + 4)
)

// Implements:
//
//	func p256PointAddAffineAsm(res, in1 *P256Point, in2 *p256AffinePoint, sign, sel, zero int)
func p256PointAddAffineAsm() {
	Implement("p256PointAddAffineAsm")
	AllocLocal(512)

	Load(Param("res"), RAX)
	Load(Param("in1"), RBX)
	Load(Param("in2"), RCX)
	Load(Param("sign"), RDX)
	Load(Param("sel"), t1_v2)
	Load(Param("zero"), t2_v2)

	MOVOU(Mem{Base: BX}.Offset(16*0), X0)
	MOVOU(Mem{Base: BX}.Offset(16*1), X1)
	MOVOU(Mem{Base: BX}.Offset(16*2), X2)
	MOVOU(Mem{Base: BX}.Offset(16*3), X3)
	MOVOU(Mem{Base: BX}.Offset(16*4), X4)
	MOVOU(Mem{Base: BX}.Offset(16*5), X5)

	MOVOU(X0, x1in_v1(16*0))
	MOVOU(X1, x1in_v1(16*1))
	MOVOU(X2, y1in_v1(16*0))
	MOVOU(X3, y1in_v1(16*1))
	MOVOU(X4, z1in_v1(16*0))
	MOVOU(X5, z1in_v1(16*1))

	MOVOU(Mem{Base: CX}.Offset(16*0), X0)
	MOVOU(Mem{Base: CX}.Offset(16*1), X1)

	MOVOU(X0, x2in_v1(16*0))
	MOVOU(X1, x2in_v1(16*1))

	Comment("Store pointer to result")
	MOVQ(mul0_v2, rptr_v1)

	// Hack to get Avo to emit:
	// 	MOVL t1, sel_save_v1
	Instruction(&ir.Instruction{
		Opcode:   "MOVL",
		Operands: []Op{t1_v2, sel_save_v1},
	})

	// Hack to get Avo to emit:
	// 	MOVL t2_v2, zero_save_v1
	Instruction(&ir.Instruction{
		Opcode:   "MOVL",
		Operands: []Op{t2_v2, zero_save_v1},
	})

	Comment("Negate y2in based on sign")
	MOVQ(Mem{Base: CX}.Offset(16*2+8*0), acc4_v2)
	MOVQ(Mem{Base: CX}.Offset(16*2+8*1), acc5_v2)
	MOVQ(Mem{Base: CX}.Offset(16*2+8*2), acc6_v2)
	MOVQ(Mem{Base: CX}.Offset(16*2+8*3), acc7_v2)
	MOVQ(I32(-1), acc0_v2)
	p256const0 := p256const0_DATA()
	MOVQ(p256const0, acc1_v2)
	MOVQ(U32(0), acc2_v2)
	p256const1 := p256const1_DATA()
	MOVQ(p256const1, acc3_v2)
	XORQ(mul0_v2, mul0_v2)

	Comment("Speculatively subtract")
	SUBQ(acc4_v2, acc0_v2)
	SBBQ(acc5_v2, acc1_v2)
	SBBQ(acc6_v2, acc2_v2)
	SBBQ(acc7_v2, acc3_v2)
	SBBQ(Imm(0), mul0_v2)
	MOVQ(acc0_v2, t0_v2)
	MOVQ(acc1_v2, t1_v2)
	MOVQ(acc2_v2, t2_v2)
	MOVQ(acc3_v2, t3_v2)

	Comment("Add in case the operand was > p256")
	ADDQ(I8(-1), acc0_v2)
	ADCQ(p256const0, acc1_v2)
	ADCQ(Imm(0), acc2_v2)
	ADCQ(p256const1, acc3_v2)
	ADCQ(Imm(0), mul0_v2)
	CMOVQNE(t0_v2, acc0_v2)
	CMOVQNE(t1_v2, acc1_v2)
	CMOVQNE(t2_v2, acc2_v2)
	CMOVQNE(t3_v2, acc3_v2)

	Comment("If condition is 0, keep original value")
	TESTQ(RDX, RDX)
	CMOVQEQ(acc4_v2, acc0_v2)
	CMOVQEQ(acc5_v2, acc1_v2)
	CMOVQEQ(acc6_v2, acc2_v2)
	CMOVQEQ(acc7_v2, acc3_v2)

	Comment("Store result")
	MOVQ(acc0_v2, y2in_v1(8*0))
	MOVQ(acc1_v2, y2in_v1(8*1))
	MOVQ(acc2_v2, y2in_v1(8*2))
	MOVQ(acc3_v2, y2in_v1(8*3))

	Comment("Begin point add")
	LDacc(z1in_v1)
	CALL(LabelRef("p256SqrInternal(SB)")) //                  z1ˆ2
	ST(z1sqr_v1)

	LDt(x2in_v1)
	CALL(LabelRef("p256MulInternal(SB)")) //             x2 * z1ˆ2

	LDt(x1in_v1)
	CALL(LabelRef("p256SubInternal(SB)")) //          h = u2 - u1)
	ST(h_v1)

	LDt(z1in_v1)
	CALL(LabelRef("p256MulInternal(SB)")) //           z3 = h * z1
	ST(zout_v1)

	LDacc(z1sqr_v1)
	CALL(LabelRef("p256MulInternal(SB)")) //                  z1ˆ3

	LDt(y2in_v1)
	CALL(LabelRef("p256MulInternal(SB)")) //        s2 = y2 * z1ˆ3
	ST(s2_v1)

	LDt(y1in_v1)
	CALL(LabelRef("p256SubInternal(SB)")) //          r = s2 - s1)
	ST(r_v1)

	CALL(LabelRef("p256SqrInternal(SB)")) //            rsqr = rˆ2
	ST(rsqr_v1)

	LDacc(h_v1)
	CALL(LabelRef("p256SqrInternal(SB)")) //            hsqr = hˆ2
	ST(hsqr_v1)

	LDt(h_v1)
	CALL(LabelRef("p256MulInternal(SB)")) //            hcub = hˆ3
	ST(hcub_v1)

	LDt(y1in_v1)
	CALL(LabelRef("p256MulInternal(SB)")) //             y1 * hˆ3
	ST(s2_v1)

	LDacc(x1in_v1)
	LDt(hsqr_v1)
	CALL(LabelRef("p256MulInternal(SB)")) //             u1 * hˆ2
	ST(h_v1)

	p256MulBy2Inline() //                    u1 * hˆ2 * 2, inline
	LDacc(rsqr_v1)
	CALL(LabelRef("p256SubInternal(SB)")) //  rˆ2 - u1 * hˆ2 * 2)

	LDt(hcub_v1)
	CALL(LabelRef("p256SubInternal(SB)"))
	ST(xout_v1)

	MOVQ(acc4_v2, t0_v2)
	MOVQ(acc5_v2, t1_v2)
	MOVQ(acc6_v2, t2_v2)
	MOVQ(acc7_v2, t3_v2)
	LDacc(h_v1)
	CALL(LabelRef("p256SubInternal(SB)"))

	LDt(r_v1)
	CALL(LabelRef("p256MulInternal(SB)"))

	LDt(s2_v1)
	CALL(LabelRef("p256SubInternal(SB)"))
	ST(yout_v1)

	Comment("Load stored values from stack")
	MOVQ(rptr_v1, RAX)
	MOVL(sel_save_v1, EBX)
	MOVL(zero_save_v1, ECX)

	Comment("The result is not valid if (sel == 0), conditional choose")
	MOVOU(xout_v1(16*0), X0)
	MOVOU(xout_v1(16*1), X1)
	MOVOU(yout_v1(16*0), X2)
	MOVOU(yout_v1(16*1), X3)
	MOVOU(zout_v1(16*0), X4)
	MOVOU(zout_v1(16*1), X5)

	// Hack to get Avo to emit:
	// 	MOVL BX, X6
	Instruction(&ir.Instruction{
		Opcode:   "MOVL",
		Operands: []Op{EBX, X6},
	})

	// Hack to get Avo to emit:
	// 	MOVL CX, X7
	Instruction(&ir.Instruction{
		Opcode:   "MOVL",
		Operands: []Op{ECX, X7},
	})

	PXOR(X8, X8)
	PCMPEQL(X9, X9)

	PSHUFD(Imm(0), X6, X6)
	PSHUFD(Imm(0), X7, X7)

	PCMPEQL(X8, X6)
	PCMPEQL(X8, X7)

	MOVOU(X6, X15)
	PANDN(X9, X15)

	MOVOU(x1in_v1(16*0), X9)
	MOVOU(x1in_v1(16*1), X10)
	MOVOU(y1in_v1(16*0), X11)
	MOVOU(y1in_v1(16*1), X12)
	MOVOU(z1in_v1(16*0), X13)
	MOVOU(z1in_v1(16*1), X14)

	PAND(X15, X0)
	PAND(X15, X1)
	PAND(X15, X2)
	PAND(X15, X3)
	PAND(X15, X4)
	PAND(X15, X5)

	PAND(X6, X9)
	PAND(X6, X10)
	PAND(X6, X11)
	PAND(X6, X12)
	PAND(X6, X13)
	PAND(X6, X14)

	PXOR(X9, X0)
	PXOR(X10, X1)
	PXOR(X11, X2)
	PXOR(X12, X3)
	PXOR(X13, X4)
	PXOR(X14, X5)

	Comment("Similarly if zero == 0")
	PCMPEQL(X9, X9)
	MOVOU(X7, X15)
	PANDN(X9, X15)

	MOVOU(x2in_v1(16*0), X9)
	MOVOU(x2in_v1(16*1), X10)
	MOVOU(y2in_v1(16*0), X11)
	MOVOU(y2in_v1(16*1), X12)
	p256one := p256one_DATA()
	MOVOU(p256one.Offset(0x00), X13)
	MOVOU(p256one.Offset(0x10), X14)

	PAND(X15, X0)
	PAND(X15, X1)
	PAND(X15, X2)
	PAND(X15, X3)
	PAND(X15, X4)
	PAND(X15, X5)

	PAND(X7, X9)
	PAND(X7, X10)
	PAND(X7, X11)
	PAND(X7, X12)
	PAND(X7, X13)
	PAND(X7, X14)

	PXOR(X9, X0)
	PXOR(X10, X1)
	PXOR(X11, X2)
	PXOR(X12, X3)
	PXOR(X13, X4)
	PXOR(X14, X5)

	Comment("Finally output the result")
	MOVOU(X0, Mem{Base: AX}.Offset(16*0))
	MOVOU(X1, Mem{Base: AX}.Offset(16*1))
	MOVOU(X2, Mem{Base: AX}.Offset(16*2))
	MOVOU(X3, Mem{Base: AX}.Offset(16*3))
	MOVOU(X4, Mem{Base: AX}.Offset(16*4))
	MOVOU(X5, Mem{Base: AX}.Offset(16*5))
	MOVQ(U32(0), rptr_v1)

	RET()
}

// p256IsZero returns 1 in AX if [acc4..acc7] represents zero and zero
// otherwise. It writes to [acc4..acc7], t0 and t1.
func p256IsZero() {
	Function("p256IsZero")
	Attributes(NOSPLIT)

	Comment("AX contains a flag that is set if the input is zero.")
	XORQ(RAX, RAX)
	MOVQ(U32(1), t1_v2)

	Comment("Check whether [acc4..acc7] are all zero.")
	MOVQ(acc4_v2, t0_v2)
	ORQ(acc5_v2, t0_v2)
	ORQ(acc6_v2, t0_v2)
	ORQ(acc7_v2, t0_v2)

	Comment("Set the zero flag if so. (CMOV of a constant to a register doesn't")
	Comment("appear to be supported in Go. Thus t1 = 1.)")
	CMOVQEQ(t1_v2, RAX)

	Comment("XOR [acc4..acc7] with P and compare with zero again.")
	XORQ(I8(-1), acc4_v2)
	p256const0 := p256const0_DATA()
	XORQ(p256const0, acc5_v2)
	p256const1 := p256const1_DATA()
	XORQ(p256const1, acc7_v2)
	ORQ(acc5_v2, acc4_v2)
	ORQ(acc6_v2, acc4_v2)
	ORQ(acc7_v2, acc4_v2)

	Comment("Set the zero flag if so.")
	CMOVQEQ(t1_v2, RAX)
	RET()
}

func x1in_v2(off int) Mem { return Mem{Base: SP}.Offset(32*0 + off) }
func y1in_v2(off int) Mem { return Mem{Base: SP}.Offset(32*1 + off) }
func z1in_v2(off int) Mem { return Mem{Base: SP}.Offset(32*2 + off) }
func x2in_v2(off int) Mem { return Mem{Base: SP}.Offset(32*3 + off) }
func y2in_v2(off int) Mem { return Mem{Base: SP}.Offset(32*4 + off) }
func z2in_v2(off int) Mem { return Mem{Base: SP}.Offset(32*5 + off) }

func xout_v2(off int) Mem { return Mem{Base: SP}.Offset(32*6 + off) }
func yout_v2(off int) Mem { return Mem{Base: SP}.Offset(32*7 + off) }
func zout_v2(off int) Mem { return Mem{Base: SP}.Offset(32*8 + off) }

func u1_v2(off int) Mem    { return Mem{Base: SP}.Offset(32*9 + off) }
func u2_v2(off int) Mem    { return Mem{Base: SP}.Offset(32*10 + off) }
func s1_v2(off int) Mem    { return Mem{Base: SP}.Offset(32*11 + off) }
func s2_v2(off int) Mem    { return Mem{Base: SP}.Offset(32*12 + off) }
func z1sqr_v2(off int) Mem { return Mem{Base: SP}.Offset(32*13 + off) }
func z2sqr_v2(off int) Mem { return Mem{Base: SP}.Offset(32*14 + off) }
func h_v2(off int) Mem     { return Mem{Base: SP}.Offset(32*15 + off) }
func r_v2(off int) Mem     { return Mem{Base: SP}.Offset(32*16 + off) }
func hsqr_v2(off int) Mem  { return Mem{Base: SP}.Offset(32*17 + off) }
func rsqr_v2(off int) Mem  { return Mem{Base: SP}.Offset(32*18 + off) }
func hcub_v2(off int) Mem  { return Mem{Base: SP}.Offset(32*19 + off) }

var (
	rptr_v2      Mem = Mem{Base: SP}.Offset(32 * 20)
	points_eq_v2     = Mem{Base: SP}.Offset(32*20 + 8)
)

// Implements:
//
//	func p256PointAddAsm(res, in1, in2 *P256Point) int
//
// See https://hyperelliptic.org/EFD/g1p/auto-shortw-jacobian-3.html#addition-add-2007-bl
func p256PointAddAsm() {
	Implement("p256PointAddAsm")
	AllocLocal(680)

	Comment("Move input to stack in order to free registers")
	Load(Param("res"), RAX)
	Load(Param("in1"), RBX)
	Load(Param("in2"), RCX)

	MOVOU(Mem{Base: BX}.Offset(16*0), X0)
	MOVOU(Mem{Base: BX}.Offset(16*1), X1)
	MOVOU(Mem{Base: BX}.Offset(16*2), X2)
	MOVOU(Mem{Base: BX}.Offset(16*3), X3)
	MOVOU(Mem{Base: BX}.Offset(16*4), X4)
	MOVOU(Mem{Base: BX}.Offset(16*5), X5)

	MOVOU(X0, x1in_v2(16*0))
	MOVOU(X1, x1in_v2(16*1))
	MOVOU(X2, y1in_v2(16*0))
	MOVOU(X3, y1in_v2(16*1))
	MOVOU(X4, z1in_v2(16*0))
	MOVOU(X5, z1in_v2(16*1))

	MOVOU(Mem{Base: CX}.Offset(16*0), X0)
	MOVOU(Mem{Base: CX}.Offset(16*1), X1)
	MOVOU(Mem{Base: CX}.Offset(16*2), X2)
	MOVOU(Mem{Base: CX}.Offset(16*3), X3)
	MOVOU(Mem{Base: CX}.Offset(16*4), X4)
	MOVOU(Mem{Base: CX}.Offset(16*5), X5)

	MOVOU(X0, x2in_v2(16*0))
	MOVOU(X1, x2in_v2(16*1))
	MOVOU(X2, y2in_v2(16*0))
	MOVOU(X3, y2in_v2(16*1))
	MOVOU(X4, z2in_v2(16*0))
	MOVOU(X5, z2in_v2(16*1))

	Comment("Store pointer to result")
	MOVQ(RAX, rptr_v2)

	Comment("Begin point add")
	LDacc(z2in_v2)
	CALL(LabelRef("p256SqrInternal(SB)")) //               z2ˆ2
	ST(z2sqr_v2)
	LDt(z2in_v2)
	CALL(LabelRef("p256MulInternal(SB)")) //               z2ˆ3
	LDt(y1in_v2)
	CALL(LabelRef("p256MulInternal(SB)")) //       s1 = z2ˆ3*y1
	ST(s1_v2)

	LDacc(z1in_v2)
	CALL(LabelRef("p256SqrInternal(SB)")) //               z1ˆ2
	ST(z1sqr_v2)
	LDt(z1in_v2)
	CALL(LabelRef("p256MulInternal(SB)")) //               z1ˆ3
	LDt(y2in_v2)
	CALL(LabelRef("p256MulInternal(SB)")) //       s2 = z1ˆ3*y2
	ST(s2_v2)

	LDt(s1_v2)
	CALL(LabelRef("p256SubInternal(SB)")) //        r = s2 - s1
	ST(r_v2)
	CALL(LabelRef("p256IsZero(SB)"))
	MOVQ(RAX, points_eq_v2)

	LDacc(z2sqr_v2)
	LDt(x1in_v2)
	CALL(LabelRef("p256MulInternal(SB)")) //     u1 = x1 * z2ˆ2
	ST(u1_v2)
	LDacc(z1sqr_v2)
	LDt(x2in_v2)
	CALL(LabelRef("p256MulInternal(SB)")) //     u2 = x2 * z1ˆ2
	ST(u2_v2)

	LDt(u1_v2)
	CALL(LabelRef("p256SubInternal(SB)")) //        h = u2 - u1
	ST(h_v2)
	CALL(LabelRef("p256IsZero(SB)"))
	ANDQ(points_eq_v2, RAX)
	MOVQ(RAX, points_eq_v2)

	LDacc(r_v2)
	CALL(LabelRef("p256SqrInternal(SB)")) //         rsqr = rˆ2
	ST(rsqr_v2)

	LDacc(h_v2)
	CALL(LabelRef("p256SqrInternal(SB)")) //         hsqr = hˆ2
	ST(hsqr_v2)

	LDt(h_v2)
	CALL(LabelRef("p256MulInternal(SB)")) //         hcub = hˆ3
	ST(hcub_v2)

	LDt(s1_v2)
	CALL(LabelRef("p256MulInternal(SB)"))
	ST(s2_v2)

	LDacc(z1in_v2)
	LDt(z2in_v2)
	CALL(LabelRef("p256MulInternal(SB)")) //            z1 * z2
	LDt(h_v2)
	CALL(LabelRef("p256MulInternal(SB)")) //        z1 * z2 * h
	ST(zout_v2)

	LDacc(hsqr_v2)
	LDt(u1_v2)
	CALL(LabelRef("p256MulInternal(SB)")) //           hˆ2 * u1
	ST(u2_v2)

	p256MulBy2Inline() //                  u1 * hˆ2 * 2, inline
	LDacc(rsqr_v2)
	CALL(LabelRef("p256SubInternal(SB)")) // rˆ2 - u1 * hˆ2 * 2

	LDt(hcub_v2)
	CALL(LabelRef("p256SubInternal(SB)"))
	ST(xout_v2)

	MOVQ(acc4_v2, t0_v2)
	MOVQ(acc5_v2, t1_v2)
	MOVQ(acc6_v2, t2_v2)
	MOVQ(acc7_v2, t3_v2)
	LDacc(u2_v2)
	CALL(LabelRef("p256SubInternal(SB)"))

	LDt(r_v2)
	CALL(LabelRef("p256MulInternal(SB)"))

	LDt(s2_v2)
	CALL(LabelRef("p256SubInternal(SB)"))
	ST(yout_v2)

	MOVOU(xout_v2(16*0), X0)
	MOVOU(xout_v2(16*1), X1)
	MOVOU(yout_v2(16*0), X2)
	MOVOU(yout_v2(16*1), X3)
	MOVOU(zout_v2(16*0), X4)
	MOVOU(zout_v2(16*1), X5)

	Comment("Finally output the result")
	MOVQ(rptr_v2, RAX)
	MOVQ(U32(0), rptr_v2)
	MOVOU(X0, Mem{Base: AX}.Offset(16*0))
	MOVOU(X1, Mem{Base: AX}.Offset(16*1))
	MOVOU(X2, Mem{Base: AX}.Offset(16*2))
	MOVOU(X3, Mem{Base: AX}.Offset(16*3))
	MOVOU(X4, Mem{Base: AX}.Offset(16*4))
	MOVOU(X5, Mem{Base: AX}.Offset(16*5))

	MOVQ(points_eq_v2, RAX)
	ret := NewParamAddr("ret", 24)
	MOVQ(RAX, ret)

	RET()
}

func x(off int) Mem { return Mem{Base: SP}.Offset(32*0 + off) }
func y(off int) Mem { return Mem{Base: SP}.Offset(32*1 + off) }
func z(off int) Mem { return Mem{Base: SP}.Offset(32*2 + off) }

func s(off int) Mem    { return Mem{Base: SP}.Offset(32*3 + off) }
func m(off int) Mem    { return Mem{Base: SP}.Offset(32*4 + off) }
func zsqr(off int) Mem { return Mem{Base: SP}.Offset(32*5 + off) }
func tmp(off int) Mem  { return Mem{Base: SP}.Offset(32*6 + off) }

var rptr_v3 = Mem{Base: SP}.Offset(32 * 7)

// Implements:
//
//	func p256PointDoubleAsm(res, in *P256Point)
func p256PointDoubleAsm() {
	Implement("p256PointDoubleAsm")
	Attributes(NOSPLIT)
	AllocLocal(256)

	Load(Param("res"), RAX)
	Load(Param("in"), RBX)

	MOVOU(Mem{Base: BX}.Offset(16*0), X0)
	MOVOU(Mem{Base: BX}.Offset(16*1), X1)
	MOVOU(Mem{Base: BX}.Offset(16*2), X2)
	MOVOU(Mem{Base: BX}.Offset(16*3), X3)
	MOVOU(Mem{Base: BX}.Offset(16*4), X4)
	MOVOU(Mem{Base: BX}.Offset(16*5), X5)

	MOVOU(X0, x(16*0))
	MOVOU(X1, x(16*1))
	MOVOU(X2, y(16*0))
	MOVOU(X3, y(16*1))
	MOVOU(X4, z(16*0))
	MOVOU(X5, z(16*1))

	Comment("Store pointer to result")
	MOVQ(RAX, rptr_v3)

	Comment("Begin point double")
	LDacc(z)
	CALL(LabelRef("p256SqrInternal(SB)"))
	ST(zsqr)

	LDt(x)
	p256AddInline()
	STt(m)

	LDacc(z)
	LDt(y)
	CALL(LabelRef("p256MulInternal(SB)"))
	p256MulBy2Inline()
	MOVQ(rptr_v3, RAX)

	Comment("Store z")
	MOVQ(t0_v2, Mem{Base: AX}.Offset(16*4+8*0))
	MOVQ(t1_v2, Mem{Base: AX}.Offset(16*4+8*1))
	MOVQ(t2_v2, Mem{Base: AX}.Offset(16*4+8*2))
	MOVQ(t3_v2, Mem{Base: AX}.Offset(16*4+8*3))

	LDacc(x)
	LDt(zsqr)
	CALL(LabelRef("p256SubInternal(SB)"))
	LDt(m)
	CALL(LabelRef("p256MulInternal(SB)"))
	ST(m)

	Comment("Multiply by 3")
	p256MulBy2Inline()
	LDacc(m)
	p256AddInline()
	STt(m)
	Comment("////////////////////////")
	LDacc(y)
	p256MulBy2Inline()
	t2acc()
	CALL(LabelRef("p256SqrInternal(SB)"))
	ST(s)
	CALL(LabelRef("p256SqrInternal(SB)"))

	Comment("Divide by 2")
	XORQ(mul0_v2, mul0_v2)
	MOVQ(acc4_v2, t0_v2)
	MOVQ(acc5_v2, t1_v2)
	MOVQ(acc6_v2, t2_v2)
	MOVQ(acc7_v2, t3_v2)

	ADDQ(I8(-1), acc4_v2)
	p256const0 := p256const0_DATA()
	ADCQ(p256const0, acc5_v2)
	ADCQ(Imm(0), acc6_v2)
	p256const1 := p256const1_DATA()
	ADCQ(p256const1, acc7_v2)
	ADCQ(Imm(0), mul0_v2)
	TESTQ(U32(1), t0_v2)

	CMOVQEQ(t0_v2, acc4_v2)
	CMOVQEQ(t1_v2, acc5_v2)
	CMOVQEQ(t2_v2, acc6_v2)
	CMOVQEQ(t3_v2, acc7_v2)
	ANDQ(t0_v2, mul0_v2)

	SHRQ(Imm(1), acc5_v2, acc4_v2)
	SHRQ(Imm(1), acc6_v2, acc5_v2)
	SHRQ(Imm(1), acc7_v2, acc6_v2)
	SHRQ(Imm(1), mul0_v2, acc7_v2)
	ST(y)
	Comment("/////////////////////////")
	LDacc(x)
	LDt(s)
	CALL(LabelRef("p256MulInternal(SB)"))
	ST(s)
	p256MulBy2Inline()
	STt(tmp)

	LDacc(m)
	CALL(LabelRef("p256SqrInternal(SB)"))
	LDt(tmp)
	CALL(LabelRef("p256SubInternal(SB)"))

	MOVQ(rptr_v3, RAX)

	Comment("Store x")
	MOVQ(acc4_v2, Mem{Base: AX}.Offset(16*0+8*0))
	MOVQ(acc5_v2, Mem{Base: AX}.Offset(16*0+8*1))
	MOVQ(acc6_v2, Mem{Base: AX}.Offset(16*0+8*2))
	MOVQ(acc7_v2, Mem{Base: AX}.Offset(16*0+8*3))

	acc2t()
	LDacc(s)
	CALL(LabelRef("p256SubInternal(SB)"))

	LDt(m)
	CALL(LabelRef("p256MulInternal(SB)"))

	LDt(y)
	CALL(LabelRef("p256SubInternal(SB)"))
	MOVQ(rptr_v3, RAX)

	Comment("Store y")
	MOVQ(acc4_v2, Mem{Base: AX}.Offset(16*2+8*0))
	MOVQ(acc5_v2, Mem{Base: AX}.Offset(16*2+8*1))
	MOVQ(acc6_v2, Mem{Base: AX}.Offset(16*2+8*2))
	MOVQ(acc7_v2, Mem{Base: AX}.Offset(16*2+8*3))
	Comment("///////////////////////")
	MOVQ(U32(0), rptr_v3)

	RET()
}

// #----------------------------DATA SECTION-----------------------------------##

// Pointers for memoizing Data section symbols
var p256const0_ptr, p256const1_ptr, p256ordK0_ptr, p256ord_ptr, p256one_ptr *Mem

func p256const0_DATA() Mem {
	if p256const0_ptr != nil {
		return *p256const0_ptr
	}

	p256const0 := GLOBL("p256const0", 8)
	p256const0_ptr = &p256const0
	DATA(0, U64(0x00000000ffffffff))
	return p256const0
}

func p256const1_DATA() Mem {
	if p256const1_ptr != nil {
		return *p256const1_ptr
	}

	p256const1 := GLOBL("p256const1", 8)
	p256const1_ptr = &p256const1
	DATA(0, U64(0xffffffff00000001))
	return p256const1
}

func p256ordK0_DATA() Mem {
	if p256ordK0_ptr != nil {
		return *p256ordK0_ptr
	}

	p256ordK0 := GLOBL("p256ordK0", 8)
	p256ordK0_ptr = &p256ordK0
	DATA(0, U64(0xccd1c8aaee00bc4f))
	return p256ordK0
}

var p256ordConstants = [4]uint64{
	0xf3b9cac2fc632551,
	0xbce6faada7179e84,
	0xffffffffffffffff,
	0xffffffff00000000,
}

func p256ord_DATA() Mem {
	if p256ord_ptr != nil {
		return *p256ord_ptr
	}

	p256ord := GLOBL("p256ord", 8)
	p256ord_ptr = &p256ord

	for i, k := range p256ordConstants {
		DATA(i*8, U64(k))
	}

	return p256ord
}

var p256oneConstants = [4]uint64{
	0x0000000000000001,
	0xffffffff00000000,
	0xffffffffffffffff,
	0x00000000fffffffe,
}

func p256one_DATA() Mem {
	if p256one_ptr != nil {
		return *p256one_ptr
	}

	p256one := GLOBL("p256one", 8)
	p256one_ptr = &p256one

	for i, k := range p256oneConstants {
		DATA(i*8, U64(k))
	}

	return p256one
}

const ThatPeskyUnicodeDot = "\u00b7"

// removePeskyUnicodeDot strips the dot from the relevant TEXT directives such that they
// can exist as internal assembly functions
//
// Avo v0.6.0 does not support the generation of internal assembly functions. Go's unicode
// dot tells the compiler to link a TEXT symbol to a function in the current Go package
// (or another package if specified). Avo unconditionally prepends the unicode dot to all
// TEXT symbols, making it impossible to emit an internal function without this hack.
//
// There is a pending PR to add internal functions to Avo:
// https://github.com/mmcloughlin/avo/pull/443
//
// If merged it should allow the usage of InternalFunction("NAME") for the specified functions
func removePeskyUnicodeDot(internalFunctions []string, target string) {
	bytes, err := os.ReadFile(target)
	if err != nil {
		panic(err)
	}

	content := string(bytes)

	for _, from := range internalFunctions {
		to := strings.ReplaceAll(from, ThatPeskyUnicodeDot, "")
		content = strings.ReplaceAll(content, from, to)
	}

	err = os.WriteFile(target, []byte(content), 0644)
	if err != nil {
		panic(err)
	}
}
