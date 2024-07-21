// Copyright 2024 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// Original source:
//	http://www.zorinaq.com/papers/md5-amd64.html
//	http://www.zorinaq.com/papers/md5-amd64.tar.bz2
//
// Translated from Perl generating GNU assembly into
// #defines generating 6a assembly by the Go Authors.

package main

import (
	. "github.com/mmcloughlin/avo/build"
	. "github.com/mmcloughlin/avo/operand"
	. "github.com/mmcloughlin/avo/reg"
)

//go:generate go run . -out ../md5block_amd64.s -pkg md5

func main() {
	Package("crypto/md5")
	ConstraintExpr("!purego")
	block()
	Generate()
}

// MD5 optimized for AMD64.
//
// Author: Marc Bevand <bevand_m (at) epita.fr>
// Licence: I hereby disclaim the copyright on this code and place it
// in the public domain.
func block() {
	Implement("block")
	Attributes(NOSPLIT)
	AllocLocal(8)

	Load(Param("dig"), RBP)
	Load(Param("p").Base(), RSI)
	Load(Param("p").Len(), RDX)
	SHRQ(Imm(6), RDX)
	SHLQ(Imm(6), RDX)

	LEAQ(Mem{Base: SI, Index: DX, Scale: 1}, RDI)
	MOVL(Mem{Base: BP}.Offset(0*4), EAX)
	MOVL(Mem{Base: BP}.Offset(1*4), EBX)
	MOVL(Mem{Base: BP}.Offset(2*4), ECX)
	MOVL(Mem{Base: BP}.Offset(3*4), EDX)
	MOVL(Imm(0xffffffff), R11L)

	CMPQ(RSI, RDI)
	JEQ(LabelRef("end"))

	loop()
	end()
}

func loop() {
	Label("loop")
	MOVL(EAX, R12L)
	MOVL(EBX, R13L)
	MOVL(ECX, R14L)
	MOVL(EDX, R15L)

	MOVL(Mem{Base: SI}.Offset(0*4), R8L)
	MOVL(EDX, R9L)

	ROUND1(EAX, EBX, ECX, EDX, 1, 0xd76aa478, 7)
	ROUND1(EDX, EAX, EBX, ECX, 2, 0xe8c7b756, 12)
	ROUND1(ECX, EDX, EAX, EBX, 3, 0x242070db, 17)
	ROUND1(EBX, ECX, EDX, EAX, 4, 0xc1bdceee, 22)
	ROUND1(EAX, EBX, ECX, EDX, 5, 0xf57c0faf, 7)
	ROUND1(EDX, EAX, EBX, ECX, 6, 0x4787c62a, 12)
	ROUND1(ECX, EDX, EAX, EBX, 7, 0xa8304613, 17)
	ROUND1(EBX, ECX, EDX, EAX, 8, 0xfd469501, 22)
	ROUND1(EAX, EBX, ECX, EDX, 9, 0x698098d8, 7)
	ROUND1(EDX, EAX, EBX, ECX, 10, 0x8b44f7af, 12)
	ROUND1(ECX, EDX, EAX, EBX, 11, 0xffff5bb1, 17)
	ROUND1(EBX, ECX, EDX, EAX, 12, 0x895cd7be, 22)
	ROUND1(EAX, EBX, ECX, EDX, 13, 0x6b901122, 7)
	ROUND1(EDX, EAX, EBX, ECX, 14, 0xfd987193, 12)
	ROUND1(ECX, EDX, EAX, EBX, 15, 0xa679438e, 17)
	ROUND1(EBX, ECX, EDX, EAX, 1, 0x49b40821, 22)

	MOVL(EDX, R9L)
	MOVL(EDX, R10L)

	ROUND2(EAX, EBX, ECX, EDX, 6, 0xf61e2562, 5)
	ROUND2(EDX, EAX, EBX, ECX, 11, 0xc040b340, 9)
	ROUND2(ECX, EDX, EAX, EBX, 0, 0x265e5a51, 14)
	ROUND2(EBX, ECX, EDX, EAX, 5, 0xe9b6c7aa, 20)
	ROUND2(EAX, EBX, ECX, EDX, 10, 0xd62f105d, 5)
	ROUND2(EDX, EAX, EBX, ECX, 15, 0x2441453, 9)
	ROUND2(ECX, EDX, EAX, EBX, 4, 0xd8a1e681, 14)
	ROUND2(EBX, ECX, EDX, EAX, 9, 0xe7d3fbc8, 20)
	ROUND2(EAX, EBX, ECX, EDX, 14, 0x21e1cde6, 5)
	ROUND2(EDX, EAX, EBX, ECX, 3, 0xc33707d6, 9)
	ROUND2(ECX, EDX, EAX, EBX, 8, 0xf4d50d87, 14)
	ROUND2(EBX, ECX, EDX, EAX, 13, 0x455a14ed, 20)
	ROUND2(EAX, EBX, ECX, EDX, 2, 0xa9e3e905, 5)
	ROUND2(EDX, EAX, EBX, ECX, 7, 0xfcefa3f8, 9)
	ROUND2(ECX, EDX, EAX, EBX, 12, 0x676f02d9, 14)
	ROUND2(EBX, ECX, EDX, EAX, 5, 0x8d2a4c8a, 20)

	MOVL(ECX, R9L)

	ROUND3FIRST(EAX, EBX, ECX, EDX, 8, 0xfffa3942, 4)
	ROUND3(EDX, EAX, EBX, ECX, 11, 0x8771f681, 11)
	ROUND3(ECX, EDX, EAX, EBX, 14, 0x6d9d6122, 16)
	ROUND3(EBX, ECX, EDX, EAX, 1, 0xfde5380c, 23)
	ROUND3(EAX, EBX, ECX, EDX, 4, 0xa4beea44, 4)
	ROUND3(EDX, EAX, EBX, ECX, 7, 0x4bdecfa9, 11)
	ROUND3(ECX, EDX, EAX, EBX, 10, 0xf6bb4b60, 16)
	ROUND3(EBX, ECX, EDX, EAX, 13, 0xbebfbc70, 23)
	ROUND3(EAX, EBX, ECX, EDX, 0, 0x289b7ec6, 4)
	ROUND3(EDX, EAX, EBX, ECX, 3, 0xeaa127fa, 11)
	ROUND3(ECX, EDX, EAX, EBX, 6, 0xd4ef3085, 16)
	ROUND3(EBX, ECX, EDX, EAX, 9, 0x4881d05, 23)
	ROUND3(EAX, EBX, ECX, EDX, 12, 0xd9d4d039, 4)
	ROUND3(EDX, EAX, EBX, ECX, 15, 0xe6db99e5, 11)
	ROUND3(ECX, EDX, EAX, EBX, 2, 0x1fa27cf8, 16)
	ROUND3(EBX, ECX, EDX, EAX, 0, 0xc4ac5665, 23)

	MOVL(R11L, R9L)
	XORL(EDX, R9L)

	ROUND4(EAX, EBX, ECX, EDX, 7, 0xf4292244, 6)
	ROUND4(EDX, EAX, EBX, ECX, 14, 0x432aff97, 10)
	ROUND4(ECX, EDX, EAX, EBX, 5, 0xab9423a7, 15)
	ROUND4(EBX, ECX, EDX, EAX, 12, 0xfc93a039, 21)
	ROUND4(EAX, EBX, ECX, EDX, 3, 0x655b59c3, 6)
	ROUND4(EDX, EAX, EBX, ECX, 10, 0x8f0ccc92, 10)
	ROUND4(ECX, EDX, EAX, EBX, 1, 0xffeff47d, 15)
	ROUND4(EBX, ECX, EDX, EAX, 8, 0x85845dd1, 21)
	ROUND4(EAX, EBX, ECX, EDX, 15, 0x6fa87e4f, 6)
	ROUND4(EDX, EAX, EBX, ECX, 6, 0xfe2ce6e0, 10)
	ROUND4(ECX, EDX, EAX, EBX, 13, 0xa3014314, 15)
	ROUND4(EBX, ECX, EDX, EAX, 4, 0x4e0811a1, 21)
	ROUND4(EAX, EBX, ECX, EDX, 11, 0xf7537e82, 6)
	ROUND4(EDX, EAX, EBX, ECX, 2, 0xbd3af235, 10)
	ROUND4(ECX, EDX, EAX, EBX, 9, 0x2ad7d2bb, 15)
	ROUND4(EBX, ECX, EDX, EAX, 0, 0xeb86d391, 21)

	ADDL(R12L, EAX)
	ADDL(R13L, EBX)
	ADDL(R14L, ECX)
	ADDL(R15L, EDX)

	ADDQ(Imm(64), RSI)
	CMPQ(RSI, RDI)
	JB(LabelRef("loop"))
}

func end() {
	Label("end")
	MOVL(EAX, Mem{Base: BP}.Offset(0*4))
	MOVL(EBX, Mem{Base: BP}.Offset(1*4))
	MOVL(ECX, Mem{Base: BP}.Offset(2*4))
	MOVL(EDX, Mem{Base: BP}.Offset(3*4))
	RET()
}

func ROUND1(a, b, c, d GPPhysical, index int, konst, shift uint64) {
	XORL(c, R9L)
	ADDL(Imm(konst), a)
	ADDL(R8L, a)
	ANDL(b, R9L)
	XORL(d, R9L)
	MOVL(Mem{Base: SI}.Offset(index*4), R8L)
	ADDL(R9L, a)
	ROLL(Imm(shift), a)
	MOVL(c, R9L)
	ADDL(b, a)
}

// Uses https://github.com/animetosho/md5-optimisation#dependency-shortcut-in-g-function
func ROUND2(a, b, c, d GPPhysical, index int, konst, shift uint64) {
	XORL(R11L, R9L)
	ADDL(Imm(konst), a)
	ADDL(R8L, a)
	ANDL(b, R10L)
	ANDL(c, R9L)
	MOVL(Mem{Base: SI}.Offset(index*4), R8L)
	ADDL(R9L, a)
	ADDL(R10L, a)
	MOVL(c, R9L)
	MOVL(c, R10L)
	ROLL(Imm(shift), a)
	ADDL(b, a)
}

// Uses https://github.com/animetosho/md5-optimisation#h-function-re-use
func ROUND3FIRST(a, b, c, d GPPhysical, index int, konst, shift uint64) {
	MOVL(d, R9L)
	XORL(c, R9L)
	XORL(b, R9L)
	ADDL(Imm(konst), a)
	ADDL(R8L, a)
	MOVL(Mem{Base: SI}.Offset(index*4), R8L)
	ADDL(R9L, a)
	ROLL(Imm(shift), a)
	ADDL(b, a)
}

func ROUND3(a, b, c, d GPPhysical, index int, konst, shift uint64) {
	XORL(a, R9L)
	XORL(b, R9L)
	ADDL(Imm(konst), a)
	ADDL(R8L, a)
	MOVL(Mem{Base: SI}.Offset(index*4), R8L)
	ADDL(R9L, a)
	ROLL(Imm(shift), a)
	ADDL(b, a)
}

func ROUND4(a, b, c, d GPPhysical, index int, konst, shift uint64) {
	ADDL(Imm(konst), a)
	ADDL(R8L, a)
	ORL(b, R9L)
	XORL(c, R9L)
	ADDL(R9L, a)
	MOVL(Mem{Base: SI}.Offset(index*4), R8L)
	MOVL(Imm(0xffffffff), R9L)
	ROLL(Imm(shift), a)
	XORL(c, R9L)
	ADDL(b, a)
}
