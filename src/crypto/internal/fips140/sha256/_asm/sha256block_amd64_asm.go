// Copyright 2024 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package main

import (
	"os"

	. "github.com/mmcloughlin/avo/build"
	. "github.com/mmcloughlin/avo/operand"
	. "github.com/mmcloughlin/avo/reg"
)

//go:generate go run . -out ../sha256block_amd64.s

// SHA256 block routine. See sha256block.go for Go equivalent.
//
// The algorithm is detailed in FIPS 180-4:
//
//  https://csrc.nist.gov/publications/fips/fips180-4/fips-180-4.pdf

// Wt = Mt; for 0 <= t <= 15
// Wt = SIGMA1(Wt-2) + SIGMA0(Wt-15) + Wt-16; for 16 <= t <= 63
//
// a = H0
// b = H1
// c = H2
// d = H3
// e = H4
// f = H5
// g = H6
// h = H7
//
// for t = 0 to 63 {
//    T1 = h + BIGSIGMA1(e) + Ch(e,f,g) + Kt + Wt
//    T2 = BIGSIGMA0(a) + Maj(a,b,c)
//    h = g
//    g = f
//    f = e
//    e = d + T1
//    d = c
//    c = b
//    b = a
//    a = T1 + T2
// }
//
// H0 = a + H0
// H1 = b + H1
// H2 = c + H2
// H3 = d + H3
// H4 = e + H4
// H5 = f + H5
// H6 = g + H6
// H7 = h + H7

func main() {
	// https://github.com/mmcloughlin/avo/issues/450
	os.Setenv("GOOS", "linux")
	os.Setenv("GOARCH", "amd64")

	Package("crypto/internal/fips140/sha256")
	ConstraintExpr("!purego")
	blockAMD64()
	blockAVX2()
	blockSHANI()
	Generate()
}

// Wt = Mt; for 0 <= t <= 15
func msgSchedule0(index int) {
	MOVL(Mem{Base: SI}.Offset(index*4), EAX)
	BSWAPL(EAX)
	MOVL(EAX, Mem{Base: BP}.Offset(index*4))
}

// Wt = SIGMA1(Wt-2) + Wt-7 + SIGMA0(Wt-15) + Wt-16; for 16 <= t <= 63
//
//	SIGMA0(x) = ROTR(7,x) XOR ROTR(18,x) XOR SHR(3,x)
//	SIGMA1(x) = ROTR(17,x) XOR ROTR(19,x) XOR SHR(10,x)
func msgSchedule1(index int) {
	MOVL(Mem{Base: BP}.Offset((index-2)*4), EAX)
	MOVL(EAX, ECX)
	RORL(Imm(17), EAX)
	MOVL(ECX, EDX)
	RORL(Imm(19), ECX)
	SHRL(Imm(10), EDX)
	MOVL(Mem{Base: BP}.Offset((index-15)*4), EBX)
	XORL(ECX, EAX)
	MOVL(EBX, ECX)
	XORL(EDX, EAX)
	RORL(Imm(7), EBX)
	MOVL(ECX, EDX)
	SHRL(Imm(3), EDX)
	RORL(Imm(18), ECX)
	ADDL(Mem{Base: BP}.Offset((index-7)*4), EAX)
	XORL(ECX, EBX)
	XORL(EDX, EBX)
	ADDL(Mem{Base: BP}.Offset((index-16)*4), EBX)
	ADDL(EBX, EAX)
	MOVL(EAX, Mem{Base: BP}.Offset((index)*4))
}

// Calculate T1 in AX - uses AX, CX and DX registers.
// h is also used as an accumulator. Wt is passed in AX.
//
//	T1 = h + BIGSIGMA1(e) + Ch(e, f, g) + Kt + Wt
//	  BIGSIGMA1(x) = ROTR(6,x) XOR ROTR(11,x) XOR ROTR(25,x)
//	  Ch(x, y, z) = (x AND y) XOR (NOT x AND z)
func sha256T1(konst uint32, e, f, g, h GPPhysical) {
	ADDL(EAX, h)
	MOVL(e, EAX)
	ADDL(U32(konst), h)
	MOVL(e, ECX)
	RORL(U8(6), EAX)
	MOVL(e, EDX)
	RORL(U8(11), ECX)
	XORL(ECX, EAX)
	MOVL(e, ECX)
	RORL(U8(25), EDX)
	ANDL(f, ECX)
	XORL(EAX, EDX)
	MOVL(e, EAX)
	NOTL(EAX)
	ADDL(EDX, h)
	ANDL(g, EAX)
	XORL(ECX, EAX)
	ADDL(h, EAX)
}

// Calculate T2 in BX - uses BX, CX, DX and DI registers.
//
//	T2 = BIGSIGMA0(a) + Maj(a, b, c)
//	  BIGSIGMA0(x) = ROTR(2,x) XOR ROTR(13,x) XOR ROTR(22,x)
//	  Maj(x, y, z) = (x AND y) XOR (x AND z) XOR (y AND z)
func sha256T2(a, b, c GPPhysical) {
	MOVL(a, EDI)
	MOVL(c, EBX)
	RORL(U8(2), EDI)
	MOVL(a, EDX)
	ANDL(b, EBX)
	RORL(U8(13), EDX)
	MOVL(a, ECX)
	ANDL(c, ECX)
	XORL(EDX, EDI)
	XORL(ECX, EBX)
	MOVL(a, EDX)
	MOVL(b, ECX)
	RORL(U8(22), EDX)
	ANDL(a, ECX)
	XORL(ECX, EBX)
	XORL(EDX, EDI)
	ADDL(EDI, EBX)
}

// Calculate T1 and T2, then e = d + T1 and a = T1 + T2.
// The values for e and a are stored in d and h, ready for rotation.
func sha256Round(index int, konst uint32, a, b, c, d, e, f, g, h GPPhysical) {
	sha256T1(konst, e, f, g, h)
	sha256T2(a, b, c)
	MOVL(EBX, h)
	ADDL(EAX, d)
	ADDL(EAX, h)
}

func sha256Round0(index int, konst uint32, a, b, c, d, e, f, g, h GPPhysical) {
	msgSchedule0(index)
	sha256Round(index, konst, a, b, c, d, e, f, g, h)
}

func sha256Round1(index int, konst uint32, a, b, c, d, e, f, g, h GPPhysical) {
	msgSchedule1(index)
	sha256Round(index, konst, a, b, c, d, e, f, g, h)
}

func blockAMD64() {
	Implement("blockAMD64")
	AllocLocal(256 + 8)

	Load(Param("p").Base(), RSI)
	Load(Param("p").Len(), RDX)
	SHRQ(Imm(6), RDX)
	SHLQ(Imm(6), RDX)

	// Return if p is empty
	LEAQ(Mem{Base: RSI, Index: RDX, Scale: 1}, RDI)
	MOVQ(RDI, Mem{Base: SP}.Offset(256))
	CMPQ(RSI, RDI)
	JEQ(LabelRef("end"))

	BP := Mem{Base: BP}
	Load(Param("dig"), RBP)
	MOVL(BP.Offset(0*4), R8L)  // a = H0
	MOVL(BP.Offset(1*4), R9L)  // b = H1
	MOVL(BP.Offset(2*4), R10L) // c = H2
	MOVL(BP.Offset(3*4), R11L) // d = H3
	MOVL(BP.Offset(4*4), R12L) // e = H4
	MOVL(BP.Offset(5*4), R13L) // f = H5
	MOVL(BP.Offset(6*4), R14L) // g = H6
	MOVL(BP.Offset(7*4), R15L) // h = H7

	loop()
	end()
}

func rotateRight(slice *[]GPPhysical) []GPPhysical {
	n := len(*slice)
	new := make([]GPPhysical, n)
	for i, reg := range *slice {
		new[(i+1)%n] = reg
	}
	return new
}

func loop() {
	Label("loop")
	MOVQ(RSP, RBP)

	regs := []GPPhysical{R8L, R9L, R10L, R11L, R12L, R13L, R14L, R15L}
	n := len(_K)

	for i := 0; i < 16; i++ {
		sha256Round0(i, _K[i], regs[0], regs[1], regs[2], regs[3], regs[4], regs[5], regs[6], regs[7])
		regs = rotateRight(&regs)
	}

	for i := 16; i < n; i++ {
		sha256Round1(i, _K[i], regs[0], regs[1], regs[2], regs[3], regs[4], regs[5], regs[6], regs[7])
		regs = rotateRight(&regs)
	}

	Load(Param("dig"), RBP)
	BP := Mem{Base: BP}
	ADDL(BP.Offset(0*4), R8L) //  H0 = a + H0
	MOVL(R8L, BP.Offset(0*4))
	ADDL(BP.Offset(1*4), R9L) //  H1 = b + H1
	MOVL(R9L, BP.Offset(1*4))
	ADDL(BP.Offset(2*4), R10L) // H2 = c + H2
	MOVL(R10L, BP.Offset(2*4))
	ADDL(BP.Offset(3*4), R11L) // H3 = d + H3
	MOVL(R11L, BP.Offset(3*4))
	ADDL(BP.Offset(4*4), R12L) // H4 = e + H4
	MOVL(R12L, BP.Offset(4*4))
	ADDL(BP.Offset(5*4), R13L) // H5 = f + H5
	MOVL(R13L, BP.Offset(5*4))
	ADDL(BP.Offset(6*4), R14L) // H6 = g + H6
	MOVL(R14L, BP.Offset(6*4))
	ADDL(BP.Offset(7*4), R15L) // H7 = h + H7
	MOVL(R15L, BP.Offset(7*4))

	ADDQ(Imm(64), RSI)
	CMPQ(RSI, Mem{Base: SP}.Offset(256))
	JB(LabelRef("loop"))
}

func end() {
	Label("end")
	RET()
}

var _K = []uint32{
	0x428a2f98,
	0x71374491,
	0xb5c0fbcf,
	0xe9b5dba5,
	0x3956c25b,
	0x59f111f1,
	0x923f82a4,
	0xab1c5ed5,
	0xd807aa98,
	0x12835b01,
	0x243185be,
	0x550c7dc3,
	0x72be5d74,
	0x80deb1fe,
	0x9bdc06a7,
	0xc19bf174,
	0xe49b69c1,
	0xefbe4786,
	0x0fc19dc6,
	0x240ca1cc,
	0x2de92c6f,
	0x4a7484aa,
	0x5cb0a9dc,
	0x76f988da,
	0x983e5152,
	0xa831c66d,
	0xb00327c8,
	0xbf597fc7,
	0xc6e00bf3,
	0xd5a79147,
	0x06ca6351,
	0x14292967,
	0x27b70a85,
	0x2e1b2138,
	0x4d2c6dfc,
	0x53380d13,
	0x650a7354,
	0x766a0abb,
	0x81c2c92e,
	0x92722c85,
	0xa2bfe8a1,
	0xa81a664b,
	0xc24b8b70,
	0xc76c51a3,
	0xd192e819,
	0xd6990624,
	0xf40e3585,
	0x106aa070,
	0x19a4c116,
	0x1e376c08,
	0x2748774c,
	0x34b0bcb5,
	0x391c0cb3,
	0x4ed8aa4a,
	0x5b9cca4f,
	0x682e6ff3,
	0x748f82ee,
	0x78a5636f,
	0x84c87814,
	0x8cc70208,
	0x90befffa,
	0xa4506ceb,
	0xbef9a3f7,
	0xc67178f2,
}
