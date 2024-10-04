// Copyright 2024 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package main

import (
	. "github.com/mmcloughlin/avo/build"
	. "github.com/mmcloughlin/avo/operand"
	. "github.com/mmcloughlin/avo/reg"
)

//go:generate go run . -out ../sha256block_amd64.s -pkg sha256

// SHA256 block routine. See sha256block.go for Go equivalent.
//
// The algorithm is detailed in FIPS 180-4:
//
//  https://csrc.nist.gov/publications/fips/fips180-4/fips-180-4.pdf

// The avx2-version is described in an Intel White-Paper:
// "Fast SHA-256 Implementations on Intel Architecture Processors"
// To find it, surf to http://www.intel.com/p/en_US/embedded
// and search for that title.
// AVX2 version by Intel, same algorithm as code in Linux kernel:
// https://github.com/torvalds/linux/blob/master/arch/x86/crypto/sha256-avx2-asm.S
// by
//     James Guilford <james.guilford@intel.com>
//     Kirk Yap <kirk.s.yap@intel.com>
//     Tim Chen <tim.c.chen@linux.intel.com>

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
	Package("crypto/sha256")
	ConstraintExpr("!purego")
	block()
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

// Definitions for AVX2 version

// addm (mem), reg
//   - Add reg to mem using reg-mem add and store
func addm(P1 Mem, P2 GPPhysical) {
	ADDL(P2, P1)
	MOVL(P1, P2)
}

var (
	XDWORD0 VecPhysical = Y4
	XDWORD1             = Y5
	XDWORD2             = Y6
	XDWORD3             = Y7

	XWORD0 = X4
	XWORD1 = X5
	XWORD2 = X6
	XWORD3 = X7

	XTMP0 = Y0
	XTMP1 = Y1
	XTMP2 = Y2
	XTMP3 = Y3
	XTMP4 = Y8
	XTMP5 = Y11

	XFER = Y9

	BYTE_FLIP_MASK   = Y13 // mask to convert LE -> BE
	X_BYTE_FLIP_MASK = X13

	NUM_BYTES GPPhysical = RDX
	INP                  = RDI

	CTX = RSI // Beginning of digest in memory (a, b, c, ... , h)

	a = EAX
	b = EBX
	c = ECX
	d = R8L
	e = EDX
	f = R9L
	g = R10L
	h = R11L

	old_h = R11L

	TBL = RBP

	SRND = RSI // SRND is same register as CTX

	T1 = R12L

	y0 = R13L
	y1 = R14L
	y2 = R15L
	y3 = EDI

	// Offsets
	XFER_SIZE    = 2 * 64 * 4
	INP_END_SIZE = 8
	INP_SIZE     = 8

	_XFER      = 0
	_INP_END   = _XFER + XFER_SIZE
	_INP       = _INP_END + INP_END_SIZE
	STACK_SIZE = _INP + INP_SIZE
)

func roundAndSchedN0(disp int, a, b, c, d, e, f, g, h GPPhysical, XDWORD0, XDWORD1, XDWORD2, XDWORD3 VecPhysical) {
	//                                                                 #############################  RND N + 0 ############################//
	MOVL(a, y3)           //                                           y3 = a
	RORXL(Imm(25), e, y0) //                                           y0 = e >> 25
	RORXL(Imm(11), e, y1) //                                           y1 = e >> 11

	ADDL(Mem{Base: SP, Disp: disp + 0*4, Scale: 1, Index: SRND}, h) // h = k + w + h
	ORL(c, y3)                                                      // y3 = a|c
	VPALIGNR(Imm(4), XDWORD2, XDWORD3, XTMP0)                       // XTMP0 = W[-7]
	MOVL(f, y2)                                                     // y2 = f
	RORXL(Imm(13), a, T1)                                           // T1 = a >> 13

	XORL(y1, y0)                  //                                   y0 = (e>>25) ^ (e>>11)
	XORL(g, y2)                   //                                   y2 = f^g
	VPADDD(XDWORD0, XTMP0, XTMP0) //                                   XTMP0 = W[-7] + W[-16]
	RORXL(Imm(6), e, y1)          //                                   y1 = (e >> 6)

	ANDL(e, y2)           //                                           y2 = (f^g)&e
	XORL(y1, y0)          //                                           y0 = (e>>25) ^ (e>>11) ^ (e>>6)
	RORXL(Imm(22), a, y1) //                                           y1 = a >> 22
	ADDL(h, d)            //                                           d = k + w + h + d

	ANDL(b, y3)                               //                       y3 = (a|c)&b
	VPALIGNR(Imm(4), XDWORD0, XDWORD1, XTMP1) //                       XTMP1 = W[-15]
	XORL(T1, y1)                              //                       y1 = (a>>22) ^ (a>>13)
	RORXL(Imm(2), a, T1)                      //                       T1 = (a >> 2)

	XORL(g, y2)                  //                                    y2 = CH = ((f^g)&e)^g
	VPSRLD(Imm(7), XTMP1, XTMP2) //
	XORL(T1, y1)                 //                                    y1 = (a>>22) ^ (a>>13) ^ (a>>2)
	MOVL(a, T1)                  //                                    T1 = a
	ANDL(c, T1)                  //                                    T1 = a&c

	ADDL(y0, y2)                    //                                 y2 = S1 + CH
	VPSLLD(Imm(32-7), XTMP1, XTMP3) //
	ORL(T1, y3)                     //                                 y3 = MAJ = (a|c)&b)|(a&c)
	ADDL(y1, h)                     //                                 h = k + w + h + S0

	ADDL(y2, d)               //                                       d = k + w + h + d + S1 + CH = d + t1
	VPOR(XTMP2, XTMP3, XTMP3) //                                       XTMP3 = W[-15] ror 7

	VPSRLD(Imm(18), XTMP1, XTMP2)
	ADDL(y2, h) //                                                     h = k + w + h + S0 + S1 + CH = t1 + S0
	ADDL(y3, h) //                                                     h = t1 + S0 + MAJ
}

func roundAndSchedN1(disp int, a, b, c, d, e, f, g, h GPPhysical, XDWORD0, XDWORD1, XDWORD2, XDWORD3 VecPhysical) {
	//                                                                 ################################### RND N + 1 ############################
	MOVL(a, y3)                                                     // y3 = a
	RORXL(Imm(25), e, y0)                                           // y0 = e >> 25
	RORXL(Imm(11), e, y1)                                           // y1 = e >> 11
	ADDL(Mem{Base: SP, Disp: disp + 1*4, Scale: 1, Index: SRND}, h) // h = k + w + h
	ORL(c, y3)                                                      // y3 = a|c

	VPSRLD(Imm(3), XTMP1, XTMP4) //                                    XTMP4 = W[-15] >> 3
	MOVL(f, y2)                  //                                    y2 = f
	RORXL(Imm(13), a, T1)        //                                    T1 = a >> 13
	XORL(y1, y0)                 //                                    y0 = (e>>25) ^ (e>>11)
	XORL(g, y2)                  //                                    y2 = f^g

	RORXL(Imm(6), e, y1)  //                                           y1 = (e >> 6)
	XORL(y1, y0)          //                                           y0 = (e>>25) ^ (e>>11) ^ (e>>6)
	RORXL(Imm(22), a, y1) //                                           y1 = a >> 22
	ANDL(e, y2)           //                                           y2 = (f^g)&e
	ADDL(h, d)            //                                           d = k + w + h + d

	VPSLLD(Imm(32-18), XTMP1, XTMP1)
	ANDL(b, y3)  //                                                    y3 = (a|c)&b
	XORL(T1, y1) //                                                    y1 = (a>>22) ^ (a>>13)

	VPXOR(XTMP1, XTMP3, XTMP3)
	RORXL(Imm(2), a, T1) //                                            T1 = (a >> 2)
	XORL(g, y2)          //                                            y2 = CH = ((f^g)&e)^g

	VPXOR(XTMP2, XTMP3, XTMP3) //                                      XTMP3 = W[-15] ror 7 ^ W[-15] ror 18
	XORL(T1, y1)               //                                      y1 = (a>>22) ^ (a>>13) ^ (a>>2)
	MOVL(a, T1)                //                                      T1 = a
	ANDL(c, T1)                //                                      T1 = a&c
	ADDL(y0, y2)               //                                      y2 = S1 + CH

	VPXOR(XTMP4, XTMP3, XTMP1)         //                              XTMP1 = s0
	VPSHUFD(Imm(0xFA), XDWORD3, XTMP2) //                              XTMP2 = W[-2] {BBAA}
	ORL(T1, y3)                        //                              y3 = MAJ = (a|c)&b)|(a&c)
	ADDL(y1, h)                        //                              h = k + w + h + S0

	VPADDD(XTMP1, XTMP0, XTMP0) //                                     XTMP0 = W[-16] + W[-7] + s0
	ADDL(y2, d)                 //                                     d = k + w + h + d + S1 + CH = d + t1
	ADDL(y2, h)                 //                                     h = k + w + h + S0 + S1 + CH = t1 + S0
	ADDL(y3, h)                 //                                     h = t1 + S0 + MAJ

	VPSRLD(Imm(10), XTMP2, XTMP4) //                                   XTMP4 = W[-2] >> 10 {BBAA}
}

func roundAndSchedN2(disp int, a, b, c, d, e, f, g, h GPPhysical, XDWORD0, XDWORD1, XDWORD2, XDWORD3 VecPhysical) {
	//                                                                 ################################### RND N + 2 ############################
	var shuff_00BA Mem = shuff_00BA_DATA()

	MOVL(a, y3)                                                     // y3 = a
	RORXL(Imm(25), e, y0)                                           // y0 = e >> 25
	ADDL(Mem{Base: SP, Disp: disp + 2*4, Scale: 1, Index: SRND}, h) // h = k + w + h

	VPSRLQ(Imm(19), XTMP2, XTMP3) //                                   XTMP3 = W[-2] ror 19 {xBxA}
	RORXL(Imm(11), e, y1)         //                                   y1 = e >> 11
	ORL(c, y3)                    //                                   y3 = a|c
	MOVL(f, y2)                   //                                   y2 = f
	XORL(g, y2)                   //                                   y2 = f^g

	RORXL(Imm(13), a, T1)         //                                   T1 = a >> 13
	XORL(y1, y0)                  //                                   y0 = (e>>25) ^ (e>>11)
	VPSRLQ(Imm(17), XTMP2, XTMP2) //                                   XTMP2 = W[-2] ror 17 {xBxA}
	ANDL(e, y2)                   //                                   y2 = (f^g)&e

	RORXL(Imm(6), e, y1) //                                            y1 = (e >> 6)
	VPXOR(XTMP3, XTMP2, XTMP2)
	ADDL(h, d)  //                                                     d = k + w + h + d
	ANDL(b, y3) //                                                     y3 = (a|c)&b

	XORL(y1, y0)               //                                      y0 = (e>>25) ^ (e>>11) ^ (e>>6)
	RORXL(Imm(22), a, y1)      //                                      y1 = a >> 22
	VPXOR(XTMP2, XTMP4, XTMP4) //                                      XTMP4 = s1 {xBxA}
	XORL(g, y2)                //                                      y2 = CH = ((f^g)&e)^g

	VPSHUFB(shuff_00BA, XTMP4, XTMP4) //                               XTMP4 = s1 {00BA}

	XORL(T1, y1)                //                                     y1 = (a>>22) ^ (a>>13)
	RORXL(Imm(2), a, T1)        //                                     T1 = (a >> 2)
	VPADDD(XTMP4, XTMP0, XTMP0) //                                     XTMP0 = {..., ..., W[1], W[0]}

	XORL(T1, y1)                   //                                  y1 = (a>>22) ^ (a>>13) ^ (a>>2)
	MOVL(a, T1)                    //                                  T1 = a
	ANDL(c, T1)                    //                                  T1 = a&c
	ADDL(y0, y2)                   //                                  y2 = S1 + CH
	VPSHUFD(Imm(80), XTMP0, XTMP2) //                                  XTMP2 = W[-2] {DDCC}

	ORL(T1, y3) //                                                     y3 = MAJ = (a|c)&b)|(a&c)
	ADDL(y1, h) //                                                     h = k + w + h + S0
	ADDL(y2, d) //                                                     d = k + w + h + d + S1 + CH = d + t1
	ADDL(y2, h) //                                                     h = k + w + h + S0 + S1 + CH = t1 + S0

	ADDL(y3, h) //                                                     h = t1 + S0 + MAJ
}

func roundAndSchedN3(disp int, a, b, c, d, e, f, g, h GPPhysical, XDWORD0, XDWORD1, XDWORD2, XDWORD3 VecPhysical) {
	//                                                                 ################################### RND N + 3 ############################
	var shuff_DC00 Mem = shuff_DC00_DATA()

	MOVL(a, y3)                                                     // y3 = a
	RORXL(Imm(25), e, y0)                                           // y0 = e >> 25
	RORXL(Imm(11), e, y1)                                           // y1 = e >> 11
	ADDL(Mem{Base: SP, Disp: disp + 3*4, Scale: 1, Index: SRND}, h) // h = k + w + h
	ORL(c, y3)                                                      // y3 = a|c

	VPSRLD(Imm(10), XTMP2, XTMP5) //                                   XTMP5 = W[-2] >> 10 {DDCC}
	MOVL(f, y2)                   //                                   y2 = f
	RORXL(Imm(13), a, T1)         //                                   T1 = a >> 13
	XORL(y1, y0)                  //                                   y0 = (e>>25) ^ (e>>11)
	XORL(g, y2)                   //                                   y2 = f^g

	VPSRLQ(Imm(19), XTMP2, XTMP3) //                                   XTMP3 = W[-2] ror 19 {xDxC}
	RORXL(Imm(6), e, y1)          //                                   y1 = (e >> 6)
	ANDL(e, y2)                   //                                   y2 = (f^g)&e
	ADDL(h, d)                    //                                   d = k + w + h + d
	ANDL(b, y3)                   //                                   y3 = (a|c)&b

	VPSRLQ(Imm(17), XTMP2, XTMP2) //                                   XTMP2 = W[-2] ror 17 {xDxC}
	XORL(y1, y0)                  //                                   y0 = (e>>25) ^ (e>>11) ^ (e>>6)
	XORL(g, y2)                   //                                   y2 = CH = ((f^g)&e)^g

	VPXOR(XTMP3, XTMP2, XTMP2)
	RORXL(Imm(22), a, y1) //                                           y1 = a >> 22
	ADDL(y0, y2)          //                                           y2 = S1 + CH

	VPXOR(XTMP2, XTMP5, XTMP5) //                                      XTMP5 = s1 {xDxC}
	XORL(T1, y1)               //                                      y1 = (a>>22) ^ (a>>13)
	ADDL(y2, d)                //                                      d = k + w + h + d + S1 + CH = d + t1

	RORXL(Imm(2), a, T1) //                                            T1 = (a >> 2)

	VPSHUFB(shuff_DC00, XTMP5, XTMP5) //                               XTMP5 = s1 {DC00}

	VPADDD(XTMP0, XTMP5, XDWORD0) //                                   XDWORD0 = {W[3], W[2], W[1], W[0]}
	XORL(T1, y1)                  //                                   y1 = (a>>22) ^ (a>>13) ^ (a>>2)
	MOVL(a, T1)                   //                                   T1 = a
	ANDL(c, T1)                   //                                   T1 = a&c
	ORL(T1, y3)                   //                                   y3 = MAJ = (a|c)&b)|(a&c)

	ADDL(y1, h) //                                                     h = k + w + h + S0
	ADDL(y2, h) //                                                     h = k + w + h + S0 + S1 + CH = t1 + S0
	ADDL(y3, h) //                                                     h = t1 + S0 + MAJ
}

func doRoundN0(disp int, a, b, c, d, e, f, g, h, old_h GPPhysical) {
	//                                                                 ################################### RND N + 0 ###########################
	MOVL(f, y2)           //                                           y2 = f
	RORXL(Imm(25), e, y0) //                                           y0 = e >> 25
	RORXL(Imm(11), e, y1) //                                           y1 = e >> 11
	XORL(g, y2)           //                                           y2 = f^g

	XORL(y1, y0)         //                                            y0 = (e>>25) ^ (e>>11)
	RORXL(Imm(6), e, y1) //                                            y1 = (e >> 6)
	ANDL(e, y2)          //                                            y2 = (f^g)&e

	XORL(y1, y0)          //                                           y0 = (e>>25) ^ (e>>11) ^ (e>>6)
	RORXL(Imm(13), a, T1) //                                           T1 = a >> 13
	XORL(g, y2)           //                                           y2 = CH = ((f^g)&e)^g
	RORXL(Imm(22), a, y1) //                                           y1 = a >> 22
	MOVL(a, y3)           //                                           y3 = a

	XORL(T1, y1)                                                    // y1 = (a>>22) ^ (a>>13)
	RORXL(Imm(2), a, T1)                                            // T1 = (a >> 2)
	ADDL(Mem{Base: SP, Disp: disp + 0*4, Scale: 1, Index: SRND}, h) // h = k + w + h
	ORL(c, y3)                                                      // y3 = a|c

	XORL(T1, y1) //                                                    y1 = (a>>22) ^ (a>>13) ^ (a>>2)
	MOVL(a, T1)  //                                                    T1 = a
	ANDL(b, y3)  //                                                    y3 = (a|c)&b
	ANDL(c, T1)  //                                                    T1 = a&c
	ADDL(y0, y2) //                                                    y2 = S1 + CH

	ADDL(h, d)  //                                                     d = k + w + h + d
	ORL(T1, y3) //                                                     y3 = MAJ = (a|c)&b)|(a&c)
	ADDL(y1, h) //                                                     h = k + w + h + S0
	ADDL(y2, d) //                                                     d = k + w + h + d + S1 + CH = d + t1
}

func doRoundN1(disp int, a, b, c, d, e, f, g, h, old_h GPPhysical) {
	//                                                                 ################################### RND N + 1 ###########################
	ADDL(y2, old_h)       //                                           h = k + w + h + S0 + S1 + CH = t1 + S0
	MOVL(f, y2)           //                                           y2 = f
	RORXL(Imm(25), e, y0) //                                           y0 = e >> 25
	RORXL(Imm(11), e, y1) //                                           y1 = e >> 11
	XORL(g, y2)           //                                           y2 = f^g

	XORL(y1, y0)         //                                            y0 = (e>>25) ^ (e>>11)
	RORXL(Imm(6), e, y1) //                                            y1 = (e >> 6)
	ANDL(e, y2)          //                                            y2 = (f^g)&e
	ADDL(y3, old_h)      //                                            h = t1 + S0 + MAJ

	XORL(y1, y0)          //                                           y0 = (e>>25) ^ (e>>11) ^ (e>>6)
	RORXL(Imm(13), a, T1) //                                           T1 = a >> 13
	XORL(g, y2)           //                                           y2 = CH = ((f^g)&e)^g
	RORXL(Imm(22), a, y1) //                                           y1 = a >> 22
	MOVL(a, y3)           //                                           y3 = a

	XORL(T1, y1)                                                    // y1 = (a>>22) ^ (a>>13)
	RORXL(Imm(2), a, T1)                                            // T1 = (a >> 2)
	ADDL(Mem{Base: SP, Disp: disp + 1*4, Scale: 1, Index: SRND}, h) // h = k + w + h
	ORL(c, y3)                                                      // y3 = a|c

	XORL(T1, y1) //                                                    y1 = (a>>22) ^ (a>>13) ^ (a>>2)
	MOVL(a, T1)  //                                                    T1 = a
	ANDL(b, y3)  //                                                    y3 = (a|c)&b
	ANDL(c, T1)  //                                                    T1 = a&c
	ADDL(y0, y2) //                                                    y2 = S1 + CH

	ADDL(h, d)  //                                                     d = k + w + h + d
	ORL(T1, y3) //                                                     y3 = MAJ = (a|c)&b)|(a&c)
	ADDL(y1, h) //                                                     h = k + w + h + S0

	ADDL(y2, d) //                                                     d = k + w + h + d + S1 + CH = d + t1
}

func doRoundN2(disp int, a, b, c, d, e, f, g, h, old_h GPPhysical) {
	//                                                                 ################################### RND N + 2 ##############################
	ADDL(y2, old_h)       //                                           h = k + w + h + S0 + S1 + CH = t1 + S0
	MOVL(f, y2)           //                                           y2 = f
	RORXL(Imm(25), e, y0) //                                           y0 = e >> 25
	RORXL(Imm(11), e, y1) //                                           y1 = e >> 11
	XORL(g, y2)           //                                           y2 = f^g

	XORL(y1, y0)         //                                            y0 = (e>>25) ^ (e>>11)
	RORXL(Imm(6), e, y1) //                                            y1 = (e >> 6)
	ANDL(e, y2)          //                                            y2 = (f^g)&e
	ADDL(y3, old_h)      //                                            h = t1 + S0 + MAJ

	XORL(y1, y0)          //                                           y0 = (e>>25) ^ (e>>11) ^ (e>>6)
	RORXL(Imm(13), a, T1) //                                           T1 = a >> 13
	XORL(g, y2)           //                                           y2 = CH = ((f^g)&e)^g
	RORXL(Imm(22), a, y1) //                                           y1 = a >> 22
	MOVL(a, y3)           //                                           y3 = a

	XORL(T1, y1)                                                    // y1 = (a>>22) ^ (a>>13)
	RORXL(Imm(2), a, T1)                                            // T1 = (a >> 2)
	ADDL(Mem{Base: SP, Disp: disp + 2*4, Scale: 1, Index: SRND}, h) // h = k + w + h
	ORL(c, y3)                                                      // y3 = a|c

	XORL(T1, y1) //                                                    y1 = (a>>22) ^ (a>>13) ^ (a>>2)
	MOVL(a, T1)  //                                                    T1 = a
	ANDL(b, y3)  //                                                    y3 = (a|c)&b
	ANDL(c, T1)  //                                                    T1 = a&c
	ADDL(y0, y2) //                                                    y2 = S1 + CH

	ADDL(h, d)  //                                                     d = k + w + h + d
	ORL(T1, y3) //                                                     y3 = MAJ = (a|c)&b)|(a&c)
	ADDL(y1, h) //                                                     h = k + w + h + S0

	ADDL(y2, d) //                                                     d = k + w + h + d + S1 + CH = d + t1
}

func doRoundN3(disp int, a, b, c, d, e, f, g, h, old_h GPPhysical) {
	//                                                                 ################################### RND N + 3 ###########################
	ADDL(y2, old_h)       //                                           h = k + w + h + S0 + S1 + CH = t1 + S0
	MOVL(f, y2)           //                                           y2 = f
	RORXL(Imm(25), e, y0) //                                           y0 = e >> 25
	RORXL(Imm(11), e, y1) //                                           y1 = e >> 11
	XORL(g, y2)           //                                           y2 = f^g

	XORL(y1, y0)         //                                            y0 = (e>>25) ^ (e>>11)
	RORXL(Imm(6), e, y1) //                                            y1 = (e >> 6)
	ANDL(e, y2)          //                                            y2 = (f^g)&e
	ADDL(y3, old_h)      //                                            h = t1 + S0 + MAJ

	XORL(y1, y0)          //                                           y0 = (e>>25) ^ (e>>11) ^ (e>>6)
	RORXL(Imm(13), a, T1) //                                           T1 = a >> 13
	XORL(g, y2)           //                                           y2 = CH = ((f^g)&e)^g
	RORXL(Imm(22), a, y1) //                                           y1 = a >> 22
	MOVL(a, y3)           //                                           y3 = a

	XORL(T1, y1)                                                    // y1 = (a>>22) ^ (a>>13)
	RORXL(Imm(2), a, T1)                                            // T1 = (a >> 2)
	ADDL(Mem{Base: SP, Disp: disp + 3*4, Scale: 1, Index: SRND}, h) // h = k + w + h
	ORL(c, y3)                                                      // y3 = a|c

	XORL(T1, y1) //                                                    y1 = (a>>22) ^ (a>>13) ^ (a>>2)
	MOVL(a, T1)  //                                                    T1 = a
	ANDL(b, y3)  //                                                    y3 = (a|c)&b
	ANDL(c, T1)  //                                                    T1 = a&c
	ADDL(y0, y2) //                                                    y2 = S1 + CH

	ADDL(h, d)  //                                                     d = k + w + h + d
	ORL(T1, y3) //                                                     y3 = MAJ = (a|c)&b)|(a&c)
	ADDL(y1, h) //                                                     h = k + w + h + S0

	ADDL(y2, d) //                                                     d = k + w + h + d + S1 + CH = d + t1

	ADDL(y2, h) //                                                     h = k + w + h + S0 + S1 + CH = t1 + S0

	ADDL(y3, h) //                                                     h = t1 + S0 + MAJ
}

// Definitions for sha-ni version
//
// The sha-ni implementation uses Intel(R) SHA extensions SHA256RNDS2, SHA256MSG1, SHA256MSG2
// It also reuses portions of the flip_mask (half) and K256 table (stride 32) from the avx2 version
//
// Reference
// S. Gulley, et al, "New Instructions Supporting the Secure Hash
// Algorithm on Intel® Architecture Processors", July 2013
// https://www.intel.com/content/www/us/en/developer/articles/technical/intel-sha-extensions.html
//

var (
	digestPtr       GPPhysical  = RDI // input/output, base pointer to digest hash vector H0, H1, ..., H7
	dataPtr                     = RSI // input, base pointer to first input data block
	numBytes                    = RDX // input, number of input bytes to be processed
	sha256Constants             = RAX // round contents from K256 table, indexed by round number x 32
	msg             VecPhysical = X0  // input data
	state0                      = X1  // round intermediates and outputs
	state1                      = X2
	m0                          = X3 //  m0, m1,... m4 -- round message temps
	m1                          = X4
	m2                          = X5
	m3                          = X6
	m4                          = X7
	shufMask                    = X8  // input data endian conversion control mask
	abefSave                    = X9  // digest hash vector inter-block buffer abef
	cdghSave                    = X10 // digest hash vector inter-block buffer cdgh
)

// nop instead of final SHA256MSG1 for first and last few rounds
func nop(m, a VecPhysical) {
}

// final SHA256MSG1 for middle rounds that require it
func sha256msg1(m, a VecPhysical) {
	SHA256MSG1(m, a)
}

// msg copy for all but rounds 12-15
func vmov(a, b VecPhysical) {
	VMOVDQA(a, b)
}

// reverse copy for rounds 12-15
func vmovrev(a, b VecPhysical) {
	VMOVDQA(b, a)
}

type VecFunc func(a, b VecPhysical)

// sha rounds 0 to 11
//
// identical with the exception of the final msg op
// which is replaced with a nop for rounds where it is not needed
// refer to Gulley, et al for more information
func rounds0to11(m, a VecPhysical, c int, sha256msg1 VecFunc) {
	VMOVDQU(Mem{Base: dataPtr}.Offset(c*16), msg)
	PSHUFB(shufMask, msg)
	VMOVDQA(msg, m)
	PADDD(Mem{Base: sha256Constants}.Offset(c*32), msg)
	SHA256RNDS2(msg, state0, state1)
	PSHUFD(U8(0x0e), msg, msg)
	SHA256RNDS2(msg, state1, state0)
	sha256msg1(m, a)
}

// sha rounds 12 to 59
//
// identical with the exception of the final msg op
// and the reverse copy(m,msg) in round 12 which is required
// after the last data load
// refer to Gulley, et al for more information
func rounds12to59(m VecPhysical, c int, a, t VecPhysical, sha256msg1, movop VecFunc) {
	movop(m, msg)
	PADDD(Mem{Base: sha256Constants}.Offset(c*32), msg)
	SHA256RNDS2(msg, state0, state1)
	VMOVDQA(m, m4)
	PALIGNR(Imm(4), a, m4)
	PADDD(m4, t)
	SHA256MSG2(m, t)
	PSHUFD(Imm(0x0e), msg, msg)
	SHA256RNDS2(msg, state1, state0)
	sha256msg1(m, a)
}

func block() {
	Implement("block")
	AllocLocal(536)

	checkArchFlags()
	sha256()
	avx2()
	sha_ni()
}

func checkArchFlags() {
	CMPB(Mem{Symbol: Symbol{Name: "·useSHA"}, Base: StaticBase}, Imm(1))
	JE(LabelRef("sha_ni"))
	CMPB(Mem{Symbol: Symbol{Name: "·useAVX2"}, Base: StaticBase}, Imm(1))
	JE(LabelRef("avx2"))
}

func sha256() {
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

func avx2() {
	Label("avx2")
	Load(Param("dig"), CTX) // d.h[8]
	Load(Param("p").Base(), INP)
	Load(Param("p").Len(), NUM_BYTES)

	LEAQ(Mem{Base: INP, Index: NUM_BYTES, Scale: 1, Disp: -64}, NUM_BYTES) // Pointer to the last block
	MOVQ(NUM_BYTES, Mem{Base: SP}.Offset(_INP_END))

	CMPQ(NUM_BYTES, INP)
	JE(LabelRef("avx2_only_one_block"))

	Comment("Load initial digest")
	CTX := Mem{Base: CTX}
	MOVL(CTX.Offset(0), a)  //  a = H0
	MOVL(CTX.Offset(4), b)  //  b = H1
	MOVL(CTX.Offset(8), c)  //  c = H2
	MOVL(CTX.Offset(12), d) //  d = H3
	MOVL(CTX.Offset(16), e) //  e = H4
	MOVL(CTX.Offset(20), f) //  f = H5
	MOVL(CTX.Offset(24), g) //  g = H6
	MOVL(CTX.Offset(28), h) //  h = H7

	avx2_loop0()
	avx2_last_block_enter()
	avx2_loop1()
	avx2_loop2()
	avx2_loop3()
	avx2_do_last_block()
	avx2_only_one_block()
	done_hash()
}

func avx2_loop0() {
	Label("avx2_loop0")
	Comment("at each iteration works with one block (512 bit)")
	VMOVDQU(Mem{Base: INP}.Offset(0*32), XTMP0)
	VMOVDQU(Mem{Base: INP}.Offset(1*32), XTMP1)
	VMOVDQU(Mem{Base: INP}.Offset(2*32), XTMP2)
	VMOVDQU(Mem{Base: INP}.Offset(3*32), XTMP3)

	flip_mask := flip_mask_DATA()

	VMOVDQU(flip_mask, BYTE_FLIP_MASK)

	Comment("Apply Byte Flip Mask: LE -> BE")
	VPSHUFB(BYTE_FLIP_MASK, XTMP0, XTMP0)
	VPSHUFB(BYTE_FLIP_MASK, XTMP1, XTMP1)
	VPSHUFB(BYTE_FLIP_MASK, XTMP2, XTMP2)
	VPSHUFB(BYTE_FLIP_MASK, XTMP3, XTMP3)

	Comment("Transpose data into high/low parts")
	VPERM2I128(Imm(0x20), XTMP2, XTMP0, XDWORD0) //  w3,  w2,  w1,  w0
	VPERM2I128(Imm(0x31), XTMP2, XTMP0, XDWORD1) //  w7,  w6,  w5,  w4
	VPERM2I128(Imm(0x20), XTMP3, XTMP1, XDWORD2) // w11, w10,  w9,  w8
	VPERM2I128(Imm(0x31), XTMP3, XTMP1, XDWORD3) // w15, w14, w13, w12

	K256 := K256_DATA()
	LEAQ(K256, TBL) // Loading address of table with round-specific constants
}

func avx2_last_block_enter() {
	Label("avx2_last_block_enter")
	ADDQ(Imm(64), INP)
	MOVQ(INP, Mem{Base: SP}.Offset(_INP))
	XORQ(SRND, SRND)
}

// for w0 - w47
func avx2_loop1() {
	Label("avx2_loop1")

	Comment("Do 4 rounds and scheduling")
	VPADDD(Mem{Base: TBL, Scale: 1, Index: SRND}.Offset((0 * 32)), XDWORD0, XFER)
	VMOVDQU(XFER, Mem{Base: SP, Scale: 1, Index: SRND}.Offset(_XFER+0*32))
	roundAndSchedN0(_XFER+0*32, a, b, c, d, e, f, g, h, XDWORD0, XDWORD1, XDWORD2, XDWORD3)
	roundAndSchedN1(_XFER+0*32, h, a, b, c, d, e, f, g, XDWORD0, XDWORD1, XDWORD2, XDWORD3)
	roundAndSchedN2(_XFER+0*32, g, h, a, b, c, d, e, f, XDWORD0, XDWORD1, XDWORD2, XDWORD3)
	roundAndSchedN3(_XFER+0*32, f, g, h, a, b, c, d, e, XDWORD0, XDWORD1, XDWORD2, XDWORD3)

	Comment("Do 4 rounds and scheduling")
	VPADDD(Mem{Base: TBL, Scale: 1, Index: SRND}.Offset(1*32), XDWORD1, XFER)
	VMOVDQU(XFER, Mem{Base: SP, Scale: 1, Index: SRND}.Offset(_XFER+1*32))
	roundAndSchedN0(_XFER+1*32, e, f, g, h, a, b, c, d, XDWORD1, XDWORD2, XDWORD3, XDWORD0)
	roundAndSchedN1(_XFER+1*32, d, e, f, g, h, a, b, c, XDWORD1, XDWORD2, XDWORD3, XDWORD0)
	roundAndSchedN2(_XFER+1*32, c, d, e, f, g, h, a, b, XDWORD1, XDWORD2, XDWORD3, XDWORD0)
	roundAndSchedN3(_XFER+1*32, b, c, d, e, f, g, h, a, XDWORD1, XDWORD2, XDWORD3, XDWORD0)

	Comment("Do 4 rounds and scheduling")
	VPADDD(Mem{Base: TBL, Scale: 1, Index: SRND}.Offset((2 * 32)), XDWORD2, XFER)
	VMOVDQU(XFER, Mem{Base: SP, Scale: 1, Index: SRND}.Offset(_XFER+2*32))
	roundAndSchedN0(_XFER+2*32, a, b, c, d, e, f, g, h, XDWORD2, XDWORD3, XDWORD0, XDWORD1)
	roundAndSchedN1(_XFER+2*32, h, a, b, c, d, e, f, g, XDWORD2, XDWORD3, XDWORD0, XDWORD1)
	roundAndSchedN2(_XFER+2*32, g, h, a, b, c, d, e, f, XDWORD2, XDWORD3, XDWORD0, XDWORD1)
	roundAndSchedN3(_XFER+2*32, f, g, h, a, b, c, d, e, XDWORD2, XDWORD3, XDWORD0, XDWORD1)

	Comment("Do 4 rounds and scheduling")
	VPADDD(Mem{Base: TBL, Scale: 1, Index: SRND}.Offset((3 * 32)), XDWORD3, XFER)
	VMOVDQU(XFER, Mem{Base: SP, Scale: 1, Index: SRND}.Offset(_XFER+3*32))
	roundAndSchedN0(_XFER+3*32, e, f, g, h, a, b, c, d, XDWORD3, XDWORD0, XDWORD1, XDWORD2)
	roundAndSchedN1(_XFER+3*32, d, e, f, g, h, a, b, c, XDWORD3, XDWORD0, XDWORD1, XDWORD2)
	roundAndSchedN2(_XFER+3*32, c, d, e, f, g, h, a, b, XDWORD3, XDWORD0, XDWORD1, XDWORD2)
	roundAndSchedN3(_XFER+3*32, b, c, d, e, f, g, h, a, XDWORD3, XDWORD0, XDWORD1, XDWORD2)

	ADDQ(Imm(4*32), SRND)
	CMPQ(SRND, U32(3*4*32))
	JB(LabelRef("avx2_loop1"))
}

// w48 - w63 processed with no scheduling (last 16 rounds)
func avx2_loop2() {
	Label("avx2_loop2")
	VPADDD(Mem{Base: TBL, Scale: 1, Index: SRND}.Offset(0*32), XDWORD0, XFER)
	VMOVDQU(XFER, Mem{Base: SP, Scale: 1, Index: SRND}.Offset(_XFER+0*32))
	doRoundN0(_XFER+0*32, a, b, c, d, e, f, g, h, h)
	doRoundN1(_XFER+0*32, h, a, b, c, d, e, f, g, h)
	doRoundN2(_XFER+0*32, g, h, a, b, c, d, e, f, g)
	doRoundN3(_XFER+0*32, f, g, h, a, b, c, d, e, f)

	VPADDD(Mem{Base: TBL, Scale: 1, Index: SRND}.Offset(1*32), XDWORD1, XFER)
	VMOVDQU(XFER, Mem{Base: SP, Scale: 1, Index: SRND}.Offset(_XFER+1*32))
	doRoundN0(_XFER+1*32, e, f, g, h, a, b, c, d, e)
	doRoundN1(_XFER+1*32, d, e, f, g, h, a, b, c, d)
	doRoundN2(_XFER+1*32, c, d, e, f, g, h, a, b, c)
	doRoundN3(_XFER+1*32, b, c, d, e, f, g, h, a, b)

	ADDQ(Imm(2*32), SRND)

	VMOVDQU(XDWORD2, XDWORD0)
	VMOVDQU(XDWORD3, XDWORD1)

	CMPQ(SRND, U32(4*4*32))
	JB(LabelRef("avx2_loop2"))

	Load(Param("dig"), CTX) // d.h[8]
	MOVQ(Mem{Base: SP}.Offset(_INP), INP)

	registers := []GPPhysical{a, b, c, d, e, f, g, h}
	for i, reg := range registers {
		addm(Mem{Base: CTX}.Offset(i*4), reg)
	}

	CMPQ(Mem{Base: SP}.Offset(_INP_END), INP)
	JB(LabelRef("done_hash"))

	XORQ(SRND, SRND)
}

// Do second block using previously scheduled results
func avx2_loop3() {
	Label("avx2_loop3")
	doRoundN0(_XFER+0*32+16, a, b, c, d, e, f, g, h, a)
	doRoundN1(_XFER+0*32+16, h, a, b, c, d, e, f, g, h)
	doRoundN2(_XFER+0*32+16, g, h, a, b, c, d, e, f, g)
	doRoundN3(_XFER+0*32+16, f, g, h, a, b, c, d, e, f)

	doRoundN0(_XFER+1*32+16, e, f, g, h, a, b, c, d, e)
	doRoundN1(_XFER+1*32+16, d, e, f, g, h, a, b, c, d)
	doRoundN2(_XFER+1*32+16, c, d, e, f, g, h, a, b, c)
	doRoundN3(_XFER+1*32+16, b, c, d, e, f, g, h, a, b)

	ADDQ(Imm(2*32), SRND)
	CMPQ(SRND, U32(4*4*32))
	JB(LabelRef("avx2_loop3"))

	Load(Param("dig"), CTX) // d.h[8]
	MOVQ(Mem{Base: SP}.Offset(_INP), INP)
	ADDQ(Imm(64), INP)

	registers := []GPPhysical{a, b, c, d, e, f, g, h}
	for i, reg := range registers {
		addm(Mem{Base: CTX}.Offset(i*4), reg)
	}

	CMPQ(Mem{Base: SP}.Offset(_INP_END), INP)
	JA(LabelRef("avx2_loop0"))
	JB(LabelRef("done_hash"))
}

func avx2_do_last_block() {
	Label("avx2_do_last_block")
	VMOVDQU(Mem{Base: INP}.Offset(0), XWORD0)
	VMOVDQU(Mem{Base: INP}.Offset(16), XWORD1)
	VMOVDQU(Mem{Base: INP}.Offset(32), XWORD2)
	VMOVDQU(Mem{Base: INP}.Offset(48), XWORD3)

	flip_mask := flip_mask_DATA()
	VMOVDQU(flip_mask, BYTE_FLIP_MASK)

	VPSHUFB(X_BYTE_FLIP_MASK, XWORD0, XWORD0)
	VPSHUFB(X_BYTE_FLIP_MASK, XWORD1, XWORD1)
	VPSHUFB(X_BYTE_FLIP_MASK, XWORD2, XWORD2)
	VPSHUFB(X_BYTE_FLIP_MASK, XWORD3, XWORD3)

	K256 := K256_DATA()
	LEAQ(K256, TBL)

	JMP(LabelRef("avx2_last_block_enter"))
}

// Load initial digest
func avx2_only_one_block() {
	Label("avx2_only_one_block")
	registers := []GPPhysical{a, b, c, d, e, f, g, h}
	for i, reg := range registers {
		MOVL(Mem{Base: CTX}.Offset(i*4), reg)
	}
	JMP(LabelRef("avx2_do_last_block"))
}

func done_hash() {
	Label("done_hash")
	VZEROUPPER()
	RET()
}

func sha_ni() {
	Label("sha_ni")
	Load(Param("dig"), digestPtr)    //                   init digest hash vector H0, H1,..., H7 pointer
	Load(Param("p").Base(), dataPtr) //                   init input data base pointer
	Load(Param("p").Len(), numBytes) //                   get number of input bytes to hash
	SHRQ(Imm(6), numBytes)           //                   force modulo 64 input buffer length
	SHLQ(Imm(6), numBytes)
	CMPQ(numBytes, Imm(0)) //                             exit early for zero-length input buffer
	JEQ(LabelRef("done"))
	ADDQ(dataPtr, numBytes)                            // point numBytes to end of input buffer
	VMOVDQU(Mem{Base: digestPtr}.Offset(0*16), state0) // load initial hash values and reorder
	VMOVDQU(Mem{Base: digestPtr}.Offset(1*16), state1) // DCBA, HGFE -> ABEF, CDGH
	PSHUFD(Imm(0xb1), state0, state0)                  // CDAB
	PSHUFD(Imm(0x1b), state1, state1)                  // EFGH
	VMOVDQA(state0, m4)
	PALIGNR(Imm(8), state1, state0) //                    ABEF
	PBLENDW(Imm(0xf0), m4, state1)  //                    CDGH
	flip_mask := flip_mask_DATA()
	VMOVDQA(flip_mask, shufMask)
	LEAQ(K256_DATA(), sha256Constants)

	roundLoop()
	done()
}

func roundLoop() {
	Label("roundLoop")
	Comment("save hash values for addition after rounds")
	VMOVDQA(state0, abefSave)
	VMOVDQA(state1, cdghSave)

	Comment("do rounds 0-59")
	rounds0to11(m0, nil, 0, nop)       //                 0-3
	rounds0to11(m1, m0, 1, sha256msg1) //                 4-7
	rounds0to11(m2, m1, 2, sha256msg1) //                8-11
	VMOVDQU(Mem{Base: dataPtr}.Offset(3*16), msg)
	PSHUFB(shufMask, msg)
	rounds12to59(m3, 3, m2, m0, sha256msg1, vmovrev) // 12-15
	rounds12to59(m0, 4, m3, m1, sha256msg1, vmov)    // 16-19
	rounds12to59(m1, 5, m0, m2, sha256msg1, vmov)    // 20-23
	rounds12to59(m2, 6, m1, m3, sha256msg1, vmov)    // 24-27
	rounds12to59(m3, 7, m2, m0, sha256msg1, vmov)    // 28-31
	rounds12to59(m0, 8, m3, m1, sha256msg1, vmov)    // 32-35
	rounds12to59(m1, 9, m0, m2, sha256msg1, vmov)    // 36-39
	rounds12to59(m2, 10, m1, m3, sha256msg1, vmov)   // 40-43
	rounds12to59(m3, 11, m2, m0, sha256msg1, vmov)   // 44-47
	rounds12to59(m0, 12, m3, m1, sha256msg1, vmov)   // 48-51
	rounds12to59(m1, 13, m0, m2, nop, vmov)          // 52-55
	rounds12to59(m2, 14, m1, m3, nop, vmov)          // 56-59

	Comment("do rounds 60-63")
	VMOVDQA(m3, msg)
	PADDD(Mem{Base: sha256Constants}.Offset(15*32), msg)
	SHA256RNDS2(msg, state0, state1)
	PSHUFD(Imm(0x0e), msg, msg)
	SHA256RNDS2(msg, state1, state0)

	Comment("add current hash values with previously saved")
	PADDD(abefSave, state0)
	PADDD(cdghSave, state1)

	Comment("advance data pointer; loop until buffer empty")
	ADDQ(Imm(64), dataPtr)
	CMPQ(numBytes, dataPtr)
	JNE(LabelRef("roundLoop"))

	Comment("write hash values back in the correct order")
	PSHUFD(Imm(0x1b), state0, state0)
	PSHUFD(Imm(0xb1), state1, state1)
	VMOVDQA(state0, m4)
	PBLENDW(Imm(0xf0), state1, state0)
	PALIGNR(Imm(8), m4, state1)
	VMOVDQU(state0, Mem{Base: digestPtr}.Offset(0*16))
	VMOVDQU(state1, Mem{Base: digestPtr}.Offset(1*16))
}

func done() {
	Label("done")
	RET()
}

/**~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~DATA SECTION~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~**/

// Pointers for memoizing Data section symbols
var flip_maskPtr, shuff_00BAPtr, shuff_DC00Ptr, K256Ptr *Mem

// shuffle byte order from LE to BE
func flip_mask_DATA() Mem {
	if flip_maskPtr != nil {
		return *flip_maskPtr
	}

	flip_mask := GLOBL("flip_mask", RODATA)
	flip_maskPtr = &flip_mask

	DATA(0x00, U64(0x0405060700010203))
	DATA(0x08, U64(0x0c0d0e0f08090a0b))
	DATA(0x10, U64(0x0405060700010203))
	DATA(0x18, U64(0x0c0d0e0f08090a0b))
	return flip_mask
}

// shuffle xBxA -> 00BA
func shuff_00BA_DATA() Mem {
	if shuff_00BAPtr != nil {
		return *shuff_00BAPtr
	}

	shuff_00BA := GLOBL("shuff_00BA", RODATA)
	shuff_00BAPtr = &shuff_00BA

	DATA(0x00, U64(0x0b0a090803020100))
	DATA(0x08, U64(0xFFFFFFFFFFFFFFFF))
	DATA(0x10, U64(0x0b0a090803020100))
	DATA(0x18, U64(0xFFFFFFFFFFFFFFFF))
	return shuff_00BA
}

// shuffle xDxC -> DC00
func shuff_DC00_DATA() Mem {
	if shuff_DC00Ptr != nil {
		return *shuff_DC00Ptr
	}

	shuff_DC00 := GLOBL("shuff_DC00", RODATA)
	shuff_DC00Ptr = &shuff_DC00

	DATA(0x00, U64(0xFFFFFFFFFFFFFFFF))
	DATA(0x08, U64(0x0b0a090803020100))
	DATA(0x10, U64(0xFFFFFFFFFFFFFFFF))
	DATA(0x18, U64(0x0b0a090803020100))
	return shuff_DC00
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

// Round specific constants
func K256_DATA() Mem {
	if K256Ptr != nil {
		return *K256Ptr
	}

	K256 := GLOBL("K256", NOPTR+RODATA)
	K256Ptr = &K256

	offset_idx := 0

	for i := 0; i < len(_K); i += 4 {
		DATA((offset_idx+0)*4, U32(_K[i+0])) // k1
		DATA((offset_idx+1)*4, U32(_K[i+1])) // k2
		DATA((offset_idx+2)*4, U32(_K[i+2])) // k3
		DATA((offset_idx+3)*4, U32(_K[i+3])) // k4

		DATA((offset_idx+4)*4, U32(_K[i+0])) // k1
		DATA((offset_idx+5)*4, U32(_K[i+1])) // k2
		DATA((offset_idx+6)*4, U32(_K[i+2])) // k3
		DATA((offset_idx+7)*4, U32(_K[i+3])) // k4
		offset_idx += 8
	}
	return K256
}
