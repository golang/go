// Copyright 2024 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package main

import (
	. "github.com/mmcloughlin/avo/build"
	. "github.com/mmcloughlin/avo/operand"
	. "github.com/mmcloughlin/avo/reg"
)

// The avx2-version is described in an Intel White-Paper:
// "Fast SHA-256 Implementations on Intel Architecture Processors"
// To find it, surf to http://www.intel.com/p/en_US/embedded
// and search for that title.
// AVX2 version by Intel, same algorithm as code in Linux kernel:
// https://github.com/torvalds/linux/blob/master/lib/crypto/x86/sha256-avx2-asm.S
// by
//     James Guilford <james.guilford@intel.com>
//     Kirk Yap <kirk.s.yap@intel.com>
//     Tim Chen <tim.c.chen@linux.intel.com>

func blockAVX2() {
	Implement("blockAVX2")
	AllocLocal(536)

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
