// Copyright 2024 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package main

import (
	. "github.com/mmcloughlin/avo/build"
	. "github.com/mmcloughlin/avo/operand"
	. "github.com/mmcloughlin/avo/reg"
)

// The sha-ni implementation uses Intel(R) SHA extensions SHA256RNDS2, SHA256MSG1, SHA256MSG2
// It also reuses portions of the flip_mask (half) and K256 table (stride 32) from the avx2 version
//
// Reference
// S. Gulley, et al, "New Instructions Supporting the Secure Hash
// Algorithm on Intel® Architecture Processors", July 2013
// https://www.intel.com/content/www/us/en/developer/articles/technical/intel-sha-extensions.html

func blockSHANI() {
	Implement("blockSHANI")
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
