// Copyright 2024 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package main

import (
	"fmt"

	. "github.com/mmcloughlin/avo/build"
	. "github.com/mmcloughlin/avo/operand"
	. "github.com/mmcloughlin/avo/reg"
)

// Implement the SHA-1 block function using the Intel(R) SHA extensions
// (SHA1RNDS4, SHA1NEXTE, SHA1MSG1, and SHA1MSG2). This implementation requires
// the AVX, SHA, SSE2, SSE4.1, and SSSE3 extensions.
//
// Reference:
// S. Gulley, et al, "New Instructions Supporting the Secure Hash
// Algorithm on Intel® Architecture Processors", July 2013
// https://www.intel.com/content/www/us/en/developer/articles/technical/intel-sha-extensions.html

func blockSHANI() {
	Implement("blockSHANI")

	digest := Load(Param("dig"), RDI)
	data := Load(Param("p").Base(), RSI)
	len := Load(Param("p").Len(), RDX)

	abcd := XMM()
	msg0, msg1, msg2, msg3 := XMM(), XMM(), XMM(), XMM()
	e0, e1 := XMM(), XMM()
	shufMask := XMM()

	CMPQ(len, Imm(0))
	JEQ(LabelRef("done"))
	ADDQ(data, len)

	stackPtr := GP64()
	{
		Comment("Allocate space on the stack for saving ABCD and E0, and align it to 16 bytes")
		local := AllocLocal(32 + 16)
		LEAQ(local.Offset(15), stackPtr)
		tmp := GP64()
		MOVQ(U64(15), tmp)
		NOTQ(tmp)
		ANDQ(tmp, stackPtr)
	}
	e0_save := Mem{Base: stackPtr}
	abcd_save := Mem{Base: stackPtr}.Offset(16)

	Comment("Load initial hash state")
	PINSRD(Imm(3), Mem{Base: digest}.Offset(16), e0)
	VMOVDQU(Mem{Base: digest}, abcd)
	PAND(upperMask(), e0)
	PSHUFD(Imm(0x1b), abcd, abcd)

	VMOVDQA(flipMask(), shufMask)

	Label("loop")

	Comment("Save ABCD and E working values")
	VMOVDQA(e0, e0_save)
	VMOVDQA(abcd, abcd_save)

	Comment("Rounds 0-3")
	VMOVDQU(Mem{Base: data}, msg0)
	PSHUFB(shufMask, msg0)
	PADDD(msg0, e0)
	VMOVDQA(abcd, e1)
	SHA1RNDS4(Imm(0), e0, abcd)

	Comment("Rounds 4-7")
	VMOVDQU(Mem{Base: data}.Offset(16), msg1)
	PSHUFB(shufMask, msg1)
	SHA1NEXTE(msg1, e1)
	VMOVDQA(abcd, e0)
	SHA1RNDS4(Imm(0), e1, abcd)
	SHA1MSG1(msg1, msg0)

	Comment("Rounds 8-11")
	VMOVDQU(Mem{Base: data}.Offset(16*2), msg2)
	PSHUFB(shufMask, msg2)
	SHA1NEXTE(msg2, e0)
	VMOVDQA(abcd, e1)
	SHA1RNDS4(Imm(0), e0, abcd)
	SHA1MSG1(msg2, msg1)
	PXOR(msg2, msg0)

	// Rounds 12 through 67 use the same repeated pattern, with e0 and e1 ping-ponging
	// back and forth, and each of the msg temporaries moving up one every four rounds.
	msgs := []VecVirtual{msg3, msg0, msg1, msg2}
	for i := range 14 {
		Comment(fmt.Sprintf("Rounds %d-%d", 12+(i*4), 12+(i*4)+3))
		a, b := e1, e0
		if i == 0 {
			VMOVDQU(Mem{Base: data}.Offset(16*3), msg3)
			PSHUFB(shufMask, msg3)
		}
		if i%2 == 1 {
			a, b = e0, e1
		}
		imm := uint64((12 + i*4) / 20)

		SHA1NEXTE(msgs[i%4], a)
		VMOVDQA(abcd, b)
		SHA1MSG2(msgs[i%4], msgs[(1+i)%4])
		SHA1RNDS4(Imm(imm), a, abcd)
		SHA1MSG1(msgs[i%4], msgs[(3+i)%4])
		PXOR(msgs[i%4], msgs[(2+i)%4])
	}

	Comment("Rounds 68-71")
	SHA1NEXTE(msg1, e1)
	VMOVDQA(abcd, e0)
	SHA1MSG2(msg1, msg2)
	SHA1RNDS4(Imm(3), e1, abcd)
	PXOR(msg1, msg3)

	Comment("Rounds 72-75")
	SHA1NEXTE(msg2, e0)
	VMOVDQA(abcd, e1)
	SHA1MSG2(msg2, msg3)
	SHA1RNDS4(Imm(3), e0, abcd)

	Comment("Rounds 76-79")
	SHA1NEXTE(msg3, e1)
	VMOVDQA(abcd, e0)
	SHA1RNDS4(Imm(3), e1, abcd)

	Comment("Add saved E and ABCD")
	SHA1NEXTE(e0_save, e0)
	PADDD(abcd_save, abcd)

	Comment("Check if we are done, if not return to the loop")
	ADDQ(Imm(64), data)
	CMPQ(data, len)
	JNE(LabelRef("loop"))

	Comment("Write the hash state back to digest")
	PSHUFD(Imm(0x1b), abcd, abcd)
	VMOVDQU(abcd, Mem{Base: digest})
	PEXTRD(Imm(3), e0, Mem{Base: digest}.Offset(16))

	Label("done")
	RET()
}

func flipMask() Mem {
	mask := GLOBL("shuffle_mask", RODATA)
	// 0x000102030405060708090a0b0c0d0e0f
	DATA(0x00, U64(0x08090a0b0c0d0e0f))
	DATA(0x08, U64(0x0001020304050607))
	return mask
}

func upperMask() Mem {
	mask := GLOBL("upper_mask", RODATA)
	// 0xFFFFFFFF000000000000000000000000
	DATA(0x00, U64(0x0000000000000000))
	DATA(0x08, U64(0xFFFFFFFF00000000))
	return mask
}
