// Copyright 2024 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package main

import (
	"fmt"
	"sync"

	. "github.com/mmcloughlin/avo/build"
	. "github.com/mmcloughlin/avo/operand"
	. "github.com/mmcloughlin/avo/reg"
)

//go:generate go run . -out ../../ctr_amd64.s

func main() {
	Package("crypto/internal/fips140/aes")
	ConstraintExpr("!purego")

	ctrBlocks(1)
	ctrBlocks(2)
	ctrBlocks(4)
	ctrBlocks(8)

	Generate()
}

func ctrBlocks(numBlocks int) {
	Implement(fmt.Sprintf("ctrBlocks%dAsm", numBlocks))

	rounds := Load(Param("nr"), GP64())
	xk := Load(Param("xk"), GP64())
	dst := Load(Param("dst"), GP64())
	src := Load(Param("src"), GP64())
	ivlo := Load(Param("ivlo"), GP64())
	ivhi := Load(Param("ivhi"), GP64())

	bswap := XMM()
	MOVOU(bswapMask(), bswap)

	blocks := make([]VecVirtual, numBlocks)

	// For the 8-block case we optimize counter generation. We build the first
	// counter as usual, then check whether the remaining seven increments will
	// overflow. When they do not (the common case) we keep the work entirely in
	// XMM registers to avoid expensive general-purpose -> XMM moves. Otherwise
	// we fall back to the traditional scalar path.
	if numBlocks == 8 {
		for i := range blocks {
			blocks[i] = XMM()
		}

		base := XMM()
		tmp := GP64()
		addVec := XMM()

		MOVQ(ivlo, blocks[0])
		PINSRQ(Imm(1), ivhi, blocks[0])
		MOVAPS(blocks[0], base)
		PSHUFB(bswap, blocks[0])

		// Check whether any of these eight counters will overflow.
		MOVQ(ivlo, tmp)
		ADDQ(Imm(uint64(numBlocks-1)), tmp)
		slowLabel := fmt.Sprintf("ctr%d_slow", numBlocks)
		doneLabel := fmt.Sprintf("ctr%d_done", numBlocks)
		JC(LabelRef(slowLabel))

		// Fast branch: create an XMM increment vector containing the value 1.
		// Adding it to the base counter yields each subsequent counter.
		XORQ(tmp, tmp)
		INCQ(tmp)
		PXOR(addVec, addVec)
		PINSRQ(Imm(0), tmp, addVec)

		for i := 1; i < numBlocks; i++ {
			PADDQ(addVec, base)
			MOVAPS(base, blocks[i])
		}
		JMP(LabelRef(doneLabel))

		Label(slowLabel)
		ADDQ(Imm(1), ivlo)
		ADCQ(Imm(0), ivhi)
		for i := 1; i < numBlocks; i++ {
			MOVQ(ivlo, blocks[i])
			PINSRQ(Imm(1), ivhi, blocks[i])
			if i < numBlocks-1 {
				ADDQ(Imm(1), ivlo)
				ADCQ(Imm(0), ivhi)
			}
		}

		Label(doneLabel)

		// Convert little-endian counters to big-endian after the branch since
		// both paths share the same shuffle sequence.
		for i := 1; i < numBlocks; i++ {
			PSHUFB(bswap, blocks[i])
		}
	} else {
		// Lay out counter block plaintext.
		for i := 0; i < numBlocks; i++ {
			x := XMM()
			blocks[i] = x

			MOVQ(ivlo, x)
			PINSRQ(Imm(1), ivhi, x)
			PSHUFB(bswap, x)
			if i < numBlocks-1 {
				ADDQ(Imm(1), ivlo)
				ADCQ(Imm(0), ivhi)
			}
		}
	}

	// Initial key add.
	aesRoundStart(blocks, Mem{Base: xk})
	ADDQ(Imm(16), xk)

	// Branch based on the number of rounds.
	SUBQ(Imm(12), rounds)
	JE(LabelRef("enc192"))
	JB(LabelRef("enc128"))

	// Two extra rounds for 256-bit keys.
	aesRound(blocks, Mem{Base: xk})
	aesRound(blocks, Mem{Base: xk}.Offset(16))
	ADDQ(Imm(32), xk)

	// Two extra rounds for 192-bit keys.
	Label("enc192")
	aesRound(blocks, Mem{Base: xk})
	aesRound(blocks, Mem{Base: xk}.Offset(16))
	ADDQ(Imm(32), xk)

	// 10 rounds for 128-bit keys (with special handling for the final round).
	Label("enc128")
	for i := 0; i < 9; i++ {
		aesRound(blocks, Mem{Base: xk}.Offset(16*i))
	}
	aesRoundLast(blocks, Mem{Base: xk}.Offset(16*9))

	// XOR state with src and write back to dst.
	for i, b := range blocks {
		x := XMM()

		MOVUPS(Mem{Base: src}.Offset(16*i), x)
		PXOR(b, x)
		MOVUPS(x, Mem{Base: dst}.Offset(16*i))
	}

	RET()
}

func aesRoundStart(blocks []VecVirtual, k Mem) {
	x := XMM()
	MOVUPS(k, x)
	for _, b := range blocks {
		PXOR(x, b)
	}
}

func aesRound(blocks []VecVirtual, k Mem) {
	x := XMM()
	MOVUPS(k, x)
	for _, b := range blocks {
		AESENC(x, b)
	}
}

func aesRoundLast(blocks []VecVirtual, k Mem) {
	x := XMM()
	MOVUPS(k, x)
	for _, b := range blocks {
		AESENCLAST(x, b)
	}
}

var bswapMask = sync.OnceValue(func() Mem {
	bswapMask := GLOBL("bswapMask", NOPTR|RODATA)
	DATA(0x00, U64(0x08090a0b0c0d0e0f))
	DATA(0x08, U64(0x0001020304050607))
	return bswapMask
})
