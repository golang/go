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

	blocks := make([]VecVirtual, 0, numBlocks)

	// Lay out counter block plaintext.
	for i := 0; i < numBlocks; i++ {
		x := XMM()
		blocks = append(blocks, x)

		MOVQ(ivlo, x)
		PINSRQ(Imm(1), ivhi, x)
		PSHUFB(bswap, x)
		if i < numBlocks-1 {
			ADDQ(Imm(1), ivlo)
			ADCQ(Imm(0), ivhi)
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
