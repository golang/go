// Copyright 2024 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package main

import (
	"os"
	"strings"

	. "github.com/mmcloughlin/avo/build"
	"github.com/mmcloughlin/avo/ir"
	. "github.com/mmcloughlin/avo/operand"
	. "github.com/mmcloughlin/avo/reg"
)

//go:generate go run . -out ../../aes_amd64.s

func main() {
	Package("crypto/aes")
	ConstraintExpr("!purego")
	encryptBlockAsm()
	decryptBlockAsm()
	expandKeyAsm()
	_expand_key_128()
	_expand_key_192a()
	_expand_key_192b()
	_expand_key_256a()
	_expand_key_256b()
	Generate()

	var internalFunctions []string = []string{
		"·_expand_key_128<>",
		"·_expand_key_192a<>",
		"·_expand_key_192b<>",
		"·_expand_key_256a<>",
		"·_expand_key_256b<>",
	}
	removePeskyUnicodeDot(internalFunctions, "../../asm_amd64.s")
}

func encryptBlockAsm() {
	Implement("encryptBlockAsm")
	Attributes(NOSPLIT)
	AllocLocal(0)

	Load(Param("nr"), RCX)
	Load(Param("xk"), RAX)
	Load(Param("dst"), RDX)
	Load(Param("src"), RBX)
	MOVUPS(Mem{Base: AX}.Offset(0), X1)
	MOVUPS(Mem{Base: BX}.Offset(0), X0)
	ADDQ(Imm(16), RAX)
	PXOR(X1, X0)
	SUBQ(Imm(12), RCX)
	JE(LabelRef("Lenc192"))
	JB(LabelRef("Lenc128"))

	Label("Lenc256")
	MOVUPS(Mem{Base: AX}.Offset(0), X1)
	AESENC(X1, X0)
	MOVUPS(Mem{Base: AX}.Offset(16), X1)
	AESENC(X1, X0)
	ADDQ(Imm(32), RAX)

	Label("Lenc192")
	MOVUPS(Mem{Base: AX}.Offset(0), X1)
	AESENC(X1, X0)
	MOVUPS(Mem{Base: AX}.Offset(16), X1)
	AESENC(X1, X0)
	ADDQ(Imm(32), RAX)

	Label("Lenc128")
	MOVUPS(Mem{Base: AX}.Offset(0), X1)
	AESENC(X1, X0)
	MOVUPS(Mem{Base: AX}.Offset(16), X1)
	AESENC(X1, X0)
	MOVUPS(Mem{Base: AX}.Offset(32), X1)
	AESENC(X1, X0)
	MOVUPS(Mem{Base: AX}.Offset(48), X1)
	AESENC(X1, X0)
	MOVUPS(Mem{Base: AX}.Offset(64), X1)
	AESENC(X1, X0)
	MOVUPS(Mem{Base: AX}.Offset(80), X1)
	AESENC(X1, X0)
	MOVUPS(Mem{Base: AX}.Offset(96), X1)
	AESENC(X1, X0)
	MOVUPS(Mem{Base: AX}.Offset(112), X1)
	AESENC(X1, X0)
	MOVUPS(Mem{Base: AX}.Offset(128), X1)
	AESENC(X1, X0)
	MOVUPS(Mem{Base: AX}.Offset(144), X1)
	AESENCLAST(X1, X0)
	MOVUPS(X0, Mem{Base: DX}.Offset(0))
	RET()
}

func decryptBlockAsm() {
	Implement("decryptBlockAsm")
	Attributes(NOSPLIT)
	AllocLocal(0)

	Load(Param("nr"), RCX)
	Load(Param("xk"), RAX)
	Load(Param("dst"), RDX)
	Load(Param("src"), RBX)

	MOVUPS(Mem{Base: AX}.Offset(0), X1)
	MOVUPS(Mem{Base: BX}.Offset(0), X0)
	ADDQ(Imm(16), RAX)
	PXOR(X1, X0)
	SUBQ(Imm(12), RCX)
	JE(LabelRef("Ldec192"))
	JB(LabelRef("Ldec128"))

	Label("Ldec256")
	MOVUPS(Mem{Base: AX}.Offset(0), X1)
	AESDEC(X1, X0)
	MOVUPS(Mem{Base: AX}.Offset(16), X1)
	AESDEC(X1, X0)
	ADDQ(Imm(32), RAX)

	Label("Ldec192")
	MOVUPS(Mem{Base: AX}.Offset(0), X1)
	AESDEC(X1, X0)
	MOVUPS(Mem{Base: AX}.Offset(16), X1)
	AESDEC(X1, X0)
	ADDQ(Imm(32), RAX)

	Label("Ldec128")
	MOVUPS(Mem{Base: AX}.Offset(0), X1)
	AESDEC(X1, X0)
	MOVUPS(Mem{Base: AX}.Offset(16), X1)
	AESDEC(X1, X0)
	MOVUPS(Mem{Base: AX}.Offset(32), X1)
	AESDEC(X1, X0)
	MOVUPS(Mem{Base: AX}.Offset(48), X1)
	AESDEC(X1, X0)
	MOVUPS(Mem{Base: AX}.Offset(64), X1)
	AESDEC(X1, X0)
	MOVUPS(Mem{Base: AX}.Offset(80), X1)
	AESDEC(X1, X0)
	MOVUPS(Mem{Base: AX}.Offset(96), X1)
	AESDEC(X1, X0)
	MOVUPS(Mem{Base: AX}.Offset(112), X1)
	AESDEC(X1, X0)
	MOVUPS(Mem{Base: AX}.Offset(128), X1)
	AESDEC(X1, X0)
	MOVUPS(Mem{Base: AX}.Offset(144), X1)
	AESDECLAST(X1, X0)
	MOVUPS(X0, Mem{Base: DX}.Offset(0))
	RET()
}

// Note that round keys are stored in uint128 format, not uint32
func expandKeyAsm() {
	Implement("expandKeyAsm")
	Attributes(NOSPLIT)
	AllocLocal(0)

	Load(Param("nr"), RCX)
	Load(Param("key"), RAX)
	Load(Param("enc"), RBX)
	Load(Param("dec"), RDX)

	MOVUPS(Mem{Base: AX}, X0)
	Comment("enc")
	MOVUPS(X0, Mem{Base: BX})
	ADDQ(Imm(16), RBX)
	PXOR(X4, X4) // _expand_key_* expect X4 to be zero
	CMPL(ECX, Imm(12))
	JE(LabelRef("Lexp_enc192"))
	JB(LabelRef("Lexp_enc128"))

	Lexp_enc256()
	Lexp_enc192()
	Lexp_enc128()
	Lexp_dec()
	Lexp_dec_loop()
}

func Lexp_enc256() {
	Label("Lexp_enc256")
	MOVUPS(Mem{Base: AX}.Offset(16), X2)
	MOVUPS(X2, Mem{Base: BX})
	ADDQ(Imm(16), RBX)

	var rcon uint64 = 1
	for i := 0; i < 6; i++ {
		AESKEYGENASSIST(Imm(rcon), X2, X1)
		CALL(LabelRef("_expand_key_256a<>(SB)"))
		AESKEYGENASSIST(Imm(rcon), X0, X1)
		CALL(LabelRef("_expand_key_256b<>(SB)"))
		rcon <<= 1
	}
	AESKEYGENASSIST(Imm(0x40), X2, X1)
	CALL(LabelRef("_expand_key_256a<>(SB)"))
	JMP(LabelRef("Lexp_dec"))
}

func Lexp_enc192() {
	Label("Lexp_enc192")
	MOVQ(Mem{Base: AX}.Offset(16), X2)

	var rcon uint64 = 1
	for i := 0; i < 8; i++ {
		AESKEYGENASSIST(Imm(rcon), X2, X1)
		if i%2 == 0 {
			CALL(LabelRef("_expand_key_192a<>(SB)"))
		} else {
			CALL(LabelRef("_expand_key_192b<>(SB)"))
		}
		rcon <<= 1
	}
	JMP(LabelRef("Lexp_dec"))
}

func Lexp_enc128() {
	Label("Lexp_enc128")
	var rcon uint64 = 1
	for i := 0; i < 8; i++ {
		AESKEYGENASSIST(Imm(rcon), X0, X1)
		CALL(LabelRef("_expand_key_128<>(SB)"))
		rcon <<= 1
	}
	AESKEYGENASSIST(Imm(0x1b), X0, X1)
	CALL(LabelRef("_expand_key_128<>(SB)"))
	AESKEYGENASSIST(Imm(0x36), X0, X1)
	CALL(LabelRef("_expand_key_128<>(SB)"))
}

func Lexp_dec() {
	Label("Lexp_dec")
	Comment("dec")
	SUBQ(Imm(16), RBX)
	MOVUPS(Mem{Base: BX}, X1)
	MOVUPS(X1, Mem{Base: DX})
	DECQ(RCX)
}

func Lexp_dec_loop() {
	Label("Lexp_dec_loop")
	MOVUPS(Mem{Base: BX}.Offset(-16), X1)
	AESIMC(X1, X0)
	MOVUPS(X0, Mem{Base: DX}.Offset(16))
	SUBQ(Imm(16), RBX)
	ADDQ(Imm(16), RDX)
	DECQ(RCX)
	JNZ(LabelRef("Lexp_dec_loop"))
	MOVUPS(Mem{Base: BX}.Offset(-16), X0)
	MOVUPS(X0, Mem{Base: DX}.Offset(16))
	RET()
}

func _expand_key_128() {
	Function("_expand_key_128<>")
	Attributes(NOSPLIT)
	AllocLocal(0)

	PSHUFD(Imm(0xff), X1, X1)
	SHUFPS(Imm(0x10), X0, X4)
	PXOR(X4, X0)
	SHUFPS(Imm(0x8c), X0, X4)
	PXOR(X4, X0)
	PXOR(X1, X0)
	MOVUPS(X0, Mem{Base: BX})
	ADDQ(Imm(16), RBX)
	RET()
}

func _expand_key_192a() {
	Function("_expand_key_192a<>")
	Attributes(NOSPLIT)
	AllocLocal(0)

	PSHUFD(Imm(0x55), X1, X1)
	SHUFPS(Imm(0x10), X0, X4)
	PXOR(X4, X0)
	SHUFPS(Imm(0x8c), X0, X4)
	PXOR(X4, X0)
	PXOR(X1, X0)

	MOVAPS(X2, X5)
	MOVAPS(X2, X6)
	PSLLDQ(Imm(0x4), X5)
	PSHUFD(Imm(0xff), X0, X3)
	PXOR(X3, X2)
	PXOR(X5, X2)

	MOVAPS(X0, X1)
	SHUFPS(Imm(0x44), X0, X6)
	MOVUPS(X6, Mem{Base: BX})
	SHUFPS(Imm(0x4e), X2, X1)
	MOVUPS(X1, Mem{Base: BX}.Offset(16))
	ADDQ(Imm(32), RBX)
	RET()
}

func _expand_key_192b() {
	Function("_expand_key_192b<>")
	Attributes(NOSPLIT)
	AllocLocal(0)

	PSHUFD(Imm(0x55), X1, X1)
	SHUFPS(Imm(0x10), X0, X4)
	PXOR(X4, X0)
	SHUFPS(Imm(0x8c), X0, X4)
	PXOR(X4, X0)
	PXOR(X1, X0)

	MOVAPS(X2, X5)
	PSLLDQ(Imm(0x4), X5)
	PSHUFD(Imm(0xff), X0, X3)
	PXOR(X3, X2)
	PXOR(X5, X2)

	MOVUPS(X0, Mem{Base: BX})
	ADDQ(Imm(16), RBX)
	RET()
}

func _expand_key_256a() {
	Function("_expand_key_256a<>")
	Attributes(NOSPLIT)
	AllocLocal(0)

	// Hack to get Avo to emit:
	// 	JMP _expand_key_128<>(SB)
	Instruction(&ir.Instruction{
		Opcode: "JMP",
		Operands: []Op{
			LabelRef("_expand_key_128<>(SB)"),
		},
	})
}

func _expand_key_256b() {
	Function("_expand_key_256b<>")
	Attributes(NOSPLIT)
	AllocLocal(0)

	PSHUFD(Imm(0xaa), X1, X1)
	SHUFPS(Imm(0x10), X2, X4)
	PXOR(X4, X2)
	SHUFPS(Imm(0x8c), X2, X4)
	PXOR(X4, X2)
	PXOR(X1, X2)

	MOVUPS(X2, Mem{Base: BX})
	ADDQ(Imm(16), RBX)
	RET()
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
