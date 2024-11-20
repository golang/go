// Copyright 2024 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// This code was translated into a form compatible with 6a from the public
// domain sources at https://github.com/gvanas/KeccakCodePackage

package main

import (
	"os"

	. "github.com/mmcloughlin/avo/build"
	. "github.com/mmcloughlin/avo/operand"
	. "github.com/mmcloughlin/avo/reg"
	_ "golang.org/x/crypto/sha3"
)

//go:generate go run . -out ../sha3_amd64.s

// Round Constants for use in the ι step.
var RoundConstants = [24]uint64{
	0x0000000000000001,
	0x0000000000008082,
	0x800000000000808A,
	0x8000000080008000,
	0x000000000000808B,
	0x0000000080000001,
	0x8000000080008081,
	0x8000000000008009,
	0x000000000000008A,
	0x0000000000000088,
	0x0000000080008009,
	0x000000008000000A,
	0x000000008000808B,
	0x800000000000008B,
	0x8000000000008089,
	0x8000000000008003,
	0x8000000000008002,
	0x8000000000000080,
	0x000000000000800A,
	0x800000008000000A,
	0x8000000080008081,
	0x8000000000008080,
	0x0000000080000001,
	0x8000000080008008,
}

var (
	// Temporary registers
	rT1 GPPhysical = RAX

	// Round vars
	rpState = Mem{Base: RDI}
	rpStack = Mem{Base: RSP}

	rDa = RBX
	rDe = RCX
	rDi = RDX
	rDo = R8
	rDu = R9

	rBa = R10
	rBe = R11
	rBi = R12
	rBo = R13
	rBu = R14

	rCa = RSI
	rCe = RBP
	rCi = rBi
	rCo = rBo
	rCu = R15
)

const (
	_ba = iota * 8
	_be
	_bi
	_bo
	_bu
	_ga
	_ge
	_gi
	_go
	_gu
	_ka
	_ke
	_ki
	_ko
	_ku
	_ma
	_me
	_mi
	_mo
	_mu
	_sa
	_se
	_si
	_so
	_su
)

func main() {
	// https://github.com/mmcloughlin/avo/issues/450
	os.Setenv("GOOS", "linux")
	os.Setenv("GOARCH", "amd64")

	Package("crypto/internal/fips/sha3")
	ConstraintExpr("!purego")
	keccakF1600()
	Generate()
}

func MOVQ_RBI_RCE() { MOVQ(rBi, rCe) }
func XORQ_RT1_RCA() { XORQ(rT1, rCa) }
func XORQ_RT1_RCE() { XORQ(rT1, rCe) }
func XORQ_RBA_RCU() { XORQ(rBa, rCu) }
func XORQ_RBE_RCU() { XORQ(rBe, rCu) }
func XORQ_RDU_RCU() { XORQ(rDu, rCu) }
func XORQ_RDA_RCA() { XORQ(rDa, rCa) }
func XORQ_RDE_RCE() { XORQ(rDe, rCe) }

type ArgMacro func()

func mKeccakRound(
	iState, oState Mem,
	rc U64,
	B_RBI_RCE, G_RT1_RCA, G_RT1_RCE, G_RBA_RCU,
	K_RT1_RCA, K_RT1_RCE, K_RBA_RCU, M_RT1_RCA,
	M_RT1_RCE, M_RBE_RCU, S_RDU_RCU, S_RDA_RCA,
	S_RDE_RCE ArgMacro,
) {
	Comment("Prepare round")
	MOVQ(rCe, rDa)
	ROLQ(Imm(1), rDa)

	MOVQ(iState.Offset(_bi), rCi)
	XORQ(iState.Offset(_gi), rDi)
	XORQ(rCu, rDa)
	XORQ(iState.Offset(_ki), rCi)
	XORQ(iState.Offset(_mi), rDi)
	XORQ(rDi, rCi)

	MOVQ(rCi, rDe)
	ROLQ(Imm(1), rDe)

	MOVQ(iState.Offset(_bo), rCo)
	XORQ(iState.Offset(_go), rDo)
	XORQ(rCa, rDe)
	XORQ(iState.Offset(_ko), rCo)
	XORQ(iState.Offset(_mo), rDo)
	XORQ(rDo, rCo)

	MOVQ(rCo, rDi)
	ROLQ(Imm(1), rDi)

	MOVQ(rCu, rDo)
	XORQ(rCe, rDi)
	ROLQ(Imm(1), rDo)

	MOVQ(rCa, rDu)
	XORQ(rCi, rDo)
	ROLQ(Imm(1), rDu)

	Comment("Result b")
	MOVQ(iState.Offset(_ba), rBa)
	MOVQ(iState.Offset(_ge), rBe)
	XORQ(rCo, rDu)
	MOVQ(iState.Offset(_ki), rBi)
	MOVQ(iState.Offset(_mo), rBo)
	MOVQ(iState.Offset(_su), rBu)
	XORQ(rDe, rBe)
	ROLQ(Imm(44), rBe)
	XORQ(rDi, rBi)
	XORQ(rDa, rBa)
	ROLQ(Imm(43), rBi)

	MOVQ(rBe, rCa)
	MOVQ(rc, rT1)
	ORQ(rBi, rCa)
	XORQ(rBa, rT1)
	XORQ(rT1, rCa)
	MOVQ(rCa, oState.Offset(_ba))

	XORQ(rDu, rBu)
	ROLQ(Imm(14), rBu)
	MOVQ(rBa, rCu)
	ANDQ(rBe, rCu)
	XORQ(rBu, rCu)
	MOVQ(rCu, oState.Offset(_bu))

	XORQ(rDo, rBo)
	ROLQ(Imm(21), rBo)
	MOVQ(rBo, rT1)
	ANDQ(rBu, rT1)
	XORQ(rBi, rT1)
	MOVQ(rT1, oState.Offset(_bi))

	NOTQ(rBi)
	ORQ(rBa, rBu)
	ORQ(rBo, rBi)
	XORQ(rBo, rBu)
	XORQ(rBe, rBi)
	MOVQ(rBu, oState.Offset(_bo))
	MOVQ(rBi, oState.Offset(_be))
	B_RBI_RCE()

	Comment("Result g")
	MOVQ(iState.Offset(_gu), rBe)
	XORQ(rDu, rBe)
	MOVQ(iState.Offset(_ka), rBi)
	ROLQ(Imm(20), rBe)
	XORQ(rDa, rBi)
	ROLQ(Imm(3), rBi)
	MOVQ(iState.Offset(_bo), rBa)
	MOVQ(rBe, rT1)
	ORQ(rBi, rT1)
	XORQ(rDo, rBa)
	MOVQ(iState.Offset(_me), rBo)
	MOVQ(iState.Offset(_si), rBu)
	ROLQ(Imm(28), rBa)
	XORQ(rBa, rT1)
	MOVQ(rT1, oState.Offset(_ga))
	G_RT1_RCA()

	XORQ(rDe, rBo)
	ROLQ(Imm(45), rBo)
	MOVQ(rBi, rT1)
	ANDQ(rBo, rT1)
	XORQ(rBe, rT1)
	MOVQ(rT1, oState.Offset(_ge))
	G_RT1_RCE()

	XORQ(rDi, rBu)
	ROLQ(Imm(61), rBu)
	MOVQ(rBu, rT1)
	ORQ(rBa, rT1)
	XORQ(rBo, rT1)
	MOVQ(rT1, oState.Offset(_go))

	ANDQ(rBe, rBa)
	XORQ(rBu, rBa)
	MOVQ(rBa, oState.Offset(_gu))
	NOTQ(rBu)
	G_RBA_RCU()

	ORQ(rBu, rBo)
	XORQ(rBi, rBo)
	MOVQ(rBo, oState.Offset(_gi))

	Comment("Result k")
	MOVQ(iState.Offset(_be), rBa)
	MOVQ(iState.Offset(_gi), rBe)
	MOVQ(iState.Offset(_ko), rBi)
	MOVQ(iState.Offset(_mu), rBo)
	MOVQ(iState.Offset(_sa), rBu)
	XORQ(rDi, rBe)
	ROLQ(Imm(6), rBe)
	XORQ(rDo, rBi)
	ROLQ(Imm(25), rBi)
	MOVQ(rBe, rT1)
	ORQ(rBi, rT1)
	XORQ(rDe, rBa)
	ROLQ(Imm(1), rBa)
	XORQ(rBa, rT1)
	MOVQ(rT1, oState.Offset(_ka))
	K_RT1_RCA()

	XORQ(rDu, rBo)
	ROLQ(Imm(8), rBo)
	MOVQ(rBi, rT1)
	ANDQ(rBo, rT1)
	XORQ(rBe, rT1)
	MOVQ(rT1, oState.Offset(_ke))
	K_RT1_RCE()

	XORQ(rDa, rBu)
	ROLQ(Imm(18), rBu)
	NOTQ(rBo)
	MOVQ(rBo, rT1)
	ANDQ(rBu, rT1)
	XORQ(rBi, rT1)
	MOVQ(rT1, oState.Offset(_ki))

	MOVQ(rBu, rT1)
	ORQ(rBa, rT1)
	XORQ(rBo, rT1)
	MOVQ(rT1, oState.Offset(_ko))

	ANDQ(rBe, rBa)
	XORQ(rBu, rBa)
	MOVQ(rBa, oState.Offset(_ku))
	K_RBA_RCU()

	Comment("Result m")
	MOVQ(iState.Offset(_ga), rBe)
	XORQ(rDa, rBe)
	MOVQ(iState.Offset(_ke), rBi)
	ROLQ(Imm(36), rBe)
	XORQ(rDe, rBi)
	MOVQ(iState.Offset(_bu), rBa)
	ROLQ(Imm(10), rBi)
	MOVQ(rBe, rT1)
	MOVQ(iState.Offset(_mi), rBo)
	ANDQ(rBi, rT1)
	XORQ(rDu, rBa)
	MOVQ(iState.Offset(_so), rBu)
	ROLQ(Imm(27), rBa)
	XORQ(rBa, rT1)
	MOVQ(rT1, oState.Offset(_ma))
	M_RT1_RCA()

	XORQ(rDi, rBo)
	ROLQ(Imm(15), rBo)
	MOVQ(rBi, rT1)
	ORQ(rBo, rT1)
	XORQ(rBe, rT1)
	MOVQ(rT1, oState.Offset(_me))
	M_RT1_RCE()

	XORQ(rDo, rBu)
	ROLQ(Imm(56), rBu)
	NOTQ(rBo)
	MOVQ(rBo, rT1)
	ORQ(rBu, rT1)
	XORQ(rBi, rT1)
	MOVQ(rT1, oState.Offset(_mi))

	ORQ(rBa, rBe)
	XORQ(rBu, rBe)
	MOVQ(rBe, oState.Offset(_mu))

	ANDQ(rBa, rBu)
	XORQ(rBo, rBu)
	MOVQ(rBu, oState.Offset(_mo))
	M_RBE_RCU()

	Comment("Result s")
	MOVQ(iState.Offset(_bi), rBa)
	MOVQ(iState.Offset(_go), rBe)
	MOVQ(iState.Offset(_ku), rBi)
	XORQ(rDi, rBa)
	MOVQ(iState.Offset(_ma), rBo)
	ROLQ(Imm(62), rBa)
	XORQ(rDo, rBe)
	MOVQ(iState.Offset(_se), rBu)
	ROLQ(Imm(55), rBe)

	XORQ(rDu, rBi)
	MOVQ(rBa, rDu)
	XORQ(rDe, rBu)
	ROLQ(Imm(2), rBu)
	ANDQ(rBe, rDu)
	XORQ(rBu, rDu)
	MOVQ(rDu, oState.Offset(_su))

	ROLQ(Imm(39), rBi)
	S_RDU_RCU()
	NOTQ(rBe)
	XORQ(rDa, rBo)
	MOVQ(rBe, rDa)
	ANDQ(rBi, rDa)
	XORQ(rBa, rDa)
	MOVQ(rDa, oState.Offset(_sa))
	S_RDA_RCA()

	ROLQ(Imm(41), rBo)
	MOVQ(rBi, rDe)
	ORQ(rBo, rDe)
	XORQ(rBe, rDe)
	MOVQ(rDe, oState.Offset(_se))
	S_RDE_RCE()

	MOVQ(rBo, rDi)
	MOVQ(rBu, rDo)
	ANDQ(rBu, rDi)
	ORQ(rBa, rDo)
	XORQ(rBi, rDi)
	XORQ(rBo, rDo)
	MOVQ(rDi, oState.Offset(_si))
	MOVQ(rDo, oState.Offset(_so))
}

// keccakF1600 applies the Keccak permutation to a 1600b-wide
// state represented as a slice of 25 uint64s.
func keccakF1600() {
	Implement("keccakF1600")
	AllocLocal(200)

	Load(Param("a"), rpState.Base)

	Comment("Convert the user state into an internal state")
	NOTQ(rpState.Offset(_be))
	NOTQ(rpState.Offset(_bi))
	NOTQ(rpState.Offset(_go))
	NOTQ(rpState.Offset(_ki))
	NOTQ(rpState.Offset(_mi))
	NOTQ(rpState.Offset(_sa))

	Comment("Execute the KeccakF permutation")
	MOVQ(rpState.Offset(_ba), rCa)
	MOVQ(rpState.Offset(_be), rCe)
	MOVQ(rpState.Offset(_bu), rCu)

	XORQ(rpState.Offset(_ga), rCa)
	XORQ(rpState.Offset(_ge), rCe)
	XORQ(rpState.Offset(_gu), rCu)

	XORQ(rpState.Offset(_ka), rCa)
	XORQ(rpState.Offset(_ke), rCe)
	XORQ(rpState.Offset(_ku), rCu)

	XORQ(rpState.Offset(_ma), rCa)
	XORQ(rpState.Offset(_me), rCe)
	XORQ(rpState.Offset(_mu), rCu)

	XORQ(rpState.Offset(_sa), rCa)
	XORQ(rpState.Offset(_se), rCe)
	MOVQ(rpState.Offset(_si), rDi)
	MOVQ(rpState.Offset(_so), rDo)
	XORQ(rpState.Offset(_su), rCu)

	for i, rc := range RoundConstants[:len(RoundConstants)-1] {
		var iState, oState Mem
		if i%2 == 0 {
			iState, oState = rpState, rpStack
		} else {
			iState, oState = rpStack, rpState
		}
		mKeccakRound(iState, oState, U64(rc), MOVQ_RBI_RCE, XORQ_RT1_RCA, XORQ_RT1_RCE, XORQ_RBA_RCU, XORQ_RT1_RCA, XORQ_RT1_RCE, XORQ_RBA_RCU, XORQ_RT1_RCA, XORQ_RT1_RCE, XORQ_RBE_RCU, XORQ_RDU_RCU, XORQ_RDA_RCA, XORQ_RDE_RCE)
	}
	mKeccakRound(rpStack, rpState, U64(RoundConstants[len(RoundConstants)-1]), NOP, NOP, NOP, NOP, NOP, NOP, NOP, NOP, NOP, NOP, NOP, NOP, NOP)

	Comment("Revert the internal state to the user state")
	NOTQ(rpState.Offset(_be))
	NOTQ(rpState.Offset(_bi))
	NOTQ(rpState.Offset(_go))
	NOTQ(rpState.Offset(_ki))
	NOTQ(rpState.Offset(_mi))
	NOTQ(rpState.Offset(_sa))

	RET()
}
