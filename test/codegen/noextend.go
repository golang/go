// asmcheck

// Copyright 2018 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package codegen

import (
	"math"
	"math/bits"
	"sync/atomic"
)

var sval64 [8]int64
var sval32 [8]int32
var sval16 [8]int16
var sval8 [8]int8
var val64 [8]uint64
var val32 [8]uint32
var val16 [8]uint16
var val8 [8]uint8

// Avoid zero/sign extensions following a load
// which has extended the value correctly.
// Note: No tests are done for int8 since
// an extra extension is usually needed due to
// no signed byte load.

func set16(x8 int8, u8 *uint8, y8 int8, z8 uint8) {
	// Truncate not needed, load does sign/zero extend

	// ppc64x:-"MOVBZ R\\d+, R\\d+"
	val16[0] = uint16(*u8)

	// AND not needed due to size
	// ppc64x:-"ANDCC"
	sval16[1] = 255 & int16(x8+y8)

	// ppc64x:-"ANDCC"
	val16[1] = 255 & uint16(*u8+z8)

}
func shiftidx(u8 *uint8, x16 *int16, u16 *uint16) {

	// ppc64x:-"MOVBZ R\\d+, R\\d+"
	val16[0] = uint16(sval16[*u8>>2])

	// ppc64x:-"MOVH R\\d+, R\\d+"
	sval16[1] = int16(val16[*x16>>1])

	// ppc64x:-"MOVHZ R\\d+, R\\d+"
	val16[1] = uint16(sval16[*u16>>2])

}

func setnox(x8 int8, u8 *uint8, y8 *int8, z8 *uint8, x16 *int16, u16 *uint16, x32 *int32, u32 *uint32) {

	// ppc64x:-"MOVBZ R\\d+, R\\d+"
	val16[0] = uint16(*u8)

	// AND not needed due to size
	// ppc64x:-"ANDCC"
	sval16[1] = 255 & int16(x8+*y8)

	// ppc64x:-"ANDCC"
	val16[1] = 255 & uint16(*u8+*z8)

	// ppc64x:-"MOVH R\\d+, R\\d+"
	sval32[1] = int32(*x16)

	// ppc64x:-"MOVBZ R\\d+, R\\d+"
	val32[0] = uint32(*u8)

	// ppc64x:-"MOVHZ R\\d+, R\\d+"
	val32[1] = uint32(*u16)

	// ppc64x:-"MOVH R\\d+, R\\d+"
	sval64[1] = int64(*x16)

	// ppc64x:-"MOVW R\\d+, R\\d+"
	sval64[2] = int64(*x32)

	// ppc64x:-"MOVBZ R\\d+, R\\d+"
	val64[0] = uint64(*u8)

	// ppc64x:-"MOVHZ R\\d+, R\\d+"
	val64[1] = uint64(*u16)

	// ppc64x:-"MOVWZ R\\d+, R\\d+"
	val64[2] = uint64(*u32)
}

func cmp16(u8 *uint8, x32 *int32, u32 *uint32, x64 *int64, u64 *uint64) bool {

	// ppc64x:-"MOVBZ R\\d+, R\\d+"
	if uint16(*u8) == val16[0] {
		return true
	}

	// ppc64x:-"MOVHZ R\\d+, R\\d+"
	if uint16(*u32>>16) == val16[0] {
		return true
	}

	// ppc64x:-"MOVHZ R\\d+, R\\d+"
	if uint16(*u64>>48) == val16[0] {
		return true
	}

	// Verify the truncates are using the correct sign.
	// ppc64x:-"MOVHZ R\\d+, R\\d+"
	if int16(*x32) == sval16[0] {
		return true
	}

	// ppc64x:-"MOVH R\\d+, R\\d+"
	if uint16(*u32) == val16[0] {
		return true
	}

	// ppc64x:-"MOVHZ R\\d+, R\\d+"
	if int16(*x64) == sval16[0] {
		return true
	}

	// ppc64x:-"MOVH R\\d+, R\\d+"
	if uint16(*u64) == val16[0] {
		return true
	}

	return false
}

func cmp32(u8 *uint8, x16 *int16, u16 *uint16, x64 *int64, u64 *uint64) bool {

	// ppc64x:-"MOVBZ R\\d+, R\\d+"
	if uint32(*u8) == val32[0] {
		return true
	}

	// ppc64x:-"MOVH R\\d+, R\\d+"
	if int32(*x16) == sval32[0] {
		return true
	}

	// ppc64x:-"MOVHZ R\\d+, R\\d+"
	if uint32(*u16) == val32[0] {
		return true
	}

	// Verify the truncates are using the correct sign.
	// ppc64x:-"MOVWZ R\\d+, R\\d+"
	if int32(*x64) == sval32[0] {
		return true
	}

	// ppc64x:-"MOVW R\\d+, R\\d+"
	if uint32(*u64) == val32[0] {
		return true
	}

	return false
}

func cmp64(u8 *uint8, x16 *int16, u16 *uint16, x32 *int32, u32 *uint32) bool {

	// ppc64x:-"MOVBZ R\\d+, R\\d+"
	if uint64(*u8) == val64[0] {
		return true
	}

	// ppc64x:-"MOVH R\\d+, R\\d+"
	if int64(*x16) == sval64[0] {
		return true
	}

	// ppc64x:-"MOVHZ R\\d+, R\\d+"
	if uint64(*u16) == val64[0] {
		return true
	}

	// ppc64x:-"MOVW R\\d+, R\\d+"
	if int64(*x32) == sval64[0] {
		return true
	}

	// ppc64x:-"MOVWZ R\\d+, R\\d+"
	if uint64(*u32) == val64[0] {
		return true
	}
	return false
}

// no unsign extension following 32 bits ops

func noUnsignEXT(t1, t2, t3, t4 uint32, k int64) uint64 {
	var ret uint64

	// arm64:"RORW" -"MOVWU"
	ret += uint64(bits.RotateLeft32(t1, 7))

	// arm64:"MULW" -"MOVWU"
	ret *= uint64(t1 * t2)

	// arm64:"MNEGW" -"MOVWU"
	ret += uint64(-t1 * t3)

	// arm64:"UDIVW" -"MOVWU"
	ret += uint64(t1 / t4)

	// arm64:-"MOVWU"
	ret += uint64(t2 % t3)

	// arm64:"MSUBW" -"MOVWU"
	ret += uint64(t1 - t2*t3)

	// arm64:"MADDW" -"MOVWU"
	ret += uint64(t3*t4 + t2)

	// arm64:"REVW" -"MOVWU"
	ret += uint64(bits.ReverseBytes32(t1))

	// arm64:"RBITW" -"MOVWU"
	ret += uint64(bits.Reverse32(t1))

	// arm64:"CLZW" -"MOVWU"
	ret += uint64(bits.LeadingZeros32(t1))

	// arm64:"REV16W" -"MOVWU"
	ret += uint64(((t1 & 0xff00ff00) >> 8) | ((t1 & 0x00ff00ff) << 8))

	// arm64:"EXTRW" -"MOVWU"
	ret += uint64((t1 << 25) | (t2 >> 7))

	return ret
}

// no sign extension when the upper bits of the result are zero

func noSignEXT(x int) int64 {
	t1 := int32(x)

	var ret int64

	// arm64:-"MOVW"
	ret += int64(t1 & 1)

	// arm64:-"MOVW"
	ret += int64(int32(x & 0x7fffffff))

	// arm64:-"MOVH"
	ret += int64(int16(x & 0x7fff))

	// arm64:-"MOVB"
	ret += int64(int8(x & 0x7f))

	return ret
}

// corner cases that sign extension must not be omitted

func shouldSignEXT(x int) int64 {
	t1 := int32(x)

	var ret int64

	// arm64:"MOVW"
	ret += int64(t1 & (-1))

	// arm64:"MOVW"
	ret += int64(int32(x & 0x80000000))

	// arm64:"MOVW"
	ret += int64(int32(x & 0x1100000011111111))

	// arm64:"MOVH"
	ret += int64(int16(x & 0x1100000000001111))

	// arm64:"MOVB"
	ret += int64(int8(x & 0x1100000000000011))

	return ret
}

func noIntermediateExtension(a, b, c uint32) uint32 {
	// arm64:-"MOVWU"
	return a*b*9 + c
}

// A zero-extension of a value already produced by an instruction that
// zeroed the upper bits folds away entirely, one test per op recognized
// by ZeroUpper*Bits. The positive check pins the producing instruction;
// the negative checks verify that no zero-extension -- emitted as a
// plain reg-to-reg MOVL on amd64, MOVWU/MOVD on arm64 -- and no other
// register copy survives.
//
// Entries with no test here: MOVLconst/MOVQconst (constant arguments
// fold before lowering), MOVLQZX and the arm64 MOV*Ureg ops as
// producers (stacked extensions already collapse in the main rewrite
// pass), SUBLconst (canonicalized to ADDLconst of the negated
// constant), the ops with signed narrow results which the guard at the
// top of ZeroUpper32Bits rejects (SARL, SARXL, CVTTS*2SL, DIVW, MODW,
// FCVTZ*W), SBBLcarrymask (feeds ANDL in shift lowerings, never a
// zero-extension directly), and the idx1/idx8 indexed forms (not
// reachable from straightforward Go code).

func noZeroExtADDL(a, b uint32) uint64 {
	// amd64:"ADDL" -`MOVL [A-Z][A-Z0-9]*, [A-Z][A-Z0-9]*`
	return uint64(a + b)
}

func noZeroExtADDLconst(a uint32) uint64 {
	// amd64:"ADDL [$]100" -`MOVL [A-Z][A-Z0-9]*, [A-Z][A-Z0-9]*`
	return uint64(a + 100)
}

func noZeroExtSUBL(a, b uint32) uint64 {
	// amd64:"SUBL" -`MOVL [A-Z][A-Z0-9]*, [A-Z][A-Z0-9]*`
	return uint64(a - b)
}

func noZeroExtANDL(a, b uint32) uint64 {
	// amd64:"ANDL" -`MOVL [A-Z][A-Z0-9]*, [A-Z][A-Z0-9]*`
	return uint64(a & b)
}

func noZeroExtANDLconst(a uint32) uint64 {
	// The mask keeps bit 31 set so that the extension is not folded
	// into the mask instead (a positive mask implies a zero upper half).
	// amd64:"ANDL [$]-15" -`MOVL [A-Z][A-Z0-9]*, [A-Z][A-Z0-9]*`
	return uint64(a & 0xfffffff1)
}

func noZeroExtORL(a, b uint32) uint64 {
	// amd64:"ORL" -`MOVL [A-Z][A-Z0-9]*, [A-Z][A-Z0-9]*`
	return uint64(a | b)
}

func noZeroExtORLconst(a uint32) uint64 {
	// amd64:"ORL [$]65521" -`MOVL [A-Z][A-Z0-9]*, [A-Z][A-Z0-9]*`
	return uint64(a | 0xfff1)
}

func noZeroExtXORL(a, b uint32) uint64 {
	// amd64:"XORL" -`MOVL [A-Z][A-Z0-9]*, [A-Z][A-Z0-9]*`
	return uint64(a ^ b)
}

func noZeroExtXORLconst(a uint32) uint64 {
	// amd64:"XORL [$]65521" -`MOVL [A-Z][A-Z0-9]*, [A-Z][A-Z0-9]*`
	return uint64(a ^ 0xfff1)
}

func noZeroExtNEGL(a uint32) uint64 {
	// amd64:"NEGL" -`MOVL [A-Z][A-Z0-9]*, [A-Z][A-Z0-9]*`
	return uint64(-a)
}

func noZeroExtNOTL(a uint32) uint64 {
	// amd64:"NOTL" -`MOVL [A-Z][A-Z0-9]*, [A-Z][A-Z0-9]*`
	return uint64(^a)
}

func noZeroExtSHLL(a uint32, n int) uint64 {
	// amd64/v1:"SHLL"
	// amd64/v3:"SHLXL"
	// amd64:-`MOVL [A-Z][A-Z0-9]*, [A-Z][A-Z0-9]*`
	return uint64(a << (n & 31))
}

func noZeroExtSHLLconst(a uint32) uint64 {
	// amd64:"SHLL [$]7" -`MOVL [A-Z][A-Z0-9]*, [A-Z][A-Z0-9]*`
	return uint64(a << 7)
}

func noZeroExtSHRL(a uint32, n int) uint64 {
	// amd64/v1:"SHRL"
	// amd64/v3:"SHRXL"
	// amd64:-`MOVL [A-Z][A-Z0-9]*, [A-Z][A-Z0-9]*`
	return uint64(a >> (n & 31))
}

func noZeroExtSHRLconst(a uint32) uint64 {
	// amd64:"SHRL [$]7" -`MOVL [A-Z][A-Z0-9]*, [A-Z][A-Z0-9]*`
	return uint64(a >> 7)
}

func noZeroExtROLL(a uint32, n int) uint64 {
	// amd64:"ROLL" -`MOVL [A-Z][A-Z0-9]*, [A-Z][A-Z0-9]*`
	// arm64:"RORW" -"MOVWU R[0-9]+, R[0-9]+" -"MOVD R[0-9]+, R[0-9]+"
	return uint64(bits.RotateLeft32(a, n))
}

func noZeroExtROLLconst(a uint32) uint64 {
	// amd64:"ROLL [$]7" -`MOVL [A-Z][A-Z0-9]*, [A-Z][A-Z0-9]*`
	// arm64:"RORW [$]25" -"MOVWU R[0-9]+, R[0-9]+" -"MOVD R[0-9]+, R[0-9]+"
	return uint64(bits.RotateLeft32(a, 7))
}

func noZeroExtRORL(a uint32, n int) uint64 {
	// amd64:"RORL" -`MOVL [A-Z][A-Z0-9]*, [A-Z][A-Z0-9]*`
	// arm64:"RORW" -"MOVWU R[0-9]+, R[0-9]+" -"MOVD R[0-9]+, R[0-9]+"
	return uint64(bits.RotateLeft32(a, -n))
}

func noZeroExtLEAL1(a, b uint32) uint64 {
	// amd64:"LEAL 1" -`MOVL [A-Z][A-Z0-9]*, [A-Z][A-Z0-9]*`
	return uint64(a + b + 1)
}

func noZeroExtLEAL2(a, b uint32) uint64 {
	// amd64:`LEAL \(.*\)\(.*\*2\)` -`MOVL [A-Z][A-Z0-9]*, [A-Z][A-Z0-9]*`
	return uint64(a + 2*b)
}

func noZeroExtLEAL4(a, b uint32) uint64 {
	// amd64:`LEAL \(.*\)\(.*\*4\)` -`MOVL [A-Z][A-Z0-9]*, [A-Z][A-Z0-9]*`
	return uint64(a + 4*b)
}

func noZeroExtLEAL8(a, b uint32) uint64 {
	// amd64:`LEAL \(.*\)\(.*\*8\)` -`MOVL [A-Z][A-Z0-9]*, [A-Z][A-Z0-9]*`
	return uint64(a + 8*b)
}

func noZeroExtIMULL(a, b uint32) uint64 {
	// amd64:"IMULL" -`MOVL [A-Z][A-Z0-9]*, [A-Z][A-Z0-9]*`
	// arm64:"MULW" -"MOVWU R[0-9]+, R[0-9]+" -"MOVD R[0-9]+, R[0-9]+"
	return uint64(a * b)
}

func noZeroExtIMUL3L(a uint32) uint64 {
	// amd64:"IMUL3L" -`MOVL [A-Z][A-Z0-9]*, [A-Z][A-Z0-9]*`
	return uint64(a * 12345)
}

func noZeroExtBSWAPL(a uint32) uint64 {
	// amd64:"BSWAPL" -`MOVL [A-Z][A-Z0-9]*, [A-Z][A-Z0-9]*`
	// arm64:"REVW" -"MOVWU R[0-9]+, R[0-9]+" -"MOVD R[0-9]+, R[0-9]+"
	return uint64(bits.ReverseBytes32(a))
}

// At v1 the POPCNT feature check turns the count into a phi of the
// POPCNTL and the fallback call, so only v2 and up fold.
func noZeroExtPOPCNTL(a uint32) uint64 {
	// amd64/v2:"POPCNTL" -`MOVL [A-Z][A-Z0-9]*, [A-Z][A-Z0-9]*`
	return uint64(uint32(bits.OnesCount32(a)))
}

func noZeroExtANDNL(a, b uint32) uint64 {
	// amd64/v3:"ANDNL"
	// amd64:-`MOVL [A-Z][A-Z0-9]*, [A-Z][A-Z0-9]*`
	return uint64(^a & b)
}

func noZeroExtBLSIL(a uint32) uint64 {
	// The negative check is v3-scoped: without BLSI the input needs a
	// two-address copy next to the NEGL on this line.
	// amd64/v3:"BLSIL" -`MOVL [A-Z][A-Z0-9]*, [A-Z][A-Z0-9]*`
	return uint64(a & -a)
}

func noZeroExtBLSMSKL(a uint32) uint64 {
	// amd64/v3:"BLSMSKL"
	// amd64:-`MOVL [A-Z][A-Z0-9]*, [A-Z][A-Z0-9]*`
	return uint64(a ^ (a - 1))
}

func noZeroExtBTSL(a uint32, n int) uint64 {
	// amd64:"BTSL" -`MOVL [A-Z][A-Z0-9]*, [A-Z][A-Z0-9]*`
	return uint64(a | 1<<(n&31))
}

func noZeroExtBTRL(a uint32, n int) uint64 {
	// amd64:"BTRL" -`MOVL [A-Z][A-Z0-9]*, [A-Z][A-Z0-9]*`
	return uint64(a &^ (1 << (n & 31)))
}

func noZeroExtBTCL(a uint32, n int) uint64 {
	// amd64:"BTCL" -`MOVL [A-Z][A-Z0-9]*, [A-Z][A-Z0-9]*`
	return uint64(a ^ 1<<(n&31))
}

func noZeroExtMOVLf2i(f float32) uint64 {
	// The negative check here spells out the GPRs so that it does not
	// match the MOVL X, R this line is expected to produce.
	// amd64:"MOVL X[0-9]+," -`MOVL (AX|BX|CX|DX|SI|DI|BP|R[0-9]+), [A-Z][A-Z0-9]*`
	return uint64(math.Float32bits(f))
}

// In 64-bit mode a 32-bit CMOV zero-extends its destination even when
// the condition is false. The compiler inverts the comparisons below,
// so each test is named after the CMOV op it actually produces, and the
// CMOV is attributed to the line the value is used on, so the checks
// sit on the converting line. arm64 lowers these to CSEL, which
// ZeroUpper32Bits does not model, so only amd64 is checked.

func noZeroExtCMOVLEQ(a, b uint32) uint64 {
	c := a
	if a == b {
		c = b
	}
	// amd64:"CMOVLEQ" -`MOVL [A-Z][A-Z0-9]*, [A-Z][A-Z0-9]*`
	return uint64(c)
}

func noZeroExtCMOVLNE(a, b uint32) uint64 {
	c := a
	if a != b {
		c = b
	}
	// amd64:"CMOVLNE" -`MOVL [A-Z][A-Z0-9]*, [A-Z][A-Z0-9]*`
	return uint64(c)
}

func noZeroExtCMOVLGT(a, b uint32) uint64 {
	c := a
	if int32(a) < int32(b) {
		c = b
	}
	// amd64:"CMOVLGT" -`MOVL [A-Z][A-Z0-9]*, [A-Z][A-Z0-9]*`
	return uint64(c)
}

func noZeroExtCMOVLLT(a, b uint32) uint64 {
	c := a
	if int32(a) > int32(b) {
		c = b
	}
	// amd64:"CMOVLLT" -`MOVL [A-Z][A-Z0-9]*, [A-Z][A-Z0-9]*`
	return uint64(c)
}

func noZeroExtCMOVLGE(a, b uint32) uint64 {
	c := a
	if int32(a) <= int32(b) {
		c = b
	}
	// amd64:"CMOVLGE" -`MOVL [A-Z][A-Z0-9]*, [A-Z][A-Z0-9]*`
	return uint64(c)
}

func noZeroExtCMOVLLE(a, b uint32) uint64 {
	c := a
	if int32(a) >= int32(b) {
		c = b
	}
	// amd64:"CMOVLLE" -`MOVL [A-Z][A-Z0-9]*, [A-Z][A-Z0-9]*`
	return uint64(c)
}

func noZeroExtCMOVLHI(a, b uint32) uint64 {
	c := a
	if a < b {
		c = b
	}
	// amd64:"CMOVLHI" -`MOVL [A-Z][A-Z0-9]*, [A-Z][A-Z0-9]*`
	return uint64(c)
}

func noZeroExtCMOVLCS(a, b uint32) uint64 {
	c := a
	if a > b {
		c = b
	}
	// amd64:"CMOVLCS" -`MOVL [A-Z][A-Z0-9]*, [A-Z][A-Z0-9]*`
	return uint64(c)
}

func noZeroExtCMOVLCC(a, b uint32) uint64 {
	c := a
	if a <= b {
		c = b
	}
	// amd64:"CMOVLCC" -`MOVL [A-Z][A-Z0-9]*, [A-Z][A-Z0-9]*`
	return uint64(c)
}

func noZeroExtCMOVLLS(a, b uint32) uint64 {
	c := a
	if a >= b {
		c = b
	}
	// amd64:"CMOVLLS" -`MOVL [A-Z][A-Z0-9]*, [A-Z][A-Z0-9]*`
	return uint64(c)
}

func noZeroExtCMOVLGTF(x, y float64, a, b uint32) uint64 {
	c := a
	if x > y {
		c = b
	}
	// CMOVLGTF assembles to CMOVLHI.
	// amd64:"CMOVLHI" -`MOVL [A-Z][A-Z0-9]*, [A-Z][A-Z0-9]*`
	return uint64(c)
}

func noZeroExtCMOVLGEF(x, y float64, a, b uint32) uint64 {
	c := a
	if x >= y {
		c = b
	}
	// CMOVLGEF assembles to CMOVLCC.
	// amd64:"CMOVLCC" -`MOVL [A-Z][A-Z0-9]*, [A-Z][A-Z0-9]*`
	return uint64(c)
}

func noZeroExtCMOVLEQF(x, y float64, a, b uint32) uint64 {
	c := a
	if x == y {
		c = b
	}
	// CMOVLEQF assembles to CMOVLNE+CMOVLPC and needs a scratch
	// register, so the copy feeding it also lands on this line.
	// amd64:"CMOVLPC"
	return uint64(c)
}

func noZeroExtCMOVLNEF(x, y float64, a, b uint32) uint64 {
	c := a
	if x != y {
		c = b
	}
	// CMOVLNEF assembles to CMOVLNE+CMOVLPS.
	// amd64:"CMOVLPS" -`MOVL [A-Z][A-Z0-9]*, [A-Z][A-Z0-9]*`
	return uint64(c)
}

// A zero-extending load already cleared the upper bits, so the
// extension folds away entirely on both architectures.

func noZeroExtLoad32(p *uint32) uint64 {
	// amd64:`MOVL \(` -`MOVL [A-Z][A-Z0-9]*, [A-Z][A-Z0-9]*`
	// arm64:`MOVWU \(` -"MOVWU R[0-9]+, R[0-9]+" -"MOVD R[0-9]+, R[0-9]+"
	return uint64(*p)
}

func noZeroExtLoad16(p *uint16) uint64 {
	// amd64:`MOVWLZX \(` -`MOVWLZX [A-Z][A-Z0-9]*, [A-Z][A-Z0-9]*`
	// arm64:`MOVHU \(` -"MOVHU R[0-9]+, R[0-9]+" -"MOVD R[0-9]+, R[0-9]+"
	return uint64(*p)
}

func noZeroExtLoad8(p *uint8) uint64 {
	// amd64:`MOVBLZX \(` -`MOVBLZX [A-Z][A-Z0-9]*, [A-Z][A-Z0-9]*`
	// arm64:`MOVBU \(` -"MOVBU R[0-9]+, R[0-9]+" -"MOVD R[0-9]+, R[0-9]+"
	return uint64(*p)
}

func noZeroExtLoadidx32(s *[8]uint32, i int) uint64 {
	// amd64:`MOVL \(.*\)\(.*\*4\)` -`MOVL [A-Z][A-Z0-9]*, [A-Z][A-Z0-9]*`
	// arm64:`MOVWU \(R[0-9]+\)\(R[0-9]+<<2\)` -"MOVWU R[0-9]+, R[0-9]+" -"MOVD R[0-9]+, R[0-9]+"
	return uint64(s[i&(len(s)-1)])
}

func noZeroExtLoadidx16(s *[8]uint16, i int) uint64 {
	// amd64:`MOVWLZX \(.*\)\(.*\*2\)` -`MOVWLZX [A-Z][A-Z0-9]*, [A-Z][A-Z0-9]*`
	// arm64:`MOVHU \(R[0-9]+\)\(R[0-9]+<<1\)` -"MOVHU R[0-9]+, R[0-9]+" -"MOVD R[0-9]+, R[0-9]+"
	return uint64(s[i&(len(s)-1)])
}

func noZeroExtLoadidx8(s *[8]uint8, i int) uint64 {
	// amd64:`MOVBLZX \(.*\)\(.*\*1\)` -`MOVBLZX [A-Z][A-Z0-9]*, [A-Z][A-Z0-9]*`
	// arm64:`MOVBU \(R[0-9]+\)\(R[0-9]+\)` -"MOVBU R[0-9]+, R[0-9]+" -"MOVD R[0-9]+, R[0-9]+"
	return uint64(s[i&(len(s)-1)])
}

func noZeroExtMOVBEL(p *uint32) uint64 {
	// amd64/v1:"BSWAPL"
	// amd64/v3:`MOVBEL \(`
	// amd64:-`MOVL [A-Z][A-Z0-9]*, [A-Z][A-Z0-9]*`
	return uint64(bits.ReverseBytes32(*p))
}

func noZeroExtMOVBELloadidx(s *[8]uint32, i int) uint64 {
	// amd64/v3:`MOVBEL \(.*\)\(.*\*4\)`
	// amd64:-`MOVL [A-Z][A-Z0-9]*, [A-Z][A-Z0-9]*`
	return uint64(bits.ReverseBytes32(s[i&(len(s)-1)]))
}

func noZeroExtADDLload(x uint32, p *uint32) uint64 {
	// amd64:`ADDL \(` -`MOVL [A-Z][A-Z0-9]*, [A-Z][A-Z0-9]*`
	return uint64(x + *p)
}

func noZeroExtSUBLload(x uint32, p *uint32) uint64 {
	// amd64:`SUBL \(` -`MOVL [A-Z][A-Z0-9]*, [A-Z][A-Z0-9]*`
	return uint64(x - *p)
}

func noZeroExtANDLload(x uint32, p *uint32) uint64 {
	// amd64:`ANDL \(` -`MOVL [A-Z][A-Z0-9]*, [A-Z][A-Z0-9]*`
	return uint64(x & *p)
}

func noZeroExtORLload(x uint32, p *uint32) uint64 {
	// amd64:`ORL \(` -`MOVL [A-Z][A-Z0-9]*, [A-Z][A-Z0-9]*`
	return uint64(x | *p)
}

func noZeroExtXORLload(x uint32, p *uint32) uint64 {
	// amd64:`XORL \(` -`MOVL [A-Z][A-Z0-9]*, [A-Z][A-Z0-9]*`
	return uint64(x ^ *p)
}

func noZeroExtADDLloadidx(x uint32, s *[8]uint32, i int) uint64 {
	// amd64:`ADDL \(.*\)\(.*\*4\)` -`MOVL [A-Z][A-Z0-9]*, [A-Z][A-Z0-9]*`
	return uint64(x + s[i&(len(s)-1)])
}

func noZeroExtSUBLloadidx(x uint32, s *[8]uint32, i int) uint64 {
	// amd64:`SUBL \(.*\)\(.*\*4\)` -`MOVL [A-Z][A-Z0-9]*, [A-Z][A-Z0-9]*`
	return uint64(x - s[i&(len(s)-1)])
}

func noZeroExtANDLloadidx(x uint32, s *[8]uint32, i int) uint64 {
	// amd64:`ANDL \(.*\)\(.*\*4\)` -`MOVL [A-Z][A-Z0-9]*, [A-Z][A-Z0-9]*`
	return uint64(x & s[i&(len(s)-1)])
}

func noZeroExtORLloadidx(x uint32, s *[8]uint32, i int) uint64 {
	// amd64:`ORL \(.*\)\(.*\*4\)` -`MOVL [A-Z][A-Z0-9]*, [A-Z][A-Z0-9]*`
	return uint64(x | s[i&(len(s)-1)])
}

func noZeroExtXORLloadidx(x uint32, s *[8]uint32, i int) uint64 {
	// amd64:`XORL \(.*\)\(.*\*4\)` -`MOVL [A-Z][A-Z0-9]*, [A-Z][A-Z0-9]*`
	return uint64(x ^ s[i&(len(s)-1)])
}

func noZeroExtSHRXLload(p *uint32, n int) uint64 {
	// amd64/v3:`SHRXL [A-Z][A-Z0-9]*, \(`
	// amd64:-`MOVL [A-Z][A-Z0-9]*, [A-Z][A-Z0-9]*`
	return uint64(*p >> (n & 31))
}

func noZeroExtSHLXLload(p *uint32, n int) uint64 {
	// amd64/v3:`SHLXL [A-Z][A-Z0-9]*, \(`
	// amd64:-`MOVL [A-Z][A-Z0-9]*, [A-Z][A-Z0-9]*`
	return uint64(*p << (n & 31))
}

func noZeroExtSHRXLloadidx(s *[8]uint32, i, n int) uint64 {
	// amd64/v3:`SHRXL [A-Z][A-Z0-9]*, \(.*\*4\)`
	// amd64:-`MOVL [A-Z][A-Z0-9]*, [A-Z][A-Z0-9]*`
	return uint64(s[i&(len(s)-1)] >> (n & 31))
}

func noZeroExtSHLXLloadidx(s *[8]uint32, i, n int) uint64 {
	// amd64/v3:`SHLXL [A-Z][A-Z0-9]*, \(.*\*4\)`
	// amd64:-`MOVL [A-Z][A-Z0-9]*, [A-Z][A-Z0-9]*`
	return uint64(s[i&(len(s)-1)] << (n & 31))
}

// The remaining arm64 ops recognized by ZeroUpper32Bits.

func noZeroExtREV16W(a uint32) uint64 {
	// arm64:"REV16W" -"MOVWU R[0-9]+, R[0-9]+" -"MOVD R[0-9]+, R[0-9]+"
	return uint64((a&0xff00ff00)>>8 | (a&0x00ff00ff)<<8)
}

func noZeroExtRBITW(a uint32) uint64 {
	// arm64:"RBITW" -"MOVWU R[0-9]+, R[0-9]+" -"MOVD R[0-9]+, R[0-9]+"
	return uint64(bits.Reverse32(a))
}

func noZeroExtCLZW(a uint32) uint64 {
	// arm64:"CLZW" -"MOVWU R[0-9]+, R[0-9]+" -"MOVD R[0-9]+, R[0-9]+"
	return uint64(uint32(bits.LeadingZeros32(a)))
}

func noZeroExtEXTRW(a, b uint32) uint64 {
	// arm64:"EXTRW [$]4" -"MOVWU R[0-9]+, R[0-9]+" -"MOVD R[0-9]+, R[0-9]+"
	return uint64(a>>4 | b<<28)
}

func noZeroExtMNEGW(a, b uint32) uint64 {
	// arm64:"MNEGW" -"MOVWU R[0-9]+, R[0-9]+" -"MOVD R[0-9]+, R[0-9]+"
	return uint64(-(a * b))
}

func noZeroExtMADDW(a, b, d uint32) uint64 {
	// arm64:"MADDW" -"MOVWU R[0-9]+, R[0-9]+" -"MOVD R[0-9]+, R[0-9]+"
	return uint64(a*b + d)
}

func noZeroExtMSUBW(a, b, d uint32) uint64 {
	// arm64:"MSUBW" -"MOVWU R[0-9]+, R[0-9]+" -"MOVD R[0-9]+, R[0-9]+"
	return uint64(d - a*b)
}

// amd64 lowers this division to DIVL and selects the quotient out of
// its tuple, recognized through the Select0 case.
func noZeroExtUDIVW(a, b uint32) uint64 {
	// amd64:"DIVL" -`MOVL [A-Z][A-Z0-9]*, [A-Z][A-Z0-9]*`
	// arm64:"UDIVW" -"MOVWU R[0-9]+, R[0-9]+" -"MOVD R[0-9]+, R[0-9]+"
	return uint64(a / b)
}

// amd64 is not checked: the remainder of DIVL is fixed in DX, so a
// move to the result register survives regardless.
func noZeroExtUMODW(a, b uint32) uint64 {
	// arm64:"UDIVW" "MSUBW" -"MOVWU R[0-9]+, R[0-9]+" -"MOVD R[0-9]+, R[0-9]+"
	return uint64(a % b)
}

// ZeroUpper48Bits: a byte-sized producer also lets a 16-to-64-bit
// zero-extension (emitted as a reg-to-reg MOVWLZX on amd64, MOVHU on
// arm64) fold away.

func noZeroExt48MOVBQZX(x uint8) uint64 {
	// amd64:"MOVBLZX" -"MOVWLZX"
	// arm64:"MOVBU R[0-9]+, R[0-9]+" -"MOVHU R[0-9]+, R[0-9]+" -"MOVD R[0-9]+, R[0-9]+"
	return uint64(uint16(x))
}

func noZeroExt48MOVBload(p *uint8) uint64 {
	// amd64:`MOVBLZX \(` -"MOVWLZX"
	// arm64:`MOVBU \(` -"MOVHU R[0-9]+, R[0-9]+" -"MOVD R[0-9]+, R[0-9]+"
	return uint64(uint16(*p))
}

func noZeroExt48MOVBloadidx(s *[8]uint8, i int) uint64 {
	// amd64:`MOVBLZX \(.*\)\(.*\*1\)` -"MOVWLZX"
	// arm64:`MOVBU \(R[0-9]+\)\(R[0-9]+\)` -"MOVHU R[0-9]+, R[0-9]+" -"MOVD R[0-9]+, R[0-9]+"
	return uint64(uint16(s[i&(len(s)-1)]))
}

// Tuple-producing ops: a zero-extension of a Select of these folds
// through the tuple (the Select cases in ZeroUpper32Bits). Besides the
// divisions above, the reachable ones are BLSRL, the atomic loads,
// XADDLlock and the arm64 pair load. XCHGL's old value and LDPW's
// words on amd64 land in registers that must then be moved to the
// result register, indistinguishable from an extension, so they are
// not checked. MULLU, NEGLflags and ADDLconstflags have no producers
// reachable from Go.

func noZeroExtBLSRL(a uint32) uint64 {
	// amd64/v3:"BLSRL"
	// amd64:-`MOVL [A-Z][A-Z0-9]*, [A-Z][A-Z0-9]*`
	return uint64(a & (a - 1))
}

func noZeroExtMOVLatomicload(p *uint32) uint64 {
	// amd64:`MOVL \(` -`MOVL [A-Z][A-Z0-9]*, [A-Z][A-Z0-9]*`
	// arm64:"LDARW" -"MOVWU R[0-9]+, R[0-9]+" -"MOVD R[0-9]+, R[0-9]+"
	return uint64(atomic.LoadUint32(p))
}

func noZeroExtXADDLlock(p *uint32) uint64 {
	// amd64:"XADDL" -`MOVL [A-Z][A-Z0-9]*, [A-Z][A-Z0-9]*`
	return uint64(atomic.AddUint32(p, 1))
}

func noZeroExtLDPW(p *[2]uint32) (uint64, uint64) {
	// arm64:"LDPW" -"MOVWU R[0-9]+, R[0-9]+" -"MOVD R[0-9]+, R[0-9]+"
	return uint64(p[0]), uint64(p[1])
}
