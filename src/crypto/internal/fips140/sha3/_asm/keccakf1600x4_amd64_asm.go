// Copyright 2026 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package main

import (
	. "github.com/mmcloughlin/avo/build"
	. "github.com/mmcloughlin/avo/operand"
	. "github.com/mmcloughlin/avo/reg"
)

// keccakF1600x4 implements the permutation for four states at once with AVX2,
// one 64-bit Keccak lane from each state per 256-bit register.
//
// The four states are kept interleaved in two ping-pong stack buffers of 25
// 32-byte rows each, with the θ D values, the current χ row, and the ρ
// byte-rotation shuffle controls in registers.
//
//	Y0-Y4    θ D values (persist for the round)
//	Y5-Y9    θ C values, then the ρ+π inputs of the current output row
//	Y10-Y11  temporaries
//	Y12-Y13  VPSHUFB controls for the ρ rotations by 8 and 56
//	Y14      round constant, χ and ι results
func keccakF1600x4() {
	Implement("keccakF1600x4AVX2")
	buf := AllocLocal(2*bufSize + 32)

	var p [4]GPPhysical = [4]GPPhysical{RDI, RSI, RDX, RCX}
	Load(Param("a0"), p[0])
	Load(Param("a1"), p[1])
	Load(Param("a2"), p[2])
	Load(Param("a3"), p[3])

	Comment("Align the state buffers to 32 bytes")
	LEAQ(buf.Offset(31), R10)
	ANDQ(I8(-32), R10)

	x4TransposeIn(p)

	VMOVDQU(rho8_DATA(), Y12)
	VMOVDQU(rho56_DATA(), Y13)

	Comment("Loop over the rounds, two per iteration to alternate buffers")
	LEAQ(rc_DATA(), R11)
	MOVQ(U32(12), R8)
	Label("loop")
	x4Round(0, 1, Mem{Base: R11})
	x4Round(1, 0, Mem{Base: R11, Disp: 8})
	ADDQ(Imm(16), R11)
	DECQ(R8)
	JNE(LabelRef("loop"))

	x4TransposeOut(p)

	VZEROUPPER()
	RET()
}

const bufSize = 25 * 32

// lane returns the 32-byte row holding lane i of the four states in the
// ping-pong buffer buf (0 or 1).
func lane(buf, i int) Mem {
	return Mem{Base: R10, Disp: buf*bufSize + i*32}
}

// x4Rows lists, for each of the five output rows of a round (b, g, k, m, s),
// the five ρ+π source lanes and their left rotations, derived from FIPS 202
// Sections 3.2.2-3.2.3: output lane (x, y) = (j, r) comes from input lane
// A[(x+3y) mod 5, x], rotated by its ρ offset. The rows match the fused round
// structure of mKeccakRound in keccakf_amd64_asm.go, although χ is computed
// directly, without the complemented lanes trick.
var x4Rows = func() (rows [5][5]struct{ src, rot int }) {
	var rho [25]int
	x, y := 1, 0
	for t := range 24 {
		rho[x+5*y] = (t + 1) * (t + 2) / 2 % 64
		x, y = y, (2*x+3*y)%5
	}
	for r := range 5 {
		for j := range 5 {
			src := (j+3*r)%5 + 5*j
			rows[r][j].src = src
			rows[r][j].rot = rho[src]
		}
	}
	return rows
}()

// x4Round applies one round, reading the interleaved state from buffer in and
// writing it to buffer out, with the round constant at rc.
func x4Round(in, out int, rc Mem) {
	C := [5]VecPhysical{Y5, Y6, Y7, Y8, Y9}
	D := [5]VecPhysical{Y0, Y1, Y2, Y3, Y4}
	T := C // reused once D is computed

	Comment("θ: C[x] = A[x,0] ^ A[x,1] ^ A[x,2] ^ A[x,3] ^ A[x,4]")
	for x := range 5 {
		VMOVDQA(lane(in, x), C[x])
		for y := 1; y < 5; y++ {
			VPXOR(lane(in, x+5*y), C[x], C[x])
		}
	}
	Comment("θ: D[x] = C[x-1] ^ (C[x+1] <<< 1)")
	for x := range 5 {
		VPSLLQ(Imm(1), C[(x+1)%5], Y10)
		VPSRLQ(Imm(63), C[(x+1)%5], Y11)
		VPOR(Y11, Y10, Y10)
		VPXOR(C[(x+4)%5], Y10, D[x])
	}

	for r := range 5 {
		Commentf("ρ and π: row %d of the fused round", r)
		for j := range 5 {
			e := x4Rows[r][j]
			VPXOR(lane(in, e.src), D[e.src%5], T[j])
			switch e.rot {
			case 0:
			case 8:
				VPSHUFB(Y12, T[j], T[j])
			case 56:
				VPSHUFB(Y13, T[j], T[j])
			default:
				VPSLLQ(Imm(uint64(e.rot)), T[j], Y10)
				VPSRLQ(Imm(uint64(64-e.rot)), T[j], T[j])
				VPOR(Y10, T[j], T[j])
			}
		}
		if r == 0 {
			VPBROADCASTQ(rc, Y14)
			Comment("χ and ι")
		} else {
			Comment("χ")
		}
		for j := range 5 {
			VPANDN(T[(j+2)%5], T[(j+1)%5], Y10)
			if r == 0 && j == 0 {
				VPXOR(Y14, Y10, Y10)
			}
			VPXOR(Y10, T[j], Y10)
			VMOVDQA(Y10, lane(out, r*5+j))
		}
	}
}

// x4TransposeIn interleaves the four sequential states into buffer 0, so that
// row i holds lane i of each of the four states.
func x4TransposeIn(p [4]GPPhysical) {
	Comment("Interleave the four states into the stack buffer")
	for q := range 6 {
		for i, pi := range p {
			VMOVDQU(Mem{Base: pi, Disp: q * 32}, [4]VecPhysical{Y0, Y1, Y2, Y3}[i])
		}
		VPUNPCKLQDQ(Y1, Y0, Y4)
		VPUNPCKHQDQ(Y1, Y0, Y5)
		VPUNPCKLQDQ(Y3, Y2, Y6)
		VPUNPCKHQDQ(Y3, Y2, Y7)
		VPERM2I128(Imm(0x20), Y6, Y4, Y8)
		VPERM2I128(Imm(0x20), Y7, Y5, Y9)
		VPERM2I128(Imm(0x31), Y6, Y4, Y10)
		VPERM2I128(Imm(0x31), Y7, Y5, Y11)
		VMOVDQA(Y8, lane(0, 4*q+0))
		VMOVDQA(Y9, lane(0, 4*q+1))
		VMOVDQA(Y10, lane(0, 4*q+2))
		VMOVDQA(Y11, lane(0, 4*q+3))
	}
	VMOVQ(Mem{Base: p[0], Disp: 192}, X0)
	VPINSRQ(Imm(1), Mem{Base: p[1], Disp: 192}, X0, X0)
	VMOVQ(Mem{Base: p[2], Disp: 192}, X1)
	VPINSRQ(Imm(1), Mem{Base: p[3], Disp: 192}, X1, X1)
	VINSERTI128(Imm(1), X1, Y0, Y0)
	VMOVDQA(Y0, lane(0, 24))
}

// x4TransposeOut writes the interleaved states in buffer 0 back out to the
// four sequential states.
func x4TransposeOut(p [4]GPPhysical) {
	Comment("Deinterleave the stack buffer back into the four states")
	for q := range 6 {
		VMOVDQA(lane(0, 4*q+0), Y0)
		VMOVDQA(lane(0, 4*q+1), Y1)
		VMOVDQA(lane(0, 4*q+2), Y2)
		VMOVDQA(lane(0, 4*q+3), Y3)
		VPUNPCKLQDQ(Y1, Y0, Y4)
		VPUNPCKHQDQ(Y1, Y0, Y5)
		VPUNPCKLQDQ(Y3, Y2, Y6)
		VPUNPCKHQDQ(Y3, Y2, Y7)
		VPERM2I128(Imm(0x20), Y6, Y4, Y8)
		VPERM2I128(Imm(0x20), Y7, Y5, Y9)
		VPERM2I128(Imm(0x31), Y6, Y4, Y10)
		VPERM2I128(Imm(0x31), Y7, Y5, Y11)
		for i, pi := range p {
			VMOVDQU([4]VecPhysical{Y8, Y9, Y10, Y11}[i], Mem{Base: pi, Disp: q * 32})
		}
	}
	VMOVDQA(lane(0, 24), Y0)
	VMOVQ(X0, Mem{Base: p[0], Disp: 192})
	VPEXTRQ(Imm(1), X0, Mem{Base: p[1], Disp: 192})
	VEXTRACTI128(Imm(1), Y0, X1)
	VMOVQ(X1, Mem{Base: p[2], Disp: 192})
	VPEXTRQ(Imm(1), X1, Mem{Base: p[3], Disp: 192})
}

var rho8Mem, rho56Mem, rcMem *Mem

// rho8_DATA returns the VPSHUFB control for a 64-bit rotation left by 8.
func rho8_DATA() Mem {
	if rho8Mem != nil {
		return *rho8Mem
	}
	rho8 := GLOBL("rho8", RODATA|NOPTR)
	rho8Mem = &rho8
	DATA(0x00, U64(0x0605040302010007))
	DATA(0x08, U64(0x0e0d0c0b0a09080f))
	DATA(0x10, U64(0x0605040302010007))
	DATA(0x18, U64(0x0e0d0c0b0a09080f))
	return rho8
}

// rho56_DATA returns the VPSHUFB control for a 64-bit rotation left by 56.
func rho56_DATA() Mem {
	if rho56Mem != nil {
		return *rho56Mem
	}
	rho56 := GLOBL("rho56", RODATA|NOPTR)
	rho56Mem = &rho56
	DATA(0x00, U64(0x0007060504030201))
	DATA(0x08, U64(0x080f0e0d0c0b0a09))
	DATA(0x10, U64(0x0007060504030201))
	DATA(0x18, U64(0x080f0e0d0c0b0a09))
	return rho56
}

// rc_DATA returns the table of round constants for the ι step.
func rc_DATA() Mem {
	if rcMem != nil {
		return *rcMem
	}
	rc := GLOBL("rc", RODATA|NOPTR)
	rcMem = &rc
	for i, c := range RoundConstants {
		DATA(8*i, U64(c))
	}
	return rc
}
