// Copyright 2026 The Go Authors. All rights reserved.
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

// ctrBlocksBulk writes
//
//	func ctrBlocksBulkAsm(nr int, xk *[60]uint32, dst, src *byte, n int, ivlo, ivhi uint64, vaes bool)
//
// which XORs n >= 1 blocks of keystream into dst, starting at counter
// ivhi*2^64 + ivlo. The low 64 bits of the counter must not wrap within the n
// blocks, and dst and src must be the same buffer or not overlap.
//
// It has two paths. With vaes, it uses VAES and AVX2: a YMM register holds
// two blocks and the loop encrypts 16 blocks per iteration. Otherwise it uses
// AES-NI, eight blocks per iteration. Both keep the counter in vector
// registers as [lo, hi], little-endian, in each 128-bit lane (adding only to
// lo, which is why the low 64 bits must not wrap), and byte-reverse each lane
// to get the AES input. Each round key is loaded once per round for all the
// blocks in flight, and the source is XORed into the last round key, so that
// AESENCLAST outputs the result. The blocks left after the loop go in groups
// of 4, 2 and 1 registers and then, with VAES, one block.
//
// The three key sizes share the code of the rounds: AES-256 starts two
// rounds before AES-192, which starts two rounds before AES-128. As in the
// fixed-size functions, the key pointer advances along the branch taken,
// so the shared rounds use the same offsets from the key pointer.
func ctrBlocksBulk() {
	Implement("ctrBlocksBulkAsm")
	g := &bulkGen{
		nr: RAX, xk: RBX, kp: RCX, dst: RDX, src: RSI, n: RDI,
		ivlo: R8, ivhi: R9,
	}
	Load(Param("nr"), g.nr)
	Load(Param("xk"), g.xk)
	Load(Param("dst"), g.dst)
	Load(Param("src"), g.src)
	Load(Param("n"), g.n)
	Load(Param("ivlo"), g.ivlo)
	Load(Param("ivhi"), g.ivhi)
	vaes := R10B
	Load(Param("vaes"), vaes)

	TESTB(vaes, vaes)
	JNZ(LabelRef("vaes"))
	g.path("aesni", 1)
	Label("vaes")
	g.path("vaes", 2)
}

// The loop encrypts this many registers per iteration.
const bulkRegs = 8

// Vector registers: the blocks in flight are in 0 to 7.
const (
	regKey  = 8  // round key
	regCtr  = 9  // counter: [lo, hi] little-endian in every 128-bit lane
	regMask = 10 // byte reversal of each 128-bit lane
	regInc  = 11 // [blocks per iteration, 0]: the counter step of one iteration
	regTmp  = 12 // source XOR last round key
	regStep = 13 // VAES: counter step of a group of the tail
)

var (
	xRegs = []VecPhysical{X0, X1, X2, X3, X4, X5, X6, X7, X8, X9, X10, X11, X12, X13, X14, X15}
	yRegs = []VecPhysical{Y0, Y1, Y2, Y3, Y4, Y5, Y6, Y7, Y8, Y9, Y10, Y11, Y12, Y13, Y14, Y15}
)

// ctrOffsets holds the 128-bit little-endian values 0, 1, ..., 16. The
// AES-NI path adds its entries with PADDQ from memory, which needs 16-byte
// alignment: the linker aligns symbols of 32 bytes or more to 32 bytes.
var ctrOffsets = sync.OnceValue(func() Mem {
	m := GLOBL("ctrOffsets", NOPTR|RODATA)
	for k := 0; k <= 16; k++ {
		DATA(16*k, U64(uint64(k)))
		DATA(16*k+8, U64(0))
	}
	return m
})

type bulkGen struct {
	nr, xk, kp, dst, src, n, ivlo, ivhi GPPhysical

	name   string // label prefix of the path
	width  int    // blocks per register: 1 (XMM, AES-NI) or 2 (YMM, VAES)
	labels int
}

func (g *bulkGen) label(s string) string {
	g.labels++
	return fmt.Sprintf("%s_%s%d", g.name, s, g.labels)
}

// vec returns vector register i holding w blocks.
func vec(w, i int) VecPhysical {
	if w == 1 {
		return xRegs[i]
	}
	return yRegs[i]
}

// path writes one path of ctrBlocksBulkAsm, ending with RET.
func (g *bulkGen) path(name string, w int) {
	g.name, g.width = name, w
	per := bulkRegs * w // blocks per iteration

	c := vec(w, regCtr)
	if w == 1 {
		MOVQ(g.ivlo, c)
		PINSRQ(Imm(1), g.ivhi, c)
		MOVOU(bswapMask(), vec(1, regMask))
	} else {
		xc := vec(1, regCtr)
		VMOVQ(g.ivlo, xc)
		VPINSRQ(Imm(1), g.ivhi, xc, xc)
		VINSERTI128(Imm(1), xc, c, c)
		VBROADCASTI128(bswapMask(), vec(2, regMask))
	}

	loop, tail := g.label("loop"), g.label("tail")
	CMPQ(g.n, U32(per))
	JB(LabelRef(tail))
	if w == 1 {
		MOVOU(ctrOffsets().Offset(16*per), vec(1, regInc))
	} else {
		VBROADCASTI128(ctrOffsets().Offset(16*per), vec(2, regInc))
	}
	Label(loop)
	g.group(bulkRegs, w)
	g.next(bulkRegs, w)
	g.addCtr(vec(w, regInc))
	SUBQ(U32(per), g.n)
	CMPQ(g.n, U32(per))
	JAE(LabelRef(loop))

	// n < per blocks are left: groups of 4, 2 and 1 registers, then, with
	// VAES, one block in an XMM register.
	Label(tail)
	for k := bulkRegs / 2; k >= 1; k /= 2 {
		g.tailGroup(k, w, w == 1 && k == 1)
	}
	if w == 2 {
		g.tailGroup(1, 1, true)
		VZEROUPPER()
	}
	RET()
}

// tailGroup encrypts k registers of ww blocks if bit k*ww of n is set. Unless
// it is the last group, it then advances src, dst and the counter.
func (g *bulkGen) tailGroup(k, ww int, last bool) {
	skip := g.label("skip")
	TESTQ(U32(k*ww), g.n)
	JZ(LabelRef(skip))
	g.group(k, ww)
	if !last {
		g.next(k, ww)
		step := ctrOffsets().Offset(16 * k * ww)
		if g.width == 1 {
			PADDQ(step, vec(1, regCtr))
		} else {
			VBROADCASTI128(step, vec(2, regStep))
			g.addCtr(vec(2, regStep))
		}
	}
	Label(skip)
}

// next advances src and dst past k registers of ww blocks.
func (g *bulkGen) next(k, ww int) {
	ADDQ(U32(16*ww*k), g.src)
	ADDQ(U32(16*ww*k), g.dst)
}

// addCtr adds step to the counter in every lane.
func (g *bulkGen) addCtr(step VecPhysical) {
	if g.width == 1 {
		PADDQ(step, vec(1, regCtr))
	} else {
		VPADDQ(step, vec(2, regCtr), vec(2, regCtr))
	}
}

// loadKey loads the round key at m into the key register, in both lanes
// for ww == 2.
func (g *bulkGen) loadKey(ww int, m Mem) {
	switch {
	case g.width == 1:
		MOVOU(m, vec(1, regKey))
	case ww == 1:
		VMOVDQU(m, vec(1, regKey))
	default:
		VBROADCASTI128(m, vec(2, regKey))
	}
}

// group encrypts k registers of ww blocks: the counters, the rounds and the
// XOR into dst.
func (g *bulkGen) group(k, ww int) {
	st := func(j int) VecPhysical { return vec(ww, j) }
	sse := g.width == 1

	// Counters: lane l of register j is the counter plus j*ww + l,
	// byte-reversed.
	if sse {
		for j := 0; j < k; j++ {
			MOVO(vec(1, regCtr), st(j))
			if j > 0 {
				PADDQ(ctrOffsets().Offset(16*j), st(j))
			}
			PSHUFB(vec(1, regMask), st(j))
		}
	} else {
		for j := 0; j < k; j++ {
			VPADDQ(ctrOffsets().Offset(16*j*ww), vec(ww, regCtr), st(j))
			VPSHUFB(vec(ww, regMask), st(j), st(j))
		}
	}

	// Round 0: XOR with key 0.
	g.loadKey(ww, Mem{Base: g.xk})
	key := vec(ww, regKey)
	for j := 0; j < k; j++ {
		if sse {
			PXOR(key, st(j))
		} else {
			VPXOR(key, st(j), st(j))
		}
	}

	// Rounds 1 to nr-1. kp starts at xk, and moves past the two extra
	// rounds of AES-256 and of AES-192, so that the last ten round keys are
	// at kp+16 to kp+160 for every key size.
	r192, r128 := g.label("r192"), g.label("r128")
	MOVQ(g.xk, g.kp)
	CMPQ(g.nr, Imm(12))
	JB(LabelRef(r128))
	JE(LabelRef(r192))
	g.round(k, ww, 16)
	g.round(k, ww, 32)
	ADDQ(Imm(32), g.kp)
	Label(r192)
	g.round(k, ww, 16)
	g.round(k, ww, 32)
	ADDQ(Imm(32), g.kp)
	Label(r128)
	for i := 1; i <= 9; i++ {
		g.round(k, ww, 16*i)
	}

	// The last round, with the source XORed into the round key.
	g.loadKey(ww, Mem{Base: g.kp, Disp: 160})
	t := vec(ww, regTmp)
	for j := 0; j < k; j++ {
		s := Mem{Base: g.src, Disp: 16 * ww * j}
		d := Mem{Base: g.dst, Disp: 16 * ww * j}
		if sse {
			MOVOU(s, t)
			PXOR(key, t)
			AESENCLAST(t, st(j))
			MOVOU(st(j), d)
		} else {
			VPXOR(s, key, t)
			VAESENCLAST(t, st(j), st(j))
			VMOVDQU(st(j), d)
		}
	}
}

// round runs one AES round with the key at kp+disp on k registers of ww
// blocks.
func (g *bulkGen) round(k, ww, disp int) {
	g.loadKey(ww, Mem{Base: g.kp, Disp: disp})
	key := vec(ww, regKey)
	for j := 0; j < k; j++ {
		if g.width == 1 {
			AESENC(key, vec(1, j))
		} else {
			VAESENC(key, vec(ww, j), vec(ww, j))
		}
	}
}
