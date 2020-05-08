// Copyright 2009 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package amd64

import (
	"cmd/compile/internal/gc"
	"cmd/internal/obj"
	"cmd/internal/obj/x86"
	"cmd/internal/objabi"
)

// no floating point in note handlers on Plan 9
var isPlan9 = objabi.GOOS == "plan9"

// DUFFZERO consists of repeated blocks of 4 MOVUPSs + LEAQ,
// See runtime/mkduff.go.
const (
	dzBlocks    = 16 // number of MOV/ADD blocks
	dzBlockLen  = 4  // number of clears per block
	dzBlockSize = 19 // size of instructions in a single block
	dzMovSize   = 4  // size of single MOV instruction w/ offset
	dzLeaqSize  = 4  // size of single LEAQ instruction
	dzClearStep = 16 // number of bytes cleared by each MOV instruction

	dzClearLen = dzClearStep * dzBlockLen // bytes cleared by one block
	dzSize     = dzBlocks * dzBlockSize
)

// dzOff returns the offset for a jump into DUFFZERO.
// b is the number of bytes to zero.
func dzOff(b int64) int64 {
	off := int64(dzSize)
	off -= b / dzClearLen * dzBlockSize
	tailLen := b % dzClearLen
	if tailLen >= dzClearStep {
		off -= dzLeaqSize + dzMovSize*(tailLen/dzClearStep)
	}
	return off
}

// duffzeroDI returns the pre-adjustment to DI for a call to DUFFZERO.
// b is the number of bytes to zero.
func dzDI(b int64) int64 {
	tailLen := b % dzClearLen
	if tailLen < dzClearStep {
		return 0
	}
	tailSteps := tailLen / dzClearStep
	return -dzClearStep * (dzBlockLen - tailSteps)
}

func zerorange(pp *gc.Progs, p *obj.Prog, off, cnt int64, state *uint32) *obj.Prog {
	const (
		ax = 1 << iota
		x0
	)

	if cnt == 0 {
		return p
	}

	if cnt%int64(gc.Widthreg) != 0 {
		// should only happen with nacl
		if cnt%int64(gc.Widthptr) != 0 {
			gc.Fatalf("zerorange count not a multiple of widthptr %d", cnt)
		}
		if *state&ax == 0 {
			p = pp.Appendpp(p, x86.AMOVQ, obj.TYPE_CONST, 0, 0, obj.TYPE_REG, x86.REG_AX, 0)
			*state |= ax
		}
		p = pp.Appendpp(p, x86.AMOVL, obj.TYPE_REG, x86.REG_AX, 0, obj.TYPE_MEM, x86.REG_SP, off)
		off += int64(gc.Widthptr)
		cnt -= int64(gc.Widthptr)
	}

	if cnt == 8 {
		if *state&ax == 0 {
			p = pp.Appendpp(p, x86.AMOVQ, obj.TYPE_CONST, 0, 0, obj.TYPE_REG, x86.REG_AX, 0)
			*state |= ax
		}
		p = pp.Appendpp(p, x86.AMOVQ, obj.TYPE_REG, x86.REG_AX, 0, obj.TYPE_MEM, x86.REG_SP, off)
	} else if !isPlan9 && cnt <= int64(8*gc.Widthreg) {
		if *state&x0 == 0 {
			p = pp.Appendpp(p, x86.AXORPS, obj.TYPE_REG, x86.REG_X0, 0, obj.TYPE_REG, x86.REG_X0, 0)
			*state |= x0
		}

		for i := int64(0); i < cnt/16; i++ {
			p = pp.Appendpp(p, x86.AMOVUPS, obj.TYPE_REG, x86.REG_X0, 0, obj.TYPE_MEM, x86.REG_SP, off+i*16)
		}

		if cnt%16 != 0 {
			p = pp.Appendpp(p, x86.AMOVUPS, obj.TYPE_REG, x86.REG_X0, 0, obj.TYPE_MEM, x86.REG_SP, off+cnt-int64(16))
		}
	} else if !isPlan9 && (cnt <= int64(128*gc.Widthreg)) {
		if *state&x0 == 0 {
			p = pp.Appendpp(p, x86.AXORPS, obj.TYPE_REG, x86.REG_X0, 0, obj.TYPE_REG, x86.REG_X0, 0)
			*state |= x0
		}
		p = pp.Appendpp(p, leaptr, obj.TYPE_MEM, x86.REG_SP, off+dzDI(cnt), obj.TYPE_REG, x86.REG_DI, 0)
		p = pp.Appendpp(p, obj.ADUFFZERO, obj.TYPE_NONE, 0, 0, obj.TYPE_ADDR, 0, dzOff(cnt))
		p.To.Sym = gc.Duffzero

		if cnt%16 != 0 {
			p = pp.Appendpp(p, x86.AMOVUPS, obj.TYPE_REG, x86.REG_X0, 0, obj.TYPE_MEM, x86.REG_DI, -int64(8))
		}
	} else {
		if *state&ax == 0 {
			p = pp.Appendpp(p, x86.AMOVQ, obj.TYPE_CONST, 0, 0, obj.TYPE_REG, x86.REG_AX, 0)
			*state |= ax
		}

		p = pp.Appendpp(p, x86.AMOVQ, obj.TYPE_CONST, 0, cnt/int64(gc.Widthreg), obj.TYPE_REG, x86.REG_CX, 0)
		p = pp.Appendpp(p, leaptr, obj.TYPE_MEM, x86.REG_SP, off, obj.TYPE_REG, x86.REG_DI, 0)
		p = pp.Appendpp(p, x86.AREP, obj.TYPE_NONE, 0, 0, obj.TYPE_NONE, 0, 0)
		p = pp.Appendpp(p, x86.ASTOSQ, obj.TYPE_NONE, 0, 0, obj.TYPE_NONE, 0, 0)
	}

	return p
}

func ginsnop(pp *gc.Progs) *obj.Prog {
	// This is a hardware nop (1-byte 0x90) instruction,
	// even though we describe it as an explicit XCHGL here.
	// Particularly, this does not zero the high 32 bits
	// like typical *L opcodes.
	// (gas assembles "xchg %eax,%eax" to 0x87 0xc0, which
	// does zero the high 32 bits.)
	p := pp.Prog(x86.AXCHGL)
	p.From.Type = obj.TYPE_REG
	p.From.Reg = x86.REG_AX
	p.To.Type = obj.TYPE_REG
	p.To.Reg = x86.REG_AX
	return p
}
