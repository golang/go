// Copyright 2009 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package amd64

import (
	"cmd/compile/internal/ir"
	"cmd/compile/internal/objw"
	"cmd/compile/internal/types"
	"cmd/internal/obj"
	"cmd/internal/obj/x86"
	"internal/buildcfg"
)

// no floating point in note handlers on Plan 9
var isPlan9 = buildcfg.GOOS == "plan9"

// DUFFZERO consists of repeated blocks of 4 MOVUPSs + LEAQ,
// See runtime/mkduff.go.
const (
	dzBlocks    = 16 // number of MOV/ADD blocks
	dzBlockLen  = 4  // number of clears per block
	dzBlockSize = 23 // size of instructions in a single block
	dzMovSize   = 5  // size of single MOV instruction w/ offset
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

func zerorange(pp *objw.Progs, p *obj.Prog, off, cnt int64, state *uint32) *obj.Prog {
	const (
		r13 = 1 << iota // if R13 is already zeroed.
	)

	if cnt == 0 {
		return p
	}

	if cnt == 8 {
		p = pp.Append(p, x86.AMOVQ, obj.TYPE_REG, x86.REG_X15, 0, obj.TYPE_MEM, x86.REG_SP, off)
	} else if !isPlan9 && cnt <= int64(8*types.RegSize) {
		for i := int64(0); i < cnt/16; i++ {
			p = pp.Append(p, x86.AMOVUPS, obj.TYPE_REG, x86.REG_X15, 0, obj.TYPE_MEM, x86.REG_SP, off+i*16)
		}

		if cnt%16 != 0 {
			p = pp.Append(p, x86.AMOVUPS, obj.TYPE_REG, x86.REG_X15, 0, obj.TYPE_MEM, x86.REG_SP, off+cnt-int64(16))
		}
	} else if !isPlan9 && (cnt <= int64(128*types.RegSize)) {
		// Save DI to r12. With the amd64 Go register abi, DI can contain
		// an incoming parameter, whereas R12 is always scratch.
		p = pp.Append(p, x86.AMOVQ, obj.TYPE_REG, x86.REG_DI, 0, obj.TYPE_REG, x86.REG_R12, 0)
		// Emit duffzero call
		p = pp.Append(p, leaptr, obj.TYPE_MEM, x86.REG_SP, off+dzDI(cnt), obj.TYPE_REG, x86.REG_DI, 0)
		p = pp.Append(p, obj.ADUFFZERO, obj.TYPE_NONE, 0, 0, obj.TYPE_ADDR, 0, dzOff(cnt))
		p.To.Sym = ir.Syms.Duffzero
		if cnt%16 != 0 {
			p = pp.Append(p, x86.AMOVUPS, obj.TYPE_REG, x86.REG_X15, 0, obj.TYPE_MEM, x86.REG_DI, -int64(8))
		}
		// Restore DI from r12
		p = pp.Append(p, x86.AMOVQ, obj.TYPE_REG, x86.REG_R12, 0, obj.TYPE_REG, x86.REG_DI, 0)

	} else {
		// When the register ABI is in effect, at this point in the
		// prolog we may have live values in all of RAX,RDI,RCX. Save
		// them off to registers before the REPSTOSQ below, then
		// restore. Note that R12 and R13 are always available as
		// scratch regs; here we also use R15 (this is safe to do
		// since there won't be any globals accessed in the prolog).
		// See rewriteToUseGot() in obj6.go for more on r15 use.

		// Save rax/rdi/rcx
		p = pp.Append(p, x86.AMOVQ, obj.TYPE_REG, x86.REG_DI, 0, obj.TYPE_REG, x86.REG_R12, 0)
		p = pp.Append(p, x86.AMOVQ, obj.TYPE_REG, x86.REG_AX, 0, obj.TYPE_REG, x86.REG_R13, 0)
		p = pp.Append(p, x86.AMOVQ, obj.TYPE_REG, x86.REG_CX, 0, obj.TYPE_REG, x86.REG_R15, 0)

		// Set up the REPSTOSQ and kick it off.
		p = pp.Append(p, x86.AXORL, obj.TYPE_REG, x86.REG_AX, 0, obj.TYPE_REG, x86.REG_AX, 0)
		p = pp.Append(p, x86.AMOVQ, obj.TYPE_CONST, 0, cnt/int64(types.RegSize), obj.TYPE_REG, x86.REG_CX, 0)
		p = pp.Append(p, leaptr, obj.TYPE_MEM, x86.REG_SP, off, obj.TYPE_REG, x86.REG_DI, 0)
		p = pp.Append(p, x86.AREP, obj.TYPE_NONE, 0, 0, obj.TYPE_NONE, 0, 0)
		p = pp.Append(p, x86.ASTOSQ, obj.TYPE_NONE, 0, 0, obj.TYPE_NONE, 0, 0)

		// Restore rax/rdi/rcx
		p = pp.Append(p, x86.AMOVQ, obj.TYPE_REG, x86.REG_R12, 0, obj.TYPE_REG, x86.REG_DI, 0)
		p = pp.Append(p, x86.AMOVQ, obj.TYPE_REG, x86.REG_R13, 0, obj.TYPE_REG, x86.REG_AX, 0)
		p = pp.Append(p, x86.AMOVQ, obj.TYPE_REG, x86.REG_R15, 0, obj.TYPE_REG, x86.REG_CX, 0)

		// Record the fact that r13 is no longer zero.
		*state &= ^uint32(r13)
	}

	return p
}

func ginsnop(pp *objw.Progs) *obj.Prog {
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
