// Copyright 2009 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package ppc64

import (
	"cmd/compile/internal/base"
	"cmd/compile/internal/ir"
	"cmd/compile/internal/objw"
	"cmd/compile/internal/types"
	"cmd/internal/obj"
	"cmd/internal/obj/ppc64"
)

func zerorange(pp *objw.Progs, p *obj.Prog, off, cnt int64, _ *uint32) *obj.Prog {
	if cnt == 0 {
		return p
	}
	if cnt < int64(4*types.PtrSize) {
		for i := int64(0); i < cnt; i += int64(types.PtrSize) {
			p = pp.Append(p, ppc64.AMOVD, obj.TYPE_REG, ppc64.REGZERO, 0, obj.TYPE_MEM, ppc64.REGSP, base.Ctxt.FixedFrameSize()+off+i)
		}
	} else if cnt <= int64(128*types.PtrSize) {
		p = pp.Append(p, ppc64.AADD, obj.TYPE_CONST, 0, base.Ctxt.FixedFrameSize()+off-8, obj.TYPE_REG, ppc64.REGRT1, 0)
		p.Reg = ppc64.REGSP
		p = pp.Append(p, obj.ADUFFZERO, obj.TYPE_NONE, 0, 0, obj.TYPE_MEM, 0, 0)
		p.To.Name = obj.NAME_EXTERN
		p.To.Sym = ir.Syms.Duffzero
		p.To.Offset = 4 * (128 - cnt/int64(types.PtrSize))
	} else {
		p = pp.Append(p, ppc64.AMOVD, obj.TYPE_CONST, 0, base.Ctxt.FixedFrameSize()+off-8, obj.TYPE_REG, ppc64.REGTMP, 0)
		p = pp.Append(p, ppc64.AADD, obj.TYPE_REG, ppc64.REGTMP, 0, obj.TYPE_REG, ppc64.REGRT1, 0)
		p.Reg = ppc64.REGSP
		p = pp.Append(p, ppc64.AMOVD, obj.TYPE_CONST, 0, cnt, obj.TYPE_REG, ppc64.REGTMP, 0)
		p = pp.Append(p, ppc64.AADD, obj.TYPE_REG, ppc64.REGTMP, 0, obj.TYPE_REG, ppc64.REGRT2, 0)
		p.Reg = ppc64.REGRT1
		p = pp.Append(p, ppc64.AMOVDU, obj.TYPE_REG, ppc64.REGZERO, 0, obj.TYPE_MEM, ppc64.REGRT1, int64(types.PtrSize))
		p1 := p
		p = pp.Append(p, ppc64.ACMP, obj.TYPE_REG, ppc64.REGRT1, 0, obj.TYPE_REG, ppc64.REGRT2, 0)
		p = pp.Append(p, ppc64.ABNE, obj.TYPE_NONE, 0, 0, obj.TYPE_BRANCH, 0, 0)
		p.To.SetTarget(p1)
	}

	return p
}

func ginsnop(pp *objw.Progs) *obj.Prog {
	p := pp.Prog(ppc64.AOR)
	p.From.Type = obj.TYPE_REG
	p.From.Reg = ppc64.REG_R0
	p.To.Type = obj.TYPE_REG
	p.To.Reg = ppc64.REG_R0
	return p
}

func ginsnopdefer(pp *objw.Progs) *obj.Prog {
	// On PPC64 two nops are required in the defer case.
	//
	// (see gc/cgen.go, gc/plive.go -- copy of comment below)
	//
	// On ppc64, when compiling Go into position
	// independent code on ppc64le we insert an
	// instruction to reload the TOC pointer from the
	// stack as well. See the long comment near
	// jmpdefer in runtime/asm_ppc64.s for why.
	// If the MOVD is not needed, insert a hardware NOP
	// so that the same number of instructions are used
	// on ppc64 in both shared and non-shared modes.

	ginsnop(pp)
	if base.Ctxt.Flag_shared {
		p := pp.Prog(ppc64.AMOVD)
		p.From.Type = obj.TYPE_MEM
		p.From.Offset = 24
		p.From.Reg = ppc64.REGSP
		p.To.Type = obj.TYPE_REG
		p.To.Reg = ppc64.REG_R2
		return p
	}
	return ginsnop(pp)
}
