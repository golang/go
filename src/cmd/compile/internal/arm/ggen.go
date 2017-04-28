// Copyright 2009 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package arm

import (
	"cmd/compile/internal/gc"
	"cmd/internal/obj"
	"cmd/internal/obj/arm"
)

func zerorange(pp *gc.Progs, p *obj.Prog, off, cnt int64, r0 *uint32) *obj.Prog {
	if cnt == 0 {
		return p
	}
	if *r0 == 0 {
		p = pp.Appendpp(p, arm.AMOVW, obj.TYPE_CONST, 0, 0, obj.TYPE_REG, arm.REG_R0, 0)
		*r0 = 1
	}

	if cnt < int64(4*gc.Widthptr) {
		for i := int64(0); i < cnt; i += int64(gc.Widthptr) {
			p = pp.Appendpp(p, arm.AMOVW, obj.TYPE_REG, arm.REG_R0, 0, obj.TYPE_MEM, arm.REGSP, 4+off+i)
		}
	} else if !gc.Nacl && (cnt <= int64(128*gc.Widthptr)) {
		p = pp.Appendpp(p, arm.AADD, obj.TYPE_CONST, 0, 4+off, obj.TYPE_REG, arm.REG_R1, 0)
		p.Reg = arm.REGSP
		p = pp.Appendpp(p, obj.ADUFFZERO, obj.TYPE_NONE, 0, 0, obj.TYPE_MEM, 0, 0)
		p.To.Name = obj.NAME_EXTERN
		p.To.Sym = gc.Duffzero
		p.To.Offset = 4 * (128 - cnt/int64(gc.Widthptr))
	} else {
		p = pp.Appendpp(p, arm.AADD, obj.TYPE_CONST, 0, 4+off, obj.TYPE_REG, arm.REG_R1, 0)
		p.Reg = arm.REGSP
		p = pp.Appendpp(p, arm.AADD, obj.TYPE_CONST, 0, cnt, obj.TYPE_REG, arm.REG_R2, 0)
		p.Reg = arm.REG_R1
		p = pp.Appendpp(p, arm.AMOVW, obj.TYPE_REG, arm.REG_R0, 0, obj.TYPE_MEM, arm.REG_R1, 4)
		p1 := p
		p.Scond |= arm.C_PBIT
		p = pp.Appendpp(p, arm.ACMP, obj.TYPE_REG, arm.REG_R1, 0, obj.TYPE_NONE, 0, 0)
		p.Reg = arm.REG_R2
		p = pp.Appendpp(p, arm.ABNE, obj.TYPE_NONE, 0, 0, obj.TYPE_BRANCH, 0, 0)
		gc.Patch(p, p1)
	}

	return p
}

func zeroAuto(pp *gc.Progs, n *gc.Node) {
	// Note: this code must not clobber any registers.
	sym := n.Sym.Linksym()
	size := n.Type.Size()
	p := pp.Prog(arm.AMOVW)
	p.From.Type = obj.TYPE_CONST
	p.From.Offset = 0
	p.To.Type = obj.TYPE_REG
	p.To.Reg = arm.REGTMP
	for i := int64(0); i < size; i += 4 {
		p := pp.Prog(arm.AMOVW)
		p.From.Type = obj.TYPE_REG
		p.From.Reg = arm.REGTMP
		p.To.Type = obj.TYPE_MEM
		p.To.Name = obj.NAME_AUTO
		p.To.Reg = arm.REGSP
		p.To.Offset = n.Xoffset + i
		p.To.Sym = sym
	}
}

func ginsnop(pp *gc.Progs) {
	p := pp.Prog(arm.AAND)
	p.From.Type = obj.TYPE_REG
	p.From.Reg = arm.REG_R0
	p.To.Type = obj.TYPE_REG
	p.To.Reg = arm.REG_R0
	p.Scond = arm.C_SCOND_EQ
}
