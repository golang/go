// Copyright 2009 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package mips64

import (
	"cmd/compile/internal/gc"
	"cmd/internal/obj"
	"cmd/internal/obj/mips"
)

func zerorange(pp *gc.Progs, p *obj.Prog, off, cnt int64, _ *uint32) *obj.Prog {
	if cnt == 0 {
		return p
	}
	if cnt < int64(4*gc.Widthptr) {
		for i := int64(0); i < cnt; i += int64(gc.Widthptr) {
			p = pp.Appendpp(p, mips.AMOVV, obj.TYPE_REG, mips.REGZERO, 0, obj.TYPE_MEM, mips.REGSP, 8+off+i)
		}
	} else if cnt <= int64(128*gc.Widthptr) {
		p = pp.Appendpp(p, mips.AADDV, obj.TYPE_CONST, 0, 8+off-8, obj.TYPE_REG, mips.REGRT1, 0)
		p.Reg = mips.REGSP
		p = pp.Appendpp(p, obj.ADUFFZERO, obj.TYPE_NONE, 0, 0, obj.TYPE_MEM, 0, 0)
		p.To.Name = obj.NAME_EXTERN
		p.To.Sym = gc.Duffzero
		p.To.Offset = 8 * (128 - cnt/int64(gc.Widthptr))
	} else {
		//	ADDV	$(8+frame+lo-8), SP, r1
		//	ADDV	$cnt, r1, r2
		// loop:
		//	MOVV	R0, (Widthptr)r1
		//	ADDV	$Widthptr, r1
		//	BNE		r1, r2, loop
		p = pp.Appendpp(p, mips.AADDV, obj.TYPE_CONST, 0, 8+off-8, obj.TYPE_REG, mips.REGRT1, 0)
		p.Reg = mips.REGSP
		p = pp.Appendpp(p, mips.AADDV, obj.TYPE_CONST, 0, cnt, obj.TYPE_REG, mips.REGRT2, 0)
		p.Reg = mips.REGRT1
		p = pp.Appendpp(p, mips.AMOVV, obj.TYPE_REG, mips.REGZERO, 0, obj.TYPE_MEM, mips.REGRT1, int64(gc.Widthptr))
		p1 := p
		p = pp.Appendpp(p, mips.AADDV, obj.TYPE_CONST, 0, int64(gc.Widthptr), obj.TYPE_REG, mips.REGRT1, 0)
		p = pp.Appendpp(p, mips.ABNE, obj.TYPE_REG, mips.REGRT1, 0, obj.TYPE_BRANCH, 0, 0)
		p.Reg = mips.REGRT2
		gc.Patch(p, p1)
	}

	return p
}

func zeroAuto(pp *gc.Progs, n *gc.Node) {
	// Note: this code must not clobber any registers.
	sym := n.Sym.Linksym()
	size := n.Type.Size()
	for i := int64(0); i < size; i += 8 {
		p := pp.Prog(mips.AMOVV)
		p.From.Type = obj.TYPE_REG
		p.From.Reg = mips.REGZERO
		p.To.Type = obj.TYPE_MEM
		p.To.Name = obj.NAME_AUTO
		p.To.Reg = mips.REGSP
		p.To.Offset = n.Xoffset + i
		p.To.Sym = sym
	}
}

func ginsnop(pp *gc.Progs) {
	p := pp.Prog(mips.ANOR)
	p.From.Type = obj.TYPE_REG
	p.From.Reg = mips.REG_R0
	p.To.Type = obj.TYPE_REG
	p.To.Reg = mips.REG_R0
}
