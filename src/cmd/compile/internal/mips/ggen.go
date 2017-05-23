// Copyright 2016 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package mips

import (
	"cmd/compile/internal/gc"
	"cmd/internal/obj"
	"cmd/internal/obj/mips"
)

func defframe(ptxt *obj.Prog) {
	// fill in argument size, stack size
	ptxt.To.Type = obj.TYPE_TEXTSIZE

	ptxt.To.Val = int32(gc.Rnd(gc.Curfn.Type.ArgWidth(), int64(gc.Widthptr)))
	frame := uint32(gc.Rnd(gc.Stksize+gc.Maxarg, int64(gc.Widthreg)))
	ptxt.To.Offset = int64(frame)

	// insert code to zero ambiguously live variables
	// so that the garbage collector only sees initialized values
	// when it looks for pointers.
	p := ptxt

	hi := int64(0)
	lo := hi

	// iterate through declarations - they are sorted in decreasing xoffset order.
	for _, n := range gc.Curfn.Func.Dcl {
		if !n.Name.Needzero {
			continue
		}
		if n.Class != gc.PAUTO {
			gc.Fatalf("needzero class %d", n.Class)
		}
		if n.Type.Width%int64(gc.Widthptr) != 0 || n.Xoffset%int64(gc.Widthptr) != 0 || n.Type.Width == 0 {
			gc.Fatalf("var %L has size %d offset %d", n, int(n.Type.Width), int(n.Xoffset))
		}

		if lo != hi && n.Xoffset+n.Type.Width >= lo-int64(2*gc.Widthreg) {
			// merge with range we already have
			lo = n.Xoffset

			continue
		}

		// zero old range
		p = zerorange(p, int64(frame), lo, hi)

		// set new range
		hi = n.Xoffset + n.Type.Width

		lo = n.Xoffset
	}

	// zero final range
	zerorange(p, int64(frame), lo, hi)
}

// TODO(mips): implement DUFFZERO
func zerorange(p *obj.Prog, frame int64, lo int64, hi int64) *obj.Prog {

	cnt := hi - lo
	if cnt == 0 {
		return p
	}
	if cnt < int64(4*gc.Widthptr) {
		for i := int64(0); i < cnt; i += int64(gc.Widthptr) {
			p = gc.Appendpp(p, mips.AMOVW, obj.TYPE_REG, mips.REGZERO, 0, obj.TYPE_MEM, mips.REGSP, gc.Ctxt.FixedFrameSize()+frame+lo+i)
		}
	} else {
		//fmt.Printf("zerorange frame:%v, lo: %v, hi:%v \n", frame ,lo, hi)
		//	ADD 	$(FIXED_FRAME+frame+lo-4), SP, r1
		//	ADD 	$cnt, r1, r2
		// loop:
		//	MOVW	R0, (Widthptr)r1
		//	ADD 	$Widthptr, r1
		//	BNE		r1, r2, loop
		p = gc.Appendpp(p, mips.AADD, obj.TYPE_CONST, 0, gc.Ctxt.FixedFrameSize()+frame+lo-4, obj.TYPE_REG, mips.REGRT1, 0)
		p.Reg = mips.REGSP
		p = gc.Appendpp(p, mips.AADD, obj.TYPE_CONST, 0, cnt, obj.TYPE_REG, mips.REGRT2, 0)
		p.Reg = mips.REGRT1
		p = gc.Appendpp(p, mips.AMOVW, obj.TYPE_REG, mips.REGZERO, 0, obj.TYPE_MEM, mips.REGRT1, int64(gc.Widthptr))
		p1 := p
		p = gc.Appendpp(p, mips.AADD, obj.TYPE_CONST, 0, int64(gc.Widthptr), obj.TYPE_REG, mips.REGRT1, 0)
		p = gc.Appendpp(p, mips.ABNE, obj.TYPE_REG, mips.REGRT1, 0, obj.TYPE_BRANCH, 0, 0)
		p.Reg = mips.REGRT2
		gc.Patch(p, p1)
	}

	return p
}

func zeroAuto(n *gc.Node, pp *obj.Prog) {
	// Note: this code must not clobber any registers.
	sym := gc.Linksym(n.Sym)
	size := n.Type.Size()
	for i := int64(0); i < size; i += 4 {
		p := gc.AddAsmAfter(mips.AMOVW, pp)
		pp = p
		p.From.Type = obj.TYPE_REG
		p.From.Reg = mips.REGZERO
		p.To.Type = obj.TYPE_MEM
		p.To.Name = obj.NAME_AUTO
		p.To.Reg = mips.REGSP
		p.To.Offset = n.Xoffset + i
		p.To.Sym = sym
	}
}

func ginsnop() {
	p := gc.Prog(mips.ANOR)
	p.From.Type = obj.TYPE_REG
	p.From.Reg = mips.REG_R0
	p.To.Type = obj.TYPE_REG
	p.To.Reg = mips.REG_R0
}
