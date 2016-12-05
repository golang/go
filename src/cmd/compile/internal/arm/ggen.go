// Copyright 2009 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package arm

import (
	"cmd/compile/internal/gc"
	"cmd/internal/obj"
	"cmd/internal/obj/arm"
)

func defframe(ptxt *obj.Prog) {
	// fill in argument size, stack size
	ptxt.To.Type = obj.TYPE_TEXTSIZE

	ptxt.To.Val = int32(gc.Rnd(gc.Curfn.Type.ArgWidth(), int64(gc.Widthptr)))
	frame := uint32(gc.Rnd(gc.Stksize+gc.Maxarg, int64(gc.Widthreg)))
	ptxt.To.Offset = int64(frame)

	// insert code to contain ambiguously live variables
	// so that garbage collector only sees initialized values
	// when it looks for pointers.
	p := ptxt

	hi := int64(0)
	lo := hi
	r0 := uint32(0)
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
		if lo != hi && n.Xoffset+n.Type.Width >= lo-int64(2*gc.Widthptr) {
			// merge with range we already have
			lo = gc.Rnd(n.Xoffset, int64(gc.Widthptr))

			continue
		}

		// zero old range
		p = zerorange(p, int64(frame), lo, hi, &r0)

		// set new range
		hi = n.Xoffset + n.Type.Width

		lo = n.Xoffset
	}

	// zero final range
	zerorange(p, int64(frame), lo, hi, &r0)
}

func zerorange(p *obj.Prog, frame int64, lo int64, hi int64, r0 *uint32) *obj.Prog {
	cnt := hi - lo
	if cnt == 0 {
		return p
	}
	if *r0 == 0 {
		p = gc.Appendpp(p, arm.AMOVW, obj.TYPE_CONST, 0, 0, obj.TYPE_REG, arm.REG_R0, 0)
		*r0 = 1
	}

	if cnt < int64(4*gc.Widthptr) {
		for i := int64(0); i < cnt; i += int64(gc.Widthptr) {
			p = gc.Appendpp(p, arm.AMOVW, obj.TYPE_REG, arm.REG_R0, 0, obj.TYPE_MEM, arm.REGSP, 4+frame+lo+i)
		}
	} else if !gc.Nacl && (cnt <= int64(128*gc.Widthptr)) {
		p = gc.Appendpp(p, arm.AADD, obj.TYPE_CONST, 0, 4+frame+lo, obj.TYPE_REG, arm.REG_R1, 0)
		p.Reg = arm.REGSP
		p = gc.Appendpp(p, obj.ADUFFZERO, obj.TYPE_NONE, 0, 0, obj.TYPE_MEM, 0, 0)
		gc.Naddr(&p.To, gc.Sysfunc("duffzero"))
		p.To.Offset = 4 * (128 - cnt/int64(gc.Widthptr))
	} else {
		p = gc.Appendpp(p, arm.AADD, obj.TYPE_CONST, 0, 4+frame+lo, obj.TYPE_REG, arm.REG_R1, 0)
		p.Reg = arm.REGSP
		p = gc.Appendpp(p, arm.AADD, obj.TYPE_CONST, 0, cnt, obj.TYPE_REG, arm.REG_R2, 0)
		p.Reg = arm.REG_R1
		p = gc.Appendpp(p, arm.AMOVW, obj.TYPE_REG, arm.REG_R0, 0, obj.TYPE_MEM, arm.REG_R1, 4)
		p1 := p
		p.Scond |= arm.C_PBIT
		p = gc.Appendpp(p, arm.ACMP, obj.TYPE_REG, arm.REG_R1, 0, obj.TYPE_NONE, 0, 0)
		p.Reg = arm.REG_R2
		p = gc.Appendpp(p, arm.ABNE, obj.TYPE_NONE, 0, 0, obj.TYPE_BRANCH, 0, 0)
		gc.Patch(p, p1)
	}

	return p
}

func ginsnop() {
	p := gc.Prog(arm.AAND)
	p.From.Type = obj.TYPE_REG
	p.From.Reg = arm.REG_R0
	p.To.Type = obj.TYPE_REG
	p.To.Reg = arm.REG_R0
	p.Scond = arm.C_SCOND_EQ
}
