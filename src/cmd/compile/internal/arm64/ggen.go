// Copyright 2009 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package arm64

import (
	"cmd/compile/internal/gc"
	"cmd/internal/obj"
	"cmd/internal/obj/arm64"
)

func defframe(ptxt *obj.Prog) {
	// fill in argument size, stack size
	ptxt.To.Type = obj.TYPE_TEXTSIZE

	ptxt.To.Val = int32(gc.Rnd(gc.Curfn.Type.ArgWidth(), int64(gc.Widthptr)))
	frame := uint32(gc.Rnd(gc.Stksize+gc.Maxarg, int64(gc.Widthreg)))

	// arm64 requires that the frame size (not counting saved LR)
	// be empty or be 8 mod 16. If not, pad it.
	if frame != 0 && frame%16 != 8 {
		frame += 8
	}

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

var darwin = obj.GOOS == "darwin"

func zerorange(p *obj.Prog, frame int64, lo int64, hi int64) *obj.Prog {
	cnt := hi - lo
	if cnt == 0 {
		return p
	}
	if cnt < int64(4*gc.Widthptr) {
		for i := int64(0); i < cnt; i += int64(gc.Widthptr) {
			p = gc.Appendpp(p, arm64.AMOVD, obj.TYPE_REG, arm64.REGZERO, 0, obj.TYPE_MEM, arm64.REGSP, 8+frame+lo+i)
		}
	} else if cnt <= int64(128*gc.Widthptr) && !darwin { // darwin ld64 cannot handle BR26 reloc with non-zero addend
		p = gc.Appendpp(p, arm64.AMOVD, obj.TYPE_REG, arm64.REGSP, 0, obj.TYPE_REG, arm64.REGRT1, 0)
		p = gc.Appendpp(p, arm64.AADD, obj.TYPE_CONST, 0, 8+frame+lo-8, obj.TYPE_REG, arm64.REGRT1, 0)
		p.Reg = arm64.REGRT1
		p = gc.Appendpp(p, obj.ADUFFZERO, obj.TYPE_NONE, 0, 0, obj.TYPE_MEM, 0, 0)
		gc.Naddr(&p.To, gc.Sysfunc("duffzero"))
		p.To.Offset = 4 * (128 - cnt/int64(gc.Widthptr))
	} else {
		p = gc.Appendpp(p, arm64.AMOVD, obj.TYPE_CONST, 0, 8+frame+lo-8, obj.TYPE_REG, arm64.REGTMP, 0)
		p = gc.Appendpp(p, arm64.AMOVD, obj.TYPE_REG, arm64.REGSP, 0, obj.TYPE_REG, arm64.REGRT1, 0)
		p = gc.Appendpp(p, arm64.AADD, obj.TYPE_REG, arm64.REGTMP, 0, obj.TYPE_REG, arm64.REGRT1, 0)
		p.Reg = arm64.REGRT1
		p = gc.Appendpp(p, arm64.AMOVD, obj.TYPE_CONST, 0, cnt, obj.TYPE_REG, arm64.REGTMP, 0)
		p = gc.Appendpp(p, arm64.AADD, obj.TYPE_REG, arm64.REGTMP, 0, obj.TYPE_REG, arm64.REGRT2, 0)
		p.Reg = arm64.REGRT1
		p = gc.Appendpp(p, arm64.AMOVD, obj.TYPE_REG, arm64.REGZERO, 0, obj.TYPE_MEM, arm64.REGRT1, int64(gc.Widthptr))
		p.Scond = arm64.C_XPRE
		p1 := p
		p = gc.Appendpp(p, arm64.ACMP, obj.TYPE_REG, arm64.REGRT1, 0, obj.TYPE_NONE, 0, 0)
		p.Reg = arm64.REGRT2
		p = gc.Appendpp(p, arm64.ABNE, obj.TYPE_NONE, 0, 0, obj.TYPE_BRANCH, 0, 0)
		gc.Patch(p, p1)
	}

	return p
}

func zeroAuto(n *gc.Node, pp *obj.Prog) {
	// Note: this code must not clobber any registers.
	sym := gc.Linksym(n.Sym)
	size := n.Type.Size()
	for i := int64(0); i < size; i += 8 {
		p := gc.AddAsmAfter(arm64.AMOVD, pp)
		pp = p
		p.From.Type = obj.TYPE_REG
		p.From.Reg = arm64.REGZERO
		p.To.Type = obj.TYPE_MEM
		p.To.Name = obj.NAME_AUTO
		p.To.Reg = arm64.REGSP
		p.To.Offset = n.Xoffset + i
		p.To.Sym = sym
	}
}

func ginsnop() {
	p := gc.Prog(arm64.AHINT)
	p.From.Type = obj.TYPE_CONST
}
