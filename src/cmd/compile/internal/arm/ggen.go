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
	var n *gc.Node

	// fill in argument size, stack size
	ptxt.To.Type = obj.TYPE_TEXTSIZE

	ptxt.To.Val = int32(gc.Rnd(gc.Curfn.Type.Argwid, int64(gc.Widthptr)))
	frame := uint32(gc.Rnd(gc.Stksize+gc.Maxarg, int64(gc.Widthreg)))
	ptxt.To.Offset = int64(frame)

	// insert code to contain ambiguously live variables
	// so that garbage collector only sees initialized values
	// when it looks for pointers.
	p := ptxt

	hi := int64(0)
	lo := hi
	r0 := uint32(0)
	for l := gc.Curfn.Func.Dcl; l != nil; l = l.Next {
		n = l.N
		if !n.Name.Needzero {
			continue
		}
		if n.Class != gc.PAUTO {
			gc.Fatalf("needzero class %d", n.Class)
		}
		if n.Type.Width%int64(gc.Widthptr) != 0 || n.Xoffset%int64(gc.Widthptr) != 0 || n.Type.Width == 0 {
			gc.Fatalf("var %v has size %d offset %d", gc.Nconv(n, obj.FmtLong), int(n.Type.Width), int(n.Xoffset))
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
		p = appendpp(p, arm.AMOVW, obj.TYPE_CONST, 0, 0, obj.TYPE_REG, arm.REG_R0, 0)
		*r0 = 1
	}

	if cnt < int64(4*gc.Widthptr) {
		for i := int64(0); i < cnt; i += int64(gc.Widthptr) {
			p = appendpp(p, arm.AMOVW, obj.TYPE_REG, arm.REG_R0, 0, obj.TYPE_MEM, arm.REGSP, int32(4+frame+lo+i))
		}
	} else if !gc.Nacl && (cnt <= int64(128*gc.Widthptr)) {
		p = appendpp(p, arm.AADD, obj.TYPE_CONST, 0, int32(4+frame+lo), obj.TYPE_REG, arm.REG_R1, 0)
		p.Reg = arm.REGSP
		p = appendpp(p, obj.ADUFFZERO, obj.TYPE_NONE, 0, 0, obj.TYPE_MEM, 0, 0)
		f := gc.Sysfunc("duffzero")
		gc.Naddr(&p.To, f)
		gc.Afunclit(&p.To, f)
		p.To.Offset = 4 * (128 - cnt/int64(gc.Widthptr))
	} else {
		p = appendpp(p, arm.AADD, obj.TYPE_CONST, 0, int32(4+frame+lo), obj.TYPE_REG, arm.REG_R1, 0)
		p.Reg = arm.REGSP
		p = appendpp(p, arm.AADD, obj.TYPE_CONST, 0, int32(cnt), obj.TYPE_REG, arm.REG_R2, 0)
		p.Reg = arm.REG_R1
		p = appendpp(p, arm.AMOVW, obj.TYPE_REG, arm.REG_R0, 0, obj.TYPE_MEM, arm.REG_R1, 4)
		p1 := p
		p.Scond |= arm.C_PBIT
		p = appendpp(p, arm.ACMP, obj.TYPE_REG, arm.REG_R1, 0, obj.TYPE_NONE, 0, 0)
		p.Reg = arm.REG_R2
		p = appendpp(p, arm.ABNE, obj.TYPE_NONE, 0, 0, obj.TYPE_BRANCH, 0, 0)
		gc.Patch(p, p1)
	}

	return p
}

func appendpp(p *obj.Prog, as int, ftype int, freg int, foffset int32, ttype int, treg int, toffset int32) *obj.Prog {
	q := gc.Ctxt.NewProg()
	gc.Clearp(q)
	q.As = int16(as)
	q.Lineno = p.Lineno
	q.From.Type = int16(ftype)
	q.From.Reg = int16(freg)
	q.From.Offset = int64(foffset)
	q.To.Type = int16(ttype)
	q.To.Reg = int16(treg)
	q.To.Offset = int64(toffset)
	q.Link = p.Link
	p.Link = q
	return q
}

/*
 * generate high multiply
 *  res = (nl * nr) >> wordsize
 */
func cgen_hmul(nl *gc.Node, nr *gc.Node, res *gc.Node) {
	if nl.Ullman < nr.Ullman {
		nl, nr = nr, nl
	}

	t := nl.Type
	w := int(t.Width * 8)
	var n1 gc.Node
	gc.Regalloc(&n1, t, res)
	gc.Cgen(nl, &n1)
	var n2 gc.Node
	gc.Regalloc(&n2, t, nil)
	gc.Cgen(nr, &n2)
	switch gc.Simtype[t.Etype] {
	case gc.TINT8,
		gc.TINT16:
		gins(optoas(gc.OMUL, t), &n2, &n1)
		gshift(arm.AMOVW, &n1, arm.SHIFT_AR, int32(w), &n1)

	case gc.TUINT8,
		gc.TUINT16:
		gins(optoas(gc.OMUL, t), &n2, &n1)
		gshift(arm.AMOVW, &n1, arm.SHIFT_LR, int32(w), &n1)

		// perform a long multiplication.
	case gc.TINT32,
		gc.TUINT32:
		var p *obj.Prog
		if gc.Issigned[t.Etype] {
			p = gins(arm.AMULL, &n2, nil)
		} else {
			p = gins(arm.AMULLU, &n2, nil)
		}

		// n2 * n1 -> (n1 n2)
		p.Reg = n1.Reg

		p.To.Type = obj.TYPE_REGREG
		p.To.Reg = n1.Reg
		p.To.Offset = int64(n2.Reg)

	default:
		gc.Fatalf("cgen_hmul %v", t)
	}

	gc.Cgen(&n1, res)
	gc.Regfree(&n1)
	gc.Regfree(&n2)
}

/*
 * generate shift according to op, one of:
 *	res = nl << nr
 *	res = nl >> nr
 */
func cgen_shift(op gc.Op, bounded bool, nl *gc.Node, nr *gc.Node, res *gc.Node) {
	if nl.Type.Width > 4 {
		gc.Fatalf("cgen_shift %v", nl.Type)
	}

	w := int(nl.Type.Width * 8)

	if op == gc.OLROT {
		v := nr.Int()
		var n1 gc.Node
		gc.Regalloc(&n1, nl.Type, res)
		if w == 32 {
			gc.Cgen(nl, &n1)
			gshift(arm.AMOVW, &n1, arm.SHIFT_RR, int32(w)-int32(v), &n1)
		} else {
			var n2 gc.Node
			gc.Regalloc(&n2, nl.Type, nil)
			gc.Cgen(nl, &n2)
			gshift(arm.AMOVW, &n2, arm.SHIFT_LL, int32(v), &n1)
			gshift(arm.AORR, &n2, arm.SHIFT_LR, int32(w)-int32(v), &n1)
			gc.Regfree(&n2)

			// Ensure sign/zero-extended result.
			gins(optoas(gc.OAS, nl.Type), &n1, &n1)
		}

		gmove(&n1, res)
		gc.Regfree(&n1)
		return
	}

	if nr.Op == gc.OLITERAL {
		var n1 gc.Node
		gc.Regalloc(&n1, nl.Type, res)
		gc.Cgen(nl, &n1)
		sc := uint64(nr.Int())
		if sc == 0 {
		} else // nothing to do
		if sc >= uint64(nl.Type.Width*8) {
			if op == gc.ORSH && gc.Issigned[nl.Type.Etype] {
				gshift(arm.AMOVW, &n1, arm.SHIFT_AR, int32(w), &n1)
			} else {
				gins(arm.AEOR, &n1, &n1)
			}
		} else {
			if op == gc.ORSH && gc.Issigned[nl.Type.Etype] {
				gshift(arm.AMOVW, &n1, arm.SHIFT_AR, int32(sc), &n1)
			} else if op == gc.ORSH {
				gshift(arm.AMOVW, &n1, arm.SHIFT_LR, int32(sc), &n1) // OLSH
			} else {
				gshift(arm.AMOVW, &n1, arm.SHIFT_LL, int32(sc), &n1)
			}
		}

		if w < 32 && op == gc.OLSH {
			gins(optoas(gc.OAS, nl.Type), &n1, &n1)
		}
		gmove(&n1, res)
		gc.Regfree(&n1)
		return
	}

	tr := nr.Type
	var t gc.Node
	var n1 gc.Node
	var n2 gc.Node
	var n3 gc.Node
	if tr.Width > 4 {
		var nt gc.Node
		gc.Tempname(&nt, nr.Type)
		if nl.Ullman >= nr.Ullman {
			gc.Regalloc(&n2, nl.Type, res)
			gc.Cgen(nl, &n2)
			gc.Cgen(nr, &nt)
			n1 = nt
		} else {
			gc.Cgen(nr, &nt)
			gc.Regalloc(&n2, nl.Type, res)
			gc.Cgen(nl, &n2)
		}

		var hi gc.Node
		var lo gc.Node
		split64(&nt, &lo, &hi)
		gc.Regalloc(&n1, gc.Types[gc.TUINT32], nil)
		gc.Regalloc(&n3, gc.Types[gc.TUINT32], nil)
		gmove(&lo, &n1)
		gmove(&hi, &n3)
		splitclean()
		gins(arm.ATST, &n3, nil)
		gc.Nodconst(&t, gc.Types[gc.TUINT32], int64(w))
		p1 := gins(arm.AMOVW, &t, &n1)
		p1.Scond = arm.C_SCOND_NE
		tr = gc.Types[gc.TUINT32]
		gc.Regfree(&n3)
	} else {
		if nl.Ullman >= nr.Ullman {
			gc.Regalloc(&n2, nl.Type, res)
			gc.Cgen(nl, &n2)
			gc.Regalloc(&n1, nr.Type, nil)
			gc.Cgen(nr, &n1)
		} else {
			gc.Regalloc(&n1, nr.Type, nil)
			gc.Cgen(nr, &n1)
			gc.Regalloc(&n2, nl.Type, res)
			gc.Cgen(nl, &n2)
		}
	}

	// test for shift being 0
	gins(arm.ATST, &n1, nil)

	p3 := gc.Gbranch(arm.ABEQ, nil, -1)

	// test and fix up large shifts
	// TODO: if(!bounded), don't emit some of this.
	gc.Regalloc(&n3, tr, nil)

	gc.Nodconst(&t, gc.Types[gc.TUINT32], int64(w))
	gmove(&t, &n3)
	gins(arm.ACMP, &n1, &n3)
	if op == gc.ORSH {
		var p1 *obj.Prog
		var p2 *obj.Prog
		if gc.Issigned[nl.Type.Etype] {
			p1 = gshift(arm.AMOVW, &n2, arm.SHIFT_AR, int32(w)-1, &n2)
			p2 = gregshift(arm.AMOVW, &n2, arm.SHIFT_AR, &n1, &n2)
		} else {
			p1 = gins(arm.AEOR, &n2, &n2)
			p2 = gregshift(arm.AMOVW, &n2, arm.SHIFT_LR, &n1, &n2)
		}

		p1.Scond = arm.C_SCOND_HS
		p2.Scond = arm.C_SCOND_LO
	} else {
		p1 := gins(arm.AEOR, &n2, &n2)
		p2 := gregshift(arm.AMOVW, &n2, arm.SHIFT_LL, &n1, &n2)
		p1.Scond = arm.C_SCOND_HS
		p2.Scond = arm.C_SCOND_LO
	}

	gc.Regfree(&n3)

	gc.Patch(p3, gc.Pc)

	// Left-shift of smaller word must be sign/zero-extended.
	if w < 32 && op == gc.OLSH {
		gins(optoas(gc.OAS, nl.Type), &n2, &n2)
	}
	gmove(&n2, res)

	gc.Regfree(&n1)
	gc.Regfree(&n2)
}

func clearfat(nl *gc.Node) {
	/* clear a fat object */
	if gc.Debug['g'] != 0 {
		gc.Dump("\nclearfat", nl)
	}

	w := uint32(nl.Type.Width)

	// Avoid taking the address for simple enough types.
	if gc.Componentgen(nil, nl) {
		return
	}

	c := w % 4 // bytes
	q := w / 4 // quads

	var r0 gc.Node
	r0.Op = gc.OREGISTER

	r0.Reg = arm.REG_R0
	var r1 gc.Node
	r1.Op = gc.OREGISTER
	r1.Reg = arm.REG_R1
	var dst gc.Node
	gc.Regalloc(&dst, gc.Types[gc.Tptr], &r1)
	gc.Agen(nl, &dst)
	var nc gc.Node
	gc.Nodconst(&nc, gc.Types[gc.TUINT32], 0)
	var nz gc.Node
	gc.Regalloc(&nz, gc.Types[gc.TUINT32], &r0)
	gc.Cgen(&nc, &nz)

	if q > 128 {
		var end gc.Node
		gc.Regalloc(&end, gc.Types[gc.Tptr], nil)
		p := gins(arm.AMOVW, &dst, &end)
		p.From.Type = obj.TYPE_ADDR
		p.From.Offset = int64(q) * 4

		p = gins(arm.AMOVW, &nz, &dst)
		p.To.Type = obj.TYPE_MEM
		p.To.Offset = 4
		p.Scond |= arm.C_PBIT
		pl := p

		p = gins(arm.ACMP, &dst, nil)
		raddr(&end, p)
		gc.Patch(gc.Gbranch(arm.ABNE, nil, 0), pl)

		gc.Regfree(&end)
	} else if q >= 4 && !gc.Nacl {
		f := gc.Sysfunc("duffzero")
		p := gins(obj.ADUFFZERO, nil, f)
		gc.Afunclit(&p.To, f)

		// 4 and 128 = magic constants: see ../../runtime/asm_arm.s
		p.To.Offset = 4 * (128 - int64(q))
	} else {
		var p *obj.Prog
		for q > 0 {
			p = gins(arm.AMOVW, &nz, &dst)
			p.To.Type = obj.TYPE_MEM
			p.To.Offset = 4
			p.Scond |= arm.C_PBIT

			//print("1. %v\n", p);
			q--
		}
	}

	var p *obj.Prog
	for c > 0 {
		p = gins(arm.AMOVB, &nz, &dst)
		p.To.Type = obj.TYPE_MEM
		p.To.Offset = 1
		p.Scond |= arm.C_PBIT

		//print("2. %v\n", p);
		c--
	}

	gc.Regfree(&dst)
	gc.Regfree(&nz)
}

// Called after regopt and peep have run.
// Expand CHECKNIL pseudo-op into actual nil pointer check.
func expandchecks(firstp *obj.Prog) {
	var reg int
	var p1 *obj.Prog

	for p := firstp; p != nil; p = p.Link {
		if p.As != obj.ACHECKNIL {
			continue
		}
		if gc.Debug_checknil != 0 && p.Lineno > 1 { // p->lineno==1 in generated wrappers
			gc.Warnl(int(p.Lineno), "generated nil check")
		}
		if p.From.Type != obj.TYPE_REG {
			gc.Fatalf("invalid nil check %v", p)
		}
		reg = int(p.From.Reg)

		// check is
		//	CMP arg, $0
		//	MOV.EQ arg, 0(arg)
		p1 = gc.Ctxt.NewProg()

		gc.Clearp(p1)
		p1.Link = p.Link
		p.Link = p1
		p1.Lineno = p.Lineno
		p1.Pc = 9999
		p1.As = arm.AMOVW
		p1.From.Type = obj.TYPE_REG
		p1.From.Reg = int16(reg)
		p1.To.Type = obj.TYPE_MEM
		p1.To.Reg = int16(reg)
		p1.To.Offset = 0
		p1.Scond = arm.C_SCOND_EQ
		p.As = arm.ACMP
		p.From.Type = obj.TYPE_CONST
		p.From.Reg = 0
		p.From.Offset = 0
		p.Reg = int16(reg)
	}
}

func ginsnop() {
	var r gc.Node
	gc.Nodreg(&r, gc.Types[gc.TINT], arm.REG_R0)
	p := gins(arm.AAND, &r, &r)
	p.Scond = arm.C_SCOND_EQ
}

/*
 * generate
 *	as $c, n
 */
func ginscon(as int, c int64, n *gc.Node) {
	var n1 gc.Node
	gc.Nodconst(&n1, gc.Types[gc.TINT32], c)
	var n2 gc.Node
	gc.Regalloc(&n2, gc.Types[gc.TINT32], nil)
	gmove(&n1, &n2)
	gins(as, &n2, n)
	gc.Regfree(&n2)
}

func ginscmp(op gc.Op, t *gc.Type, n1, n2 *gc.Node, likely int) *obj.Prog {
	if gc.Isint[t.Etype] && n1.Op == gc.OLITERAL && n1.Int() == 0 && n2.Op != gc.OLITERAL {
		op = gc.Brrev(op)
		n1, n2 = n2, n1
	}
	var r1, r2, g1, g2 gc.Node
	gc.Regalloc(&r1, t, n1)
	gc.Regalloc(&g1, n1.Type, &r1)
	gc.Cgen(n1, &g1)
	gmove(&g1, &r1)
	if gc.Isint[t.Etype] && n2.Op == gc.OLITERAL && n2.Int() == 0 {
		gins(arm.ACMP, &r1, n2)
	} else {
		gc.Regalloc(&r2, t, n2)
		gc.Regalloc(&g2, n1.Type, &r2)
		gc.Cgen(n2, &g2)
		gmove(&g2, &r2)
		gins(optoas(gc.OCMP, t), &r1, &r2)
		gc.Regfree(&g2)
		gc.Regfree(&r2)
	}
	gc.Regfree(&g1)
	gc.Regfree(&r1)
	return gc.Gbranch(optoas(op, t), nil, likely)
}

// addr += index*width if possible.
func addindex(index *gc.Node, width int64, addr *gc.Node) bool {
	switch width {
	case 2:
		gshift(arm.AADD, index, arm.SHIFT_LL, 1, addr)
		return true
	case 4:
		gshift(arm.AADD, index, arm.SHIFT_LL, 2, addr)
		return true
	case 8:
		gshift(arm.AADD, index, arm.SHIFT_LL, 3, addr)
		return true
	}
	return false
}

// res = runtime.getg()
func getg(res *gc.Node) {
	var n1 gc.Node
	gc.Nodreg(&n1, res.Type, arm.REGG)
	gmove(&n1, res)
}
