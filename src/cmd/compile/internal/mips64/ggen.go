// Copyright 2009 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package mips64

import (
	"cmd/compile/internal/gc"
	"cmd/internal/obj"
	"cmd/internal/obj/mips"
	"fmt"
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
			gc.Fatalf("var %v has size %d offset %d", gc.Nconv(n, gc.FmtLong), int(n.Type.Width), int(n.Xoffset))
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

func zerorange(p *obj.Prog, frame int64, lo int64, hi int64) *obj.Prog {
	cnt := hi - lo
	if cnt == 0 {
		return p
	}
	if cnt < int64(4*gc.Widthptr) {
		for i := int64(0); i < cnt; i += int64(gc.Widthptr) {
			p = appendpp(p, mips.AMOVV, obj.TYPE_REG, mips.REGZERO, 0, obj.TYPE_MEM, mips.REGSP, 8+frame+lo+i)
		}
		// TODO(dfc): https://golang.org/issue/12108
		// If DUFFZERO is used inside a tail call (see genwrapper) it will
		// overwrite the link register.
	} else if false && cnt <= int64(128*gc.Widthptr) {
		p = appendpp(p, mips.AADDV, obj.TYPE_CONST, 0, 8+frame+lo-8, obj.TYPE_REG, mips.REGRT1, 0)
		p.Reg = mips.REGSP
		p = appendpp(p, obj.ADUFFZERO, obj.TYPE_NONE, 0, 0, obj.TYPE_MEM, 0, 0)
		f := gc.Sysfunc("duffzero")
		gc.Naddr(&p.To, f)
		gc.Afunclit(&p.To, f)
		p.To.Offset = 8 * (128 - cnt/int64(gc.Widthptr))
	} else {
		//	ADDV	$(8+frame+lo-8), SP, r1
		//	ADDV	$cnt, r1, r2
		// loop:
		//	MOVV	R0, (Widthptr)r1
		//	ADDV	$Widthptr, r1
		//	BNE		r1, r2, loop
		p = appendpp(p, mips.AADDV, obj.TYPE_CONST, 0, 8+frame+lo-8, obj.TYPE_REG, mips.REGRT1, 0)
		p.Reg = mips.REGSP
		p = appendpp(p, mips.AADDV, obj.TYPE_CONST, 0, cnt, obj.TYPE_REG, mips.REGRT2, 0)
		p.Reg = mips.REGRT1
		p = appendpp(p, mips.AMOVV, obj.TYPE_REG, mips.REGZERO, 0, obj.TYPE_MEM, mips.REGRT1, int64(gc.Widthptr))
		p1 := p
		p = appendpp(p, mips.AADDV, obj.TYPE_CONST, 0, int64(gc.Widthptr), obj.TYPE_REG, mips.REGRT1, 0)
		p = appendpp(p, mips.ABNE, obj.TYPE_REG, mips.REGRT1, 0, obj.TYPE_BRANCH, 0, 0)
		p.Reg = mips.REGRT2
		gc.Patch(p, p1)
	}

	return p
}

func appendpp(p *obj.Prog, as obj.As, ftype obj.AddrType, freg int, foffset int64, ttype obj.AddrType, treg int, toffset int64) *obj.Prog {
	q := gc.Ctxt.NewProg()
	gc.Clearp(q)
	q.As = as
	q.Lineno = p.Lineno
	q.From.Type = ftype
	q.From.Reg = int16(freg)
	q.From.Offset = foffset
	q.To.Type = ttype
	q.To.Reg = int16(treg)
	q.To.Offset = toffset
	q.Link = p.Link
	p.Link = q
	return q
}

func ginsnop() {
	var reg gc.Node
	gc.Nodreg(&reg, gc.Types[gc.TINT], mips.REG_R0)
	gins(mips.ANOR, &reg, &reg)
}

var panicdiv *gc.Node

/*
 * generate division.
 * generates one of:
 *	res = nl / nr
 *	res = nl % nr
 * according to op.
 */
func dodiv(op gc.Op, nl *gc.Node, nr *gc.Node, res *gc.Node) {
	t := nl.Type

	t0 := t

	if t.Width < 8 {
		if t.IsSigned() {
			t = gc.Types[gc.TINT64]
		} else {
			t = gc.Types[gc.TUINT64]
		}
	}

	a := optoas(gc.ODIV, t)

	var tl gc.Node
	gc.Regalloc(&tl, t0, nil)
	var tr gc.Node
	gc.Regalloc(&tr, t0, nil)
	if nl.Ullman >= nr.Ullman {
		gc.Cgen(nl, &tl)
		gc.Cgen(nr, &tr)
	} else {
		gc.Cgen(nr, &tr)
		gc.Cgen(nl, &tl)
	}

	if t != t0 {
		// Convert
		tl2 := tl

		tr2 := tr
		tl.Type = t
		tr.Type = t
		gmove(&tl2, &tl)
		gmove(&tr2, &tr)
	}

	// Handle divide-by-zero panic.
	p1 := ginsbranch(mips.ABNE, nil, &tr, nil, 0)
	if panicdiv == nil {
		panicdiv = gc.Sysfunc("panicdivide")
	}
	gc.Ginscall(panicdiv, -1)
	gc.Patch(p1, gc.Pc)

	gins3(a, &tr, &tl, nil)
	gc.Regfree(&tr)
	if op == gc.ODIV {
		var lo gc.Node
		gc.Nodreg(&lo, gc.Types[gc.TUINT64], mips.REG_LO)
		gins(mips.AMOVV, &lo, &tl)
	} else { // remainder in REG_HI
		var hi gc.Node
		gc.Nodreg(&hi, gc.Types[gc.TUINT64], mips.REG_HI)
		gins(mips.AMOVV, &hi, &tl)
	}
	gmove(&tl, res)
	gc.Regfree(&tl)
}

/*
 * generate high multiply:
 *   res = (nl*nr) >> width
 */
func cgen_hmul(nl *gc.Node, nr *gc.Node, res *gc.Node) {
	// largest ullman on left.
	if nl.Ullman < nr.Ullman {
		nl, nr = nr, nl
	}

	t := nl.Type
	w := t.Width * 8
	var n1 gc.Node
	gc.Cgenr(nl, &n1, res)
	var n2 gc.Node
	gc.Cgenr(nr, &n2, nil)
	switch gc.Simtype[t.Etype] {
	case gc.TINT8,
		gc.TINT16,
		gc.TINT32:
		gins3(optoas(gc.OMUL, t), &n2, &n1, nil)
		var lo gc.Node
		gc.Nodreg(&lo, gc.Types[gc.TUINT64], mips.REG_LO)
		gins(mips.AMOVV, &lo, &n1)
		p := gins(mips.ASRAV, nil, &n1)
		p.From.Type = obj.TYPE_CONST
		p.From.Offset = w

	case gc.TUINT8,
		gc.TUINT16,
		gc.TUINT32:
		gins3(optoas(gc.OMUL, t), &n2, &n1, nil)
		var lo gc.Node
		gc.Nodreg(&lo, gc.Types[gc.TUINT64], mips.REG_LO)
		gins(mips.AMOVV, &lo, &n1)
		p := gins(mips.ASRLV, nil, &n1)
		p.From.Type = obj.TYPE_CONST
		p.From.Offset = w

	case gc.TINT64,
		gc.TUINT64:
		if t.IsSigned() {
			gins3(mips.AMULV, &n2, &n1, nil)
		} else {
			gins3(mips.AMULVU, &n2, &n1, nil)
		}
		var hi gc.Node
		gc.Nodreg(&hi, gc.Types[gc.TUINT64], mips.REG_HI)
		gins(mips.AMOVV, &hi, &n1)

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
	a := optoas(op, nl.Type)

	if nr.Op == gc.OLITERAL {
		var n1 gc.Node
		gc.Regalloc(&n1, nl.Type, res)
		gc.Cgen(nl, &n1)
		sc := uint64(nr.Int64())
		if sc >= uint64(nl.Type.Width*8) {
			// large shift gets 2 shifts by width-1
			var n3 gc.Node
			gc.Nodconst(&n3, gc.Types[gc.TUINT32], nl.Type.Width*8-1)

			gins(a, &n3, &n1)
			gins(a, &n3, &n1)
		} else {
			gins(a, nr, &n1)
		}
		gmove(&n1, res)
		gc.Regfree(&n1)
		return
	}

	if nl.Ullman >= gc.UINF {
		var n4 gc.Node
		gc.Tempname(&n4, nl.Type)
		gc.Cgen(nl, &n4)
		nl = &n4
	}

	if nr.Ullman >= gc.UINF {
		var n5 gc.Node
		gc.Tempname(&n5, nr.Type)
		gc.Cgen(nr, &n5)
		nr = &n5
	}

	// Allow either uint32 or uint64 as shift type,
	// to avoid unnecessary conversion from uint32 to uint64
	// just to do the comparison.
	tcount := gc.Types[gc.Simtype[nr.Type.Etype]]

	if tcount.Etype < gc.TUINT32 {
		tcount = gc.Types[gc.TUINT32]
	}

	var n1 gc.Node
	gc.Regalloc(&n1, nr.Type, nil) // to hold the shift type in CX
	var n3 gc.Node
	gc.Regalloc(&n3, tcount, &n1) // to clear high bits of CX

	var n2 gc.Node
	gc.Regalloc(&n2, nl.Type, res)

	if nl.Ullman >= nr.Ullman {
		gc.Cgen(nl, &n2)
		gc.Cgen(nr, &n1)
		gmove(&n1, &n3)
	} else {
		gc.Cgen(nr, &n1)
		gmove(&n1, &n3)
		gc.Cgen(nl, &n2)
	}

	gc.Regfree(&n3)

	// test and fix up large shifts
	if !bounded {
		var rtmp gc.Node
		gc.Nodreg(&rtmp, tcount, mips.REGTMP)
		gc.Nodconst(&n3, tcount, nl.Type.Width*8)
		gins3(mips.ASGTU, &n3, &n1, &rtmp)
		p1 := ginsbranch(mips.ABNE, nil, &rtmp, nil, 0)
		if op == gc.ORSH && nl.Type.IsSigned() {
			gc.Nodconst(&n3, gc.Types[gc.TUINT32], nl.Type.Width*8-1)
			gins(a, &n3, &n2)
		} else {
			gc.Nodconst(&n3, nl.Type, 0)
			gmove(&n3, &n2)
		}

		gc.Patch(p1, gc.Pc)
	}

	gins(a, &n1, &n2)

	gmove(&n2, res)

	gc.Regfree(&n1)
	gc.Regfree(&n2)
}

func clearfat(nl *gc.Node) {
	/* clear a fat object */
	if gc.Debug['g'] != 0 {
		fmt.Printf("clearfat %v (%v, size: %d)\n", nl, nl.Type, nl.Type.Width)
	}

	w := uint64(nl.Type.Width)

	// Avoid taking the address for simple enough types.
	if gc.Componentgen(nil, nl) {
		return
	}

	c := w % 8 // bytes
	q := w / 8 // dwords

	if gc.Reginuse(mips.REGRT1) {
		gc.Fatalf("%v in use during clearfat", obj.Rconv(mips.REGRT1))
	}

	var r0 gc.Node
	gc.Nodreg(&r0, gc.Types[gc.TUINT64], mips.REGZERO)
	var dst gc.Node
	gc.Nodreg(&dst, gc.Types[gc.Tptr], mips.REGRT1)
	gc.Regrealloc(&dst)
	gc.Agen(nl, &dst)

	var boff uint64
	if q > 128 {
		p := gins(mips.ASUBV, nil, &dst)
		p.From.Type = obj.TYPE_CONST
		p.From.Offset = 8

		var end gc.Node
		gc.Regalloc(&end, gc.Types[gc.Tptr], nil)
		p = gins(mips.AMOVV, &dst, &end)
		p.From.Type = obj.TYPE_ADDR
		p.From.Offset = int64(q * 8)

		p = gins(mips.AMOVV, &r0, &dst)
		p.To.Type = obj.TYPE_MEM
		p.To.Offset = 8
		pl := p

		p = gins(mips.AADDV, nil, &dst)
		p.From.Type = obj.TYPE_CONST
		p.From.Offset = 8

		gc.Patch(ginsbranch(mips.ABNE, nil, &dst, &end, 0), pl)

		gc.Regfree(&end)

		// The loop leaves R1 on the last zeroed dword
		boff = 8
		// TODO(dfc): https://golang.org/issue/12108
		// If DUFFZERO is used inside a tail call (see genwrapper) it will
		// overwrite the link register.
	} else if false && q >= 4 {
		p := gins(mips.ASUBV, nil, &dst)
		p.From.Type = obj.TYPE_CONST
		p.From.Offset = 8
		f := gc.Sysfunc("duffzero")
		p = gins(obj.ADUFFZERO, nil, f)
		gc.Afunclit(&p.To, f)

		// 8 and 128 = magic constants: see ../../runtime/asm_mips64x.s
		p.To.Offset = int64(8 * (128 - q))

		// duffzero leaves R1 on the last zeroed dword
		boff = 8
	} else {
		var p *obj.Prog
		for t := uint64(0); t < q; t++ {
			p = gins(mips.AMOVV, &r0, &dst)
			p.To.Type = obj.TYPE_MEM
			p.To.Offset = int64(8 * t)
		}

		boff = 8 * q
	}

	var p *obj.Prog
	for t := uint64(0); t < c; t++ {
		p = gins(mips.AMOVB, &r0, &dst)
		p.To.Type = obj.TYPE_MEM
		p.To.Offset = int64(t + boff)
	}

	gc.Regfree(&dst)
}

// Called after regopt and peep have run.
// Expand CHECKNIL pseudo-op into actual nil pointer check.
func expandchecks(firstp *obj.Prog) {
	var p1 *obj.Prog

	for p := firstp; p != nil; p = p.Link {
		if gc.Debug_checknil != 0 && gc.Ctxt.Debugvlog != 0 {
			fmt.Printf("expandchecks: %v\n", p)
		}
		if p.As != obj.ACHECKNIL {
			continue
		}
		if gc.Debug_checknil != 0 && p.Lineno > 1 { // p->lineno==1 in generated wrappers
			gc.Warnl(p.Lineno, "generated nil check")
		}
		if p.From.Type != obj.TYPE_REG {
			gc.Fatalf("invalid nil check %v\n", p)
		}

		// check is
		//	BNE arg, 2(PC)
		//	MOVV R0, 0(R0)
		p1 = gc.Ctxt.NewProg()
		gc.Clearp(p1)
		p1.Link = p.Link
		p.Link = p1
		p1.Lineno = p.Lineno
		p1.Pc = 9999

		p.As = mips.ABNE
		p.To.Type = obj.TYPE_BRANCH
		p.To.Val = p1.Link

		// crash by write to memory address 0.
		p1.As = mips.AMOVV
		p1.From.Type = obj.TYPE_REG
		p1.From.Reg = mips.REGZERO
		p1.To.Type = obj.TYPE_MEM
		p1.To.Reg = mips.REGZERO
		p1.To.Offset = 0
	}
}

// res = runtime.getg()
func getg(res *gc.Node) {
	var n1 gc.Node
	gc.Nodreg(&n1, res.Type, mips.REGG)
	gmove(&n1, res)
}
