// Copyright 2016 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package s390x

import (
	"cmd/compile/internal/gc"
	"cmd/internal/obj"
	"cmd/internal/obj/s390x"
	"fmt"
)

// clearLoopCutOff is the (somewhat arbitrary) value above which it is better
// to have a loop of clear instructions (e.g. XCs) rather than just generating
// multiple instructions (i.e. loop unrolling).
// Must be between 256 and 4096.
const clearLoopCutoff = 1024

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

// zerorange clears the stack in the given range.
func zerorange(p *obj.Prog, frame int64, lo int64, hi int64) *obj.Prog {
	cnt := hi - lo
	if cnt == 0 {
		return p
	}

	// Adjust the frame to account for LR.
	frame += gc.Ctxt.FixedFrameSize()
	offset := frame + lo
	reg := int16(s390x.REGSP)

	// If the offset cannot fit in a 12-bit unsigned displacement then we
	// need to create a copy of the stack pointer that we can adjust.
	// We also need to do this if we are going to loop.
	if offset < 0 || offset > 4096-clearLoopCutoff || cnt > clearLoopCutoff {
		p = appendpp(p, s390x.AADD, obj.TYPE_CONST, 0, offset, obj.TYPE_REG, s390x.REGRT1, 0)
		p.Reg = int16(s390x.REGSP)
		reg = s390x.REGRT1
		offset = 0
	}

	// Generate a loop of large clears.
	if cnt > clearLoopCutoff {
		n := cnt - (cnt % 256)
		end := int16(s390x.REGRT2)
		p = appendpp(p, s390x.AADD, obj.TYPE_CONST, 0, offset+n, obj.TYPE_REG, end, 0)
		p.Reg = reg
		p = appendpp(p, s390x.AXC, obj.TYPE_MEM, reg, offset, obj.TYPE_MEM, reg, offset)
		p.From3 = new(obj.Addr)
		p.From3.Type = obj.TYPE_CONST
		p.From3.Offset = 256
		pl := p
		p = appendpp(p, s390x.AADD, obj.TYPE_CONST, 0, 256, obj.TYPE_REG, reg, 0)
		p = appendpp(p, s390x.ACMP, obj.TYPE_REG, reg, 0, obj.TYPE_REG, end, 0)
		p = appendpp(p, s390x.ABNE, obj.TYPE_NONE, 0, 0, obj.TYPE_BRANCH, 0, 0)
		gc.Patch(p, pl)

		cnt -= n
	}

	// Generate remaining clear instructions without a loop.
	for cnt > 0 {
		n := cnt

		// Can clear at most 256 bytes per instruction.
		if n > 256 {
			n = 256
		}

		switch n {
		// Handle very small clears with move instructions.
		case 8, 4, 2, 1:
			ins := s390x.AMOVB
			switch n {
			case 8:
				ins = s390x.AMOVD
			case 4:
				ins = s390x.AMOVW
			case 2:
				ins = s390x.AMOVH
			}
			p = appendpp(p, ins, obj.TYPE_CONST, 0, 0, obj.TYPE_MEM, reg, offset)

		// Handle clears that would require multiple move instructions with XC.
		default:
			p = appendpp(p, s390x.AXC, obj.TYPE_MEM, reg, offset, obj.TYPE_MEM, reg, offset)
			p.From3 = new(obj.Addr)
			p.From3.Type = obj.TYPE_CONST
			p.From3.Offset = n
		}

		cnt -= n
		offset += n
	}

	return p
}

func appendpp(p *obj.Prog, as obj.As, ftype obj.AddrType, freg int16, foffset int64, ttype obj.AddrType, treg int16, toffset int64) *obj.Prog {
	q := gc.Ctxt.NewProg()
	gc.Clearp(q)
	q.As = as
	q.Lineno = p.Lineno
	q.From.Type = ftype
	q.From.Reg = freg
	q.From.Offset = foffset
	q.To.Type = ttype
	q.To.Reg = treg
	q.To.Offset = toffset
	q.Link = p.Link
	p.Link = q
	return q
}

func ginsnop() {
	var reg gc.Node
	gc.Nodreg(&reg, gc.Types[gc.TINT], s390x.REG_R0)
	gins(s390x.AOR, &reg, &reg)
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
	// Have to be careful about handling
	// most negative int divided by -1 correctly.
	// The hardware will generate undefined result.
	// Also need to explicitly trap on division on zero,
	// the hardware will silently generate undefined result.
	// DIVW will leave unpredicable result in higher 32-bit,
	// so always use DIVD/DIVDU.
	t := nl.Type

	t0 := t
	check := 0
	if t.IsSigned() {
		check = 1
		if gc.Isconst(nl, gc.CTINT) && nl.Int64() != -(1<<uint64(t.Width*8-1)) {
			check = 0
		} else if gc.Isconst(nr, gc.CTINT) && nr.Int64() != -1 {
			check = 0
		}
	}

	if t.Width < 8 {
		if t.IsSigned() {
			t = gc.Types[gc.TINT64]
		} else {
			t = gc.Types[gc.TUINT64]
		}
		check = 0
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
	p1 := gins(optoas(gc.OCMP, t), &tr, nil)

	p1.To.Type = obj.TYPE_REG
	p1.To.Reg = s390x.REGZERO
	p1 = gc.Gbranch(optoas(gc.ONE, t), nil, +1)
	if panicdiv == nil {
		panicdiv = gc.Sysfunc("panicdivide")
	}
	gc.Ginscall(panicdiv, -1)
	gc.Patch(p1, gc.Pc)

	var p2 *obj.Prog
	if check != 0 {
		var nm1 gc.Node
		gc.Nodconst(&nm1, t, -1)
		gins(optoas(gc.OCMP, t), &tr, &nm1)
		p1 := gc.Gbranch(optoas(gc.ONE, t), nil, +1)
		if op == gc.ODIV {
			// a / (-1) is -a.
			gins(optoas(gc.OMINUS, t), nil, &tl)

			gmove(&tl, res)
		} else {
			// a % (-1) is 0.
			var nz gc.Node
			gc.Nodconst(&nz, t, 0)

			gmove(&nz, res)
		}

		p2 = gc.Gbranch(obj.AJMP, nil, 0)
		gc.Patch(p1, gc.Pc)
	}

	p1 = gins(a, &tr, &tl)
	if op == gc.ODIV {
		gc.Regfree(&tr)
		gmove(&tl, res)
	} else {
		// A%B = A-(A/B*B)
		var tm gc.Node
		gc.Regalloc(&tm, t, nil)

		// patch div to use the 3 register form
		// TODO(minux): add gins3?
		p1.Reg = p1.To.Reg

		p1.To.Reg = tm.Reg
		gins(optoas(gc.OMUL, t), &tr, &tm)
		gc.Regfree(&tr)
		gins(optoas(gc.OSUB, t), &tm, &tl)
		gc.Regfree(&tm)
		gmove(&tl, res)
	}

	gc.Regfree(&tl)
	if check != 0 {
		gc.Patch(p2, gc.Pc)
	}
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
	w := int(t.Width) * 8
	var n1 gc.Node
	gc.Cgenr(nl, &n1, res)
	var n2 gc.Node
	gc.Cgenr(nr, &n2, nil)
	switch gc.Simtype[t.Etype] {
	case gc.TINT8,
		gc.TINT16,
		gc.TINT32:
		gins(optoas(gc.OMUL, t), &n2, &n1)
		p := gins(s390x.ASRAD, nil, &n1)
		p.From.Type = obj.TYPE_CONST
		p.From.Offset = int64(w)

	case gc.TUINT8,
		gc.TUINT16,
		gc.TUINT32:
		gins(optoas(gc.OMUL, t), &n2, &n1)
		p := gins(s390x.ASRD, nil, &n1)
		p.From.Type = obj.TYPE_CONST
		p.From.Offset = int64(w)

	case gc.TINT64:
		gins(s390x.AMULHD, &n2, &n1)

	case gc.TUINT64:
		gins(s390x.AMULHDU, &n2, &n1)

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
		gc.Nodconst(&n3, tcount, nl.Type.Width*8)
		gins(optoas(gc.OCMP, tcount), &n1, &n3)
		p1 := gc.Gbranch(optoas(gc.OLT, tcount), nil, 1)
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

// clearfat clears (i.e. replaces with zeros) the value pointed to by nl.
func clearfat(nl *gc.Node) {
	if gc.Debug['g'] != 0 {
		fmt.Printf("clearfat %v (%v, size: %d)\n", nl, nl.Type, nl.Type.Width)
	}

	// Avoid taking the address for simple enough types.
	if gc.Componentgen(nil, nl) {
		return
	}

	var dst gc.Node
	gc.Regalloc(&dst, gc.Types[gc.Tptr], nil)
	gc.Agen(nl, &dst)

	var boff int64
	w := nl.Type.Width
	if w > clearLoopCutoff {
		// Generate a loop clearing 256 bytes per iteration using XCs.
		var end gc.Node
		gc.Regalloc(&end, gc.Types[gc.Tptr], nil)
		p := gins(s390x.AMOVD, &dst, &end)
		p.From.Type = obj.TYPE_ADDR
		p.From.Offset = w - (w % 256)

		p = gins(s390x.AXC, &dst, &dst)
		p.From.Type = obj.TYPE_MEM
		p.From.Offset = 0
		p.To.Type = obj.TYPE_MEM
		p.To.Offset = 0
		p.From3 = new(obj.Addr)
		p.From3.Offset = 256
		p.From3.Type = obj.TYPE_CONST
		pl := p

		ginscon(s390x.AADD, 256, &dst)
		gins(s390x.ACMP, &dst, &end)
		gc.Patch(gc.Gbranch(s390x.ABNE, nil, 0), pl)
		gc.Regfree(&end)
		w = w % 256
	}

	// Generate instructions to clear the remaining memory.
	for w > 0 {
		n := w

		// Can clear at most 256 bytes per instruction.
		if n > 256 {
			n = 256
		}

		switch n {
		// Handle very small clears using moves.
		case 8, 4, 2, 1:
			ins := s390x.AMOVB
			switch n {
			case 8:
				ins = s390x.AMOVD
			case 4:
				ins = s390x.AMOVW
			case 2:
				ins = s390x.AMOVH
			}
			p := gins(ins, nil, &dst)
			p.From.Type = obj.TYPE_CONST
			p.From.Offset = 0
			p.To.Type = obj.TYPE_MEM
			p.To.Offset = boff

		// Handle clears that would require multiple moves with a XC.
		default:
			p := gins(s390x.AXC, &dst, &dst)
			p.From.Type = obj.TYPE_MEM
			p.From.Offset = boff
			p.To.Type = obj.TYPE_MEM
			p.To.Offset = boff
			p.From3 = new(obj.Addr)
			p.From3.Offset = n
			p.From3.Type = obj.TYPE_CONST
		}

		boff += n
		w -= n
	}

	gc.Regfree(&dst)
}

// Called after regopt and peep have run.
// Expand CHECKNIL pseudo-op into actual nil pointer check.
func expandchecks(firstp *obj.Prog) {
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
		//	CMPBNE arg, $0, 2(PC) [likely]
		//	MOVD   R0, 0(R0)
		p1 := gc.Ctxt.NewProg()

		gc.Clearp(p1)
		p1.Link = p.Link
		p.Link = p1
		p1.Lineno = p.Lineno
		p1.Pc = 9999
		p.As = s390x.ACMPBNE
		p.From3 = new(obj.Addr)
		p.From3.Type = obj.TYPE_CONST
		p.From3.Offset = 0

		p.To.Type = obj.TYPE_BRANCH
		p.To.Val = p1.Link

		// crash by write to memory address 0.
		p1.As = s390x.AMOVD

		p1.From.Type = obj.TYPE_REG
		p1.From.Reg = s390x.REGZERO
		p1.To.Type = obj.TYPE_MEM
		p1.To.Reg = s390x.REGZERO
		p1.To.Offset = 0
	}
}

// res = runtime.getg()
func getg(res *gc.Node) {
	var n1 gc.Node
	gc.Nodreg(&n1, res.Type, s390x.REGG)
	gmove(&n1, res)
}
