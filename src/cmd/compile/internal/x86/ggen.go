// Copyright 2009 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package x86

import (
	"cmd/compile/internal/gc"
	"cmd/internal/obj"
	"cmd/internal/obj/x86"
)

func defframe(ptxt *obj.Prog) {
	var n *gc.Node

	// fill in argument size, stack size
	ptxt.To.Type = obj.TYPE_TEXTSIZE

	ptxt.To.Val = int32(gc.Rnd(gc.Curfn.Type.Argwid, int64(gc.Widthptr)))
	frame := uint32(gc.Rnd(gc.Stksize+gc.Maxarg, int64(gc.Widthreg)))
	ptxt.To.Offset = int64(frame)

	// insert code to zero ambiguously live variables
	// so that the garbage collector only sees initialized values
	// when it looks for pointers.
	p := ptxt

	hi := int64(0)
	lo := hi
	ax := uint32(0)
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
		if lo != hi && n.Xoffset+n.Type.Width == lo-int64(2*gc.Widthptr) {
			// merge with range we already have
			lo = n.Xoffset

			continue
		}

		// zero old range
		p = zerorange(p, int64(frame), lo, hi, &ax)

		// set new range
		hi = n.Xoffset + n.Type.Width

		lo = n.Xoffset
	}

	// zero final range
	zerorange(p, int64(frame), lo, hi, &ax)
}

func zerorange(p *obj.Prog, frame int64, lo int64, hi int64, ax *uint32) *obj.Prog {
	cnt := hi - lo
	if cnt == 0 {
		return p
	}
	if *ax == 0 {
		p = appendpp(p, x86.AMOVL, obj.TYPE_CONST, 0, 0, obj.TYPE_REG, x86.REG_AX, 0)
		*ax = 1
	}

	if cnt <= int64(4*gc.Widthreg) {
		for i := int64(0); i < cnt; i += int64(gc.Widthreg) {
			p = appendpp(p, x86.AMOVL, obj.TYPE_REG, x86.REG_AX, 0, obj.TYPE_MEM, x86.REG_SP, frame+lo+i)
		}
	} else if !gc.Nacl && cnt <= int64(128*gc.Widthreg) {
		p = appendpp(p, x86.ALEAL, obj.TYPE_MEM, x86.REG_SP, frame+lo, obj.TYPE_REG, x86.REG_DI, 0)
		p = appendpp(p, obj.ADUFFZERO, obj.TYPE_NONE, 0, 0, obj.TYPE_ADDR, 0, 1*(128-cnt/int64(gc.Widthreg)))
		p.To.Sym = gc.Linksym(gc.Pkglookup("duffzero", gc.Runtimepkg))
	} else {
		p = appendpp(p, x86.AMOVL, obj.TYPE_CONST, 0, cnt/int64(gc.Widthreg), obj.TYPE_REG, x86.REG_CX, 0)
		p = appendpp(p, x86.ALEAL, obj.TYPE_MEM, x86.REG_SP, frame+lo, obj.TYPE_REG, x86.REG_DI, 0)
		p = appendpp(p, x86.AREP, obj.TYPE_NONE, 0, 0, obj.TYPE_NONE, 0, 0)
		p = appendpp(p, x86.ASTOSL, obj.TYPE_NONE, 0, 0, obj.TYPE_NONE, 0, 0)
	}

	return p
}

func appendpp(p *obj.Prog, as int, ftype int, freg int, foffset int64, ttype int, treg int, toffset int64) *obj.Prog {
	q := gc.Ctxt.NewProg()
	gc.Clearp(q)
	q.As = int16(as)
	q.Lineno = p.Lineno
	q.From.Type = int16(ftype)
	q.From.Reg = int16(freg)
	q.From.Offset = foffset
	q.To.Type = int16(ttype)
	q.To.Reg = int16(treg)
	q.To.Offset = toffset
	q.Link = p.Link
	p.Link = q
	return q
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

	if q < 4 {
		// Write sequence of MOV 0, off(base) instead of using STOSL.
		// The hope is that although the code will be slightly longer,
		// the MOVs will have no dependencies and pipeline better
		// than the unrolled STOSL loop.
		// NOTE: Must use agen, not igen, so that optimizer sees address
		// being taken. We are not writing on field boundaries.
		var n1 gc.Node
		gc.Regalloc(&n1, gc.Types[gc.Tptr], nil)

		gc.Agen(nl, &n1)
		n1.Op = gc.OINDREG
		var z gc.Node
		gc.Nodconst(&z, gc.Types[gc.TUINT64], 0)
		for ; q > 0; q-- {
			n1.Type = z.Type
			gins(x86.AMOVL, &z, &n1)
			n1.Xoffset += 4
		}

		gc.Nodconst(&z, gc.Types[gc.TUINT8], 0)
		for ; c > 0; c-- {
			n1.Type = z.Type
			gins(x86.AMOVB, &z, &n1)
			n1.Xoffset++
		}

		gc.Regfree(&n1)
		return
	}

	var n1 gc.Node
	gc.Nodreg(&n1, gc.Types[gc.Tptr], x86.REG_DI)
	gc.Agen(nl, &n1)
	gconreg(x86.AMOVL, 0, x86.REG_AX)

	if q > 128 || (q >= 4 && gc.Nacl) {
		gconreg(x86.AMOVL, int64(q), x86.REG_CX)
		gins(x86.AREP, nil, nil)   // repeat
		gins(x86.ASTOSL, nil, nil) // STOL AL,*(DI)+
	} else if q >= 4 {
		p := gins(obj.ADUFFZERO, nil, nil)
		p.To.Type = obj.TYPE_ADDR
		p.To.Sym = gc.Linksym(gc.Pkglookup("duffzero", gc.Runtimepkg))

		// 1 and 128 = magic constants: see ../../runtime/asm_386.s
		p.To.Offset = 1 * (128 - int64(q))
	} else {
		for q > 0 {
			gins(x86.ASTOSL, nil, nil) // STOL AL,*(DI)+
			q--
		}
	}

	for c > 0 {
		gins(x86.ASTOSB, nil, nil) // STOB AL,*(DI)+
		c--
	}
}

var panicdiv *gc.Node

/*
 * generate division.
 * caller must set:
 *	ax = allocated AX register
 *	dx = allocated DX register
 * generates one of:
 *	res = nl / nr
 *	res = nl % nr
 * according to op.
 */
func dodiv(op gc.Op, nl *gc.Node, nr *gc.Node, res *gc.Node, ax *gc.Node, dx *gc.Node) {
	// Have to be careful about handling
	// most negative int divided by -1 correctly.
	// The hardware will trap.
	// Also the byte divide instruction needs AH,
	// which we otherwise don't have to deal with.
	// Easiest way to avoid for int8, int16: use int32.
	// For int32 and int64, use explicit test.
	// Could use int64 hw for int32.
	t := nl.Type

	t0 := t
	check := false
	if gc.Issigned[t.Etype] {
		check = true
		if gc.Isconst(nl, gc.CTINT) && nl.Int() != -1<<uint64(t.Width*8-1) {
			check = false
		} else if gc.Isconst(nr, gc.CTINT) && nr.Int() != -1 {
			check = false
		}
	}

	if t.Width < 4 {
		if gc.Issigned[t.Etype] {
			t = gc.Types[gc.TINT32]
		} else {
			t = gc.Types[gc.TUINT32]
		}
		check = false
	}

	var t1 gc.Node
	gc.Tempname(&t1, t)
	var t2 gc.Node
	gc.Tempname(&t2, t)
	if t0 != t {
		var t3 gc.Node
		gc.Tempname(&t3, t0)
		var t4 gc.Node
		gc.Tempname(&t4, t0)
		gc.Cgen(nl, &t3)
		gc.Cgen(nr, &t4)

		// Convert.
		gmove(&t3, &t1)

		gmove(&t4, &t2)
	} else {
		gc.Cgen(nl, &t1)
		gc.Cgen(nr, &t2)
	}

	var n1 gc.Node
	if !gc.Samereg(ax, res) && !gc.Samereg(dx, res) {
		gc.Regalloc(&n1, t, res)
	} else {
		gc.Regalloc(&n1, t, nil)
	}
	gmove(&t2, &n1)
	gmove(&t1, ax)
	var p2 *obj.Prog
	var n4 gc.Node
	if gc.Nacl {
		// Native Client does not relay the divide-by-zero trap
		// to the executing program, so we must insert a check
		// for ourselves.
		gc.Nodconst(&n4, t, 0)

		gins(optoas(gc.OCMP, t), &n1, &n4)
		p1 := gc.Gbranch(optoas(gc.ONE, t), nil, +1)
		if panicdiv == nil {
			panicdiv = gc.Sysfunc("panicdivide")
		}
		gc.Ginscall(panicdiv, -1)
		gc.Patch(p1, gc.Pc)
	}

	if check {
		gc.Nodconst(&n4, t, -1)
		gins(optoas(gc.OCMP, t), &n1, &n4)
		p1 := gc.Gbranch(optoas(gc.ONE, t), nil, +1)
		if op == gc.ODIV {
			// a / (-1) is -a.
			gins(optoas(gc.OMINUS, t), nil, ax)

			gmove(ax, res)
		} else {
			// a % (-1) is 0.
			gc.Nodconst(&n4, t, 0)

			gmove(&n4, res)
		}

		p2 = gc.Gbranch(obj.AJMP, nil, 0)
		gc.Patch(p1, gc.Pc)
	}

	if !gc.Issigned[t.Etype] {
		var nz gc.Node
		gc.Nodconst(&nz, t, 0)
		gmove(&nz, dx)
	} else {
		gins(optoas(gc.OEXTEND, t), nil, nil)
	}
	gins(optoas(op, t), &n1, nil)
	gc.Regfree(&n1)

	if op == gc.ODIV {
		gmove(ax, res)
	} else {
		gmove(dx, res)
	}
	if check {
		gc.Patch(p2, gc.Pc)
	}
}

func savex(dr int, x *gc.Node, oldx *gc.Node, res *gc.Node, t *gc.Type) {
	r := gc.GetReg(dr)
	gc.Nodreg(x, gc.Types[gc.TINT32], dr)

	// save current ax and dx if they are live
	// and not the destination
	*oldx = gc.Node{}

	if r > 0 && !gc.Samereg(x, res) {
		gc.Tempname(oldx, gc.Types[gc.TINT32])
		gmove(x, oldx)
	}

	gc.Regalloc(x, t, x)
}

func restx(x *gc.Node, oldx *gc.Node) {
	gc.Regfree(x)

	if oldx.Op != 0 {
		x.Type = gc.Types[gc.TINT32]
		gmove(oldx, x)
	}
}

/*
 * generate division according to op, one of:
 *	res = nl / nr
 *	res = nl % nr
 */
func cgen_div(op gc.Op, nl *gc.Node, nr *gc.Node, res *gc.Node) {
	if gc.Is64(nl.Type) {
		gc.Fatalf("cgen_div %v", nl.Type)
	}

	var t *gc.Type
	if gc.Issigned[nl.Type.Etype] {
		t = gc.Types[gc.TINT32]
	} else {
		t = gc.Types[gc.TUINT32]
	}
	var ax gc.Node
	var oldax gc.Node
	savex(x86.REG_AX, &ax, &oldax, res, t)
	var olddx gc.Node
	var dx gc.Node
	savex(x86.REG_DX, &dx, &olddx, res, t)
	dodiv(op, nl, nr, res, &ax, &dx)
	restx(&dx, &olddx)
	restx(&ax, &oldax)
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

	a := optoas(op, nl.Type)

	if nr.Op == gc.OLITERAL {
		var n2 gc.Node
		gc.Tempname(&n2, nl.Type)
		gc.Cgen(nl, &n2)
		var n1 gc.Node
		gc.Regalloc(&n1, nl.Type, res)
		gmove(&n2, &n1)
		sc := uint64(nr.Int())
		if sc >= uint64(nl.Type.Width*8) {
			// large shift gets 2 shifts by width-1
			gins(a, ncon(uint32(w)-1), &n1)

			gins(a, ncon(uint32(w)-1), &n1)
		} else {
			gins(a, nr, &n1)
		}
		gmove(&n1, res)
		gc.Regfree(&n1)
		return
	}

	var oldcx gc.Node
	var cx gc.Node
	gc.Nodreg(&cx, gc.Types[gc.TUINT32], x86.REG_CX)
	if gc.GetReg(x86.REG_CX) > 1 && !gc.Samereg(&cx, res) {
		gc.Tempname(&oldcx, gc.Types[gc.TUINT32])
		gmove(&cx, &oldcx)
	}

	var n1 gc.Node
	var nt gc.Node
	if nr.Type.Width > 4 {
		gc.Tempname(&nt, nr.Type)
		n1 = nt
	} else {
		gc.Nodreg(&n1, gc.Types[gc.TUINT32], x86.REG_CX)
		gc.Regalloc(&n1, nr.Type, &n1) // to hold the shift type in CX
	}

	var n2 gc.Node
	if gc.Samereg(&cx, res) {
		gc.Regalloc(&n2, nl.Type, nil)
	} else {
		gc.Regalloc(&n2, nl.Type, res)
	}
	if nl.Ullman >= nr.Ullman {
		gc.Cgen(nl, &n2)
		gc.Cgen(nr, &n1)
	} else {
		gc.Cgen(nr, &n1)
		gc.Cgen(nl, &n2)
	}

	// test and fix up large shifts
	if bounded {
		if nr.Type.Width > 4 {
			// delayed reg alloc
			gc.Nodreg(&n1, gc.Types[gc.TUINT32], x86.REG_CX)

			gc.Regalloc(&n1, gc.Types[gc.TUINT32], &n1) // to hold the shift type in CX
			var lo gc.Node
			var hi gc.Node
			split64(&nt, &lo, &hi)
			gmove(&lo, &n1)
			splitclean()
		}
	} else {
		var p1 *obj.Prog
		if nr.Type.Width > 4 {
			// delayed reg alloc
			gc.Nodreg(&n1, gc.Types[gc.TUINT32], x86.REG_CX)

			gc.Regalloc(&n1, gc.Types[gc.TUINT32], &n1) // to hold the shift type in CX
			var lo gc.Node
			var hi gc.Node
			split64(&nt, &lo, &hi)
			gmove(&lo, &n1)
			gins(optoas(gc.OCMP, gc.Types[gc.TUINT32]), &hi, ncon(0))
			p2 := gc.Gbranch(optoas(gc.ONE, gc.Types[gc.TUINT32]), nil, +1)
			gins(optoas(gc.OCMP, gc.Types[gc.TUINT32]), &n1, ncon(uint32(w)))
			p1 = gc.Gbranch(optoas(gc.OLT, gc.Types[gc.TUINT32]), nil, +1)
			splitclean()
			gc.Patch(p2, gc.Pc)
		} else {
			gins(optoas(gc.OCMP, nr.Type), &n1, ncon(uint32(w)))
			p1 = gc.Gbranch(optoas(gc.OLT, gc.Types[gc.TUINT32]), nil, +1)
		}

		if op == gc.ORSH && gc.Issigned[nl.Type.Etype] {
			gins(a, ncon(uint32(w)-1), &n2)
		} else {
			gmove(ncon(0), &n2)
		}

		gc.Patch(p1, gc.Pc)
	}

	gins(a, &n1, &n2)

	if oldcx.Op != 0 {
		gmove(&oldcx, &cx)
	}

	gmove(&n2, res)

	gc.Regfree(&n1)
	gc.Regfree(&n2)
}

/*
 * generate byte multiply:
 *	res = nl * nr
 * there is no 2-operand byte multiply instruction so
 * we do a full-width multiplication and truncate afterwards.
 */
func cgen_bmul(op gc.Op, nl *gc.Node, nr *gc.Node, res *gc.Node) bool {
	if optoas(op, nl.Type) != x86.AIMULB {
		return false
	}

	// copy from byte to full registers
	t := gc.Types[gc.TUINT32]

	if gc.Issigned[nl.Type.Etype] {
		t = gc.Types[gc.TINT32]
	}

	// largest ullman on left.
	if nl.Ullman < nr.Ullman {
		nl, nr = nr, nl
	}

	var nt gc.Node
	gc.Tempname(&nt, nl.Type)
	gc.Cgen(nl, &nt)
	var n1 gc.Node
	gc.Regalloc(&n1, t, res)
	gc.Cgen(nr, &n1)
	var n2 gc.Node
	gc.Regalloc(&n2, t, nil)
	gmove(&nt, &n2)
	a := optoas(op, t)
	gins(a, &n2, &n1)
	gc.Regfree(&n2)
	gmove(&n1, res)
	gc.Regfree(&n1)

	return true
}

/*
 * generate high multiply:
 *   res = (nl*nr) >> width
 */
func cgen_hmul(nl *gc.Node, nr *gc.Node, res *gc.Node) {
	var n1 gc.Node
	var n2 gc.Node

	t := nl.Type
	a := optoas(gc.OHMUL, t)

	// gen nl in n1.
	gc.Tempname(&n1, t)
	gc.Cgen(nl, &n1)

	// gen nr in n2.
	gc.Regalloc(&n2, t, res)
	gc.Cgen(nr, &n2)

	var ax, oldax, dx, olddx gc.Node
	savex(x86.REG_AX, &ax, &oldax, res, gc.Types[gc.TUINT32])
	savex(x86.REG_DX, &dx, &olddx, res, gc.Types[gc.TUINT32])

	gmove(&n2, &ax)
	gins(a, &n1, nil)
	gc.Regfree(&n2)

	if t.Width == 1 {
		// byte multiply behaves differently.
		var byteAH, byteDX gc.Node
		gc.Nodreg(&byteAH, t, x86.REG_AH)
		gc.Nodreg(&byteDX, t, x86.REG_DX)
		gmove(&byteAH, &byteDX)
	}

	gmove(&dx, res)

	restx(&ax, &oldax)
	restx(&dx, &olddx)
}

/*
 * generate floating-point operation.
 */
func cgen_float(n *gc.Node, res *gc.Node) {
	nl := n.Left
	switch n.Op {
	case gc.OEQ,
		gc.ONE,
		gc.OLT,
		gc.OLE,
		gc.OGE:
		p1 := gc.Gbranch(obj.AJMP, nil, 0)
		p2 := gc.Pc
		gmove(gc.Nodbool(true), res)
		p3 := gc.Gbranch(obj.AJMP, nil, 0)
		gc.Patch(p1, gc.Pc)
		gc.Bgen(n, true, 0, p2)
		gmove(gc.Nodbool(false), res)
		gc.Patch(p3, gc.Pc)
		return

	case gc.OPLUS:
		gc.Cgen(nl, res)
		return

	case gc.OCONV:
		if gc.Eqtype(n.Type, nl.Type) || gc.Noconv(n.Type, nl.Type) {
			gc.Cgen(nl, res)
			return
		}

		var n2 gc.Node
		gc.Tempname(&n2, n.Type)
		var n1 gc.Node
		gc.Mgen(nl, &n1, res)
		gmove(&n1, &n2)
		gmove(&n2, res)
		gc.Mfree(&n1)
		return
	}

	if gc.Thearch.Use387 {
		cgen_float387(n, res)
	} else {
		cgen_floatsse(n, res)
	}
}

// floating-point.  387 (not SSE2)
func cgen_float387(n *gc.Node, res *gc.Node) {
	var f0 gc.Node
	var f1 gc.Node

	nl := n.Left
	nr := n.Right
	gc.Nodreg(&f0, nl.Type, x86.REG_F0)
	gc.Nodreg(&f1, n.Type, x86.REG_F0+1)
	if nr != nil {
		// binary
		if nl.Ullman >= nr.Ullman {
			gc.Cgen(nl, &f0)
			if nr.Addable {
				gins(foptoas(n.Op, n.Type, 0), nr, &f0)
			} else {
				gc.Cgen(nr, &f0)
				gins(foptoas(n.Op, n.Type, Fpop), &f0, &f1)
			}
		} else {
			gc.Cgen(nr, &f0)
			if nl.Addable {
				gins(foptoas(n.Op, n.Type, Frev), nl, &f0)
			} else {
				gc.Cgen(nl, &f0)
				gins(foptoas(n.Op, n.Type, Frev|Fpop), &f0, &f1)
			}
		}

		gmove(&f0, res)
		return
	}

	// unary
	gc.Cgen(nl, &f0)

	if n.Op != gc.OCONV && n.Op != gc.OPLUS {
		gins(foptoas(n.Op, n.Type, 0), nil, nil)
	}
	gmove(&f0, res)
	return
}

func cgen_floatsse(n *gc.Node, res *gc.Node) {
	var a int

	nl := n.Left
	nr := n.Right
	switch n.Op {
	default:
		gc.Dump("cgen_floatsse", n)
		gc.Fatalf("cgen_floatsse %v", gc.Oconv(int(n.Op), 0))
		return

	case gc.OMINUS,
		gc.OCOM:
		nr = gc.Nodintconst(-1)
		gc.Convlit(&nr, n.Type)
		a = foptoas(gc.OMUL, nl.Type, 0)
		goto sbop

		// symmetric binary
	case gc.OADD,
		gc.OMUL:
		a = foptoas(n.Op, nl.Type, 0)

		goto sbop

		// asymmetric binary
	case gc.OSUB,
		gc.OMOD,
		gc.ODIV:
		a = foptoas(n.Op, nl.Type, 0)

		goto abop
	}

sbop: // symmetric binary
	if nl.Ullman < nr.Ullman || nl.Op == gc.OLITERAL {
		nl, nr = nr, nl
	}

abop: // asymmetric binary
	if nl.Ullman >= nr.Ullman {
		var nt gc.Node
		gc.Tempname(&nt, nl.Type)
		gc.Cgen(nl, &nt)
		var n2 gc.Node
		gc.Mgen(nr, &n2, nil)
		var n1 gc.Node
		gc.Regalloc(&n1, nl.Type, res)
		gmove(&nt, &n1)
		gins(a, &n2, &n1)
		gmove(&n1, res)
		gc.Regfree(&n1)
		gc.Mfree(&n2)
	} else {
		var n2 gc.Node
		gc.Regalloc(&n2, nr.Type, res)
		gc.Cgen(nr, &n2)
		var n1 gc.Node
		gc.Regalloc(&n1, nl.Type, nil)
		gc.Cgen(nl, &n1)
		gins(a, &n2, &n1)
		gc.Regfree(&n2)
		gmove(&n1, res)
		gc.Regfree(&n1)
	}

	return
}

func bgen_float(n *gc.Node, wantTrue bool, likely int, to *obj.Prog) {
	nl := n.Left
	nr := n.Right
	op := n.Op
	if !wantTrue {
		// brcom is not valid on floats when NaN is involved.
		p1 := gc.Gbranch(obj.AJMP, nil, 0)
		p2 := gc.Gbranch(obj.AJMP, nil, 0)
		gc.Patch(p1, gc.Pc)

		// No need to avoid re-genning ninit.
		bgen_float(n, true, -likely, p2)

		gc.Patch(gc.Gbranch(obj.AJMP, nil, 0), to)
		gc.Patch(p2, gc.Pc)
		return
	}

	if gc.Thearch.Use387 {
		op = gc.Brrev(op) // because the args are stacked
		if op == gc.OGE || op == gc.OGT {
			// only < and <= work right with NaN; reverse if needed
			nl, nr = nr, nl
			op = gc.Brrev(op)
		}

		var ax, n2, tmp gc.Node
		gc.Nodreg(&tmp, nr.Type, x86.REG_F0)
		gc.Nodreg(&n2, nr.Type, x86.REG_F0+1)
		gc.Nodreg(&ax, gc.Types[gc.TUINT16], x86.REG_AX)
		if gc.Simsimtype(nr.Type) == gc.TFLOAT64 {
			if nl.Ullman > nr.Ullman {
				gc.Cgen(nl, &tmp)
				gc.Cgen(nr, &tmp)
				gins(x86.AFXCHD, &tmp, &n2)
			} else {
				gc.Cgen(nr, &tmp)
				gc.Cgen(nl, &tmp)
			}
			gins(x86.AFUCOMPP, &tmp, &n2)
		} else {
			// TODO(rsc): The moves back and forth to memory
			// here are for truncating the value to 32 bits.
			// This handles 32-bit comparison but presumably
			// all the other ops have the same problem.
			// We need to figure out what the right general
			// solution is, besides telling people to use float64.
			var t1 gc.Node
			gc.Tempname(&t1, gc.Types[gc.TFLOAT32])

			var t2 gc.Node
			gc.Tempname(&t2, gc.Types[gc.TFLOAT32])
			gc.Cgen(nr, &t1)
			gc.Cgen(nl, &t2)
			gmove(&t2, &tmp)
			gins(x86.AFCOMFP, &t1, &tmp)
		}
		gins(x86.AFSTSW, nil, &ax)
		gins(x86.ASAHF, nil, nil)
	} else {
		// Not 387
		if !nl.Addable {
			nl = gc.CgenTemp(nl)
		}
		if !nr.Addable {
			nr = gc.CgenTemp(nr)
		}

		var n2 gc.Node
		gc.Regalloc(&n2, nr.Type, nil)
		gmove(nr, &n2)
		nr = &n2

		if nl.Op != gc.OREGISTER {
			var n3 gc.Node
			gc.Regalloc(&n3, nl.Type, nil)
			gmove(nl, &n3)
			nl = &n3
		}

		if op == gc.OGE || op == gc.OGT {
			// only < and <= work right with NopN; reverse if needed
			nl, nr = nr, nl
			op = gc.Brrev(op)
		}

		gins(foptoas(gc.OCMP, nr.Type, 0), nl, nr)
		if nl.Op == gc.OREGISTER {
			gc.Regfree(nl)
		}
		gc.Regfree(nr)
	}

	switch op {
	case gc.OEQ:
		// neither NE nor P
		p1 := gc.Gbranch(x86.AJNE, nil, -likely)
		p2 := gc.Gbranch(x86.AJPS, nil, -likely)
		gc.Patch(gc.Gbranch(obj.AJMP, nil, 0), to)
		gc.Patch(p1, gc.Pc)
		gc.Patch(p2, gc.Pc)
	case gc.ONE:
		// either NE or P
		gc.Patch(gc.Gbranch(x86.AJNE, nil, likely), to)
		gc.Patch(gc.Gbranch(x86.AJPS, nil, likely), to)
	default:
		gc.Patch(gc.Gbranch(optoas(op, nr.Type), nil, likely), to)
	}
}

// Called after regopt and peep have run.
// Expand CHECKNIL pseudo-op into actual nil pointer check.
func expandchecks(firstp *obj.Prog) {
	var p1 *obj.Prog
	var p2 *obj.Prog

	for p := firstp; p != nil; p = p.Link {
		if p.As != obj.ACHECKNIL {
			continue
		}
		if gc.Debug_checknil != 0 && p.Lineno > 1 { // p->lineno==1 in generated wrappers
			gc.Warnl(int(p.Lineno), "generated nil check")
		}

		// check is
		//	CMP arg, $0
		//	JNE 2(PC) (likely)
		//	MOV AX, 0
		p1 = gc.Ctxt.NewProg()

		p2 = gc.Ctxt.NewProg()
		gc.Clearp(p1)
		gc.Clearp(p2)
		p1.Link = p2
		p2.Link = p.Link
		p.Link = p1
		p1.Lineno = p.Lineno
		p2.Lineno = p.Lineno
		p1.Pc = 9999
		p2.Pc = 9999
		p.As = x86.ACMPL
		p.To.Type = obj.TYPE_CONST
		p.To.Offset = 0
		p1.As = x86.AJNE
		p1.From.Type = obj.TYPE_CONST
		p1.From.Offset = 1 // likely
		p1.To.Type = obj.TYPE_BRANCH
		p1.To.Val = p2.Link

		// crash by write to memory address 0.
		// if possible, since we know arg is 0, use 0(arg),
		// which will be shorter to encode than plain 0.
		p2.As = x86.AMOVL

		p2.From.Type = obj.TYPE_REG
		p2.From.Reg = x86.REG_AX
		if regtyp(&p.From) {
			p2.To.Type = obj.TYPE_MEM
			p2.To.Reg = p.From.Reg
		} else {
			p2.To.Type = obj.TYPE_MEM
		}
		p2.To.Offset = 0
	}
}

// addr += index*width if possible.
func addindex(index *gc.Node, width int64, addr *gc.Node) bool {
	switch width {
	case 1, 2, 4, 8:
		p1 := gins(x86.ALEAL, index, addr)
		p1.From.Type = obj.TYPE_MEM
		p1.From.Scale = int16(width)
		p1.From.Index = p1.From.Reg
		p1.From.Reg = p1.To.Reg
		return true
	}
	return false
}

// res = runtime.getg()
func getg(res *gc.Node) {
	var n1 gc.Node
	gc.Regalloc(&n1, res.Type, res)
	mov := optoas(gc.OAS, gc.Types[gc.Tptr])
	p := gins(mov, nil, &n1)
	p.From.Type = obj.TYPE_REG
	p.From.Reg = x86.REG_TLS
	p = gins(mov, nil, &n1)
	p.From = p.To
	p.From.Type = obj.TYPE_MEM
	p.From.Index = x86.REG_TLS
	p.From.Scale = 1
	gmove(&n1, res)
	gc.Regfree(&n1)
}
