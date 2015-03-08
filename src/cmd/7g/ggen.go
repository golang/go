// Copyright 2009 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package main

import (
	"cmd/internal/gc"
	"cmd/internal/obj"
	"cmd/internal/obj/arm64"
	"fmt"
)

func defframe(ptxt *obj.Prog) {
	var n *gc.Node

	// fill in argument size, stack size
	ptxt.To.Type = obj.TYPE_TEXTSIZE

	ptxt.To.U.Argsize = int32(gc.Rnd(gc.Curfn.Type.Argwid, int64(gc.Widthptr)))
	frame := uint32(gc.Rnd(gc.Stksize+gc.Maxarg, int64(gc.Widthreg)))
	ptxt.To.Offset = int64(frame)

	// insert code to zero ambiguously live variables
	// so that the garbage collector only sees initialized values
	// when it looks for pointers.
	p := ptxt

	hi := int64(0)
	lo := hi

	// iterate through declarations - they are sorted in decreasing xoffset order.
	for l := gc.Curfn.Dcl; l != nil; l = l.Next {
		n = l.N
		if !n.Needzero {
			continue
		}
		if n.Class != gc.PAUTO {
			gc.Fatal("needzero class %d", n.Class)
		}
		if n.Type.Width%int64(gc.Widthptr) != 0 || n.Xoffset%int64(gc.Widthptr) != 0 || n.Type.Width == 0 {
			gc.Fatal("var %v has size %d offset %d", gc.Nconv(n, obj.FmtLong), int(n.Type.Width), int(n.Xoffset))
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
			p = appendpp(p, arm64.AMOVD, obj.TYPE_REG, arm64.REGZERO, 0, obj.TYPE_MEM, arm64.REGSP, 8+frame+lo+i)
		}
	} else if cnt <= int64(128*gc.Widthptr) {
		p = appendpp(p, arm64.AMOVD, obj.TYPE_REG, arm64.REGSP, 0, obj.TYPE_REG, arm64.REGRT1, 0)
		p = appendpp(p, arm64.AADD, obj.TYPE_CONST, 0, 8+frame+lo-8, obj.TYPE_REG, arm64.REGRT1, 0)
		p.Reg = arm64.REGRT1
		p = appendpp(p, obj.ADUFFZERO, obj.TYPE_NONE, 0, 0, obj.TYPE_MEM, 0, 0)
		f := gc.Sysfunc("duffzero")
		p.To = gc.Naddr(f)
		gc.Afunclit(&p.To, f)
		p.To.Offset = 4 * (128 - cnt/int64(gc.Widthptr))
	} else {
		p = appendpp(p, arm64.AMOVD, obj.TYPE_CONST, 0, 8+frame+lo-8, obj.TYPE_REG, arm64.REGTMP, 0)
		p = appendpp(p, arm64.AMOVD, obj.TYPE_REG, arm64.REGSP, 0, obj.TYPE_REG, arm64.REGRT1, 0)
		p = appendpp(p, arm64.AADD, obj.TYPE_REG, arm64.REGTMP, 0, obj.TYPE_REG, arm64.REGRT1, 0)
		p.Reg = arm64.REGRT1
		p = appendpp(p, arm64.AMOVD, obj.TYPE_CONST, 0, cnt, obj.TYPE_REG, arm64.REGTMP, 0)
		p = appendpp(p, arm64.AADD, obj.TYPE_REG, arm64.REGTMP, 0, obj.TYPE_REG, arm64.REGRT2, 0)
		p.Reg = arm64.REGRT1
		p = appendpp(p, arm64.AMOVD, obj.TYPE_REG, arm64.REGZERO, 0, obj.TYPE_MEM, arm64.REGRT1, int64(gc.Widthptr))
		p.Scond = arm64.C_XPRE
		p1 := p
		p = appendpp(p, arm64.ACMP, obj.TYPE_REG, arm64.REGRT1, 0, obj.TYPE_NONE, 0, 0)
		p.Reg = arm64.REGRT2
		p = appendpp(p, arm64.ABNE, obj.TYPE_NONE, 0, 0, obj.TYPE_BRANCH, 0, 0)
		gc.Patch(p, p1)
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

/*
 * generate:
 *	call f
 *	proc=-1	normal call but no return
 *	proc=0	normal call
 *	proc=1	goroutine run in new proc
 *	proc=2	defer call save away stack
  *	proc=3	normal call to C pointer (not Go func value)
*/
func ginscall(f *gc.Node, proc int) {
	if f.Type != nil {
		extra := int32(0)
		if proc == 1 || proc == 2 {
			extra = 2 * int32(gc.Widthptr)
		}
		gc.Setmaxarg(f.Type, extra)
	}

	switch proc {
	default:
		gc.Fatal("ginscall: bad proc %d", proc)

	case 0, // normal call
		-1: // normal call but no return
		if f.Op == gc.ONAME && f.Class == gc.PFUNC {
			if f == gc.Deferreturn {
				// Deferred calls will appear to be returning to
				// the CALL deferreturn(SB) that we are about to emit.
				// However, the stack trace code will show the line
				// of the instruction byte before the return PC.
				// To avoid that being an unrelated instruction,
				// insert a arm64 NOP that we will have the right line number.
				// The arm64 NOP is really or HINT $0; use that description
				// because the NOP pseudo-instruction would be removed by
				// the linker.
				var con gc.Node
				gc.Nodconst(&con, gc.Types[gc.TINT], 0)
				gins(arm64.AHINT, &con, nil)
			}

			p := gins(arm64.ABL, nil, f)
			gc.Afunclit(&p.To, f)
			if proc == -1 || gc.Noreturn(p) {
				gins(obj.AUNDEF, nil, nil)
			}
			break
		}

		var reg gc.Node
		gc.Nodreg(&reg, gc.Types[gc.Tptr], arm64.REGCTXT)
		var r1 gc.Node
		gc.Nodreg(&r1, gc.Types[gc.Tptr], arm64.REGRT1)
		gmove(f, &reg)
		reg.Op = gc.OINDREG
		gmove(&reg, &r1)
		r1.Op = gc.OINDREG
		gins(arm64.ABL, nil, &r1)

	case 3: // normal call of c function pointer
		gins(arm64.ABL, nil, f)

	case 1, // call in new proc (go)
		2: // deferred call (defer)
		var con gc.Node
		gc.Nodconst(&con, gc.Types[gc.TINT64], int64(gc.Argsize(f.Type)))

		var reg gc.Node
		gc.Nodreg(&reg, gc.Types[gc.TINT64], arm64.REGRT1)
		var reg2 gc.Node
		gc.Nodreg(&reg2, gc.Types[gc.TINT64], arm64.REGRT2)
		gmove(f, &reg)

		gmove(&con, &reg2)
		p := gins(arm64.AMOVW, &reg2, nil)
		p.To.Type = obj.TYPE_MEM
		p.To.Reg = arm64.REGSP
		p.To.Offset = 8

		p = gins(arm64.AMOVD, &reg, nil)
		p.To.Type = obj.TYPE_MEM
		p.To.Reg = arm64.REGSP
		p.To.Offset = 16

		if proc == 1 {
			ginscall(gc.Newproc, 0)
		} else {
			if gc.Hasdefer == 0 {
				gc.Fatal("hasdefer=0 but has defer")
			}
			ginscall(gc.Deferproc, 0)
		}

		if proc == 2 {
			gc.Nodreg(&reg, gc.Types[gc.TINT64], arm64.REG_R0) // R0 should match runtime.return0
			p := gins(arm64.ACMP, &reg, nil)
			p.Reg = arm64.REGZERO
			p = gc.Gbranch(arm64.ABEQ, nil, +1)
			cgen_ret(nil)
			gc.Patch(p, gc.Pc)
		}
	}
}

/*
 * n is call to interface method.
 * generate res = n.
 */
func cgen_callinter(n *gc.Node, res *gc.Node, proc int) {
	i := n.Left
	if i.Op != gc.ODOTINTER {
		gc.Fatal("cgen_callinter: not ODOTINTER %v", gc.Oconv(int(i.Op), 0))
	}

	f := i.Right // field
	if f.Op != gc.ONAME {
		gc.Fatal("cgen_callinter: not ONAME %v", gc.Oconv(int(f.Op), 0))
	}

	i = i.Left // interface

	if i.Addable == 0 {
		var tmpi gc.Node
		gc.Tempname(&tmpi, i.Type)
		cgen(i, &tmpi)
		i = &tmpi
	}

	gc.Genlist(n.List) // assign the args

	// i is now addable, prepare an indirected
	// register to hold its address.
	var nodi gc.Node
	igen(i, &nodi, res) // REG = &inter

	var nodsp gc.Node
	gc.Nodindreg(&nodsp, gc.Types[gc.Tptr], arm64.REGSP)

	nodsp.Xoffset = int64(gc.Widthptr)
	if proc != 0 {
		nodsp.Xoffset += 2 * int64(gc.Widthptr) // leave room for size & fn
	}
	nodi.Type = gc.Types[gc.Tptr]
	nodi.Xoffset += int64(gc.Widthptr)
	cgen(&nodi, &nodsp) // {8 or 24}(SP) = 8(REG) -- i.data

	var nodo gc.Node
	regalloc(&nodo, gc.Types[gc.Tptr], res)

	nodi.Type = gc.Types[gc.Tptr]
	nodi.Xoffset -= int64(gc.Widthptr)
	cgen(&nodi, &nodo) // REG = 0(REG) -- i.tab
	regfree(&nodi)

	var nodr gc.Node
	regalloc(&nodr, gc.Types[gc.Tptr], &nodo)
	if n.Left.Xoffset == gc.BADWIDTH {
		gc.Fatal("cgen_callinter: badwidth")
	}
	gc.Cgen_checknil(&nodo) // in case offset is huge
	nodo.Op = gc.OINDREG
	nodo.Xoffset = n.Left.Xoffset + 3*int64(gc.Widthptr) + 8
	if proc == 0 {
		// plain call: use direct c function pointer - more efficient
		cgen(&nodo, &nodr) // REG = 32+offset(REG) -- i.tab->fun[f]
		proc = 3
	} else {
		// go/defer. generate go func value.
		p := gins(arm64.AMOVD, &nodo, &nodr) // REG = &(32+offset(REG)) -- i.tab->fun[f]
		p.From.Type = obj.TYPE_ADDR
	}

	nodr.Type = n.Left.Type
	ginscall(&nodr, proc)

	regfree(&nodr)
	regfree(&nodo)
}

/*
 * generate function call;
 *	proc=0	normal call
 *	proc=1	goroutine run in new proc
 *	proc=2	defer call save away stack
 */
func cgen_call(n *gc.Node, proc int) {
	if n == nil {
		return
	}

	var afun gc.Node
	if n.Left.Ullman >= gc.UINF {
		// if name involves a fn call
		// precompute the address of the fn
		gc.Tempname(&afun, gc.Types[gc.Tptr])

		cgen(n.Left, &afun)
	}

	gc.Genlist(n.List) // assign the args
	t := n.Left.Type

	// call tempname pointer
	if n.Left.Ullman >= gc.UINF {
		var nod gc.Node
		regalloc(&nod, gc.Types[gc.Tptr], nil)
		gc.Cgen_as(&nod, &afun)
		nod.Type = t
		ginscall(&nod, proc)
		regfree(&nod)
		return
	}

	// call pointer
	if n.Left.Op != gc.ONAME || n.Left.Class != gc.PFUNC {
		var nod gc.Node
		regalloc(&nod, gc.Types[gc.Tptr], nil)
		gc.Cgen_as(&nod, n.Left)
		nod.Type = t
		ginscall(&nod, proc)
		regfree(&nod)
		return
	}

	// call direct
	n.Left.Method = 1

	ginscall(n.Left, proc)
}

/*
 * call to n has already been generated.
 * generate:
 *	res = return value from call.
 */
func cgen_callret(n *gc.Node, res *gc.Node) {
	t := n.Left.Type
	if t.Etype == gc.TPTR32 || t.Etype == gc.TPTR64 {
		t = t.Type
	}

	var flist gc.Iter
	fp := gc.Structfirst(&flist, gc.Getoutarg(t))
	if fp == nil {
		gc.Fatal("cgen_callret: nil")
	}

	var nod gc.Node
	nod.Op = gc.OINDREG
	nod.Val.U.Reg = arm64.REGSP
	nod.Addable = 1

	nod.Xoffset = fp.Width + int64(gc.Widthptr) // +widthptr: saved LR at 0(R1)
	nod.Type = fp.Type
	gc.Cgen_as(res, &nod)
}

/*
 * call to n has already been generated.
 * generate:
 *	res = &return value from call.
 */
func cgen_aret(n *gc.Node, res *gc.Node) {
	t := n.Left.Type
	if gc.Isptr[t.Etype] {
		t = t.Type
	}

	var flist gc.Iter
	fp := gc.Structfirst(&flist, gc.Getoutarg(t))
	if fp == nil {
		gc.Fatal("cgen_aret: nil")
	}

	var nod1 gc.Node
	nod1.Op = gc.OINDREG
	nod1.Val.U.Reg = arm64.REGSP
	nod1.Addable = 1

	nod1.Xoffset = fp.Width + int64(gc.Widthptr) // +widthptr: saved lr at 0(SP)
	nod1.Type = fp.Type

	if res.Op != gc.OREGISTER {
		var nod2 gc.Node
		regalloc(&nod2, gc.Types[gc.Tptr], res)
		agen(&nod1, &nod2)
		gins(arm64.AMOVD, &nod2, res)
		regfree(&nod2)
	} else {
		agen(&nod1, res)
	}
}

/*
 * generate return.
 * n->left is assignments to return values.
 */
func cgen_ret(n *gc.Node) {
	if n != nil {
		gc.Genlist(n.List) // copy out args
	}
	if gc.Hasdefer != 0 {
		ginscall(gc.Deferreturn, 0)
	}
	gc.Genlist(gc.Curfn.Exit)
	p := gins(obj.ARET, nil, nil)
	if n != nil && n.Op == gc.ORETJMP {
		p.To.Name = obj.NAME_EXTERN
		p.To.Type = obj.TYPE_ADDR
		p.To.Sym = gc.Linksym(n.Left.Sym)
	}
}

/*
 * generate division.
 * generates one of:
 *	res = nl / nr
 *	res = nl % nr
 * according to op.
 */
func dodiv(op int, nl *gc.Node, nr *gc.Node, res *gc.Node) {
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
	if gc.Issigned[t.Etype] {
		check = 1
		if gc.Isconst(nl, gc.CTINT) && gc.Mpgetfix(nl.Val.U.Xval) != -(1<<uint64(t.Width*8-1)) {
			check = 0
		} else if gc.Isconst(nr, gc.CTINT) && gc.Mpgetfix(nr.Val.U.Xval) != -1 {
			check = 0
		}
	}

	if t.Width < 8 {
		if gc.Issigned[t.Etype] {
			t = gc.Types[gc.TINT64]
		} else {
			t = gc.Types[gc.TUINT64]
		}
		check = 0
	}

	a := optoas(gc.ODIV, t)

	var tl gc.Node
	regalloc(&tl, t0, nil)
	var tr gc.Node
	regalloc(&tr, t0, nil)
	if nl.Ullman >= nr.Ullman {
		cgen(nl, &tl)
		cgen(nr, &tr)
	} else {
		cgen(nr, &tr)
		cgen(nl, &tl)
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
	p1.Reg = arm64.REGZERO
	p1 = gc.Gbranch(optoas(gc.ONE, t), nil, +1)
	if panicdiv == nil {
		panicdiv = gc.Sysfunc("panicdivide")
	}
	ginscall(panicdiv, -1)
	gc.Patch(p1, gc.Pc)

	var p2 *obj.Prog
	if check != 0 {
		var nm1 gc.Node
		gc.Nodconst(&nm1, t, -1)
		gcmp(optoas(gc.OCMP, t), &tr, &nm1)
		p1 := gc.Gbranch(optoas(gc.ONE, t), nil, +1)
		if op == gc.ODIV {
			// a / (-1) is -a.
			gins(optoas(gc.OMINUS, t), &tl, &tl)

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
		regfree(&tr)
		gmove(&tl, res)
	} else {
		// A%B = A-(A/B*B)
		var tm gc.Node
		regalloc(&tm, t, nil)

		// patch div to use the 3 register form
		// TODO(minux): add gins3?
		p1.Reg = p1.To.Reg

		p1.To.Reg = tm.Val.U.Reg
		gins(optoas(gc.OMUL, t), &tr, &tm)
		regfree(&tr)
		gins(optoas(gc.OSUB, t), &tm, &tl)
		regfree(&tm)
		gmove(&tl, res)
	}

	regfree(&tl)
	if check != 0 {
		gc.Patch(p2, gc.Pc)
	}
}

/*
 * generate division according to op, one of:
 *	res = nl / nr
 *	res = nl % nr
 */
func cgen_div(op int, nl *gc.Node, nr *gc.Node, res *gc.Node) {
	// TODO(minux): enable division by magic multiply (also need to fix longmod below)
	//if(nr->op != OLITERAL)
	// division and mod using (slow) hardware instruction
	dodiv(op, nl, nr, res)

	return
}

/*
 * generate high multiply:
 *   res = (nl*nr) >> width
 */
func cgen_hmul(nl *gc.Node, nr *gc.Node, res *gc.Node) {
	// largest ullman on left.
	if nl.Ullman < nr.Ullman {
		tmp := (*gc.Node)(nl)
		nl = nr
		nr = tmp
	}

	t := (*gc.Type)(nl.Type)
	w := int(int(t.Width * 8))
	var n1 gc.Node
	cgenr(nl, &n1, res)
	var n2 gc.Node
	cgenr(nr, &n2, nil)
	switch gc.Simtype[t.Etype] {
	case gc.TINT8,
		gc.TINT16,
		gc.TINT32:
		gins(optoas(gc.OMUL, t), &n2, &n1)
		p := (*obj.Prog)(gins(arm64.AASR, nil, &n1))
		p.From.Type = obj.TYPE_CONST
		p.From.Offset = int64(w)

	case gc.TUINT8,
		gc.TUINT16,
		gc.TUINT32:
		gins(optoas(gc.OMUL, t), &n2, &n1)
		p := (*obj.Prog)(gins(arm64.ALSR, nil, &n1))
		p.From.Type = obj.TYPE_CONST
		p.From.Offset = int64(w)

	case gc.TINT64,
		gc.TUINT64:
		if gc.Issigned[t.Etype] {
			gins(arm64.ASMULH, &n2, &n1)
		} else {
			gins(arm64.AUMULH, &n2, &n1)
		}

	default:
		gc.Fatal("cgen_hmul %v", gc.Tconv(t, 0))
	}

	cgen(&n1, res)
	regfree(&n1)
	regfree(&n2)
}

/*
 * generate shift according to op, one of:
 *	res = nl << nr
 *	res = nl >> nr
 */
func cgen_shift(op int, bounded bool, nl *gc.Node, nr *gc.Node, res *gc.Node) {
	a := int(optoas(op, nl.Type))

	if nr.Op == gc.OLITERAL {
		var n1 gc.Node
		regalloc(&n1, nl.Type, res)
		cgen(nl, &n1)
		sc := uint64(uint64(gc.Mpgetfix(nr.Val.U.Xval)))
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
		regfree(&n1)
		return
	}

	if nl.Ullman >= gc.UINF {
		var n4 gc.Node
		gc.Tempname(&n4, nl.Type)
		cgen(nl, &n4)
		nl = &n4
	}

	if nr.Ullman >= gc.UINF {
		var n5 gc.Node
		gc.Tempname(&n5, nr.Type)
		cgen(nr, &n5)
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
	regalloc(&n1, nr.Type, nil) // to hold the shift type in CX
	var n3 gc.Node
	regalloc(&n3, tcount, &n1) // to clear high bits of CX

	var n2 gc.Node
	regalloc(&n2, nl.Type, res)

	if nl.Ullman >= nr.Ullman {
		cgen(nl, &n2)
		cgen(nr, &n1)
		gmove(&n1, &n3)
	} else {
		cgen(nr, &n1)
		gmove(&n1, &n3)
		cgen(nl, &n2)
	}

	regfree(&n3)

	// test and fix up large shifts
	if !bounded {
		gc.Nodconst(&n3, tcount, nl.Type.Width*8)
		gcmp(optoas(gc.OCMP, tcount), &n1, &n3)
		p1 := (*obj.Prog)(gc.Gbranch(optoas(gc.OLT, tcount), nil, +1))
		if op == gc.ORSH && gc.Issigned[nl.Type.Etype] {
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

	regfree(&n1)
	regfree(&n2)
}

func clearfat(nl *gc.Node) {
	/* clear a fat object */
	if gc.Debug['g'] != 0 {
		fmt.Printf("clearfat %v (%v, size: %d)\n", gc.Nconv(nl, 0), gc.Tconv(nl.Type, 0), nl.Type.Width)
	}

	w := uint64(uint64(nl.Type.Width))

	// Avoid taking the address for simple enough types.
	//if(componentgen(N, nl))
	//	return;

	c := uint64(w % 8) // bytes
	q := uint64(w / 8) // dwords

	if reg[arm64.REGRT1-arm64.REG_R0] > 0 {
		gc.Fatal("R%d in use during clearfat", arm64.REGRT1-arm64.REG_R0)
	}

	var r0 gc.Node
	gc.Nodreg(&r0, gc.Types[gc.TUINT64], arm64.REGZERO)
	var dst gc.Node
	gc.Nodreg(&dst, gc.Types[gc.Tptr], arm64.REGRT1)
	reg[arm64.REGRT1-arm64.REG_R0]++
	agen(nl, &dst)

	var boff uint64
	if q > 128 {
		p := gins(arm64.ASUB, nil, &dst)
		p.From.Type = obj.TYPE_CONST
		p.From.Offset = 8

		var end gc.Node
		regalloc(&end, gc.Types[gc.Tptr], nil)
		p = gins(arm64.AMOVD, &dst, &end)
		p.From.Type = obj.TYPE_ADDR
		p.From.Offset = int64(q * 8)

		p = gins(arm64.AMOVD, &r0, &dst)
		p.To.Type = obj.TYPE_MEM
		p.To.Offset = 8
		p.Scond = arm64.C_XPRE
		pl := (*obj.Prog)(p)

		p = gcmp(arm64.ACMP, &dst, &end)
		gc.Patch(gc.Gbranch(arm64.ABNE, nil, 0), pl)

		regfree(&end)

		// The loop leaves R16 on the last zeroed dword
		boff = 8
	} else if q >= 4 {
		p := gins(arm64.ASUB, nil, &dst)
		p.From.Type = obj.TYPE_CONST
		p.From.Offset = 8
		f := (*gc.Node)(gc.Sysfunc("duffzero"))
		p = gins(obj.ADUFFZERO, nil, f)
		gc.Afunclit(&p.To, f)

		// 4 and 128 = magic constants: see ../../runtime/asm_arm64x.s
		p.To.Offset = int64(4 * (128 - q))

		// duffzero leaves R16 on the last zeroed dword
		boff = 8
	} else {
		var p *obj.Prog
		for t := uint64(0); t < q; t++ {
			p = gins(arm64.AMOVD, &r0, &dst)
			p.To.Type = obj.TYPE_MEM
			p.To.Offset = int64(8 * t)
		}

		boff = 8 * q
	}

	var p *obj.Prog
	for t := uint64(0); t < c; t++ {
		p = gins(arm64.AMOVB, &r0, &dst)
		p.To.Type = obj.TYPE_MEM
		p.To.Offset = int64(t + boff)
	}

	reg[arm64.REGRT1-arm64.REG_R0]--
}

// Called after regopt and peep have run.
// Expand CHECKNIL pseudo-op into actual nil pointer check.
func expandchecks(firstp *obj.Prog) {
	var p1 *obj.Prog
	var p2 *obj.Prog

	for p := (*obj.Prog)(firstp); p != nil; p = p.Link {
		if gc.Debug_checknil != 0 && gc.Ctxt.Debugvlog != 0 {
			fmt.Printf("expandchecks: %v\n", p)
		}
		if p.As != obj.ACHECKNIL {
			continue
		}
		if gc.Debug_checknil != 0 && p.Lineno > 1 { // p->lineno==1 in generated wrappers
			gc.Warnl(int(p.Lineno), "generated nil check")
		}
		if p.From.Type != obj.TYPE_REG {
			gc.Fatal("invalid nil check %v\n", p)
		}

		// check is
		//	CMP arg, ZR
		//	BNE 2(PC) [likely]
		//	MOVD ZR, 0(arg)
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
		p.As = arm64.ACMP
		p.Reg = arm64.REGZERO
		p1.As = arm64.ABNE

		//p1->from.type = TYPE_CONST;
		//p1->from.offset = 1; // likely
		p1.To.Type = obj.TYPE_BRANCH

		p1.To.U.Branch = p2.Link

		// crash by write to memory address 0.
		p2.As = arm64.AMOVD
		p2.From.Type = obj.TYPE_REG
		p2.From.Reg = arm64.REGZERO
		p2.To.Type = obj.TYPE_MEM
		p2.To.Reg = p.From.Reg
		p2.To.Offset = 0
	}
}
