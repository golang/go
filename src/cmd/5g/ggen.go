// Copyright 2009 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package main

import (
	"cmd/internal/obj"
	"cmd/internal/obj/arm"
)
import "cmd/internal/gc"

func defframe(ptxt *obj.Prog) {
	var n *gc.Node

	// fill in argument size, stack size
	ptxt.To.Type = obj.TYPE_TEXTSIZE

	ptxt.To.U.Argsize = int32(gc.Rnd(gc.Curfn.Type.Argwid, int64(gc.Widthptr)))
	frame := uint32(gc.Rnd(gc.Stksize+gc.Maxarg, int64(gc.Widthreg)))
	ptxt.To.Offset = int64(frame)

	// insert code to contain ambiguously live variables
	// so that garbage collector only sees initialized values
	// when it looks for pointers.
	p := ptxt

	hi := int64(0)
	lo := hi
	r0 := uint32(0)
	for l := gc.Curfn.Dcl; l != nil; l = l.Next {
		n = l.N
		if n.Needzero == 0 {
			continue
		}
		if n.Class != gc.PAUTO {
			gc.Fatal("needzero class %d", n.Class)
		}
		if n.Type.Width%int64(gc.Widthptr) != 0 || n.Xoffset%int64(gc.Widthptr) != 0 || n.Type.Width == 0 {
			gc.Fatal("var %v has size %d offset %d", gc.Nconv(n, obj.FmtLong), int(n.Type.Width), int(n.Xoffset))
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
		gc.Naddr(f, &p.To, 1)
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
				// the BL deferreturn(SB) that we are about to emit.
				// However, the stack trace code will show the line
				// of the instruction before that return PC.
				// To avoid that instruction being an unrelated instruction,
				// insert a NOP so that we will have the right line number.
				// ARM NOP 0x00000000 is really AND.EQ R0, R0, R0.
				// Use the latter form because the NOP pseudo-instruction
				// would be removed by the linker.
				var r gc.Node
				gc.Nodreg(&r, gc.Types[gc.TINT], arm.REG_R0)

				p := gins(arm.AAND, &r, &r)
				p.Scond = arm.C_SCOND_EQ
			}

			p := gins(arm.ABL, nil, f)
			gc.Afunclit(&p.To, f)
			if proc == -1 || gc.Noreturn(p) {
				gins(obj.AUNDEF, nil, nil)
			}
			break
		}

		var r gc.Node
		gc.Nodreg(&r, gc.Types[gc.Tptr], arm.REG_R7)
		var r1 gc.Node
		gc.Nodreg(&r1, gc.Types[gc.Tptr], arm.REG_R1)
		gmove(f, &r)
		r.Op = gc.OINDREG
		gmove(&r, &r1)
		r.Op = gc.OREGISTER
		r1.Op = gc.OINDREG
		gins(arm.ABL, &r, &r1)

	case 3: // normal call of c function pointer
		gins(arm.ABL, nil, f)

	case 1, // call in new proc (go)
		2: // deferred call (defer)
		var r gc.Node
		regalloc(&r, gc.Types[gc.Tptr], nil)

		var con gc.Node
		gc.Nodconst(&con, gc.Types[gc.TINT32], int64(gc.Argsize(f.Type)))
		gins(arm.AMOVW, &con, &r)
		p := gins(arm.AMOVW, &r, nil)
		p.To.Type = obj.TYPE_MEM
		p.To.Reg = arm.REGSP
		p.To.Offset = 4

		gins(arm.AMOVW, f, &r)
		p = gins(arm.AMOVW, &r, nil)
		p.To.Type = obj.TYPE_MEM
		p.To.Reg = arm.REGSP
		p.To.Offset = 8

		regfree(&r)

		if proc == 1 {
			ginscall(gc.Newproc, 0)
		} else {
			ginscall(gc.Deferproc, 0)
		}

		if proc == 2 {
			gc.Nodconst(&con, gc.Types[gc.TINT32], 0)
			p := gins(arm.ACMP, &con, nil)
			p.Reg = arm.REG_R0
			p = gc.Gbranch(arm.ABEQ, nil, +1)
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

	// Release res register during genlist and cgen,
	// which might have their own function calls.
	r := -1

	if res != nil && (res.Op == gc.OREGISTER || res.Op == gc.OINDREG) {
		r = int(res.Val.U.Reg)
		reg[r]--
	}

	if i.Addable == 0 {
		var tmpi gc.Node
		gc.Tempname(&tmpi, i.Type)
		cgen(i, &tmpi)
		i = &tmpi
	}

	gc.Genlist(n.List) // args
	if r >= 0 {
		reg[r]++
	}

	var nodr gc.Node
	regalloc(&nodr, gc.Types[gc.Tptr], res)
	var nodo gc.Node
	regalloc(&nodo, gc.Types[gc.Tptr], &nodr)
	nodo.Op = gc.OINDREG

	agen(i, &nodr) // REG = &inter

	var nodsp gc.Node
	gc.Nodindreg(&nodsp, gc.Types[gc.Tptr], arm.REGSP)

	nodsp.Xoffset = int64(gc.Widthptr)
	if proc != 0 {
		nodsp.Xoffset += 2 * int64(gc.Widthptr) // leave room for size & fn
	}
	nodo.Xoffset += int64(gc.Widthptr)
	cgen(&nodo, &nodsp) // {4 or 12}(SP) = 4(REG) -- i.data

	nodo.Xoffset -= int64(gc.Widthptr)

	cgen(&nodo, &nodr)      // REG = 0(REG) -- i.tab
	gc.Cgen_checknil(&nodr) // in case offset is huge

	nodo.Xoffset = n.Left.Xoffset + 3*int64(gc.Widthptr) + 8

	if proc == 0 {
		// plain call: use direct c function pointer - more efficient
		cgen(&nodo, &nodr) // REG = 20+offset(REG) -- i.tab->fun[f]
		nodr.Op = gc.OINDREG
		proc = 3
	} else {
		// go/defer. generate go func value.
		p := gins(arm.AMOVW, &nodo, &nodr)

		p.From.Type = obj.TYPE_ADDR // REG = &(20+offset(REG)) -- i.tab->fun[f]
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
		goto ret
	}

	// call pointer
	if n.Left.Op != gc.ONAME || n.Left.Class != gc.PFUNC {
		var nod gc.Node
		regalloc(&nod, gc.Types[gc.Tptr], nil)
		gc.Cgen_as(&nod, n.Left)
		nod.Type = t
		ginscall(&nod, proc)
		regfree(&nod)
		goto ret
	}

	// call direct
	n.Left.Method = 1

	ginscall(n.Left, proc)

ret:
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

	nod := gc.Node{}
	nod.Op = gc.OINDREG
	nod.Val.U.Reg = arm.REGSP
	nod.Addable = 1

	nod.Xoffset = fp.Width + 4 // +4: saved lr at 0(SP)
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
	if gc.Isptr[t.Etype] != 0 {
		t = t.Type
	}

	var flist gc.Iter
	fp := gc.Structfirst(&flist, gc.Getoutarg(t))
	if fp == nil {
		gc.Fatal("cgen_aret: nil")
	}

	nod1 := gc.Node{}
	nod1.Op = gc.OINDREG
	nod1.Val.U.Reg = arm.REGSP
	nod1.Addable = 1

	nod1.Xoffset = fp.Width + 4 // +4: saved lr at 0(SP)
	nod1.Type = fp.Type

	if res.Op != gc.OREGISTER {
		var nod2 gc.Node
		regalloc(&nod2, gc.Types[gc.Tptr], res)
		agen(&nod1, &nod2)
		gins(arm.AMOVW, &nod2, res)
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
 * generate high multiply
 *  res = (nl * nr) >> wordsize
 */
func cgen_hmul(nl *gc.Node, nr *gc.Node, res *gc.Node) {
	if nl.Ullman < nr.Ullman {
		tmp := nl
		nl = nr
		nr = tmp
	}

	t := nl.Type
	w := int(t.Width * 8)
	var n1 gc.Node
	regalloc(&n1, t, res)
	cgen(nl, &n1)
	var n2 gc.Node
	regalloc(&n2, t, nil)
	cgen(nr, &n2)
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
		if gc.Issigned[t.Etype] != 0 {
			p = gins(arm.AMULL, &n2, nil)
		} else {
			p = gins(arm.AMULLU, &n2, nil)
		}

		// n2 * n1 -> (n1 n2)
		p.Reg = n1.Val.U.Reg

		p.To.Type = obj.TYPE_REGREG
		p.To.Reg = n1.Val.U.Reg
		p.To.Offset = int64(n2.Val.U.Reg)

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
	if nl.Type.Width > 4 {
		gc.Fatal("cgen_shift %v", gc.Tconv(nl.Type, 0))
	}

	w := int(nl.Type.Width * 8)

	if op == gc.OLROT {
		v := int(gc.Mpgetfix(nr.Val.U.Xval))
		var n1 gc.Node
		regalloc(&n1, nl.Type, res)
		if w == 32 {
			cgen(nl, &n1)
			gshift(arm.AMOVW, &n1, arm.SHIFT_RR, int32(w)-int32(v), &n1)
		} else {
			var n2 gc.Node
			regalloc(&n2, nl.Type, nil)
			cgen(nl, &n2)
			gshift(arm.AMOVW, &n2, arm.SHIFT_LL, int32(v), &n1)
			gshift(arm.AORR, &n2, arm.SHIFT_LR, int32(w)-int32(v), &n1)
			regfree(&n2)

			// Ensure sign/zero-extended result.
			gins(optoas(gc.OAS, nl.Type), &n1, &n1)
		}

		gmove(&n1, res)
		regfree(&n1)
		return
	}

	if nr.Op == gc.OLITERAL {
		var n1 gc.Node
		regalloc(&n1, nl.Type, res)
		cgen(nl, &n1)
		sc := uint64(gc.Mpgetfix(nr.Val.U.Xval))
		if sc == 0 {
		} else // nothing to do
		if sc >= uint64(nl.Type.Width*8) {
			if op == gc.ORSH && gc.Issigned[nl.Type.Etype] != 0 {
				gshift(arm.AMOVW, &n1, arm.SHIFT_AR, int32(w), &n1)
			} else {
				gins(arm.AEOR, &n1, &n1)
			}
		} else {
			if op == gc.ORSH && gc.Issigned[nl.Type.Etype] != 0 {
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
		regfree(&n1)
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
			regalloc(&n2, nl.Type, res)
			cgen(nl, &n2)
			cgen(nr, &nt)
			n1 = nt
		} else {
			cgen(nr, &nt)
			regalloc(&n2, nl.Type, res)
			cgen(nl, &n2)
		}

		var hi gc.Node
		var lo gc.Node
		split64(&nt, &lo, &hi)
		regalloc(&n1, gc.Types[gc.TUINT32], nil)
		regalloc(&n3, gc.Types[gc.TUINT32], nil)
		gmove(&lo, &n1)
		gmove(&hi, &n3)
		splitclean()
		gins(arm.ATST, &n3, nil)
		gc.Nodconst(&t, gc.Types[gc.TUINT32], int64(w))
		p1 := gins(arm.AMOVW, &t, &n1)
		p1.Scond = arm.C_SCOND_NE
		tr = gc.Types[gc.TUINT32]
		regfree(&n3)
	} else {
		if nl.Ullman >= nr.Ullman {
			regalloc(&n2, nl.Type, res)
			cgen(nl, &n2)
			regalloc(&n1, nr.Type, nil)
			cgen(nr, &n1)
		} else {
			regalloc(&n1, nr.Type, nil)
			cgen(nr, &n1)
			regalloc(&n2, nl.Type, res)
			cgen(nl, &n2)
		}
	}

	// test for shift being 0
	gins(arm.ATST, &n1, nil)

	p3 := gc.Gbranch(arm.ABEQ, nil, -1)

	// test and fix up large shifts
	// TODO: if(!bounded), don't emit some of this.
	regalloc(&n3, tr, nil)

	gc.Nodconst(&t, gc.Types[gc.TUINT32], int64(w))
	gmove(&t, &n3)
	gcmp(arm.ACMP, &n1, &n3)
	if op == gc.ORSH {
		var p1 *obj.Prog
		var p2 *obj.Prog
		if gc.Issigned[nl.Type.Etype] != 0 {
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

	regfree(&n3)

	gc.Patch(p3, gc.Pc)

	// Left-shift of smaller word must be sign/zero-extended.
	if w < 32 && op == gc.OLSH {
		gins(optoas(gc.OAS, nl.Type), &n2, &n2)
	}
	gmove(&n2, res)

	regfree(&n1)
	regfree(&n2)
}

func clearfat(nl *gc.Node) {
	/* clear a fat object */
	if gc.Debug['g'] != 0 {
		gc.Dump("\nclearfat", nl)
	}

	w := uint32(nl.Type.Width)

	// Avoid taking the address for simple enough types.
	if componentgen(nil, nl) {
		return
	}

	c := w % 4 // bytes
	q := w / 4 // quads

	var r0 gc.Node
	r0.Op = gc.OREGISTER

	r0.Val.U.Reg = REGALLOC_R0
	var r1 gc.Node
	r1.Op = gc.OREGISTER
	r1.Val.U.Reg = REGALLOC_R0 + 1
	var dst gc.Node
	regalloc(&dst, gc.Types[gc.Tptr], &r1)
	agen(nl, &dst)
	var nc gc.Node
	gc.Nodconst(&nc, gc.Types[gc.TUINT32], 0)
	var nz gc.Node
	regalloc(&nz, gc.Types[gc.TUINT32], &r0)
	cgen(&nc, &nz)

	if q > 128 {
		var end gc.Node
		regalloc(&end, gc.Types[gc.Tptr], nil)
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

		regfree(&end)
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

			//print("1. %P\n", p);
			q--
		}
	}

	var p *obj.Prog
	for c > 0 {
		p = gins(arm.AMOVB, &nz, &dst)
		p.To.Type = obj.TYPE_MEM
		p.To.Offset = 1
		p.Scond |= arm.C_PBIT

		//print("2. %P\n", p);
		c--
	}

	regfree(&dst)
	regfree(&nz)
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
			gc.Fatal("invalid nil check %v", p)
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
