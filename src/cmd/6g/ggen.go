// Copyright 2009 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package main

import (
	"cmd/internal/obj"
	"cmd/internal/obj/x86"
)
import "cmd/internal/gc"

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
	ax := uint32(0)

	// iterate through declarations - they are sorted in decreasing xoffset order.
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

		if lo != hi && n.Xoffset+n.Type.Width >= lo-int64(2*gc.Widthreg) {
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
		p = appendpp(p, x86.AMOVQ, obj.TYPE_CONST, 0, 0, obj.TYPE_REG, x86.REG_AX, 0)
		*ax = 1
	}

	if cnt%int64(gc.Widthreg) != 0 {
		// should only happen with nacl
		if cnt%int64(gc.Widthptr) != 0 {
			gc.Fatal("zerorange count not a multiple of widthptr %d", cnt)
		}
		p = appendpp(p, x86.AMOVL, obj.TYPE_REG, x86.REG_AX, 0, obj.TYPE_MEM, x86.REG_SP, frame+lo)
		lo += int64(gc.Widthptr)
		cnt -= int64(gc.Widthptr)
	}

	if cnt <= int64(4*gc.Widthreg) {
		for i := int64(0); i < cnt; i += int64(gc.Widthreg) {
			p = appendpp(p, x86.AMOVQ, obj.TYPE_REG, x86.REG_AX, 0, obj.TYPE_MEM, x86.REG_SP, frame+lo+i)
		}
	} else if !gc.Nacl && (cnt <= int64(128*gc.Widthreg)) {
		p = appendpp(p, leaptr, obj.TYPE_MEM, x86.REG_SP, frame+lo, obj.TYPE_REG, x86.REG_DI, 0)
		p = appendpp(p, obj.ADUFFZERO, obj.TYPE_NONE, 0, 0, obj.TYPE_ADDR, 0, 2*(128-cnt/int64(gc.Widthreg)))
		p.To.Sym = gc.Linksym(gc.Pkglookup("duffzero", gc.Runtimepkg))
	} else {
		p = appendpp(p, x86.AMOVQ, obj.TYPE_CONST, 0, cnt/int64(gc.Widthreg), obj.TYPE_REG, x86.REG_CX, 0)
		p = appendpp(p, leaptr, obj.TYPE_MEM, x86.REG_SP, frame+lo, obj.TYPE_REG, x86.REG_DI, 0)
		p = appendpp(p, x86.AREP, obj.TYPE_NONE, 0, 0, obj.TYPE_NONE, 0, 0)
		p = appendpp(p, x86.ASTOSQ, obj.TYPE_NONE, 0, 0, obj.TYPE_NONE, 0, 0)
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
				// insert an x86 NOP that we will have the right line number.
				// x86 NOP 0x90 is really XCHG AX, AX; use that description
				// because the NOP pseudo-instruction would be removed by
				// the linker.
				var reg gc.Node
				gc.Nodreg(&reg, gc.Types[gc.TINT], x86.REG_AX)

				gins(x86.AXCHGL, &reg, &reg)
			}

			p := gins(obj.ACALL, nil, f)
			gc.Afunclit(&p.To, f)
			if proc == -1 || gc.Noreturn(p) {
				gins(obj.AUNDEF, nil, nil)
			}
			break
		}

		var reg gc.Node
		gc.Nodreg(&reg, gc.Types[gc.Tptr], x86.REG_DX)
		var r1 gc.Node
		gc.Nodreg(&r1, gc.Types[gc.Tptr], x86.REG_BX)
		gmove(f, &reg)
		reg.Op = gc.OINDREG
		gmove(&reg, &r1)
		reg.Op = gc.OREGISTER
		gins(obj.ACALL, &reg, &r1)

	case 3: // normal call of c function pointer
		gins(obj.ACALL, nil, f)

	case 1, // call in new proc (go)
		2: // deferred call (defer)
		stk := gc.Node{}

		stk.Op = gc.OINDREG
		stk.Val.U.Reg = x86.REG_SP
		stk.Xoffset = 0

		var reg gc.Node
		if gc.Widthptr == 8 {
			// size of arguments at 0(SP)
			ginscon(x86.AMOVQ, int64(gc.Argsize(f.Type)), &stk)

			// FuncVal* at 8(SP)
			stk.Xoffset = int64(gc.Widthptr)

			gc.Nodreg(&reg, gc.Types[gc.TINT64], x86.REG_AX)
			gmove(f, &reg)
			gins(x86.AMOVQ, &reg, &stk)
		} else {
			// size of arguments at 0(SP)
			ginscon(x86.AMOVL, int64(gc.Argsize(f.Type)), &stk)

			// FuncVal* at 4(SP)
			stk.Xoffset = int64(gc.Widthptr)

			gc.Nodreg(&reg, gc.Types[gc.TINT32], x86.REG_AX)
			gmove(f, &reg)
			gins(x86.AMOVL, &reg, &stk)
		}

		if proc == 1 {
			ginscall(gc.Newproc, 0)
		} else {
			if gc.Hasdefer == 0 {
				gc.Fatal("hasdefer=0 but has defer")
			}
			ginscall(gc.Deferproc, 0)
		}

		if proc == 2 {
			gc.Nodreg(&reg, gc.Types[gc.TINT32], x86.REG_AX)
			gins(x86.ATESTL, &reg, &reg)
			p := gc.Gbranch(x86.AJEQ, nil, +1)
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
	gc.Nodindreg(&nodsp, gc.Types[gc.Tptr], x86.REG_SP)

	nodsp.Xoffset = 0
	if proc != 0 {
		nodsp.Xoffset += 2 * int64(gc.Widthptr) // leave room for size & fn
	}
	nodi.Type = gc.Types[gc.Tptr]
	nodi.Xoffset += int64(gc.Widthptr)
	cgen(&nodi, &nodsp) // {0, 8(nacl), or 16}(SP) = 8(REG) -- i.data

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
		gins(x86.ALEAQ, &nodo, &nodr) // REG = &(32+offset(REG)) -- i.tab->fun[f]
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

	nod := gc.Node{}
	nod.Op = gc.OINDREG
	nod.Val.U.Reg = x86.REG_SP
	nod.Addable = 1

	nod.Xoffset = fp.Width
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
	nod1.Val.U.Reg = x86.REG_SP
	nod1.Addable = 1

	nod1.Xoffset = fp.Width
	nod1.Type = fp.Type

	if res.Op != gc.OREGISTER {
		var nod2 gc.Node
		regalloc(&nod2, gc.Types[gc.Tptr], res)
		gins(leaptr, &nod1, &nod2)
		gins(movptr, &nod2, res)
		regfree(&nod2)
	} else {
		gins(leaptr, &nod1, res)
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
		p.To.Type = obj.TYPE_MEM
		p.To.Name = obj.NAME_EXTERN
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
	// The hardware will trap.
	// Also the byte divide instruction needs AH,
	// which we otherwise don't have to deal with.
	// Easiest way to avoid for int8, int16: use int32.
	// For int32 and int64, use explicit test.
	// Could use int64 hw for int32.
	t := nl.Type

	t0 := t
	check := 0
	if gc.Issigned[t.Etype] != 0 {
		check = 1
		if gc.Isconst(nl, gc.CTINT) && gc.Mpgetfix(nl.Val.U.Xval) != -(1<<uint64(t.Width*8-1)) {
			check = 0
		} else if gc.Isconst(nr, gc.CTINT) && gc.Mpgetfix(nr.Val.U.Xval) != -1 {
			check = 0
		}
	}

	if t.Width < 4 {
		if gc.Issigned[t.Etype] != 0 {
			t = gc.Types[gc.TINT32]
		} else {
			t = gc.Types[gc.TUINT32]
		}
		check = 0
	}

	a := optoas(op, t)

	var n3 gc.Node
	regalloc(&n3, t0, nil)
	var ax gc.Node
	var oldax gc.Node
	if nl.Ullman >= nr.Ullman {
		savex(x86.REG_AX, &ax, &oldax, res, t0)
		cgen(nl, &ax)
		regalloc(&ax, t0, &ax) // mark ax live during cgen
		cgen(nr, &n3)
		regfree(&ax)
	} else {
		cgen(nr, &n3)
		savex(x86.REG_AX, &ax, &oldax, res, t0)
		cgen(nl, &ax)
	}

	if t != t0 {
		// Convert
		ax1 := ax

		n31 := n3
		ax.Type = t
		n3.Type = t
		gmove(&ax1, &ax)
		gmove(&n31, &n3)
	}

	p2 := (*obj.Prog)(nil)
	var n4 gc.Node
	if gc.Nacl {
		// Native Client does not relay the divide-by-zero trap
		// to the executing program, so we must insert a check
		// for ourselves.
		gc.Nodconst(&n4, t, 0)

		gins(optoas(gc.OCMP, t), &n3, &n4)
		p1 := gc.Gbranch(optoas(gc.ONE, t), nil, +1)
		if panicdiv == nil {
			panicdiv = gc.Sysfunc("panicdivide")
		}
		ginscall(panicdiv, -1)
		gc.Patch(p1, gc.Pc)
	}

	if check != 0 {
		gc.Nodconst(&n4, t, -1)
		gins(optoas(gc.OCMP, t), &n3, &n4)
		p1 := gc.Gbranch(optoas(gc.ONE, t), nil, +1)
		if op == gc.ODIV {
			// a / (-1) is -a.
			gins(optoas(gc.OMINUS, t), nil, &ax)

			gmove(&ax, res)
		} else {
			// a % (-1) is 0.
			gc.Nodconst(&n4, t, 0)

			gmove(&n4, res)
		}

		p2 = gc.Gbranch(obj.AJMP, nil, 0)
		gc.Patch(p1, gc.Pc)
	}

	var olddx gc.Node
	var dx gc.Node
	savex(x86.REG_DX, &dx, &olddx, res, t)
	if gc.Issigned[t.Etype] == 0 {
		gc.Nodconst(&n4, t, 0)
		gmove(&n4, &dx)
	} else {
		gins(optoas(gc.OEXTEND, t), nil, nil)
	}
	gins(a, &n3, nil)
	regfree(&n3)
	if op == gc.ODIV {
		gmove(&ax, res)
	} else {
		gmove(&dx, res)
	}
	restx(&dx, &olddx)
	if check != 0 {
		gc.Patch(p2, gc.Pc)
	}
	restx(&ax, &oldax)
}

/*
 * register dr is one of the special ones (AX, CX, DI, SI, etc.).
 * we need to use it.  if it is already allocated as a temporary
 * (r > 1; can only happen if a routine like sgen passed a
 * special as cgen's res and then cgen used regalloc to reuse
 * it as its own temporary), then move it for now to another
 * register.  caller must call restx to move it back.
 * the move is not necessary if dr == res, because res is
 * known to be dead.
 */
func savex(dr int, x *gc.Node, oldx *gc.Node, res *gc.Node, t *gc.Type) {
	r := int(reg[dr])

	// save current ax and dx if they are live
	// and not the destination
	*oldx = gc.Node{}

	gc.Nodreg(x, t, dr)
	if r > 1 && !gc.Samereg(x, res) {
		regalloc(oldx, gc.Types[gc.TINT64], nil)
		x.Type = gc.Types[gc.TINT64]
		gmove(x, oldx)
		x.Type = t
		oldx.Ostk = int32(r) // squirrel away old r value
		reg[dr] = 1
	}
}

func restx(x *gc.Node, oldx *gc.Node) {
	if oldx.Op != 0 {
		x.Type = gc.Types[gc.TINT64]
		reg[x.Val.U.Reg] = uint8(oldx.Ostk)
		gmove(oldx, x)
		regfree(oldx)
	}
}

/*
 * generate division according to op, one of:
 *	res = nl / nr
 *	res = nl % nr
 */
func cgen_div(op int, nl *gc.Node, nr *gc.Node, res *gc.Node) {
	var w int

	if nr.Op != gc.OLITERAL {
		goto longdiv
	}
	w = int(nl.Type.Width * 8)

	// Front end handled 32-bit division. We only need to handle 64-bit.
	// try to do division by multiply by (2^w)/d
	// see hacker's delight chapter 10
	switch gc.Simtype[nl.Type.Etype] {
	default:
		goto longdiv

	case gc.TUINT64:
		var m gc.Magic
		m.W = w
		m.Ud = uint64(gc.Mpgetfix(nr.Val.U.Xval))
		gc.Umagic(&m)
		if m.Bad != 0 {
			break
		}
		if op == gc.OMOD {
			goto longmod
		}

		var n1 gc.Node
		cgenr(nl, &n1, nil)
		var n2 gc.Node
		gc.Nodconst(&n2, nl.Type, int64(m.Um))
		var n3 gc.Node
		regalloc(&n3, nl.Type, res)
		cgen_hmul(&n1, &n2, &n3)

		if m.Ua != 0 {
			// need to add numerator accounting for overflow
			gins(optoas(gc.OADD, nl.Type), &n1, &n3)

			gc.Nodconst(&n2, nl.Type, 1)
			gins(optoas(gc.ORROTC, nl.Type), &n2, &n3)
			gc.Nodconst(&n2, nl.Type, int64(m.S)-1)
			gins(optoas(gc.ORSH, nl.Type), &n2, &n3)
		} else {
			gc.Nodconst(&n2, nl.Type, int64(m.S))
			gins(optoas(gc.ORSH, nl.Type), &n2, &n3) // shift dx
		}

		gmove(&n3, res)
		regfree(&n1)
		regfree(&n3)
		return

	case gc.TINT64:
		var m gc.Magic
		m.W = w
		m.Sd = gc.Mpgetfix(nr.Val.U.Xval)
		gc.Smagic(&m)
		if m.Bad != 0 {
			break
		}
		if op == gc.OMOD {
			goto longmod
		}

		var n1 gc.Node
		cgenr(nl, &n1, res)
		var n2 gc.Node
		gc.Nodconst(&n2, nl.Type, m.Sm)
		var n3 gc.Node
		regalloc(&n3, nl.Type, nil)
		cgen_hmul(&n1, &n2, &n3)

		if m.Sm < 0 {
			// need to add numerator
			gins(optoas(gc.OADD, nl.Type), &n1, &n3)
		}

		gc.Nodconst(&n2, nl.Type, int64(m.S))
		gins(optoas(gc.ORSH, nl.Type), &n2, &n3) // shift n3

		gc.Nodconst(&n2, nl.Type, int64(w)-1)

		gins(optoas(gc.ORSH, nl.Type), &n2, &n1) // -1 iff num is neg
		gins(optoas(gc.OSUB, nl.Type), &n1, &n3) // added

		if m.Sd < 0 {
			// this could probably be removed
			// by factoring it into the multiplier
			gins(optoas(gc.OMINUS, nl.Type), nil, &n3)
		}

		gmove(&n3, res)
		regfree(&n1)
		regfree(&n3)
		return
	}

	goto longdiv

	// division and mod using (slow) hardware instruction
longdiv:
	dodiv(op, nl, nr, res)

	return

	// mod using formula A%B = A-(A/B*B) but
	// we know that there is a fast algorithm for A/B
longmod:
	var n1 gc.Node
	regalloc(&n1, nl.Type, res)

	cgen(nl, &n1)
	var n2 gc.Node
	regalloc(&n2, nl.Type, nil)
	cgen_div(gc.ODIV, &n1, nr, &n2)
	a := optoas(gc.OMUL, nl.Type)
	if w == 8 {
		// use 2-operand 16-bit multiply
		// because there is no 2-operand 8-bit multiply
		a = x86.AIMULW
	}

	if !gc.Smallintconst(nr) {
		var n3 gc.Node
		regalloc(&n3, nl.Type, nil)
		cgen(nr, &n3)
		gins(a, &n3, &n2)
		regfree(&n3)
	} else {
		gins(a, nr, &n2)
	}
	gins(optoas(gc.OSUB, nl.Type), &n2, &n1)
	gmove(&n1, res)
	regfree(&n1)
	regfree(&n2)
}

/*
 * generate high multiply:
 *   res = (nl*nr) >> width
 */
func cgen_hmul(nl *gc.Node, nr *gc.Node, res *gc.Node) {
	t := nl.Type
	a := optoas(gc.OHMUL, t)
	if nl.Ullman < nr.Ullman {
		tmp := nl
		nl = nr
		nr = tmp
	}

	var n1 gc.Node
	cgenr(nl, &n1, res)
	var n2 gc.Node
	cgenr(nr, &n2, nil)
	var ax gc.Node
	gc.Nodreg(&ax, t, x86.REG_AX)
	gmove(&n1, &ax)
	gins(a, &n2, nil)
	regfree(&n2)
	regfree(&n1)

	var dx gc.Node
	if t.Width == 1 {
		// byte multiply behaves differently.
		gc.Nodreg(&ax, t, x86.REG_AH)

		gc.Nodreg(&dx, t, x86.REG_DX)
		gmove(&ax, &dx)
	}

	gc.Nodreg(&dx, t, x86.REG_DX)
	gmove(&dx, res)
}

/*
 * generate shift according to op, one of:
 *	res = nl << nr
 *	res = nl >> nr
 */
func cgen_shift(op int, bounded bool, nl *gc.Node, nr *gc.Node, res *gc.Node) {
	var n1 gc.Node
	var n2 gc.Node
	var n3 gc.Node
	var cx gc.Node
	var oldcx gc.Node
	var rcx int
	var tcount *gc.Type

	a := optoas(op, nl.Type)

	if nr.Op == gc.OLITERAL {
		var n1 gc.Node
		regalloc(&n1, nl.Type, res)
		cgen(nl, &n1)
		sc := uint64(gc.Mpgetfix(nr.Val.U.Xval))
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
		goto ret
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

	rcx = int(reg[x86.REG_CX])
	gc.Nodreg(&n1, gc.Types[gc.TUINT32], x86.REG_CX)

	// Allow either uint32 or uint64 as shift type,
	// to avoid unnecessary conversion from uint32 to uint64
	// just to do the comparison.
	tcount = gc.Types[gc.Simtype[nr.Type.Etype]]

	if tcount.Etype < gc.TUINT32 {
		tcount = gc.Types[gc.TUINT32]
	}

	regalloc(&n1, nr.Type, &n1) // to hold the shift type in CX
	regalloc(&n3, tcount, &n1)  // to clear high bits of CX

	gc.Nodreg(&cx, gc.Types[gc.TUINT64], x86.REG_CX)

	oldcx = gc.Node{}
	if rcx > 0 && !gc.Samereg(&cx, res) {
		regalloc(&oldcx, gc.Types[gc.TUINT64], nil)
		gmove(&cx, &oldcx)
	}

	cx.Type = tcount

	if gc.Samereg(&cx, res) {
		regalloc(&n2, nl.Type, nil)
	} else {
		regalloc(&n2, nl.Type, res)
	}
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
		gins(optoas(gc.OCMP, tcount), &n1, &n3)
		p1 := gc.Gbranch(optoas(gc.OLT, tcount), nil, +1)
		if op == gc.ORSH && gc.Issigned[nl.Type.Etype] != 0 {
			gc.Nodconst(&n3, gc.Types[gc.TUINT32], nl.Type.Width*8-1)
			gins(a, &n3, &n2)
		} else {
			gc.Nodconst(&n3, nl.Type, 0)
			gmove(&n3, &n2)
		}

		gc.Patch(p1, gc.Pc)
	}

	gins(a, &n1, &n2)

	if oldcx.Op != 0 {
		cx.Type = gc.Types[gc.TUINT64]
		gmove(&oldcx, &cx)
		regfree(&oldcx)
	}

	gmove(&n2, res)

	regfree(&n1)
	regfree(&n2)

ret:
}

/*
 * generate byte multiply:
 *	res = nl * nr
 * there is no 2-operand byte multiply instruction so
 * we do a full-width multiplication and truncate afterwards.
 */
func cgen_bmul(op int, nl *gc.Node, nr *gc.Node, res *gc.Node) {
	// largest ullman on left.
	if nl.Ullman < nr.Ullman {
		tmp := nl
		nl = nr
		nr = tmp
	}

	// generate operands in "8-bit" registers.
	var n1b gc.Node
	regalloc(&n1b, nl.Type, res)

	cgen(nl, &n1b)
	var n2b gc.Node
	regalloc(&n2b, nr.Type, nil)
	cgen(nr, &n2b)

	// perform full-width multiplication.
	t := gc.Types[gc.TUINT64]

	if gc.Issigned[nl.Type.Etype] != 0 {
		t = gc.Types[gc.TINT64]
	}
	var n1 gc.Node
	gc.Nodreg(&n1, t, int(n1b.Val.U.Reg))
	var n2 gc.Node
	gc.Nodreg(&n2, t, int(n2b.Val.U.Reg))
	a := optoas(op, t)
	gins(a, &n2, &n1)

	// truncate.
	gmove(&n1, res)

	regfree(&n1b)
	regfree(&n2b)
}

func clearfat(nl *gc.Node) {
	/* clear a fat object */
	if gc.Debug['g'] != 0 {
		gc.Dump("\nclearfat", nl)
	}

	w := nl.Type.Width

	// Avoid taking the address for simple enough types.
	if componentgen(nil, nl) {
		return
	}

	c := w % 8 // bytes
	q := w / 8 // quads

	if q < 4 {
		// Write sequence of MOV 0, off(base) instead of using STOSQ.
		// The hope is that although the code will be slightly longer,
		// the MOVs will have no dependencies and pipeline better
		// than the unrolled STOSQ loop.
		// NOTE: Must use agen, not igen, so that optimizer sees address
		// being taken. We are not writing on field boundaries.
		var n1 gc.Node
		agenr(nl, &n1, nil)

		n1.Op = gc.OINDREG
		var z gc.Node
		gc.Nodconst(&z, gc.Types[gc.TUINT64], 0)
		for {
			tmp14 := q
			q--
			if tmp14 <= 0 {
				break
			}
			n1.Type = z.Type
			gins(x86.AMOVQ, &z, &n1)
			n1.Xoffset += 8
		}

		if c >= 4 {
			gc.Nodconst(&z, gc.Types[gc.TUINT32], 0)
			n1.Type = z.Type
			gins(x86.AMOVL, &z, &n1)
			n1.Xoffset += 4
			c -= 4
		}

		gc.Nodconst(&z, gc.Types[gc.TUINT8], 0)
		for {
			tmp15 := c
			c--
			if tmp15 <= 0 {
				break
			}
			n1.Type = z.Type
			gins(x86.AMOVB, &z, &n1)
			n1.Xoffset++
		}

		regfree(&n1)
		return
	}

	var oldn1 gc.Node
	var n1 gc.Node
	savex(x86.REG_DI, &n1, &oldn1, nil, gc.Types[gc.Tptr])
	agen(nl, &n1)

	var ax gc.Node
	var oldax gc.Node
	savex(x86.REG_AX, &ax, &oldax, nil, gc.Types[gc.Tptr])
	gconreg(x86.AMOVL, 0, x86.REG_AX)

	if q > 128 || gc.Nacl {
		gconreg(movptr, q, x86.REG_CX)
		gins(x86.AREP, nil, nil)   // repeat
		gins(x86.ASTOSQ, nil, nil) // STOQ AL,*(DI)+
	} else {
		p := gins(obj.ADUFFZERO, nil, nil)
		p.To.Type = obj.TYPE_ADDR
		p.To.Sym = gc.Linksym(gc.Pkglookup("duffzero", gc.Runtimepkg))

		// 2 and 128 = magic constants: see ../../runtime/asm_amd64.s
		p.To.Offset = 2 * (128 - q)
	}

	z := ax
	di := n1
	if w >= 8 && c >= 4 {
		di.Op = gc.OINDREG
		z.Type = gc.Types[gc.TINT64]
		di.Type = z.Type
		p := gins(x86.AMOVQ, &z, &di)
		p.To.Scale = 1
		p.To.Offset = c - 8
	} else if c >= 4 {
		di.Op = gc.OINDREG
		z.Type = gc.Types[gc.TINT32]
		di.Type = z.Type
		gins(x86.AMOVL, &z, &di)
		if c > 4 {
			p := gins(x86.AMOVL, &z, &di)
			p.To.Scale = 1
			p.To.Offset = c - 4
		}
	} else {
		for c > 0 {
			gins(x86.ASTOSB, nil, nil) // STOB AL,*(DI)+
			c--
		}
	}

	restx(&n1, &oldn1)
	restx(&ax, &oldax)
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
		p.As = int16(cmpptr)
		p.To.Type = obj.TYPE_CONST
		p.To.Offset = 0
		p1.As = x86.AJNE
		p1.From.Type = obj.TYPE_CONST
		p1.From.Offset = 1 // likely
		p1.To.Type = obj.TYPE_BRANCH
		p1.To.U.Branch = p2.Link

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
			p2.To.Reg = x86.REG_NONE
		}

		p2.To.Offset = 0
	}
}
