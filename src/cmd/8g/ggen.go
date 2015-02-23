// Copyright 2009 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package main

import (
	"cmd/internal/obj"
	"cmd/internal/obj/i386"
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
		p = appendpp(p, i386.AMOVL, obj.TYPE_CONST, 0, 0, obj.TYPE_REG, i386.REG_AX, 0)
		*ax = 1
	}

	if cnt <= int64(4*gc.Widthreg) {
		for i := int64(0); i < cnt; i += int64(gc.Widthreg) {
			p = appendpp(p, i386.AMOVL, obj.TYPE_REG, i386.REG_AX, 0, obj.TYPE_MEM, i386.REG_SP, frame+lo+i)
		}
	} else if !gc.Nacl && cnt <= int64(128*gc.Widthreg) {
		p = appendpp(p, i386.ALEAL, obj.TYPE_MEM, i386.REG_SP, frame+lo, obj.TYPE_REG, i386.REG_DI, 0)
		p = appendpp(p, obj.ADUFFZERO, obj.TYPE_NONE, 0, 0, obj.TYPE_ADDR, 0, 1*(128-cnt/int64(gc.Widthreg)))
		p.To.Sym = gc.Linksym(gc.Pkglookup("duffzero", gc.Runtimepkg))
	} else {
		p = appendpp(p, i386.AMOVL, obj.TYPE_CONST, 0, cnt/int64(gc.Widthreg), obj.TYPE_REG, i386.REG_CX, 0)
		p = appendpp(p, i386.ALEAL, obj.TYPE_MEM, i386.REG_SP, frame+lo, obj.TYPE_REG, i386.REG_DI, 0)
		p = appendpp(p, i386.AREP, obj.TYPE_NONE, 0, 0, obj.TYPE_NONE, 0, 0)
		p = appendpp(p, i386.ASTOSL, obj.TYPE_NONE, 0, 0, obj.TYPE_NONE, 0, 0)
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
	if componentgen(nil, nl) {
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
		regalloc(&n1, gc.Types[gc.Tptr], nil)

		agen(nl, &n1)
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
			gins(i386.AMOVL, &z, &n1)
			n1.Xoffset += 4
		}

		gc.Nodconst(&z, gc.Types[gc.TUINT8], 0)
		for {
			tmp15 := c
			c--
			if tmp15 <= 0 {
				break
			}
			n1.Type = z.Type
			gins(i386.AMOVB, &z, &n1)
			n1.Xoffset++
		}

		regfree(&n1)
		return
	}

	var n1 gc.Node
	gc.Nodreg(&n1, gc.Types[gc.Tptr], i386.REG_DI)
	agen(nl, &n1)
	gconreg(i386.AMOVL, 0, i386.REG_AX)

	if q > 128 || (q >= 4 && gc.Nacl) {
		gconreg(i386.AMOVL, int64(q), i386.REG_CX)
		gins(i386.AREP, nil, nil)   // repeat
		gins(i386.ASTOSL, nil, nil) // STOL AL,*(DI)+
	} else if q >= 4 {
		p := gins(obj.ADUFFZERO, nil, nil)
		p.To.Type = obj.TYPE_ADDR
		p.To.Sym = gc.Linksym(gc.Pkglookup("duffzero", gc.Runtimepkg))

		// 1 and 128 = magic constants: see ../../runtime/asm_386.s
		p.To.Offset = 1 * (128 - int64(q))
	} else {
		for q > 0 {
			gins(i386.ASTOSL, nil, nil) // STOL AL,*(DI)+
			q--
		}
	}

	for c > 0 {
		gins(i386.ASTOSB, nil, nil) // STOB AL,*(DI)+
		c--
	}
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
				// because the NOP pseudo-instruction will be removed by
				// the linker.
				var reg gc.Node
				gc.Nodreg(&reg, gc.Types[gc.TINT], i386.REG_AX)

				gins(i386.AXCHGL, &reg, &reg)
			}

			p := gins(obj.ACALL, nil, f)
			gc.Afunclit(&p.To, f)
			if proc == -1 || gc.Noreturn(p) {
				gins(obj.AUNDEF, nil, nil)
			}
			break
		}

		var reg gc.Node
		gc.Nodreg(&reg, gc.Types[gc.Tptr], i386.REG_DX)
		var r1 gc.Node
		gc.Nodreg(&r1, gc.Types[gc.Tptr], i386.REG_BX)
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
		stk.Val.U.Reg = i386.REG_SP
		stk.Xoffset = 0

		// size of arguments at 0(SP)
		var con gc.Node
		gc.Nodconst(&con, gc.Types[gc.TINT32], int64(gc.Argsize(f.Type)))

		gins(i386.AMOVL, &con, &stk)

		// FuncVal* at 4(SP)
		stk.Xoffset = int64(gc.Widthptr)

		gins(i386.AMOVL, f, &stk)

		if proc == 1 {
			ginscall(gc.Newproc, 0)
		} else {
			ginscall(gc.Deferproc, 0)
		}
		if proc == 2 {
			var reg gc.Node
			gc.Nodreg(&reg, gc.Types[gc.TINT32], i386.REG_AX)
			gins(i386.ATESTL, &reg, &reg)
			p := gc.Gbranch(i386.AJEQ, nil, +1)
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
	gc.Nodindreg(&nodsp, gc.Types[gc.Tptr], i386.REG_SP)

	nodsp.Xoffset = 0
	if proc != 0 {
		nodsp.Xoffset += 2 * int64(gc.Widthptr) // leave room for size & fn
	}
	nodi.Type = gc.Types[gc.Tptr]
	nodi.Xoffset += int64(gc.Widthptr)
	cgen(&nodi, &nodsp) // {0 or 8}(SP) = 4(REG) -- i.data

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
	gc.Cgen_checknil(&nodo)
	nodo.Op = gc.OINDREG
	nodo.Xoffset = n.Left.Xoffset + 3*int64(gc.Widthptr) + 8

	if proc == 0 {
		// plain call: use direct c function pointer - more efficient
		cgen(&nodo, &nodr) // REG = 20+offset(REG) -- i.tab->fun[f]
		proc = 3
	} else {
		// go/defer. generate go func value.
		gins(i386.ALEAL, &nodo, &nodr) // REG = &(20+offset(REG)) -- i.tab->fun[f]
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
	nod.Val.U.Reg = i386.REG_SP
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
	nod1.Val.U.Reg = i386.REG_SP
	nod1.Addable = 1

	nod1.Xoffset = fp.Width
	nod1.Type = fp.Type

	if res.Op != gc.OREGISTER {
		var nod2 gc.Node
		regalloc(&nod2, gc.Types[gc.Tptr], res)
		gins(i386.ALEAL, &nod1, &nod2)
		gins(i386.AMOVL, &nod2, res)
		regfree(&nod2)
	} else {
		gins(i386.ALEAL, &nod1, res)
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
 * caller must set:
 *	ax = allocated AX register
 *	dx = allocated DX register
 * generates one of:
 *	res = nl / nr
 *	res = nl % nr
 * according to op.
 */
func dodiv(op int, nl *gc.Node, nr *gc.Node, res *gc.Node, ax *gc.Node, dx *gc.Node) {
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
		if gc.Isconst(nl, gc.CTINT) && gc.Mpgetfix(nl.Val.U.Xval) != -1<<uint64(t.Width*8-1) {
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

	var t1 gc.Node
	gc.Tempname(&t1, t)
	var t2 gc.Node
	gc.Tempname(&t2, t)
	if t0 != t {
		var t3 gc.Node
		gc.Tempname(&t3, t0)
		var t4 gc.Node
		gc.Tempname(&t4, t0)
		cgen(nl, &t3)
		cgen(nr, &t4)

		// Convert.
		gmove(&t3, &t1)

		gmove(&t4, &t2)
	} else {
		cgen(nl, &t1)
		cgen(nr, &t2)
	}

	var n1 gc.Node
	if !gc.Samereg(ax, res) && !gc.Samereg(dx, res) {
		regalloc(&n1, t, res)
	} else {
		regalloc(&n1, t, nil)
	}
	gmove(&t2, &n1)
	gmove(&t1, ax)
	p2 := (*obj.Prog)(nil)
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
		ginscall(panicdiv, -1)
		gc.Patch(p1, gc.Pc)
	}

	if check != 0 {
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

	if gc.Issigned[t.Etype] == 0 {
		var nz gc.Node
		gc.Nodconst(&nz, t, 0)
		gmove(&nz, dx)
	} else {
		gins(optoas(gc.OEXTEND, t), nil, nil)
	}
	gins(optoas(op, t), &n1, nil)
	regfree(&n1)

	if op == gc.ODIV {
		gmove(ax, res)
	} else {
		gmove(dx, res)
	}
	if check != 0 {
		gc.Patch(p2, gc.Pc)
	}
}

func savex(dr int, x *gc.Node, oldx *gc.Node, res *gc.Node, t *gc.Type) {
	r := int(reg[dr])
	gc.Nodreg(x, gc.Types[gc.TINT32], dr)

	// save current ax and dx if they are live
	// and not the destination
	*oldx = gc.Node{}

	if r > 0 && !gc.Samereg(x, res) {
		gc.Tempname(oldx, gc.Types[gc.TINT32])
		gmove(x, oldx)
	}

	regalloc(x, t, x)
}

func restx(x *gc.Node, oldx *gc.Node) {
	regfree(x)

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
func cgen_div(op int, nl *gc.Node, nr *gc.Node, res *gc.Node) {
	if gc.Is64(nl.Type) {
		gc.Fatal("cgen_div %v", gc.Tconv(nl.Type, 0))
	}

	var t *gc.Type
	if gc.Issigned[nl.Type.Etype] != 0 {
		t = gc.Types[gc.TINT32]
	} else {
		t = gc.Types[gc.TUINT32]
	}
	var ax gc.Node
	var oldax gc.Node
	savex(i386.REG_AX, &ax, &oldax, res, t)
	var olddx gc.Node
	var dx gc.Node
	savex(i386.REG_DX, &dx, &olddx, res, t)
	dodiv(op, nl, nr, res, &ax, &dx)
	restx(&dx, &olddx)
	restx(&ax, &oldax)
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

	a := optoas(op, nl.Type)

	if nr.Op == gc.OLITERAL {
		var n2 gc.Node
		gc.Tempname(&n2, nl.Type)
		cgen(nl, &n2)
		var n1 gc.Node
		regalloc(&n1, nl.Type, res)
		gmove(&n2, &n1)
		sc := uint64(gc.Mpgetfix(nr.Val.U.Xval))
		if sc >= uint64(nl.Type.Width*8) {
			// large shift gets 2 shifts by width-1
			gins(a, ncon(uint32(w)-1), &n1)

			gins(a, ncon(uint32(w)-1), &n1)
		} else {
			gins(a, nr, &n1)
		}
		gmove(&n1, res)
		regfree(&n1)
		return
	}

	oldcx := gc.Node{}
	var cx gc.Node
	gc.Nodreg(&cx, gc.Types[gc.TUINT32], i386.REG_CX)
	if reg[i386.REG_CX] > 1 && !gc.Samereg(&cx, res) {
		gc.Tempname(&oldcx, gc.Types[gc.TUINT32])
		gmove(&cx, &oldcx)
	}

	var n1 gc.Node
	var nt gc.Node
	if nr.Type.Width > 4 {
		gc.Tempname(&nt, nr.Type)
		n1 = nt
	} else {
		gc.Nodreg(&n1, gc.Types[gc.TUINT32], i386.REG_CX)
		regalloc(&n1, nr.Type, &n1) // to hold the shift type in CX
	}

	var n2 gc.Node
	if gc.Samereg(&cx, res) {
		regalloc(&n2, nl.Type, nil)
	} else {
		regalloc(&n2, nl.Type, res)
	}
	if nl.Ullman >= nr.Ullman {
		cgen(nl, &n2)
		cgen(nr, &n1)
	} else {
		cgen(nr, &n1)
		cgen(nl, &n2)
	}

	// test and fix up large shifts
	if bounded {
		if nr.Type.Width > 4 {
			// delayed reg alloc
			gc.Nodreg(&n1, gc.Types[gc.TUINT32], i386.REG_CX)

			regalloc(&n1, gc.Types[gc.TUINT32], &n1) // to hold the shift type in CX
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
			gc.Nodreg(&n1, gc.Types[gc.TUINT32], i386.REG_CX)

			regalloc(&n1, gc.Types[gc.TUINT32], &n1) // to hold the shift type in CX
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

		if op == gc.ORSH && gc.Issigned[nl.Type.Etype] != 0 {
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

	regfree(&n1)
	regfree(&n2)
}

/*
 * generate byte multiply:
 *	res = nl * nr
 * there is no 2-operand byte multiply instruction so
 * we do a full-width multiplication and truncate afterwards.
 */
func cgen_bmul(op int, nl *gc.Node, nr *gc.Node, res *gc.Node) {
	// copy from byte to full registers
	t := gc.Types[gc.TUINT32]

	if gc.Issigned[nl.Type.Etype] != 0 {
		t = gc.Types[gc.TINT32]
	}

	// largest ullman on left.
	if nl.Ullman < nr.Ullman {
		tmp := nl
		nl = nr
		nr = tmp
	}

	var nt gc.Node
	gc.Tempname(&nt, nl.Type)
	cgen(nl, &nt)
	var n1 gc.Node
	regalloc(&n1, t, res)
	cgen(nr, &n1)
	var n2 gc.Node
	regalloc(&n2, t, nil)
	gmove(&nt, &n2)
	a := optoas(op, t)
	gins(a, &n2, &n1)
	regfree(&n2)
	gmove(&n1, res)
	regfree(&n1)
}

/*
 * generate high multiply:
 *   res = (nl*nr) >> width
 */
func cgen_hmul(nl *gc.Node, nr *gc.Node, res *gc.Node) {
	var n1 gc.Node
	var n2 gc.Node
	var ax gc.Node
	var dx gc.Node

	t := nl.Type
	a := optoas(gc.OHMUL, t)

	// gen nl in n1.
	gc.Tempname(&n1, t)

	cgen(nl, &n1)

	// gen nr in n2.
	regalloc(&n2, t, res)

	cgen(nr, &n2)

	// multiply.
	gc.Nodreg(&ax, t, i386.REG_AX)

	gmove(&n2, &ax)
	gins(a, &n1, nil)
	regfree(&n2)

	if t.Width == 1 {
		// byte multiply behaves differently.
		gc.Nodreg(&ax, t, i386.REG_AH)

		gc.Nodreg(&dx, t, i386.REG_DX)
		gmove(&ax, &dx)
	}

	gc.Nodreg(&dx, t, i386.REG_DX)
	gmove(&dx, res)
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
		bgen(n, true, 0, p2)
		gmove(gc.Nodbool(false), res)
		gc.Patch(p3, gc.Pc)
		return

	case gc.OPLUS:
		cgen(nl, res)
		return

	case gc.OCONV:
		if gc.Eqtype(n.Type, nl.Type) || gc.Noconv(n.Type, nl.Type) {
			cgen(nl, res)
			return
		}

		var n2 gc.Node
		gc.Tempname(&n2, n.Type)
		var n1 gc.Node
		mgen(nl, &n1, res)
		gmove(&n1, &n2)
		gmove(&n2, res)
		mfree(&n1)
		return
	}

	if gc.Use_sse != 0 {
		cgen_floatsse(n, res)
	} else {
		cgen_float387(n, res)
	}
}

// floating-point.  387 (not SSE2)
func cgen_float387(n *gc.Node, res *gc.Node) {
	var f0 gc.Node
	var f1 gc.Node

	nl := n.Left
	nr := n.Right
	gc.Nodreg(&f0, nl.Type, i386.REG_F0)
	gc.Nodreg(&f1, n.Type, i386.REG_F0+1)
	if nr != nil {
		goto flt2
	}

	// unary
	cgen(nl, &f0)

	if n.Op != gc.OCONV && n.Op != gc.OPLUS {
		gins(foptoas(int(n.Op), n.Type, 0), nil, nil)
	}
	gmove(&f0, res)
	return

flt2: // binary
	if nl.Ullman >= nr.Ullman {
		cgen(nl, &f0)
		if nr.Addable != 0 {
			gins(foptoas(int(n.Op), n.Type, 0), nr, &f0)
		} else {
			cgen(nr, &f0)
			gins(foptoas(int(n.Op), n.Type, Fpop), &f0, &f1)
		}
	} else {
		cgen(nr, &f0)
		if nl.Addable != 0 {
			gins(foptoas(int(n.Op), n.Type, Frev), nl, &f0)
		} else {
			cgen(nl, &f0)
			gins(foptoas(int(n.Op), n.Type, Frev|Fpop), &f0, &f1)
		}
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
		gc.Fatal("cgen_floatsse %v", gc.Oconv(int(n.Op), 0))
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
		a = foptoas(int(n.Op), nl.Type, 0)

		goto sbop

		// asymmetric binary
	case gc.OSUB,
		gc.OMOD,
		gc.ODIV:
		a = foptoas(int(n.Op), nl.Type, 0)

		goto abop
	}

sbop: // symmetric binary
	if nl.Ullman < nr.Ullman || nl.Op == gc.OLITERAL {
		r := nl
		nl = nr
		nr = r
	}

abop: // asymmetric binary
	if nl.Ullman >= nr.Ullman {
		var nt gc.Node
		gc.Tempname(&nt, nl.Type)
		cgen(nl, &nt)
		var n2 gc.Node
		mgen(nr, &n2, nil)
		var n1 gc.Node
		regalloc(&n1, nl.Type, res)
		gmove(&nt, &n1)
		gins(a, &n2, &n1)
		gmove(&n1, res)
		regfree(&n1)
		mfree(&n2)
	} else {
		var n2 gc.Node
		regalloc(&n2, nr.Type, res)
		cgen(nr, &n2)
		var n1 gc.Node
		regalloc(&n1, nl.Type, nil)
		cgen(nl, &n1)
		gins(a, &n2, &n1)
		regfree(&n2)
		gmove(&n1, res)
		regfree(&n1)
	}

	return
}

func bgen_float(n *gc.Node, true_ int, likely int, to *obj.Prog) {
	nl := n.Left
	nr := n.Right
	a := int(n.Op)
	if true_ == 0 {
		// brcom is not valid on floats when NaN is involved.
		p1 := gc.Gbranch(obj.AJMP, nil, 0)

		p2 := gc.Gbranch(obj.AJMP, nil, 0)
		gc.Patch(p1, gc.Pc)

		// No need to avoid re-genning ninit.
		bgen_float(n, 1, -likely, p2)

		gc.Patch(gc.Gbranch(obj.AJMP, nil, 0), to)
		gc.Patch(p2, gc.Pc)
		return
	}

	var tmp gc.Node
	var et int
	var n2 gc.Node
	var ax gc.Node
	if gc.Use_sse != 0 {
		goto sse
	} else {
		goto x87
	}

x87:
	a = gc.Brrev(a) // because the args are stacked
	if a == gc.OGE || a == gc.OGT {
		// only < and <= work right with NaN; reverse if needed
		r := nr

		nr = nl
		nl = r
		a = gc.Brrev(a)
	}

	gc.Nodreg(&tmp, nr.Type, i386.REG_F0)
	gc.Nodreg(&n2, nr.Type, i386.REG_F0+1)
	gc.Nodreg(&ax, gc.Types[gc.TUINT16], i386.REG_AX)
	et = gc.Simsimtype(nr.Type)
	if et == gc.TFLOAT64 {
		if nl.Ullman > nr.Ullman {
			cgen(nl, &tmp)
			cgen(nr, &tmp)
			gins(i386.AFXCHD, &tmp, &n2)
		} else {
			cgen(nr, &tmp)
			cgen(nl, &tmp)
		}

		gins(i386.AFUCOMIP, &tmp, &n2)
		gins(i386.AFMOVDP, &tmp, &tmp) // annoying pop but still better than STSW+SAHF
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
		cgen(nr, &t1)
		cgen(nl, &t2)
		gmove(&t2, &tmp)
		gins(i386.AFCOMFP, &t1, &tmp)
		gins(i386.AFSTSW, nil, &ax)
		gins(i386.ASAHF, nil, nil)
	}

	goto ret

sse:
	if nl.Addable == 0 {
		var n1 gc.Node
		gc.Tempname(&n1, nl.Type)
		cgen(nl, &n1)
		nl = &n1
	}

	if nr.Addable == 0 {
		var tmp gc.Node
		gc.Tempname(&tmp, nr.Type)
		cgen(nr, &tmp)
		nr = &tmp
	}

	regalloc(&n2, nr.Type, nil)
	gmove(nr, &n2)
	nr = &n2

	if nl.Op != gc.OREGISTER {
		var n3 gc.Node
		regalloc(&n3, nl.Type, nil)
		gmove(nl, &n3)
		nl = &n3
	}

	if a == gc.OGE || a == gc.OGT {
		// only < and <= work right with NaN; reverse if needed
		r := nr

		nr = nl
		nl = r
		a = gc.Brrev(a)
	}

	gins(foptoas(gc.OCMP, nr.Type, 0), nl, nr)
	if nl.Op == gc.OREGISTER {
		regfree(nl)
	}
	regfree(nr)

ret:
	if a == gc.OEQ {
		// neither NE nor P
		p1 := gc.Gbranch(i386.AJNE, nil, -likely)

		p2 := gc.Gbranch(i386.AJPS, nil, -likely)
		gc.Patch(gc.Gbranch(obj.AJMP, nil, 0), to)
		gc.Patch(p1, gc.Pc)
		gc.Patch(p2, gc.Pc)
	} else if a == gc.ONE {
		// either NE or P
		gc.Patch(gc.Gbranch(i386.AJNE, nil, likely), to)

		gc.Patch(gc.Gbranch(i386.AJPS, nil, likely), to)
	} else {
		gc.Patch(gc.Gbranch(optoas(a, nr.Type), nil, likely), to)
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
		p.As = i386.ACMPL
		p.To.Type = obj.TYPE_CONST
		p.To.Offset = 0
		p1.As = i386.AJNE
		p1.From.Type = obj.TYPE_CONST
		p1.From.Offset = 1 // likely
		p1.To.Type = obj.TYPE_BRANCH
		p1.To.U.Branch = p2.Link

		// crash by write to memory address 0.
		// if possible, since we know arg is 0, use 0(arg),
		// which will be shorter to encode than plain 0.
		p2.As = i386.AMOVL

		p2.From.Type = obj.TYPE_REG
		p2.From.Reg = i386.REG_AX
		if regtyp(&p.From) {
			p2.To.Type = obj.TYPE_MEM
			p2.To.Reg = p.From.Reg
		} else {
			p2.To.Type = obj.TYPE_MEM
		}
		p2.To.Offset = 0
	}
}
