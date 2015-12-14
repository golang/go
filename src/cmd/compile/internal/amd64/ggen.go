// Copyright 2009 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package amd64

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
	x0 := uint32(0)

	// iterate through declarations - they are sorted in decreasing xoffset order.
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

		if lo != hi && n.Xoffset+n.Type.Width >= lo-int64(2*gc.Widthreg) {
			// merge with range we already have
			lo = n.Xoffset

			continue
		}

		// zero old range
		p = zerorange(p, int64(frame), lo, hi, &ax, &x0)

		// set new range
		hi = n.Xoffset + n.Type.Width

		lo = n.Xoffset
	}

	// zero final range
	zerorange(p, int64(frame), lo, hi, &ax, &x0)
}

// DUFFZERO consists of repeated blocks of 4 MOVUPSs + ADD,
// See runtime/mkduff.go.
const (
	dzBlocks    = 16 // number of MOV/ADD blocks
	dzBlockLen  = 4  // number of clears per block
	dzBlockSize = 19 // size of instructions in a single block
	dzMovSize   = 4  // size of single MOV instruction w/ offset
	dzAddSize   = 4  // size of single ADD instruction
	dzClearStep = 16 // number of bytes cleared by each MOV instruction

	dzClearLen = dzClearStep * dzBlockLen // bytes cleared by one block
	dzSize     = dzBlocks * dzBlockSize
)

// dzOff returns the offset for a jump into DUFFZERO.
// b is the number of bytes to zero.
func dzOff(b int64) int64 {
	off := int64(dzSize)
	off -= b / dzClearLen * dzBlockSize
	tailLen := b % dzClearLen
	if tailLen >= dzClearStep {
		off -= dzAddSize + dzMovSize*(tailLen/dzClearStep)
	}
	return off
}

// duffzeroDI returns the pre-adjustment to DI for a call to DUFFZERO.
// b is the number of bytes to zero.
func dzDI(b int64) int64 {
	tailLen := b % dzClearLen
	if tailLen < dzClearStep {
		return 0
	}
	tailSteps := tailLen / dzClearStep
	return -dzClearStep * (dzBlockLen - tailSteps)
}

func zerorange(p *obj.Prog, frame int64, lo int64, hi int64, ax *uint32, x0 *uint32) *obj.Prog {
	cnt := hi - lo
	if cnt == 0 {
		return p
	}

	if cnt%int64(gc.Widthreg) != 0 {
		// should only happen with nacl
		if cnt%int64(gc.Widthptr) != 0 {
			gc.Fatalf("zerorange count not a multiple of widthptr %d", cnt)
		}
		if *ax == 0 {
			p = appendpp(p, x86.AMOVQ, obj.TYPE_CONST, 0, 0, obj.TYPE_REG, x86.REG_AX, 0)
			*ax = 1
		}
		p = appendpp(p, x86.AMOVL, obj.TYPE_REG, x86.REG_AX, 0, obj.TYPE_MEM, x86.REG_SP, frame+lo)
		lo += int64(gc.Widthptr)
		cnt -= int64(gc.Widthptr)
	}

	if cnt == 8 {
		if *ax == 0 {
			p = appendpp(p, x86.AMOVQ, obj.TYPE_CONST, 0, 0, obj.TYPE_REG, x86.REG_AX, 0)
			*ax = 1
		}
		p = appendpp(p, x86.AMOVQ, obj.TYPE_REG, x86.REG_AX, 0, obj.TYPE_MEM, x86.REG_SP, frame+lo)
	} else if cnt <= int64(8*gc.Widthreg) {
		if *x0 == 0 {
			p = appendpp(p, x86.AXORPS, obj.TYPE_REG, x86.REG_X0, 0, obj.TYPE_REG, x86.REG_X0, 0)
			*x0 = 1
		}

		for i := int64(0); i < cnt/16; i++ {
			p = appendpp(p, x86.AMOVUPS, obj.TYPE_REG, x86.REG_X0, 0, obj.TYPE_MEM, x86.REG_SP, frame+lo+i*16)
		}

		if cnt%16 != 0 {
			p = appendpp(p, x86.AMOVUPS, obj.TYPE_REG, x86.REG_X0, 0, obj.TYPE_MEM, x86.REG_SP, frame+lo+cnt-int64(16))
		}
	} else if !gc.Nacl && (cnt <= int64(128*gc.Widthreg)) {
		if *x0 == 0 {
			p = appendpp(p, x86.AXORPS, obj.TYPE_REG, x86.REG_X0, 0, obj.TYPE_REG, x86.REG_X0, 0)
			*x0 = 1
		}

		p = appendpp(p, leaptr, obj.TYPE_MEM, x86.REG_SP, frame+lo+dzDI(cnt), obj.TYPE_REG, x86.REG_DI, 0)
		p = appendpp(p, obj.ADUFFZERO, obj.TYPE_NONE, 0, 0, obj.TYPE_ADDR, 0, dzOff(cnt))
		p.To.Sym = gc.Linksym(gc.Pkglookup("duffzero", gc.Runtimepkg))

		if cnt%16 != 0 {
			p = appendpp(p, x86.AMOVUPS, obj.TYPE_REG, x86.REG_X0, 0, obj.TYPE_MEM, x86.REG_DI, -int64(8))
		}
	} else {
		if *ax == 0 {
			p = appendpp(p, x86.AMOVQ, obj.TYPE_CONST, 0, 0, obj.TYPE_REG, x86.REG_AX, 0)
			*ax = 1
		}

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
		if gc.Isconst(nl, gc.CTINT) && nl.Int() != -(1<<uint64(t.Width*8-1)) {
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

	a := optoas(op, t)

	var n3 gc.Node
	gc.Regalloc(&n3, t0, nil)
	var ax gc.Node
	var oldax gc.Node
	if nl.Ullman >= nr.Ullman {
		savex(x86.REG_AX, &ax, &oldax, res, t0)
		gc.Cgen(nl, &ax)
		gc.Regalloc(&ax, t0, &ax) // mark ax live during cgen
		gc.Cgen(nr, &n3)
		gc.Regfree(&ax)
	} else {
		gc.Cgen(nr, &n3)
		savex(x86.REG_AX, &ax, &oldax, res, t0)
		gc.Cgen(nl, &ax)
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
		gc.Ginscall(panicdiv, -1)
		gc.Patch(p1, gc.Pc)
	}

	var p2 *obj.Prog
	if check {
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
	if !gc.Issigned[t.Etype] {
		gc.Nodconst(&n4, t, 0)
		gmove(&n4, &dx)
	} else {
		gins(optoas(gc.OEXTEND, t), nil, nil)
	}
	gins(a, &n3, nil)
	gc.Regfree(&n3)
	if op == gc.ODIV {
		gmove(&ax, res)
	} else {
		gmove(&dx, res)
	}
	restx(&dx, &olddx)
	if check {
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
	r := uint8(gc.GetReg(dr))

	// save current ax and dx if they are live
	// and not the destination
	*oldx = gc.Node{}

	gc.Nodreg(x, t, dr)
	if r > 1 && !gc.Samereg(x, res) {
		gc.Regalloc(oldx, gc.Types[gc.TINT64], nil)
		x.Type = gc.Types[gc.TINT64]
		gmove(x, oldx)
		x.Type = t
		// TODO(marvin): Fix Node.EType type union.
		oldx.Etype = gc.EType(r) // squirrel away old r value
		gc.SetReg(dr, 1)
	}
}

func restx(x *gc.Node, oldx *gc.Node) {
	if oldx.Op != 0 {
		x.Type = gc.Types[gc.TINT64]
		gc.SetReg(int(x.Reg), int(oldx.Etype))
		gmove(oldx, x)
		gc.Regfree(oldx)
	}
}

/*
 * generate high multiply:
 *   res = (nl*nr) >> width
 */
func cgen_hmul(nl *gc.Node, nr *gc.Node, res *gc.Node) {
	t := nl.Type
	a := optoas(gc.OHMUL, t)
	if nl.Ullman < nr.Ullman {
		nl, nr = nr, nl
	}

	var n1 gc.Node
	gc.Cgenr(nl, &n1, res)
	var n2 gc.Node
	gc.Cgenr(nr, &n2, nil)
	var ax, oldax, dx, olddx gc.Node
	savex(x86.REG_AX, &ax, &oldax, res, gc.Types[gc.TUINT64])
	savex(x86.REG_DX, &dx, &olddx, res, gc.Types[gc.TUINT64])
	gmove(&n1, &ax)
	gins(a, &n2, nil)
	gc.Regfree(&n2)
	gc.Regfree(&n1)

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
		sc := uint64(nr.Int())
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

	rcx := gc.GetReg(x86.REG_CX)
	var n1 gc.Node
	gc.Nodreg(&n1, gc.Types[gc.TUINT32], x86.REG_CX)

	// Allow either uint32 or uint64 as shift type,
	// to avoid unnecessary conversion from uint32 to uint64
	// just to do the comparison.
	tcount := gc.Types[gc.Simtype[nr.Type.Etype]]

	if tcount.Etype < gc.TUINT32 {
		tcount = gc.Types[gc.TUINT32]
	}

	gc.Regalloc(&n1, nr.Type, &n1) // to hold the shift type in CX
	var n3 gc.Node
	gc.Regalloc(&n3, tcount, &n1) // to clear high bits of CX

	var cx gc.Node
	gc.Nodreg(&cx, gc.Types[gc.TUINT64], x86.REG_CX)

	var oldcx gc.Node
	if rcx > 0 && !gc.Samereg(&cx, res) {
		gc.Regalloc(&oldcx, gc.Types[gc.TUINT64], nil)
		gmove(&cx, &oldcx)
	}

	cx.Type = tcount

	var n2 gc.Node
	if gc.Samereg(&cx, res) {
		gc.Regalloc(&n2, nl.Type, nil)
	} else {
		gc.Regalloc(&n2, nl.Type, res)
	}
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
		p1 := gc.Gbranch(optoas(gc.OLT, tcount), nil, +1)
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

	if oldcx.Op != 0 {
		cx.Type = gc.Types[gc.TUINT64]
		gmove(&oldcx, &cx)
		gc.Regfree(&oldcx)
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

	// largest ullman on left.
	if nl.Ullman < nr.Ullman {
		nl, nr = nr, nl
	}

	// generate operands in "8-bit" registers.
	var n1b gc.Node
	gc.Regalloc(&n1b, nl.Type, res)

	gc.Cgen(nl, &n1b)
	var n2b gc.Node
	gc.Regalloc(&n2b, nr.Type, nil)
	gc.Cgen(nr, &n2b)

	// perform full-width multiplication.
	t := gc.Types[gc.TUINT64]

	if gc.Issigned[nl.Type.Etype] {
		t = gc.Types[gc.TINT64]
	}
	var n1 gc.Node
	gc.Nodreg(&n1, t, int(n1b.Reg))
	var n2 gc.Node
	gc.Nodreg(&n2, t, int(n2b.Reg))
	a := optoas(op, t)
	gins(a, &n2, &n1)

	// truncate.
	gmove(&n1, res)

	gc.Regfree(&n1b)
	gc.Regfree(&n2b)
	return true
}

func clearfat(nl *gc.Node) {
	/* clear a fat object */
	if gc.Debug['g'] != 0 {
		gc.Dump("\nclearfat", nl)
	}

	// Avoid taking the address for simple enough types.
	if gc.Componentgen(nil, nl) {
		return
	}

	w := nl.Type.Width

	if w > 1024 || (gc.Nacl && w >= 64) {
		var oldn1 gc.Node
		var n1 gc.Node
		savex(x86.REG_DI, &n1, &oldn1, nil, gc.Types[gc.Tptr])
		gc.Agen(nl, &n1)

		var ax gc.Node
		var oldax gc.Node
		savex(x86.REG_AX, &ax, &oldax, nil, gc.Types[gc.Tptr])
		gconreg(x86.AMOVL, 0, x86.REG_AX)
		gconreg(movptr, w/8, x86.REG_CX)

		gins(x86.AREP, nil, nil)   // repeat
		gins(x86.ASTOSQ, nil, nil) // STOQ AL,*(DI)+

		if w%8 != 0 {
			n1.Op = gc.OINDREG
			clearfat_tail(&n1, w%8)
		}

		restx(&n1, &oldn1)
		restx(&ax, &oldax)
		return
	}

	if w >= 64 {
		var oldn1 gc.Node
		var n1 gc.Node
		savex(x86.REG_DI, &n1, &oldn1, nil, gc.Types[gc.Tptr])
		gc.Agen(nl, &n1)

		var vec_zero gc.Node
		var old_x0 gc.Node
		savex(x86.REG_X0, &vec_zero, &old_x0, nil, gc.Types[gc.TFLOAT64])
		gins(x86.AXORPS, &vec_zero, &vec_zero)

		if di := dzDI(w); di != 0 {
			gconreg(addptr, di, x86.REG_DI)
		}
		p := gins(obj.ADUFFZERO, nil, nil)
		p.To.Type = obj.TYPE_ADDR
		p.To.Sym = gc.Linksym(gc.Pkglookup("duffzero", gc.Runtimepkg))
		p.To.Offset = dzOff(w)

		if w%16 != 0 {
			n1.Op = gc.OINDREG
			n1.Xoffset -= 16 - w%16
			gins(x86.AMOVUPS, &vec_zero, &n1)
		}

		restx(&vec_zero, &old_x0)
		restx(&n1, &oldn1)
		return
	}

	// NOTE: Must use agen, not igen, so that optimizer sees address
	// being taken. We are not writing on field boundaries.
	var n1 gc.Node
	gc.Agenr(nl, &n1, nil)
	n1.Op = gc.OINDREG

	clearfat_tail(&n1, w)

	gc.Regfree(&n1)
}

func clearfat_tail(n1 *gc.Node, b int64) {
	if b >= 16 {
		var vec_zero gc.Node
		gc.Regalloc(&vec_zero, gc.Types[gc.TFLOAT64], nil)
		gins(x86.AXORPS, &vec_zero, &vec_zero)

		for b >= 16 {
			gins(x86.AMOVUPS, &vec_zero, n1)
			n1.Xoffset += 16
			b -= 16
		}

		// MOVUPS X0, off(base) is a few bytes shorter than MOV 0, off(base)
		if b != 0 {
			n1.Xoffset -= 16 - b
			gins(x86.AMOVUPS, &vec_zero, n1)
		}

		gc.Regfree(&vec_zero)
		return
	}

	// Write sequence of MOV 0, off(base) instead of using STOSQ.
	// The hope is that although the code will be slightly longer,
	// the MOVs will have no dependencies and pipeline better
	// than the unrolled STOSQ loop.
	var z gc.Node
	gc.Nodconst(&z, gc.Types[gc.TUINT64], 0)
	if b >= 8 {
		n1.Type = z.Type
		gins(x86.AMOVQ, &z, n1)
		n1.Xoffset += 8
		b -= 8

		if b != 0 {
			n1.Xoffset -= 8 - b
			gins(x86.AMOVQ, &z, n1)
		}
		return
	}

	if b >= 4 {
		gc.Nodconst(&z, gc.Types[gc.TUINT32], 0)
		n1.Type = z.Type
		gins(x86.AMOVL, &z, n1)
		n1.Xoffset += 4
		b -= 4

		if b != 0 {
			n1.Xoffset -= 4 - b
			gins(x86.AMOVL, &z, n1)
		}
		return
	}

	if b >= 2 {
		gc.Nodconst(&z, gc.Types[gc.TUINT16], 0)
		n1.Type = z.Type
		gins(x86.AMOVW, &z, n1)
		n1.Xoffset += 2
		b -= 2
	}

	gc.Nodconst(&z, gc.Types[gc.TUINT8], 0)
	for b > 0 {
		n1.Type = z.Type
		gins(x86.AMOVB, &z, n1)
		n1.Xoffset++
		b--
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
		p.As = int16(cmpptr)
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
			p2.To.Reg = x86.REG_NONE
		}

		p2.To.Offset = 0
	}
}

// addr += index*width if possible.
func addindex(index *gc.Node, width int64, addr *gc.Node) bool {
	switch width {
	case 1, 2, 4, 8:
		p1 := gins(x86.ALEAQ, index, addr)
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
