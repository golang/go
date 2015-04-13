// Derived from Inferno utils/6c/txt.c
// http://code.google.com/p/inferno-os/source/browse/utils/6c/txt.c
//
//	Copyright © 1994-1999 Lucent Technologies Inc.  All rights reserved.
//	Portions Copyright © 1995-1997 C H Forsyth (forsyth@terzarima.net)
//	Portions Copyright © 1997-1999 Vita Nuova Limited
//	Portions Copyright © 2000-2007 Vita Nuova Holdings Limited (www.vitanuova.com)
//	Portions Copyright © 2004,2006 Bruce Ellis
//	Portions Copyright © 2005-2007 C H Forsyth (forsyth@terzarima.net)
//	Revisions Copyright © 2000-2007 Lucent Technologies Inc. and others
//	Portions Copyright © 2009 The Go Authors.  All rights reserved.
//
// Permission is hereby granted, free of charge, to any person obtaining a copy
// of this software and associated documentation files (the "Software"), to deal
// in the Software without restriction, including without limitation the rights
// to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
// copies of the Software, and to permit persons to whom the Software is
// furnished to do so, subject to the following conditions:
//
// The above copyright notice and this permission notice shall be included in
// all copies or substantial portions of the Software.
//
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
// IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
// FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.  IN NO EVENT SHALL THE
// AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
// LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
// THE SOFTWARE.

package main

import (
	"cmd/internal/gc"
	"cmd/internal/obj"
	"cmd/internal/obj/x86"
	"fmt"
)

var resvd = []int{
	x86.REG_DI, // for movstring
	x86.REG_SI, // for movstring

	x86.REG_AX, // for divide
	x86.REG_CX, // for shift
	x86.REG_DX, // for divide
	x86.REG_SP, // for stack
}

/*
 * generate
 *	as $c, reg
 */
func gconreg(as int, c int64, reg int) {
	var nr gc.Node

	switch as {
	case x86.AADDL,
		x86.AMOVL,
		x86.ALEAL:
		gc.Nodreg(&nr, gc.Types[gc.TINT32], reg)

	default:
		gc.Nodreg(&nr, gc.Types[gc.TINT64], reg)
	}

	ginscon(as, c, &nr)
}

/*
 * generate
 *	as $c, n
 */
func ginscon(as int, c int64, n2 *gc.Node) {
	var n1 gc.Node

	switch as {
	case x86.AADDL,
		x86.AMOVL,
		x86.ALEAL:
		gc.Nodconst(&n1, gc.Types[gc.TINT32], c)

	default:
		gc.Nodconst(&n1, gc.Types[gc.TINT64], c)
	}

	if as != x86.AMOVQ && (c < -(1<<31) || c >= 1<<31) {
		// cannot have 64-bit immediate in ADD, etc.
		// instead, MOV into register first.
		var ntmp gc.Node
		gc.Regalloc(&ntmp, gc.Types[gc.TINT64], nil)

		gins(x86.AMOVQ, &n1, &ntmp)
		gins(as, &ntmp, n2)
		gc.Regfree(&ntmp)
		return
	}

	gins(as, &n1, n2)
}

/*
 * set up nodes representing 2^63
 */
var bigi gc.Node

var bigf gc.Node

var bignodes_did int

func bignodes() {
	if bignodes_did != 0 {
		return
	}
	bignodes_did = 1

	gc.Nodconst(&bigi, gc.Types[gc.TUINT64], 1)
	gc.Mpshiftfix(bigi.Val.U.Xval, 63)

	bigf = bigi
	bigf.Type = gc.Types[gc.TFLOAT64]
	bigf.Val.Ctype = gc.CTFLT
	bigf.Val.U.Fval = new(gc.Mpflt)
	gc.Mpmovefixflt(bigf.Val.U.Fval, bigi.Val.U.Xval)
}

/*
 * generate move:
 *	t = f
 * hard part is conversions.
 */
func gmove(f *gc.Node, t *gc.Node) {
	if gc.Debug['M'] != 0 {
		fmt.Printf("gmove %v -> %v\n", gc.Nconv(f, obj.FmtLong), gc.Nconv(t, obj.FmtLong))
	}

	ft := gc.Simsimtype(f.Type)
	tt := gc.Simsimtype(t.Type)
	cvt := t.Type

	if gc.Iscomplex[ft] || gc.Iscomplex[tt] {
		gc.Complexmove(f, t)
		return
	}

	// cannot have two memory operands
	var a int
	if gc.Ismem(f) && gc.Ismem(t) {
		goto hard
	}

	// convert constant to desired type
	if f.Op == gc.OLITERAL {
		var con gc.Node
		gc.Convconst(&con, t.Type, &f.Val)
		f = &con
		ft = tt // so big switch will choose a simple mov

		// some constants can't move directly to memory.
		if gc.Ismem(t) {
			// float constants come from memory.
			if gc.Isfloat[tt] {
				goto hard
			}

			// 64-bit immediates are really 32-bit sign-extended
			// unless moving into a register.
			if gc.Isint[tt] {
				if gc.Mpcmpfixfix(con.Val.U.Xval, gc.Minintval[gc.TINT32]) < 0 {
					goto hard
				}
				if gc.Mpcmpfixfix(con.Val.U.Xval, gc.Maxintval[gc.TINT32]) > 0 {
					goto hard
				}
			}
		}
	}

	// value -> value copy, only one memory operand.
	// figure out the instruction to use.
	// break out of switch for one-instruction gins.
	// goto rdst for "destination must be register".
	// goto hard for "convert to cvt type first".
	// otherwise handle and return.

	switch uint32(ft)<<16 | uint32(tt) {
	default:
		gc.Fatal("gmove %v -> %v", gc.Tconv(f.Type, obj.FmtLong), gc.Tconv(t.Type, obj.FmtLong))

		/*
		 * integer copy and truncate
		 */
	case gc.TINT8<<16 | gc.TINT8, // same size
		gc.TINT8<<16 | gc.TUINT8,
		gc.TUINT8<<16 | gc.TINT8,
		gc.TUINT8<<16 | gc.TUINT8,
		gc.TINT16<<16 | gc.TINT8,
		// truncate
		gc.TUINT16<<16 | gc.TINT8,
		gc.TINT32<<16 | gc.TINT8,
		gc.TUINT32<<16 | gc.TINT8,
		gc.TINT64<<16 | gc.TINT8,
		gc.TUINT64<<16 | gc.TINT8,
		gc.TINT16<<16 | gc.TUINT8,
		gc.TUINT16<<16 | gc.TUINT8,
		gc.TINT32<<16 | gc.TUINT8,
		gc.TUINT32<<16 | gc.TUINT8,
		gc.TINT64<<16 | gc.TUINT8,
		gc.TUINT64<<16 | gc.TUINT8:
		a = x86.AMOVB

	case gc.TINT16<<16 | gc.TINT16, // same size
		gc.TINT16<<16 | gc.TUINT16,
		gc.TUINT16<<16 | gc.TINT16,
		gc.TUINT16<<16 | gc.TUINT16,
		gc.TINT32<<16 | gc.TINT16,
		// truncate
		gc.TUINT32<<16 | gc.TINT16,
		gc.TINT64<<16 | gc.TINT16,
		gc.TUINT64<<16 | gc.TINT16,
		gc.TINT32<<16 | gc.TUINT16,
		gc.TUINT32<<16 | gc.TUINT16,
		gc.TINT64<<16 | gc.TUINT16,
		gc.TUINT64<<16 | gc.TUINT16:
		a = x86.AMOVW

	case gc.TINT32<<16 | gc.TINT32, // same size
		gc.TINT32<<16 | gc.TUINT32,
		gc.TUINT32<<16 | gc.TINT32,
		gc.TUINT32<<16 | gc.TUINT32:
		a = x86.AMOVL

	case gc.TINT64<<16 | gc.TINT32, // truncate
		gc.TUINT64<<16 | gc.TINT32,
		gc.TINT64<<16 | gc.TUINT32,
		gc.TUINT64<<16 | gc.TUINT32:
		a = x86.AMOVQL

	case gc.TINT64<<16 | gc.TINT64, // same size
		gc.TINT64<<16 | gc.TUINT64,
		gc.TUINT64<<16 | gc.TINT64,
		gc.TUINT64<<16 | gc.TUINT64:
		a = x86.AMOVQ

		/*
		 * integer up-conversions
		 */
	case gc.TINT8<<16 | gc.TINT16, // sign extend int8
		gc.TINT8<<16 | gc.TUINT16:
		a = x86.AMOVBWSX

		goto rdst

	case gc.TINT8<<16 | gc.TINT32,
		gc.TINT8<<16 | gc.TUINT32:
		a = x86.AMOVBLSX
		goto rdst

	case gc.TINT8<<16 | gc.TINT64,
		gc.TINT8<<16 | gc.TUINT64:
		a = x86.AMOVBQSX
		goto rdst

	case gc.TUINT8<<16 | gc.TINT16, // zero extend uint8
		gc.TUINT8<<16 | gc.TUINT16:
		a = x86.AMOVBWZX

		goto rdst

	case gc.TUINT8<<16 | gc.TINT32,
		gc.TUINT8<<16 | gc.TUINT32:
		a = x86.AMOVBLZX
		goto rdst

	case gc.TUINT8<<16 | gc.TINT64,
		gc.TUINT8<<16 | gc.TUINT64:
		a = x86.AMOVBQZX
		goto rdst

	case gc.TINT16<<16 | gc.TINT32, // sign extend int16
		gc.TINT16<<16 | gc.TUINT32:
		a = x86.AMOVWLSX

		goto rdst

	case gc.TINT16<<16 | gc.TINT64,
		gc.TINT16<<16 | gc.TUINT64:
		a = x86.AMOVWQSX
		goto rdst

	case gc.TUINT16<<16 | gc.TINT32, // zero extend uint16
		gc.TUINT16<<16 | gc.TUINT32:
		a = x86.AMOVWLZX

		goto rdst

	case gc.TUINT16<<16 | gc.TINT64,
		gc.TUINT16<<16 | gc.TUINT64:
		a = x86.AMOVWQZX
		goto rdst

	case gc.TINT32<<16 | gc.TINT64, // sign extend int32
		gc.TINT32<<16 | gc.TUINT64:
		a = x86.AMOVLQSX

		goto rdst

		// AMOVL into a register zeros the top of the register,
	// so this is not always necessary, but if we rely on AMOVL
	// the optimizer is almost certain to screw with us.
	case gc.TUINT32<<16 | gc.TINT64, // zero extend uint32
		gc.TUINT32<<16 | gc.TUINT64:
		a = x86.AMOVLQZX

		goto rdst

		/*
		* float to integer
		 */
	case gc.TFLOAT32<<16 | gc.TINT32:
		a = x86.ACVTTSS2SL

		goto rdst

	case gc.TFLOAT64<<16 | gc.TINT32:
		a = x86.ACVTTSD2SL
		goto rdst

	case gc.TFLOAT32<<16 | gc.TINT64:
		a = x86.ACVTTSS2SQ
		goto rdst

	case gc.TFLOAT64<<16 | gc.TINT64:
		a = x86.ACVTTSD2SQ
		goto rdst

		// convert via int32.
	case gc.TFLOAT32<<16 | gc.TINT16,
		gc.TFLOAT32<<16 | gc.TINT8,
		gc.TFLOAT32<<16 | gc.TUINT16,
		gc.TFLOAT32<<16 | gc.TUINT8,
		gc.TFLOAT64<<16 | gc.TINT16,
		gc.TFLOAT64<<16 | gc.TINT8,
		gc.TFLOAT64<<16 | gc.TUINT16,
		gc.TFLOAT64<<16 | gc.TUINT8:
		cvt = gc.Types[gc.TINT32]

		goto hard

		// convert via int64.
	case gc.TFLOAT32<<16 | gc.TUINT32,
		gc.TFLOAT64<<16 | gc.TUINT32:
		cvt = gc.Types[gc.TINT64]

		goto hard

		// algorithm is:
	//	if small enough, use native float64 -> int64 conversion.
	//	otherwise, subtract 2^63, convert, and add it back.
	case gc.TFLOAT32<<16 | gc.TUINT64,
		gc.TFLOAT64<<16 | gc.TUINT64:
		a := x86.ACVTTSS2SQ

		if ft == gc.TFLOAT64 {
			a = x86.ACVTTSD2SQ
		}
		bignodes()
		var r1 gc.Node
		gc.Regalloc(&r1, gc.Types[ft], nil)
		var r2 gc.Node
		gc.Regalloc(&r2, gc.Types[tt], t)
		var r3 gc.Node
		gc.Regalloc(&r3, gc.Types[ft], nil)
		var r4 gc.Node
		gc.Regalloc(&r4, gc.Types[tt], nil)
		gins(optoas(gc.OAS, f.Type), f, &r1)
		gins(optoas(gc.OCMP, f.Type), &bigf, &r1)
		p1 := gc.Gbranch(optoas(gc.OLE, f.Type), nil, +1)
		gins(a, &r1, &r2)
		p2 := gc.Gbranch(obj.AJMP, nil, 0)
		gc.Patch(p1, gc.Pc)
		gins(optoas(gc.OAS, f.Type), &bigf, &r3)
		gins(optoas(gc.OSUB, f.Type), &r3, &r1)
		gins(a, &r1, &r2)
		gins(x86.AMOVQ, &bigi, &r4)
		gins(x86.AXORQ, &r4, &r2)
		gc.Patch(p2, gc.Pc)
		gmove(&r2, t)
		gc.Regfree(&r4)
		gc.Regfree(&r3)
		gc.Regfree(&r2)
		gc.Regfree(&r1)
		return

		/*
		 * integer to float
		 */
	case gc.TINT32<<16 | gc.TFLOAT32:
		a = x86.ACVTSL2SS

		goto rdst

	case gc.TINT32<<16 | gc.TFLOAT64:
		a = x86.ACVTSL2SD
		goto rdst

	case gc.TINT64<<16 | gc.TFLOAT32:
		a = x86.ACVTSQ2SS
		goto rdst

	case gc.TINT64<<16 | gc.TFLOAT64:
		a = x86.ACVTSQ2SD
		goto rdst

		// convert via int32
	case gc.TINT16<<16 | gc.TFLOAT32,
		gc.TINT16<<16 | gc.TFLOAT64,
		gc.TINT8<<16 | gc.TFLOAT32,
		gc.TINT8<<16 | gc.TFLOAT64,
		gc.TUINT16<<16 | gc.TFLOAT32,
		gc.TUINT16<<16 | gc.TFLOAT64,
		gc.TUINT8<<16 | gc.TFLOAT32,
		gc.TUINT8<<16 | gc.TFLOAT64:
		cvt = gc.Types[gc.TINT32]

		goto hard

		// convert via int64.
	case gc.TUINT32<<16 | gc.TFLOAT32,
		gc.TUINT32<<16 | gc.TFLOAT64:
		cvt = gc.Types[gc.TINT64]

		goto hard

		// algorithm is:
	//	if small enough, use native int64 -> uint64 conversion.
	//	otherwise, halve (rounding to odd?), convert, and double.
	case gc.TUINT64<<16 | gc.TFLOAT32,
		gc.TUINT64<<16 | gc.TFLOAT64:
		a := x86.ACVTSQ2SS

		if tt == gc.TFLOAT64 {
			a = x86.ACVTSQ2SD
		}
		var zero gc.Node
		gc.Nodconst(&zero, gc.Types[gc.TUINT64], 0)
		var one gc.Node
		gc.Nodconst(&one, gc.Types[gc.TUINT64], 1)
		var r1 gc.Node
		gc.Regalloc(&r1, f.Type, f)
		var r2 gc.Node
		gc.Regalloc(&r2, t.Type, t)
		var r3 gc.Node
		gc.Regalloc(&r3, f.Type, nil)
		var r4 gc.Node
		gc.Regalloc(&r4, f.Type, nil)
		gmove(f, &r1)
		gins(x86.ACMPQ, &r1, &zero)
		p1 := gc.Gbranch(x86.AJLT, nil, +1)
		gins(a, &r1, &r2)
		p2 := gc.Gbranch(obj.AJMP, nil, 0)
		gc.Patch(p1, gc.Pc)
		gmove(&r1, &r3)
		gins(x86.ASHRQ, &one, &r3)
		gmove(&r1, &r4)
		gins(x86.AANDL, &one, &r4)
		gins(x86.AORQ, &r4, &r3)
		gins(a, &r3, &r2)
		gins(optoas(gc.OADD, t.Type), &r2, &r2)
		gc.Patch(p2, gc.Pc)
		gmove(&r2, t)
		gc.Regfree(&r4)
		gc.Regfree(&r3)
		gc.Regfree(&r2)
		gc.Regfree(&r1)
		return

		/*
		 * float to float
		 */
	case gc.TFLOAT32<<16 | gc.TFLOAT32:
		a = x86.AMOVSS

	case gc.TFLOAT64<<16 | gc.TFLOAT64:
		a = x86.AMOVSD

	case gc.TFLOAT32<<16 | gc.TFLOAT64:
		a = x86.ACVTSS2SD
		goto rdst

	case gc.TFLOAT64<<16 | gc.TFLOAT32:
		a = x86.ACVTSD2SS
		goto rdst
	}

	gins(a, f, t)
	return

	// requires register destination
rdst:
	{
		var r1 gc.Node
		gc.Regalloc(&r1, t.Type, t)

		gins(a, f, &r1)
		gmove(&r1, t)
		gc.Regfree(&r1)
		return
	}

	// requires register intermediate
hard:
	var r1 gc.Node
	gc.Regalloc(&r1, cvt, t)

	gmove(f, &r1)
	gmove(&r1, t)
	gc.Regfree(&r1)
	return
}

func samaddr(f *gc.Node, t *gc.Node) bool {
	if f.Op != t.Op {
		return false
	}

	switch f.Op {
	case gc.OREGISTER:
		if f.Reg != t.Reg {
			break
		}
		return true
	}

	return false
}

/*
 * generate one instruction:
 *	as f, t
 */
func gins(as int, f *gc.Node, t *gc.Node) *obj.Prog {
	//	Node nod;

	//	if(f != N && f->op == OINDEX) {
	//		gc.Regalloc(&nod, &regnode, Z);
	//		v = constnode.vconst;
	//		gc.Cgen(f->right, &nod);
	//		constnode.vconst = v;
	//		idx.reg = nod.reg;
	//		gc.Regfree(&nod);
	//	}
	//	if(t != N && t->op == OINDEX) {
	//		gc.Regalloc(&nod, &regnode, Z);
	//		v = constnode.vconst;
	//		gc.Cgen(t->right, &nod);
	//		constnode.vconst = v;
	//		idx.reg = nod.reg;
	//		gc.Regfree(&nod);
	//	}

	if f != nil && f.Op == gc.OADDR && (as == x86.AMOVL || as == x86.AMOVQ) {
		// Turn MOVL $xxx into LEAL xxx.
		// These should be equivalent but most of the backend
		// only expects to see LEAL, because that's what we had
		// historically generated. Various hidden assumptions are baked in by now.
		if as == x86.AMOVL {
			as = x86.ALEAL
		} else {
			as = x86.ALEAQ
		}
		f = f.Left
	}

	switch as {
	case x86.AMOVB,
		x86.AMOVW,
		x86.AMOVL,
		x86.AMOVQ,
		x86.AMOVSS,
		x86.AMOVSD:
		if f != nil && t != nil && samaddr(f, t) {
			return nil
		}

	case x86.ALEAQ:
		if f != nil && gc.Isconst(f, gc.CTNIL) {
			gc.Fatal("gins LEAQ nil %v", gc.Tconv(f.Type, 0))
		}
	}

	p := gc.Prog(as)
	gc.Naddr(&p.From, f)
	gc.Naddr(&p.To, t)

	if gc.Debug['g'] != 0 {
		fmt.Printf("%v\n", p)
	}

	w := int32(0)
	switch as {
	case x86.AMOVB:
		w = 1

	case x86.AMOVW:
		w = 2

	case x86.AMOVL:
		w = 4

	case x86.AMOVQ:
		w = 8
	}

	if w != 0 && ((f != nil && p.From.Width < int64(w)) || (t != nil && p.To.Width > int64(w))) {
		gc.Dump("f", f)
		gc.Dump("t", t)
		gc.Fatal("bad width: %v (%d, %d)\n", p, p.From.Width, p.To.Width)
	}

	if p.To.Type == obj.TYPE_ADDR && w > 0 {
		gc.Fatal("bad use of addr: %v", p)
	}

	return p
}

func ginsnop() {
	// This is actually not the x86 NOP anymore,
	// but at the point where it gets used, AX is dead
	// so it's okay if we lose the high bits.
	var reg gc.Node
	gc.Nodreg(&reg, gc.Types[gc.TINT], x86.REG_AX)
	gins(x86.AXCHGL, &reg, &reg)
}

/*
 * return Axxx for Oxxx on type t.
 */
func optoas(op int, t *gc.Type) int {
	if t == nil {
		gc.Fatal("optoas: t is nil")
	}

	a := obj.AXXX
	switch uint32(op)<<16 | uint32(gc.Simtype[t.Etype]) {
	default:
		gc.Fatal("optoas: no entry %v-%v", gc.Oconv(int(op), 0), gc.Tconv(t, 0))

	case gc.OADDR<<16 | gc.TPTR32:
		a = x86.ALEAL

	case gc.OADDR<<16 | gc.TPTR64:
		a = x86.ALEAQ

	case gc.OEQ<<16 | gc.TBOOL,
		gc.OEQ<<16 | gc.TINT8,
		gc.OEQ<<16 | gc.TUINT8,
		gc.OEQ<<16 | gc.TINT16,
		gc.OEQ<<16 | gc.TUINT16,
		gc.OEQ<<16 | gc.TINT32,
		gc.OEQ<<16 | gc.TUINT32,
		gc.OEQ<<16 | gc.TINT64,
		gc.OEQ<<16 | gc.TUINT64,
		gc.OEQ<<16 | gc.TPTR32,
		gc.OEQ<<16 | gc.TPTR64,
		gc.OEQ<<16 | gc.TFLOAT32,
		gc.OEQ<<16 | gc.TFLOAT64:
		a = x86.AJEQ

	case gc.ONE<<16 | gc.TBOOL,
		gc.ONE<<16 | gc.TINT8,
		gc.ONE<<16 | gc.TUINT8,
		gc.ONE<<16 | gc.TINT16,
		gc.ONE<<16 | gc.TUINT16,
		gc.ONE<<16 | gc.TINT32,
		gc.ONE<<16 | gc.TUINT32,
		gc.ONE<<16 | gc.TINT64,
		gc.ONE<<16 | gc.TUINT64,
		gc.ONE<<16 | gc.TPTR32,
		gc.ONE<<16 | gc.TPTR64,
		gc.ONE<<16 | gc.TFLOAT32,
		gc.ONE<<16 | gc.TFLOAT64:
		a = x86.AJNE

	case gc.OPS<<16 | gc.TBOOL,
		gc.OPS<<16 | gc.TINT8,
		gc.OPS<<16 | gc.TUINT8,
		gc.OPS<<16 | gc.TINT16,
		gc.OPS<<16 | gc.TUINT16,
		gc.OPS<<16 | gc.TINT32,
		gc.OPS<<16 | gc.TUINT32,
		gc.OPS<<16 | gc.TINT64,
		gc.OPS<<16 | gc.TUINT64,
		gc.OPS<<16 | gc.TPTR32,
		gc.OPS<<16 | gc.TPTR64,
		gc.OPS<<16 | gc.TFLOAT32,
		gc.OPS<<16 | gc.TFLOAT64:
		a = x86.AJPS

	case gc.OLT<<16 | gc.TINT8,
		gc.OLT<<16 | gc.TINT16,
		gc.OLT<<16 | gc.TINT32,
		gc.OLT<<16 | gc.TINT64:
		a = x86.AJLT

	case gc.OLT<<16 | gc.TUINT8,
		gc.OLT<<16 | gc.TUINT16,
		gc.OLT<<16 | gc.TUINT32,
		gc.OLT<<16 | gc.TUINT64:
		a = x86.AJCS

	case gc.OLE<<16 | gc.TINT8,
		gc.OLE<<16 | gc.TINT16,
		gc.OLE<<16 | gc.TINT32,
		gc.OLE<<16 | gc.TINT64:
		a = x86.AJLE

	case gc.OLE<<16 | gc.TUINT8,
		gc.OLE<<16 | gc.TUINT16,
		gc.OLE<<16 | gc.TUINT32,
		gc.OLE<<16 | gc.TUINT64:
		a = x86.AJLS

	case gc.OGT<<16 | gc.TINT8,
		gc.OGT<<16 | gc.TINT16,
		gc.OGT<<16 | gc.TINT32,
		gc.OGT<<16 | gc.TINT64:
		a = x86.AJGT

	case gc.OGT<<16 | gc.TUINT8,
		gc.OGT<<16 | gc.TUINT16,
		gc.OGT<<16 | gc.TUINT32,
		gc.OGT<<16 | gc.TUINT64,
		gc.OLT<<16 | gc.TFLOAT32,
		gc.OLT<<16 | gc.TFLOAT64:
		a = x86.AJHI

	case gc.OGE<<16 | gc.TINT8,
		gc.OGE<<16 | gc.TINT16,
		gc.OGE<<16 | gc.TINT32,
		gc.OGE<<16 | gc.TINT64:
		a = x86.AJGE

	case gc.OGE<<16 | gc.TUINT8,
		gc.OGE<<16 | gc.TUINT16,
		gc.OGE<<16 | gc.TUINT32,
		gc.OGE<<16 | gc.TUINT64,
		gc.OLE<<16 | gc.TFLOAT32,
		gc.OLE<<16 | gc.TFLOAT64:
		a = x86.AJCC

	case gc.OCMP<<16 | gc.TBOOL,
		gc.OCMP<<16 | gc.TINT8,
		gc.OCMP<<16 | gc.TUINT8:
		a = x86.ACMPB

	case gc.OCMP<<16 | gc.TINT16,
		gc.OCMP<<16 | gc.TUINT16:
		a = x86.ACMPW

	case gc.OCMP<<16 | gc.TINT32,
		gc.OCMP<<16 | gc.TUINT32,
		gc.OCMP<<16 | gc.TPTR32:
		a = x86.ACMPL

	case gc.OCMP<<16 | gc.TINT64,
		gc.OCMP<<16 | gc.TUINT64,
		gc.OCMP<<16 | gc.TPTR64:
		a = x86.ACMPQ

	case gc.OCMP<<16 | gc.TFLOAT32:
		a = x86.AUCOMISS

	case gc.OCMP<<16 | gc.TFLOAT64:
		a = x86.AUCOMISD

	case gc.OAS<<16 | gc.TBOOL,
		gc.OAS<<16 | gc.TINT8,
		gc.OAS<<16 | gc.TUINT8:
		a = x86.AMOVB

	case gc.OAS<<16 | gc.TINT16,
		gc.OAS<<16 | gc.TUINT16:
		a = x86.AMOVW

	case gc.OAS<<16 | gc.TINT32,
		gc.OAS<<16 | gc.TUINT32,
		gc.OAS<<16 | gc.TPTR32:
		a = x86.AMOVL

	case gc.OAS<<16 | gc.TINT64,
		gc.OAS<<16 | gc.TUINT64,
		gc.OAS<<16 | gc.TPTR64:
		a = x86.AMOVQ

	case gc.OAS<<16 | gc.TFLOAT32:
		a = x86.AMOVSS

	case gc.OAS<<16 | gc.TFLOAT64:
		a = x86.AMOVSD

	case gc.OADD<<16 | gc.TINT8,
		gc.OADD<<16 | gc.TUINT8:
		a = x86.AADDB

	case gc.OADD<<16 | gc.TINT16,
		gc.OADD<<16 | gc.TUINT16:
		a = x86.AADDW

	case gc.OADD<<16 | gc.TINT32,
		gc.OADD<<16 | gc.TUINT32,
		gc.OADD<<16 | gc.TPTR32:
		a = x86.AADDL

	case gc.OADD<<16 | gc.TINT64,
		gc.OADD<<16 | gc.TUINT64,
		gc.OADD<<16 | gc.TPTR64:
		a = x86.AADDQ

	case gc.OADD<<16 | gc.TFLOAT32:
		a = x86.AADDSS

	case gc.OADD<<16 | gc.TFLOAT64:
		a = x86.AADDSD

	case gc.OSUB<<16 | gc.TINT8,
		gc.OSUB<<16 | gc.TUINT8:
		a = x86.ASUBB

	case gc.OSUB<<16 | gc.TINT16,
		gc.OSUB<<16 | gc.TUINT16:
		a = x86.ASUBW

	case gc.OSUB<<16 | gc.TINT32,
		gc.OSUB<<16 | gc.TUINT32,
		gc.OSUB<<16 | gc.TPTR32:
		a = x86.ASUBL

	case gc.OSUB<<16 | gc.TINT64,
		gc.OSUB<<16 | gc.TUINT64,
		gc.OSUB<<16 | gc.TPTR64:
		a = x86.ASUBQ

	case gc.OSUB<<16 | gc.TFLOAT32:
		a = x86.ASUBSS

	case gc.OSUB<<16 | gc.TFLOAT64:
		a = x86.ASUBSD

	case gc.OINC<<16 | gc.TINT8,
		gc.OINC<<16 | gc.TUINT8:
		a = x86.AINCB

	case gc.OINC<<16 | gc.TINT16,
		gc.OINC<<16 | gc.TUINT16:
		a = x86.AINCW

	case gc.OINC<<16 | gc.TINT32,
		gc.OINC<<16 | gc.TUINT32,
		gc.OINC<<16 | gc.TPTR32:
		a = x86.AINCL

	case gc.OINC<<16 | gc.TINT64,
		gc.OINC<<16 | gc.TUINT64,
		gc.OINC<<16 | gc.TPTR64:
		a = x86.AINCQ

	case gc.ODEC<<16 | gc.TINT8,
		gc.ODEC<<16 | gc.TUINT8:
		a = x86.ADECB

	case gc.ODEC<<16 | gc.TINT16,
		gc.ODEC<<16 | gc.TUINT16:
		a = x86.ADECW

	case gc.ODEC<<16 | gc.TINT32,
		gc.ODEC<<16 | gc.TUINT32,
		gc.ODEC<<16 | gc.TPTR32:
		a = x86.ADECL

	case gc.ODEC<<16 | gc.TINT64,
		gc.ODEC<<16 | gc.TUINT64,
		gc.ODEC<<16 | gc.TPTR64:
		a = x86.ADECQ

	case gc.OMINUS<<16 | gc.TINT8,
		gc.OMINUS<<16 | gc.TUINT8:
		a = x86.ANEGB

	case gc.OMINUS<<16 | gc.TINT16,
		gc.OMINUS<<16 | gc.TUINT16:
		a = x86.ANEGW

	case gc.OMINUS<<16 | gc.TINT32,
		gc.OMINUS<<16 | gc.TUINT32,
		gc.OMINUS<<16 | gc.TPTR32:
		a = x86.ANEGL

	case gc.OMINUS<<16 | gc.TINT64,
		gc.OMINUS<<16 | gc.TUINT64,
		gc.OMINUS<<16 | gc.TPTR64:
		a = x86.ANEGQ

	case gc.OAND<<16 | gc.TINT8,
		gc.OAND<<16 | gc.TUINT8:
		a = x86.AANDB

	case gc.OAND<<16 | gc.TINT16,
		gc.OAND<<16 | gc.TUINT16:
		a = x86.AANDW

	case gc.OAND<<16 | gc.TINT32,
		gc.OAND<<16 | gc.TUINT32,
		gc.OAND<<16 | gc.TPTR32:
		a = x86.AANDL

	case gc.OAND<<16 | gc.TINT64,
		gc.OAND<<16 | gc.TUINT64,
		gc.OAND<<16 | gc.TPTR64:
		a = x86.AANDQ

	case gc.OOR<<16 | gc.TINT8,
		gc.OOR<<16 | gc.TUINT8:
		a = x86.AORB

	case gc.OOR<<16 | gc.TINT16,
		gc.OOR<<16 | gc.TUINT16:
		a = x86.AORW

	case gc.OOR<<16 | gc.TINT32,
		gc.OOR<<16 | gc.TUINT32,
		gc.OOR<<16 | gc.TPTR32:
		a = x86.AORL

	case gc.OOR<<16 | gc.TINT64,
		gc.OOR<<16 | gc.TUINT64,
		gc.OOR<<16 | gc.TPTR64:
		a = x86.AORQ

	case gc.OXOR<<16 | gc.TINT8,
		gc.OXOR<<16 | gc.TUINT8:
		a = x86.AXORB

	case gc.OXOR<<16 | gc.TINT16,
		gc.OXOR<<16 | gc.TUINT16:
		a = x86.AXORW

	case gc.OXOR<<16 | gc.TINT32,
		gc.OXOR<<16 | gc.TUINT32,
		gc.OXOR<<16 | gc.TPTR32:
		a = x86.AXORL

	case gc.OXOR<<16 | gc.TINT64,
		gc.OXOR<<16 | gc.TUINT64,
		gc.OXOR<<16 | gc.TPTR64:
		a = x86.AXORQ

	case gc.OLROT<<16 | gc.TINT8,
		gc.OLROT<<16 | gc.TUINT8:
		a = x86.AROLB

	case gc.OLROT<<16 | gc.TINT16,
		gc.OLROT<<16 | gc.TUINT16:
		a = x86.AROLW

	case gc.OLROT<<16 | gc.TINT32,
		gc.OLROT<<16 | gc.TUINT32,
		gc.OLROT<<16 | gc.TPTR32:
		a = x86.AROLL

	case gc.OLROT<<16 | gc.TINT64,
		gc.OLROT<<16 | gc.TUINT64,
		gc.OLROT<<16 | gc.TPTR64:
		a = x86.AROLQ

	case gc.OLSH<<16 | gc.TINT8,
		gc.OLSH<<16 | gc.TUINT8:
		a = x86.ASHLB

	case gc.OLSH<<16 | gc.TINT16,
		gc.OLSH<<16 | gc.TUINT16:
		a = x86.ASHLW

	case gc.OLSH<<16 | gc.TINT32,
		gc.OLSH<<16 | gc.TUINT32,
		gc.OLSH<<16 | gc.TPTR32:
		a = x86.ASHLL

	case gc.OLSH<<16 | gc.TINT64,
		gc.OLSH<<16 | gc.TUINT64,
		gc.OLSH<<16 | gc.TPTR64:
		a = x86.ASHLQ

	case gc.ORSH<<16 | gc.TUINT8:
		a = x86.ASHRB

	case gc.ORSH<<16 | gc.TUINT16:
		a = x86.ASHRW

	case gc.ORSH<<16 | gc.TUINT32,
		gc.ORSH<<16 | gc.TPTR32:
		a = x86.ASHRL

	case gc.ORSH<<16 | gc.TUINT64,
		gc.ORSH<<16 | gc.TPTR64:
		a = x86.ASHRQ

	case gc.ORSH<<16 | gc.TINT8:
		a = x86.ASARB

	case gc.ORSH<<16 | gc.TINT16:
		a = x86.ASARW

	case gc.ORSH<<16 | gc.TINT32:
		a = x86.ASARL

	case gc.ORSH<<16 | gc.TINT64:
		a = x86.ASARQ

	case gc.ORROTC<<16 | gc.TINT8,
		gc.ORROTC<<16 | gc.TUINT8:
		a = x86.ARCRB

	case gc.ORROTC<<16 | gc.TINT16,
		gc.ORROTC<<16 | gc.TUINT16:
		a = x86.ARCRW

	case gc.ORROTC<<16 | gc.TINT32,
		gc.ORROTC<<16 | gc.TUINT32:
		a = x86.ARCRL

	case gc.ORROTC<<16 | gc.TINT64,
		gc.ORROTC<<16 | gc.TUINT64:
		a = x86.ARCRQ

	case gc.OHMUL<<16 | gc.TINT8,
		gc.OMUL<<16 | gc.TINT8,
		gc.OMUL<<16 | gc.TUINT8:
		a = x86.AIMULB

	case gc.OHMUL<<16 | gc.TINT16,
		gc.OMUL<<16 | gc.TINT16,
		gc.OMUL<<16 | gc.TUINT16:
		a = x86.AIMULW

	case gc.OHMUL<<16 | gc.TINT32,
		gc.OMUL<<16 | gc.TINT32,
		gc.OMUL<<16 | gc.TUINT32,
		gc.OMUL<<16 | gc.TPTR32:
		a = x86.AIMULL

	case gc.OHMUL<<16 | gc.TINT64,
		gc.OMUL<<16 | gc.TINT64,
		gc.OMUL<<16 | gc.TUINT64,
		gc.OMUL<<16 | gc.TPTR64:
		a = x86.AIMULQ

	case gc.OHMUL<<16 | gc.TUINT8:
		a = x86.AMULB

	case gc.OHMUL<<16 | gc.TUINT16:
		a = x86.AMULW

	case gc.OHMUL<<16 | gc.TUINT32,
		gc.OHMUL<<16 | gc.TPTR32:
		a = x86.AMULL

	case gc.OHMUL<<16 | gc.TUINT64,
		gc.OHMUL<<16 | gc.TPTR64:
		a = x86.AMULQ

	case gc.OMUL<<16 | gc.TFLOAT32:
		a = x86.AMULSS

	case gc.OMUL<<16 | gc.TFLOAT64:
		a = x86.AMULSD

	case gc.ODIV<<16 | gc.TINT8,
		gc.OMOD<<16 | gc.TINT8:
		a = x86.AIDIVB

	case gc.ODIV<<16 | gc.TUINT8,
		gc.OMOD<<16 | gc.TUINT8:
		a = x86.ADIVB

	case gc.ODIV<<16 | gc.TINT16,
		gc.OMOD<<16 | gc.TINT16:
		a = x86.AIDIVW

	case gc.ODIV<<16 | gc.TUINT16,
		gc.OMOD<<16 | gc.TUINT16:
		a = x86.ADIVW

	case gc.ODIV<<16 | gc.TINT32,
		gc.OMOD<<16 | gc.TINT32:
		a = x86.AIDIVL

	case gc.ODIV<<16 | gc.TUINT32,
		gc.ODIV<<16 | gc.TPTR32,
		gc.OMOD<<16 | gc.TUINT32,
		gc.OMOD<<16 | gc.TPTR32:
		a = x86.ADIVL

	case gc.ODIV<<16 | gc.TINT64,
		gc.OMOD<<16 | gc.TINT64:
		a = x86.AIDIVQ

	case gc.ODIV<<16 | gc.TUINT64,
		gc.ODIV<<16 | gc.TPTR64,
		gc.OMOD<<16 | gc.TUINT64,
		gc.OMOD<<16 | gc.TPTR64:
		a = x86.ADIVQ

	case gc.OEXTEND<<16 | gc.TINT16:
		a = x86.ACWD

	case gc.OEXTEND<<16 | gc.TINT32:
		a = x86.ACDQ

	case gc.OEXTEND<<16 | gc.TINT64:
		a = x86.ACQO

	case gc.ODIV<<16 | gc.TFLOAT32:
		a = x86.ADIVSS

	case gc.ODIV<<16 | gc.TFLOAT64:
		a = x86.ADIVSD

	case gc.OSQRT<<16 | gc.TFLOAT64:
		a = x86.ASQRTSD
	}

	return a
}

const (
	ODynam   = 1 << 0
	OAddable = 1 << 1
)

var clean [20]gc.Node

var cleani int = 0

func sudoclean() {
	if clean[cleani-1].Op != gc.OEMPTY {
		gc.Regfree(&clean[cleani-1])
	}
	if clean[cleani-2].Op != gc.OEMPTY {
		gc.Regfree(&clean[cleani-2])
	}
	cleani -= 2
}

/*
 * generate code to compute address of n,
 * a reference to a (perhaps nested) field inside
 * an array or struct.
 * return 0 on failure, 1 on success.
 * on success, leaves usable address in a.
 *
 * caller is responsible for calling sudoclean
 * after successful sudoaddable,
 * to release the register used for a.
 */
func sudoaddable(as int, n *gc.Node, a *obj.Addr) bool {
	if n.Type == nil {
		return false
	}

	*a = obj.Addr{}

	switch n.Op {
	case gc.OLITERAL:
		if !gc.Isconst(n, gc.CTINT) {
			break
		}
		v := gc.Mpgetfix(n.Val.U.Xval)
		if v >= 32000 || v <= -32000 {
			break
		}
		switch as {
		default:
			return false

		case x86.AADDB,
			x86.AADDW,
			x86.AADDL,
			x86.AADDQ,
			x86.ASUBB,
			x86.ASUBW,
			x86.ASUBL,
			x86.ASUBQ,
			x86.AANDB,
			x86.AANDW,
			x86.AANDL,
			x86.AANDQ,
			x86.AORB,
			x86.AORW,
			x86.AORL,
			x86.AORQ,
			x86.AXORB,
			x86.AXORW,
			x86.AXORL,
			x86.AXORQ,
			x86.AINCB,
			x86.AINCW,
			x86.AINCL,
			x86.AINCQ,
			x86.ADECB,
			x86.ADECW,
			x86.ADECL,
			x86.ADECQ,
			x86.AMOVB,
			x86.AMOVW,
			x86.AMOVL,
			x86.AMOVQ:
			break
		}

		cleani += 2
		reg := &clean[cleani-1]
		reg1 := &clean[cleani-2]
		reg.Op = gc.OEMPTY
		reg1.Op = gc.OEMPTY
		gc.Naddr(a, n)
		return true

	case gc.ODOT,
		gc.ODOTPTR:
		cleani += 2
		reg := &clean[cleani-1]
		reg1 := &clean[cleani-2]
		reg.Op = gc.OEMPTY
		reg1.Op = gc.OEMPTY
		var nn *gc.Node
		var oary [10]int64
		o := gc.Dotoffset(n, oary[:], &nn)
		if nn == nil {
			sudoclean()
			return false
		}

		if nn.Addable && o == 1 && oary[0] >= 0 {
			// directly addressable set of DOTs
			n1 := *nn

			n1.Type = n.Type
			n1.Xoffset += oary[0]
			gc.Naddr(a, &n1)
			return true
		}

		gc.Regalloc(reg, gc.Types[gc.Tptr], nil)
		n1 := *reg
		n1.Op = gc.OINDREG
		if oary[0] >= 0 {
			gc.Agen(nn, reg)
			n1.Xoffset = oary[0]
		} else {
			gc.Cgen(nn, reg)
			gc.Cgen_checknil(reg)
			n1.Xoffset = -(oary[0] + 1)
		}

		for i := 1; i < o; i++ {
			if oary[i] >= 0 {
				gc.Fatal("can't happen")
			}
			gins(movptr, &n1, reg)
			gc.Cgen_checknil(reg)
			n1.Xoffset = -(oary[i] + 1)
		}

		a.Type = obj.TYPE_NONE
		a.Index = obj.TYPE_NONE
		gc.Fixlargeoffset(&n1)
		gc.Naddr(a, &n1)
		return true

	case gc.OINDEX:
		return false
	}

	return false
}
