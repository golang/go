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
//	Portions Copyright © 2009 The Go Authors. All rights reserved.
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

package amd64

import (
	"cmd/compile/internal/big"
	"cmd/compile/internal/gc"
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
func gconreg(as obj.As, c int64, reg int) {
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
func ginscon(as obj.As, c int64, n2 *gc.Node) {
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

func ginscmp(op gc.Op, t *gc.Type, n1, n2 *gc.Node, likely int) *obj.Prog {
	if t.IsInteger() && n1.Op == gc.OLITERAL && gc.Smallintconst(n1) && n2.Op != gc.OLITERAL {
		// Reverse comparison to place constant last.
		op = gc.Brrev(op)
		n1, n2 = n2, n1
	}
	// General case.
	var r1, r2, g1, g2 gc.Node

	// A special case to make write barriers more efficient.
	// Comparing the first field of a named struct can be done directly.
	base := n1
	if n1.Op == gc.ODOT && n1.Left.Type.IsStruct() && n1.Left.Type.Field(0).Sym == n1.Sym {
		base = n1.Left
	}

	if base.Op == gc.ONAME && base.Class != gc.PAUTOHEAP || n1.Op == gc.OINDREG {
		r1 = *n1
	} else {
		gc.Regalloc(&r1, t, n1)
		gc.Regalloc(&g1, n1.Type, &r1)
		gc.Cgen(n1, &g1)
		gmove(&g1, &r1)
	}
	if n2.Op == gc.OLITERAL && t.IsInteger() && gc.Smallintconst(n2) {
		r2 = *n2
	} else {
		gc.Regalloc(&r2, t, n2)
		gc.Regalloc(&g2, n1.Type, &r2)
		gc.Cgen(n2, &g2)
		gmove(&g2, &r2)
	}
	gins(optoas(gc.OCMP, t), &r1, &r2)
	if r1.Op == gc.OREGISTER {
		gc.Regfree(&g1)
		gc.Regfree(&r1)
	}
	if r2.Op == gc.OREGISTER {
		gc.Regfree(&g2)
		gc.Regfree(&r2)
	}
	return gc.Gbranch(optoas(op, t), nil, likely)
}

func ginsboolval(a obj.As, n *gc.Node) {
	gins(jmptoset(a), nil, n)
}

// set up nodes representing 2^63
var (
	bigi         gc.Node
	bigf         gc.Node
	bignodes_did bool
)

func bignodes() {
	if bignodes_did {
		return
	}
	bignodes_did = true

	var i big.Int
	i.SetInt64(1)
	i.Lsh(&i, 63)

	gc.Nodconst(&bigi, gc.Types[gc.TUINT64], 0)
	bigi.SetBigInt(&i)

	bigi.Convconst(&bigf, gc.Types[gc.TFLOAT64])
}

/*
 * generate move:
 *	t = f
 * hard part is conversions.
 */
func gmove(f *gc.Node, t *gc.Node) {
	if gc.Debug['M'] != 0 {
		fmt.Printf("gmove %v -> %v\n", gc.Nconv(f, gc.FmtLong), gc.Nconv(t, gc.FmtLong))
	}

	ft := gc.Simsimtype(f.Type)
	tt := gc.Simsimtype(t.Type)
	cvt := t.Type

	if gc.Iscomplex[ft] || gc.Iscomplex[tt] {
		gc.Complexmove(f, t)
		return
	}

	// cannot have two memory operands
	var a obj.As
	if gc.Ismem(f) && gc.Ismem(t) {
		goto hard
	}

	// convert constant to desired type
	if f.Op == gc.OLITERAL {
		var con gc.Node
		f.Convconst(&con, t.Type)
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
				if i := con.Int64(); int64(int32(i)) != i {
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
		gc.Dump("f", f)
		gc.Dump("t", t)
		gc.Fatalf("gmove %v -> %v", gc.Tconv(f.Type, gc.FmtLong), gc.Tconv(t.Type, gc.FmtLong))

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
func gins(as obj.As, f *gc.Node, t *gc.Node) *obj.Prog {
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
			gc.Fatalf("gins LEAQ nil %v", f.Type)
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
		gc.Fatalf("bad width: %v (%d, %d)\n", p, p.From.Width, p.To.Width)
	}

	if p.To.Type == obj.TYPE_ADDR && w > 0 {
		gc.Fatalf("bad use of addr: %v", p)
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
func optoas(op gc.Op, t *gc.Type) obj.As {
	if t == nil {
		gc.Fatalf("optoas: t is nil")
	}

	// avoid constant conversions in switches below
	const (
		OMINUS_  = uint32(gc.OMINUS) << 16
		OLSH_    = uint32(gc.OLSH) << 16
		ORSH_    = uint32(gc.ORSH) << 16
		OADD_    = uint32(gc.OADD) << 16
		OSUB_    = uint32(gc.OSUB) << 16
		OMUL_    = uint32(gc.OMUL) << 16
		ODIV_    = uint32(gc.ODIV) << 16
		OMOD_    = uint32(gc.OMOD) << 16
		OOR_     = uint32(gc.OOR) << 16
		OAND_    = uint32(gc.OAND) << 16
		OXOR_    = uint32(gc.OXOR) << 16
		OEQ_     = uint32(gc.OEQ) << 16
		ONE_     = uint32(gc.ONE) << 16
		OLT_     = uint32(gc.OLT) << 16
		OLE_     = uint32(gc.OLE) << 16
		OGE_     = uint32(gc.OGE) << 16
		OGT_     = uint32(gc.OGT) << 16
		OCMP_    = uint32(gc.OCMP) << 16
		OPS_     = uint32(gc.OPS) << 16
		OPC_     = uint32(gc.OPC) << 16
		OAS_     = uint32(gc.OAS) << 16
		OHMUL_   = uint32(gc.OHMUL) << 16
		OSQRT_   = uint32(gc.OSQRT) << 16
		OADDR_   = uint32(gc.OADDR) << 16
		OINC_    = uint32(gc.OINC) << 16
		ODEC_    = uint32(gc.ODEC) << 16
		OLROT_   = uint32(gc.OLROT) << 16
		ORROTC_  = uint32(gc.ORROTC) << 16
		OEXTEND_ = uint32(gc.OEXTEND) << 16
	)

	a := obj.AXXX
	switch uint32(op)<<16 | uint32(gc.Simtype[t.Etype]) {
	default:
		gc.Fatalf("optoas: no entry %v-%v", op, t)

	case OADDR_ | gc.TPTR32:
		a = x86.ALEAL

	case OADDR_ | gc.TPTR64:
		a = x86.ALEAQ

	case OEQ_ | gc.TBOOL,
		OEQ_ | gc.TINT8,
		OEQ_ | gc.TUINT8,
		OEQ_ | gc.TINT16,
		OEQ_ | gc.TUINT16,
		OEQ_ | gc.TINT32,
		OEQ_ | gc.TUINT32,
		OEQ_ | gc.TINT64,
		OEQ_ | gc.TUINT64,
		OEQ_ | gc.TPTR32,
		OEQ_ | gc.TPTR64,
		OEQ_ | gc.TFLOAT32,
		OEQ_ | gc.TFLOAT64:
		a = x86.AJEQ

	case ONE_ | gc.TBOOL,
		ONE_ | gc.TINT8,
		ONE_ | gc.TUINT8,
		ONE_ | gc.TINT16,
		ONE_ | gc.TUINT16,
		ONE_ | gc.TINT32,
		ONE_ | gc.TUINT32,
		ONE_ | gc.TINT64,
		ONE_ | gc.TUINT64,
		ONE_ | gc.TPTR32,
		ONE_ | gc.TPTR64,
		ONE_ | gc.TFLOAT32,
		ONE_ | gc.TFLOAT64:
		a = x86.AJNE

	case OPS_ | gc.TBOOL,
		OPS_ | gc.TINT8,
		OPS_ | gc.TUINT8,
		OPS_ | gc.TINT16,
		OPS_ | gc.TUINT16,
		OPS_ | gc.TINT32,
		OPS_ | gc.TUINT32,
		OPS_ | gc.TINT64,
		OPS_ | gc.TUINT64,
		OPS_ | gc.TPTR32,
		OPS_ | gc.TPTR64,
		OPS_ | gc.TFLOAT32,
		OPS_ | gc.TFLOAT64:
		a = x86.AJPS

	case OPC_ | gc.TBOOL,
		OPC_ | gc.TINT8,
		OPC_ | gc.TUINT8,
		OPC_ | gc.TINT16,
		OPC_ | gc.TUINT16,
		OPC_ | gc.TINT32,
		OPC_ | gc.TUINT32,
		OPC_ | gc.TINT64,
		OPC_ | gc.TUINT64,
		OPC_ | gc.TPTR32,
		OPC_ | gc.TPTR64,
		OPC_ | gc.TFLOAT32,
		OPC_ | gc.TFLOAT64:
		a = x86.AJPC

	case OLT_ | gc.TINT8,
		OLT_ | gc.TINT16,
		OLT_ | gc.TINT32,
		OLT_ | gc.TINT64:
		a = x86.AJLT

	case OLT_ | gc.TUINT8,
		OLT_ | gc.TUINT16,
		OLT_ | gc.TUINT32,
		OLT_ | gc.TUINT64:
		a = x86.AJCS

	case OLE_ | gc.TINT8,
		OLE_ | gc.TINT16,
		OLE_ | gc.TINT32,
		OLE_ | gc.TINT64:
		a = x86.AJLE

	case OLE_ | gc.TUINT8,
		OLE_ | gc.TUINT16,
		OLE_ | gc.TUINT32,
		OLE_ | gc.TUINT64:
		a = x86.AJLS

	case OGT_ | gc.TINT8,
		OGT_ | gc.TINT16,
		OGT_ | gc.TINT32,
		OGT_ | gc.TINT64:
		a = x86.AJGT

	case OGT_ | gc.TUINT8,
		OGT_ | gc.TUINT16,
		OGT_ | gc.TUINT32,
		OGT_ | gc.TUINT64,
		OLT_ | gc.TFLOAT32,
		OLT_ | gc.TFLOAT64:
		a = x86.AJHI

	case OGE_ | gc.TINT8,
		OGE_ | gc.TINT16,
		OGE_ | gc.TINT32,
		OGE_ | gc.TINT64:
		a = x86.AJGE

	case OGE_ | gc.TUINT8,
		OGE_ | gc.TUINT16,
		OGE_ | gc.TUINT32,
		OGE_ | gc.TUINT64,
		OLE_ | gc.TFLOAT32,
		OLE_ | gc.TFLOAT64:
		a = x86.AJCC

	case OCMP_ | gc.TBOOL,
		OCMP_ | gc.TINT8,
		OCMP_ | gc.TUINT8:
		a = x86.ACMPB

	case OCMP_ | gc.TINT16,
		OCMP_ | gc.TUINT16:
		a = x86.ACMPW

	case OCMP_ | gc.TINT32,
		OCMP_ | gc.TUINT32,
		OCMP_ | gc.TPTR32:
		a = x86.ACMPL

	case OCMP_ | gc.TINT64,
		OCMP_ | gc.TUINT64,
		OCMP_ | gc.TPTR64:
		a = x86.ACMPQ

	case OCMP_ | gc.TFLOAT32:
		a = x86.AUCOMISS

	case OCMP_ | gc.TFLOAT64:
		a = x86.AUCOMISD

	case OAS_ | gc.TBOOL,
		OAS_ | gc.TINT8,
		OAS_ | gc.TUINT8:
		a = x86.AMOVB

	case OAS_ | gc.TINT16,
		OAS_ | gc.TUINT16:
		a = x86.AMOVW

	case OAS_ | gc.TINT32,
		OAS_ | gc.TUINT32,
		OAS_ | gc.TPTR32:
		a = x86.AMOVL

	case OAS_ | gc.TINT64,
		OAS_ | gc.TUINT64,
		OAS_ | gc.TPTR64:
		a = x86.AMOVQ

	case OAS_ | gc.TFLOAT32:
		a = x86.AMOVSS

	case OAS_ | gc.TFLOAT64:
		a = x86.AMOVSD

	case OADD_ | gc.TINT8,
		OADD_ | gc.TUINT8:
		a = x86.AADDB

	case OADD_ | gc.TINT16,
		OADD_ | gc.TUINT16:
		a = x86.AADDW

	case OADD_ | gc.TINT32,
		OADD_ | gc.TUINT32,
		OADD_ | gc.TPTR32:
		a = x86.AADDL

	case OADD_ | gc.TINT64,
		OADD_ | gc.TUINT64,
		OADD_ | gc.TPTR64:
		a = x86.AADDQ

	case OADD_ | gc.TFLOAT32:
		a = x86.AADDSS

	case OADD_ | gc.TFLOAT64:
		a = x86.AADDSD

	case OSUB_ | gc.TINT8,
		OSUB_ | gc.TUINT8:
		a = x86.ASUBB

	case OSUB_ | gc.TINT16,
		OSUB_ | gc.TUINT16:
		a = x86.ASUBW

	case OSUB_ | gc.TINT32,
		OSUB_ | gc.TUINT32,
		OSUB_ | gc.TPTR32:
		a = x86.ASUBL

	case OSUB_ | gc.TINT64,
		OSUB_ | gc.TUINT64,
		OSUB_ | gc.TPTR64:
		a = x86.ASUBQ

	case OSUB_ | gc.TFLOAT32:
		a = x86.ASUBSS

	case OSUB_ | gc.TFLOAT64:
		a = x86.ASUBSD

	case OINC_ | gc.TINT8,
		OINC_ | gc.TUINT8:
		a = x86.AINCB

	case OINC_ | gc.TINT16,
		OINC_ | gc.TUINT16:
		a = x86.AINCW

	case OINC_ | gc.TINT32,
		OINC_ | gc.TUINT32,
		OINC_ | gc.TPTR32:
		a = x86.AINCL

	case OINC_ | gc.TINT64,
		OINC_ | gc.TUINT64,
		OINC_ | gc.TPTR64:
		a = x86.AINCQ

	case ODEC_ | gc.TINT8,
		ODEC_ | gc.TUINT8:
		a = x86.ADECB

	case ODEC_ | gc.TINT16,
		ODEC_ | gc.TUINT16:
		a = x86.ADECW

	case ODEC_ | gc.TINT32,
		ODEC_ | gc.TUINT32,
		ODEC_ | gc.TPTR32:
		a = x86.ADECL

	case ODEC_ | gc.TINT64,
		ODEC_ | gc.TUINT64,
		ODEC_ | gc.TPTR64:
		a = x86.ADECQ

	case OMINUS_ | gc.TINT8,
		OMINUS_ | gc.TUINT8:
		a = x86.ANEGB

	case OMINUS_ | gc.TINT16,
		OMINUS_ | gc.TUINT16:
		a = x86.ANEGW

	case OMINUS_ | gc.TINT32,
		OMINUS_ | gc.TUINT32,
		OMINUS_ | gc.TPTR32:
		a = x86.ANEGL

	case OMINUS_ | gc.TINT64,
		OMINUS_ | gc.TUINT64,
		OMINUS_ | gc.TPTR64:
		a = x86.ANEGQ

	case OAND_ | gc.TBOOL,
		OAND_ | gc.TINT8,
		OAND_ | gc.TUINT8:
		a = x86.AANDB

	case OAND_ | gc.TINT16,
		OAND_ | gc.TUINT16:
		a = x86.AANDW

	case OAND_ | gc.TINT32,
		OAND_ | gc.TUINT32,
		OAND_ | gc.TPTR32:
		a = x86.AANDL

	case OAND_ | gc.TINT64,
		OAND_ | gc.TUINT64,
		OAND_ | gc.TPTR64:
		a = x86.AANDQ

	case OOR_ | gc.TBOOL,
		OOR_ | gc.TINT8,
		OOR_ | gc.TUINT8:
		a = x86.AORB

	case OOR_ | gc.TINT16,
		OOR_ | gc.TUINT16:
		a = x86.AORW

	case OOR_ | gc.TINT32,
		OOR_ | gc.TUINT32,
		OOR_ | gc.TPTR32:
		a = x86.AORL

	case OOR_ | gc.TINT64,
		OOR_ | gc.TUINT64,
		OOR_ | gc.TPTR64:
		a = x86.AORQ

	case OXOR_ | gc.TINT8,
		OXOR_ | gc.TUINT8:
		a = x86.AXORB

	case OXOR_ | gc.TINT16,
		OXOR_ | gc.TUINT16:
		a = x86.AXORW

	case OXOR_ | gc.TINT32,
		OXOR_ | gc.TUINT32,
		OXOR_ | gc.TPTR32:
		a = x86.AXORL

	case OXOR_ | gc.TINT64,
		OXOR_ | gc.TUINT64,
		OXOR_ | gc.TPTR64:
		a = x86.AXORQ

	case OLROT_ | gc.TINT8,
		OLROT_ | gc.TUINT8:
		a = x86.AROLB

	case OLROT_ | gc.TINT16,
		OLROT_ | gc.TUINT16:
		a = x86.AROLW

	case OLROT_ | gc.TINT32,
		OLROT_ | gc.TUINT32,
		OLROT_ | gc.TPTR32:
		a = x86.AROLL

	case OLROT_ | gc.TINT64,
		OLROT_ | gc.TUINT64,
		OLROT_ | gc.TPTR64:
		a = x86.AROLQ

	case OLSH_ | gc.TINT8,
		OLSH_ | gc.TUINT8:
		a = x86.ASHLB

	case OLSH_ | gc.TINT16,
		OLSH_ | gc.TUINT16:
		a = x86.ASHLW

	case OLSH_ | gc.TINT32,
		OLSH_ | gc.TUINT32,
		OLSH_ | gc.TPTR32:
		a = x86.ASHLL

	case OLSH_ | gc.TINT64,
		OLSH_ | gc.TUINT64,
		OLSH_ | gc.TPTR64:
		a = x86.ASHLQ

	case ORSH_ | gc.TUINT8:
		a = x86.ASHRB

	case ORSH_ | gc.TUINT16:
		a = x86.ASHRW

	case ORSH_ | gc.TUINT32,
		ORSH_ | gc.TPTR32:
		a = x86.ASHRL

	case ORSH_ | gc.TUINT64,
		ORSH_ | gc.TPTR64:
		a = x86.ASHRQ

	case ORSH_ | gc.TINT8:
		a = x86.ASARB

	case ORSH_ | gc.TINT16:
		a = x86.ASARW

	case ORSH_ | gc.TINT32:
		a = x86.ASARL

	case ORSH_ | gc.TINT64:
		a = x86.ASARQ

	case ORROTC_ | gc.TINT8,
		ORROTC_ | gc.TUINT8:
		a = x86.ARCRB

	case ORROTC_ | gc.TINT16,
		ORROTC_ | gc.TUINT16:
		a = x86.ARCRW

	case ORROTC_ | gc.TINT32,
		ORROTC_ | gc.TUINT32:
		a = x86.ARCRL

	case ORROTC_ | gc.TINT64,
		ORROTC_ | gc.TUINT64:
		a = x86.ARCRQ

	case OHMUL_ | gc.TINT8,
		OMUL_ | gc.TINT8,
		OMUL_ | gc.TUINT8:
		a = x86.AIMULB

	case OHMUL_ | gc.TINT16,
		OMUL_ | gc.TINT16,
		OMUL_ | gc.TUINT16:
		a = x86.AIMULW

	case OHMUL_ | gc.TINT32,
		OMUL_ | gc.TINT32,
		OMUL_ | gc.TUINT32,
		OMUL_ | gc.TPTR32:
		a = x86.AIMULL

	case OHMUL_ | gc.TINT64,
		OMUL_ | gc.TINT64,
		OMUL_ | gc.TUINT64,
		OMUL_ | gc.TPTR64:
		a = x86.AIMULQ

	case OHMUL_ | gc.TUINT8:
		a = x86.AMULB

	case OHMUL_ | gc.TUINT16:
		a = x86.AMULW

	case OHMUL_ | gc.TUINT32,
		OHMUL_ | gc.TPTR32:
		a = x86.AMULL

	case OHMUL_ | gc.TUINT64,
		OHMUL_ | gc.TPTR64:
		a = x86.AMULQ

	case OMUL_ | gc.TFLOAT32:
		a = x86.AMULSS

	case OMUL_ | gc.TFLOAT64:
		a = x86.AMULSD

	case ODIV_ | gc.TINT8,
		OMOD_ | gc.TINT8:
		a = x86.AIDIVB

	case ODIV_ | gc.TUINT8,
		OMOD_ | gc.TUINT8:
		a = x86.ADIVB

	case ODIV_ | gc.TINT16,
		OMOD_ | gc.TINT16:
		a = x86.AIDIVW

	case ODIV_ | gc.TUINT16,
		OMOD_ | gc.TUINT16:
		a = x86.ADIVW

	case ODIV_ | gc.TINT32,
		OMOD_ | gc.TINT32:
		a = x86.AIDIVL

	case ODIV_ | gc.TUINT32,
		ODIV_ | gc.TPTR32,
		OMOD_ | gc.TUINT32,
		OMOD_ | gc.TPTR32:
		a = x86.ADIVL

	case ODIV_ | gc.TINT64,
		OMOD_ | gc.TINT64:
		a = x86.AIDIVQ

	case ODIV_ | gc.TUINT64,
		ODIV_ | gc.TPTR64,
		OMOD_ | gc.TUINT64,
		OMOD_ | gc.TPTR64:
		a = x86.ADIVQ

	case OEXTEND_ | gc.TINT16:
		a = x86.ACWD

	case OEXTEND_ | gc.TINT32:
		a = x86.ACDQ

	case OEXTEND_ | gc.TINT64:
		a = x86.ACQO

	case ODIV_ | gc.TFLOAT32:
		a = x86.ADIVSS

	case ODIV_ | gc.TFLOAT64:
		a = x86.ADIVSD

	case OSQRT_ | gc.TFLOAT64:
		a = x86.ASQRTSD
	}

	return a
}

// jmptoset returns ASETxx for AJxx.
func jmptoset(jmp obj.As) obj.As {
	switch jmp {
	case x86.AJEQ:
		return x86.ASETEQ
	case x86.AJNE:
		return x86.ASETNE
	case x86.AJLT:
		return x86.ASETLT
	case x86.AJCS:
		return x86.ASETCS
	case x86.AJLE:
		return x86.ASETLE
	case x86.AJLS:
		return x86.ASETLS
	case x86.AJGT:
		return x86.ASETGT
	case x86.AJHI:
		return x86.ASETHI
	case x86.AJGE:
		return x86.ASETGE
	case x86.AJCC:
		return x86.ASETCC
	case x86.AJMI:
		return x86.ASETMI
	case x86.AJOC:
		return x86.ASETOC
	case x86.AJOS:
		return x86.ASETOS
	case x86.AJPC:
		return x86.ASETPC
	case x86.AJPL:
		return x86.ASETPL
	case x86.AJPS:
		return x86.ASETPS
	}
	gc.Fatalf("jmptoset: no entry for %v", jmp)
	panic("unreachable")
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
func sudoaddable(as obj.As, n *gc.Node, a *obj.Addr) bool {
	if n.Type == nil {
		return false
	}

	*a = obj.Addr{}

	switch n.Op {
	case gc.OLITERAL:
		if !gc.Isconst(n, gc.CTINT) {
			break
		}
		v := n.Int64()
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
				gc.Fatalf("can't happen")
			}
			gins(movptr, &n1, reg)
			gc.Cgen_checknil(reg)
			n1.Xoffset = -(oary[i] + 1)
		}

		a.Type = obj.TYPE_NONE
		a.Index = x86.REG_NONE
		gc.Fixlargeoffset(&n1)
		gc.Naddr(a, &n1)
		return true

	case gc.OINDEX:
		return false
	}

	return false
}
