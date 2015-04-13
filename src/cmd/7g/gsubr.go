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
	"cmd/internal/obj/arm64"
	"fmt"
)

var resvd = []int{
	arm64.REGTMP,
	arm64.REGG,
	arm64.REGRT1,
	arm64.REGRT2,
	arm64.REG_R31, // REGZERO and REGSP
	arm64.FREGZERO,
	arm64.FREGHALF,
	arm64.FREGONE,
	arm64.FREGTWO,
}

/*
 * generate
 *	as $c, n
 */
func ginscon(as int, c int64, n2 *gc.Node) {
	var n1 gc.Node

	gc.Nodconst(&n1, gc.Types[gc.TINT64], c)

	if as != arm64.AMOVD && (c < -arm64.BIG || c > arm64.BIG) || as == arm64.AMUL || n2 != nil && n2.Op != gc.OREGISTER {
		// cannot have more than 16-bit of immediate in ADD, etc.
		// instead, MOV into register first.
		var ntmp gc.Node
		gc.Regalloc(&ntmp, gc.Types[gc.TINT64], nil)

		gins(arm64.AMOVD, &n1, &ntmp)
		gins(as, &ntmp, n2)
		gc.Regfree(&ntmp)
		return
	}

	rawgins(as, &n1, n2)
}

/*
 * generate
 *	as n, $c (CMP)
 */
func ginscon2(as int, n2 *gc.Node, c int64) {
	var n1 gc.Node

	gc.Nodconst(&n1, gc.Types[gc.TINT64], c)

	switch as {
	default:
		gc.Fatal("ginscon2")

	case arm64.ACMP:
		if -arm64.BIG <= c && c <= arm64.BIG {
			gcmp(as, n2, &n1)
			return
		}
	}

	// MOV n1 into register first
	var ntmp gc.Node
	gc.Regalloc(&ntmp, gc.Types[gc.TINT64], nil)

	rawgins(arm64.AMOVD, &n1, &ntmp)
	gcmp(as, n2, &ntmp)
	gc.Regfree(&ntmp)
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

	ft := int(gc.Simsimtype(f.Type))
	tt := int(gc.Simsimtype(t.Type))
	cvt := (*gc.Type)(t.Type)

	if gc.Iscomplex[ft] || gc.Iscomplex[tt] {
		gc.Complexmove(f, t)
		return
	}

	// cannot have two memory operands
	var r1 gc.Node
	var a int
	if gc.Ismem(f) && gc.Ismem(t) {
		goto hard
	}

	// convert constant to desired type
	if f.Op == gc.OLITERAL {
		var con gc.Node
		switch tt {
		default:
			gc.Convconst(&con, t.Type, &f.Val)

		case gc.TINT32,
			gc.TINT16,
			gc.TINT8:
			var con gc.Node
			gc.Convconst(&con, gc.Types[gc.TINT64], &f.Val)
			var r1 gc.Node
			gc.Regalloc(&r1, con.Type, t)
			gins(arm64.AMOVD, &con, &r1)
			gmove(&r1, t)
			gc.Regfree(&r1)
			return

		case gc.TUINT32,
			gc.TUINT16,
			gc.TUINT8:
			var con gc.Node
			gc.Convconst(&con, gc.Types[gc.TUINT64], &f.Val)
			var r1 gc.Node
			gc.Regalloc(&r1, con.Type, t)
			gins(arm64.AMOVD, &con, &r1)
			gmove(&r1, t)
			gc.Regfree(&r1)
			return
		}

		f = &con
		ft = tt // so big switch will choose a simple mov

		// constants can't move directly to memory.
		if gc.Ismem(t) {
			goto hard
		}
	}

	// value -> value copy, first operand in memory.
	// any floating point operand requires register
	// src, so goto hard to copy to register first.
	if gc.Ismem(f) && ft != tt && (gc.Isfloat[ft] || gc.Isfloat[tt]) {
		cvt = gc.Types[ft]
		goto hard
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
		gc.TUINT8<<16 | gc.TINT8,
		gc.TINT16<<16 | gc.TINT8,
		// truncate
		gc.TUINT16<<16 | gc.TINT8,
		gc.TINT32<<16 | gc.TINT8,
		gc.TUINT32<<16 | gc.TINT8,
		gc.TINT64<<16 | gc.TINT8,
		gc.TUINT64<<16 | gc.TINT8:
		a = arm64.AMOVB

	case gc.TINT8<<16 | gc.TUINT8, // same size
		gc.TUINT8<<16 | gc.TUINT8,
		gc.TINT16<<16 | gc.TUINT8,
		// truncate
		gc.TUINT16<<16 | gc.TUINT8,
		gc.TINT32<<16 | gc.TUINT8,
		gc.TUINT32<<16 | gc.TUINT8,
		gc.TINT64<<16 | gc.TUINT8,
		gc.TUINT64<<16 | gc.TUINT8:
		a = arm64.AMOVBU

	case gc.TINT16<<16 | gc.TINT16, // same size
		gc.TUINT16<<16 | gc.TINT16,
		gc.TINT32<<16 | gc.TINT16,
		// truncate
		gc.TUINT32<<16 | gc.TINT16,
		gc.TINT64<<16 | gc.TINT16,
		gc.TUINT64<<16 | gc.TINT16:
		a = arm64.AMOVH

	case gc.TINT16<<16 | gc.TUINT16, // same size
		gc.TUINT16<<16 | gc.TUINT16,
		gc.TINT32<<16 | gc.TUINT16,
		// truncate
		gc.TUINT32<<16 | gc.TUINT16,
		gc.TINT64<<16 | gc.TUINT16,
		gc.TUINT64<<16 | gc.TUINT16:
		a = arm64.AMOVHU

	case gc.TINT32<<16 | gc.TINT32, // same size
		gc.TUINT32<<16 | gc.TINT32,
		gc.TINT64<<16 | gc.TINT32,
		// truncate
		gc.TUINT64<<16 | gc.TINT32:
		a = arm64.AMOVW

	case gc.TINT32<<16 | gc.TUINT32, // same size
		gc.TUINT32<<16 | gc.TUINT32,
		gc.TINT64<<16 | gc.TUINT32,
		gc.TUINT64<<16 | gc.TUINT32:
		a = arm64.AMOVWU

	case gc.TINT64<<16 | gc.TINT64, // same size
		gc.TINT64<<16 | gc.TUINT64,
		gc.TUINT64<<16 | gc.TINT64,
		gc.TUINT64<<16 | gc.TUINT64:
		a = arm64.AMOVD

		/*
		 * integer up-conversions
		 */
	case gc.TINT8<<16 | gc.TINT16, // sign extend int8
		gc.TINT8<<16 | gc.TUINT16,
		gc.TINT8<<16 | gc.TINT32,
		gc.TINT8<<16 | gc.TUINT32,
		gc.TINT8<<16 | gc.TINT64,
		gc.TINT8<<16 | gc.TUINT64:
		a = arm64.AMOVB

		goto rdst

	case gc.TUINT8<<16 | gc.TINT16, // zero extend uint8
		gc.TUINT8<<16 | gc.TUINT16,
		gc.TUINT8<<16 | gc.TINT32,
		gc.TUINT8<<16 | gc.TUINT32,
		gc.TUINT8<<16 | gc.TINT64,
		gc.TUINT8<<16 | gc.TUINT64:
		a = arm64.AMOVBU

		goto rdst

	case gc.TINT16<<16 | gc.TINT32, // sign extend int16
		gc.TINT16<<16 | gc.TUINT32,
		gc.TINT16<<16 | gc.TINT64,
		gc.TINT16<<16 | gc.TUINT64:
		a = arm64.AMOVH

		goto rdst

	case gc.TUINT16<<16 | gc.TINT32, // zero extend uint16
		gc.TUINT16<<16 | gc.TUINT32,
		gc.TUINT16<<16 | gc.TINT64,
		gc.TUINT16<<16 | gc.TUINT64:
		a = arm64.AMOVHU

		goto rdst

	case gc.TINT32<<16 | gc.TINT64, // sign extend int32
		gc.TINT32<<16 | gc.TUINT64:
		a = arm64.AMOVW

		goto rdst

	case gc.TUINT32<<16 | gc.TINT64, // zero extend uint32
		gc.TUINT32<<16 | gc.TUINT64:
		a = arm64.AMOVWU

		goto rdst

	/*
	* float to integer
	 */
	case gc.TFLOAT32<<16 | gc.TINT32:
		a = arm64.AFCVTZSSW
		goto rdst

	case gc.TFLOAT64<<16 | gc.TINT32:
		a = arm64.AFCVTZSDW
		goto rdst

	case gc.TFLOAT32<<16 | gc.TINT64:
		a = arm64.AFCVTZSS
		goto rdst

	case gc.TFLOAT64<<16 | gc.TINT64:
		a = arm64.AFCVTZSD
		goto rdst

	case gc.TFLOAT32<<16 | gc.TUINT32:
		a = arm64.AFCVTZUSW
		goto rdst

	case gc.TFLOAT64<<16 | gc.TUINT32:
		a = arm64.AFCVTZUDW
		goto rdst

	case gc.TFLOAT32<<16 | gc.TUINT64:
		a = arm64.AFCVTZUS
		goto rdst

	case gc.TFLOAT64<<16 | gc.TUINT64:
		a = arm64.AFCVTZUD
		goto rdst

	case gc.TFLOAT32<<16 | gc.TINT16,
		gc.TFLOAT32<<16 | gc.TINT8,
		gc.TFLOAT64<<16 | gc.TINT16,
		gc.TFLOAT64<<16 | gc.TINT8:
		cvt = gc.Types[gc.TINT32]

		goto hard

	case gc.TFLOAT32<<16 | gc.TUINT16,
		gc.TFLOAT32<<16 | gc.TUINT8,
		gc.TFLOAT64<<16 | gc.TUINT16,
		gc.TFLOAT64<<16 | gc.TUINT8:
		cvt = gc.Types[gc.TUINT32]

		goto hard

	/*
	 * integer to float
	 */
	case gc.TINT8<<16 | gc.TFLOAT32,
		gc.TINT16<<16 | gc.TFLOAT32,
		gc.TINT32<<16 | gc.TFLOAT32:
		a = arm64.ASCVTFWS

		goto rdst

	case gc.TINT8<<16 | gc.TFLOAT64,
		gc.TINT16<<16 | gc.TFLOAT64,
		gc.TINT32<<16 | gc.TFLOAT64:
		a = arm64.ASCVTFWD

		goto rdst

	case gc.TINT64<<16 | gc.TFLOAT32:
		a = arm64.ASCVTFS
		goto rdst

	case gc.TINT64<<16 | gc.TFLOAT64:
		a = arm64.ASCVTFD
		goto rdst

	case gc.TUINT8<<16 | gc.TFLOAT32,
		gc.TUINT16<<16 | gc.TFLOAT32,
		gc.TUINT32<<16 | gc.TFLOAT32:
		a = arm64.AUCVTFWS

		goto rdst

	case gc.TUINT8<<16 | gc.TFLOAT64,
		gc.TUINT16<<16 | gc.TFLOAT64,
		gc.TUINT32<<16 | gc.TFLOAT64:
		a = arm64.AUCVTFWD

		goto rdst

	case gc.TUINT64<<16 | gc.TFLOAT32:
		a = arm64.AUCVTFS
		goto rdst

	case gc.TUINT64<<16 | gc.TFLOAT64:
		a = arm64.AUCVTFD
		goto rdst

		/*
		 * float to float
		 */
	case gc.TFLOAT32<<16 | gc.TFLOAT32:
		a = arm64.AFMOVS

	case gc.TFLOAT64<<16 | gc.TFLOAT64:
		a = arm64.AFMOVD

	case gc.TFLOAT32<<16 | gc.TFLOAT64:
		a = arm64.AFCVTSD
		goto rdst

	case gc.TFLOAT64<<16 | gc.TFLOAT32:
		a = arm64.AFCVTDS
		goto rdst
	}

	gins(a, f, t)
	return

	// requires register destination
rdst:
	gc.Regalloc(&r1, t.Type, t)

	gins(a, f, &r1)
	gmove(&r1, t)
	gc.Regfree(&r1)
	return

	// requires register intermediate
hard:
	gc.Regalloc(&r1, cvt, t)

	gmove(f, &r1)
	gmove(&r1, t)
	gc.Regfree(&r1)
	return
}

func intLiteral(n *gc.Node) (x int64, ok bool) {
	if n == nil || n.Op != gc.OLITERAL {
		return
	}
	switch n.Val.Ctype {
	case gc.CTINT, gc.CTRUNE:
		return gc.Mpgetfix(n.Val.U.Xval), true
	case gc.CTBOOL:
		return int64(bool2int(n.Val.U.Bval)), true
	}
	return
}

// gins is called by the front end.
// It synthesizes some multiple-instruction sequences
// so the front end can stay simpler.
func gins(as int, f, t *gc.Node) *obj.Prog {
	if as >= obj.A_ARCHSPECIFIC {
		if x, ok := intLiteral(f); ok {
			ginscon(as, x, t)
			return nil // caller must not use
		}
	}
	if as == arm64.ACMP {
		if x, ok := intLiteral(t); ok {
			ginscon2(as, f, x)
			return nil // caller must not use
		}
	}
	return rawgins(as, f, t)
}

/*
 * generate one instruction:
 *	as f, t
 */
func rawgins(as int, f *gc.Node, t *gc.Node) *obj.Prog {
	// TODO(austin): Add self-move test like in 6g (but be careful
	// of truncation moves)

	p := gc.Prog(as)
	gc.Naddr(&p.From, f)
	gc.Naddr(&p.To, t)

	switch as {
	case arm64.ACMP, arm64.AFCMPS, arm64.AFCMPD:
		if t != nil {
			if f.Op != gc.OREGISTER {
				gc.Fatal("bad operands to gcmp")
			}
			p.From = p.To
			p.To = obj.Addr{}
			raddr(f, p)
		}
	}

	// Bad things the front end has done to us. Crash to find call stack.
	switch as {
	case arm64.AAND, arm64.AMUL:
		if p.From.Type == obj.TYPE_CONST {
			gc.Debug['h'] = 1
			gc.Fatal("bad inst: %v", p)
		}
	case arm64.ACMP:
		if p.From.Type == obj.TYPE_MEM || p.To.Type == obj.TYPE_MEM {
			gc.Debug['h'] = 1
			gc.Fatal("bad inst: %v", p)
		}
	}

	if gc.Debug['g'] != 0 {
		fmt.Printf("%v\n", p)
	}

	w := int32(0)
	switch as {
	case arm64.AMOVB,
		arm64.AMOVBU:
		w = 1

	case arm64.AMOVH,
		arm64.AMOVHU:
		w = 2

	case arm64.AMOVW,
		arm64.AMOVWU:
		w = 4

	case arm64.AMOVD:
		if p.From.Type == obj.TYPE_CONST || p.From.Type == obj.TYPE_ADDR {
			break
		}
		w = 8
	}

	if w != 0 && ((f != nil && p.From.Width < int64(w)) || (t != nil && p.To.Type != obj.TYPE_REG && p.To.Width > int64(w))) {
		gc.Dump("f", f)
		gc.Dump("t", t)
		gc.Fatal("bad width: %v (%d, %d)\n", p, p.From.Width, p.To.Width)
	}

	return p
}

func fixlargeoffset(n *gc.Node) {
	if n == nil {
		return
	}
	if n.Op != gc.OINDREG {
		return
	}
	if -4096 <= n.Xoffset && n.Xoffset < 4096 {
		return
	}
	a := gc.Node(*n)
	a.Op = gc.OREGISTER
	a.Type = gc.Types[gc.Tptr]
	a.Xoffset = 0
	gc.Cgen_checknil(&a)
	ginscon(optoas(gc.OADD, gc.Types[gc.Tptr]), n.Xoffset, &a)
	n.Xoffset = 0
}

/*
 * insert n into reg slot of p
 */
func raddr(n *gc.Node, p *obj.Prog) {
	var a obj.Addr

	gc.Naddr(&a, n)
	if a.Type != obj.TYPE_REG {
		if n != nil {
			gc.Fatal("bad in raddr: %v", gc.Oconv(int(n.Op), 0))
		} else {
			gc.Fatal("bad in raddr: <null>")
		}
		p.Reg = 0
	} else {
		p.Reg = a.Reg
	}
}

func gcmp(as int, lhs *gc.Node, rhs *gc.Node) *obj.Prog {
	if lhs.Op != gc.OREGISTER {
		gc.Fatal("bad operands to gcmp: %v %v", gc.Oconv(int(lhs.Op), 0), gc.Oconv(int(rhs.Op), 0))
	}

	p := rawgins(as, rhs, nil)
	raddr(lhs, p)
	return p
}

/*
 * return Axxx for Oxxx on type t.
 */
func optoas(op int, t *gc.Type) int {
	if t == nil {
		gc.Fatal("optoas: t is nil")
	}

	a := int(obj.AXXX)
	switch uint32(op)<<16 | uint32(gc.Simtype[t.Etype]) {
	default:
		gc.Fatal("optoas: no entry for op=%v type=%v", gc.Oconv(int(op), 0), gc.Tconv(t, 0))

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
		a = arm64.ABEQ

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
		a = arm64.ABNE

	case gc.OLT<<16 | gc.TINT8,
		gc.OLT<<16 | gc.TINT16,
		gc.OLT<<16 | gc.TINT32,
		gc.OLT<<16 | gc.TINT64:
		a = arm64.ABLT

	case gc.OLT<<16 | gc.TUINT8,
		gc.OLT<<16 | gc.TUINT16,
		gc.OLT<<16 | gc.TUINT32,
		gc.OLT<<16 | gc.TUINT64,
		gc.OLT<<16 | gc.TFLOAT32,
		gc.OLT<<16 | gc.TFLOAT64:
		a = arm64.ABLO

	case gc.OLE<<16 | gc.TINT8,
		gc.OLE<<16 | gc.TINT16,
		gc.OLE<<16 | gc.TINT32,
		gc.OLE<<16 | gc.TINT64:
		a = arm64.ABLE

	case gc.OLE<<16 | gc.TUINT8,
		gc.OLE<<16 | gc.TUINT16,
		gc.OLE<<16 | gc.TUINT32,
		gc.OLE<<16 | gc.TUINT64,
		gc.OLE<<16 | gc.TFLOAT32,
		gc.OLE<<16 | gc.TFLOAT64:
		a = arm64.ABLS

	case gc.OGT<<16 | gc.TINT8,
		gc.OGT<<16 | gc.TINT16,
		gc.OGT<<16 | gc.TINT32,
		gc.OGT<<16 | gc.TINT64,
		gc.OGT<<16 | gc.TFLOAT32,
		gc.OGT<<16 | gc.TFLOAT64:
		a = arm64.ABGT

	case gc.OGT<<16 | gc.TUINT8,
		gc.OGT<<16 | gc.TUINT16,
		gc.OGT<<16 | gc.TUINT32,
		gc.OGT<<16 | gc.TUINT64:
		a = arm64.ABHI

	case gc.OGE<<16 | gc.TINT8,
		gc.OGE<<16 | gc.TINT16,
		gc.OGE<<16 | gc.TINT32,
		gc.OGE<<16 | gc.TINT64,
		gc.OGE<<16 | gc.TFLOAT32,
		gc.OGE<<16 | gc.TFLOAT64:
		a = arm64.ABGE

	case gc.OGE<<16 | gc.TUINT8,
		gc.OGE<<16 | gc.TUINT16,
		gc.OGE<<16 | gc.TUINT32,
		gc.OGE<<16 | gc.TUINT64:
		a = arm64.ABHS

	case gc.OCMP<<16 | gc.TBOOL,
		gc.OCMP<<16 | gc.TINT8,
		gc.OCMP<<16 | gc.TINT16,
		gc.OCMP<<16 | gc.TINT32,
		gc.OCMP<<16 | gc.TPTR32,
		gc.OCMP<<16 | gc.TINT64,
		gc.OCMP<<16 | gc.TUINT8,
		gc.OCMP<<16 | gc.TUINT16,
		gc.OCMP<<16 | gc.TUINT32,
		gc.OCMP<<16 | gc.TUINT64,
		gc.OCMP<<16 | gc.TPTR64:
		a = arm64.ACMP

	case gc.OCMP<<16 | gc.TFLOAT32:
		a = arm64.AFCMPS

	case gc.OCMP<<16 | gc.TFLOAT64:
		a = arm64.AFCMPD

	case gc.OAS<<16 | gc.TBOOL,
		gc.OAS<<16 | gc.TINT8:
		a = arm64.AMOVB

	case gc.OAS<<16 | gc.TUINT8:
		a = arm64.AMOVBU

	case gc.OAS<<16 | gc.TINT16:
		a = arm64.AMOVH

	case gc.OAS<<16 | gc.TUINT16:
		a = arm64.AMOVHU

	case gc.OAS<<16 | gc.TINT32:
		a = arm64.AMOVW

	case gc.OAS<<16 | gc.TUINT32,
		gc.OAS<<16 | gc.TPTR32:
		a = arm64.AMOVWU

	case gc.OAS<<16 | gc.TINT64,
		gc.OAS<<16 | gc.TUINT64,
		gc.OAS<<16 | gc.TPTR64:
		a = arm64.AMOVD

	case gc.OAS<<16 | gc.TFLOAT32:
		a = arm64.AFMOVS

	case gc.OAS<<16 | gc.TFLOAT64:
		a = arm64.AFMOVD

	case gc.OADD<<16 | gc.TINT8,
		gc.OADD<<16 | gc.TUINT8,
		gc.OADD<<16 | gc.TINT16,
		gc.OADD<<16 | gc.TUINT16,
		gc.OADD<<16 | gc.TINT32,
		gc.OADD<<16 | gc.TUINT32,
		gc.OADD<<16 | gc.TPTR32,
		gc.OADD<<16 | gc.TINT64,
		gc.OADD<<16 | gc.TUINT64,
		gc.OADD<<16 | gc.TPTR64:
		a = arm64.AADD

	case gc.OADD<<16 | gc.TFLOAT32:
		a = arm64.AFADDS

	case gc.OADD<<16 | gc.TFLOAT64:
		a = arm64.AFADDD

	case gc.OSUB<<16 | gc.TINT8,
		gc.OSUB<<16 | gc.TUINT8,
		gc.OSUB<<16 | gc.TINT16,
		gc.OSUB<<16 | gc.TUINT16,
		gc.OSUB<<16 | gc.TINT32,
		gc.OSUB<<16 | gc.TUINT32,
		gc.OSUB<<16 | gc.TPTR32,
		gc.OSUB<<16 | gc.TINT64,
		gc.OSUB<<16 | gc.TUINT64,
		gc.OSUB<<16 | gc.TPTR64:
		a = arm64.ASUB

	case gc.OSUB<<16 | gc.TFLOAT32:
		a = arm64.AFSUBS

	case gc.OSUB<<16 | gc.TFLOAT64:
		a = arm64.AFSUBD

	case gc.OMINUS<<16 | gc.TINT8,
		gc.OMINUS<<16 | gc.TUINT8,
		gc.OMINUS<<16 | gc.TINT16,
		gc.OMINUS<<16 | gc.TUINT16,
		gc.OMINUS<<16 | gc.TINT32,
		gc.OMINUS<<16 | gc.TUINT32,
		gc.OMINUS<<16 | gc.TPTR32,
		gc.OMINUS<<16 | gc.TINT64,
		gc.OMINUS<<16 | gc.TUINT64,
		gc.OMINUS<<16 | gc.TPTR64:
		a = arm64.ANEG

	case gc.OMINUS<<16 | gc.TFLOAT32:
		a = arm64.AFNEGS

	case gc.OMINUS<<16 | gc.TFLOAT64:
		a = arm64.AFNEGD

	case gc.OAND<<16 | gc.TINT8,
		gc.OAND<<16 | gc.TUINT8,
		gc.OAND<<16 | gc.TINT16,
		gc.OAND<<16 | gc.TUINT16,
		gc.OAND<<16 | gc.TINT32,
		gc.OAND<<16 | gc.TUINT32,
		gc.OAND<<16 | gc.TPTR32,
		gc.OAND<<16 | gc.TINT64,
		gc.OAND<<16 | gc.TUINT64,
		gc.OAND<<16 | gc.TPTR64:
		a = arm64.AAND

	case gc.OOR<<16 | gc.TINT8,
		gc.OOR<<16 | gc.TUINT8,
		gc.OOR<<16 | gc.TINT16,
		gc.OOR<<16 | gc.TUINT16,
		gc.OOR<<16 | gc.TINT32,
		gc.OOR<<16 | gc.TUINT32,
		gc.OOR<<16 | gc.TPTR32,
		gc.OOR<<16 | gc.TINT64,
		gc.OOR<<16 | gc.TUINT64,
		gc.OOR<<16 | gc.TPTR64:
		a = arm64.AORR

	case gc.OXOR<<16 | gc.TINT8,
		gc.OXOR<<16 | gc.TUINT8,
		gc.OXOR<<16 | gc.TINT16,
		gc.OXOR<<16 | gc.TUINT16,
		gc.OXOR<<16 | gc.TINT32,
		gc.OXOR<<16 | gc.TUINT32,
		gc.OXOR<<16 | gc.TPTR32,
		gc.OXOR<<16 | gc.TINT64,
		gc.OXOR<<16 | gc.TUINT64,
		gc.OXOR<<16 | gc.TPTR64:
		a = arm64.AEOR

		// TODO(minux): handle rotates
	//case CASE(OLROT, TINT8):
	//case CASE(OLROT, TUINT8):
	//case CASE(OLROT, TINT16):
	//case CASE(OLROT, TUINT16):
	//case CASE(OLROT, TINT32):
	//case CASE(OLROT, TUINT32):
	//case CASE(OLROT, TPTR32):
	//case CASE(OLROT, TINT64):
	//case CASE(OLROT, TUINT64):
	//case CASE(OLROT, TPTR64):
	//	a = 0//???; RLDC?
	//	break;

	case gc.OLSH<<16 | gc.TINT8,
		gc.OLSH<<16 | gc.TUINT8,
		gc.OLSH<<16 | gc.TINT16,
		gc.OLSH<<16 | gc.TUINT16,
		gc.OLSH<<16 | gc.TINT32,
		gc.OLSH<<16 | gc.TUINT32,
		gc.OLSH<<16 | gc.TPTR32,
		gc.OLSH<<16 | gc.TINT64,
		gc.OLSH<<16 | gc.TUINT64,
		gc.OLSH<<16 | gc.TPTR64:
		a = arm64.ALSL

	case gc.ORSH<<16 | gc.TUINT8,
		gc.ORSH<<16 | gc.TUINT16,
		gc.ORSH<<16 | gc.TUINT32,
		gc.ORSH<<16 | gc.TPTR32,
		gc.ORSH<<16 | gc.TUINT64,
		gc.ORSH<<16 | gc.TPTR64:
		a = arm64.ALSR

	case gc.ORSH<<16 | gc.TINT8,
		gc.ORSH<<16 | gc.TINT16,
		gc.ORSH<<16 | gc.TINT32,
		gc.ORSH<<16 | gc.TINT64:
		a = arm64.AASR

		// TODO(minux): handle rotates
	//case CASE(ORROTC, TINT8):
	//case CASE(ORROTC, TUINT8):
	//case CASE(ORROTC, TINT16):
	//case CASE(ORROTC, TUINT16):
	//case CASE(ORROTC, TINT32):
	//case CASE(ORROTC, TUINT32):
	//case CASE(ORROTC, TINT64):
	//case CASE(ORROTC, TUINT64):
	//	a = 0//??? RLDC??
	//	break;

	case gc.OHMUL<<16 | gc.TINT64:
		a = arm64.ASMULH

	case gc.OHMUL<<16 | gc.TUINT64,
		gc.OHMUL<<16 | gc.TPTR64:
		a = arm64.AUMULH

	case gc.OMUL<<16 | gc.TINT8,
		gc.OMUL<<16 | gc.TINT16,
		gc.OMUL<<16 | gc.TINT32:
		a = arm64.ASMULL

	case gc.OMUL<<16 | gc.TINT64:
		a = arm64.AMUL

	case gc.OMUL<<16 | gc.TUINT8,
		gc.OMUL<<16 | gc.TUINT16,
		gc.OMUL<<16 | gc.TUINT32,
		gc.OMUL<<16 | gc.TPTR32:
		// don't use word multiply, the high 32-bit are undefined.
		a = arm64.AUMULL

	case gc.OMUL<<16 | gc.TUINT64,
		gc.OMUL<<16 | gc.TPTR64:
		a = arm64.AMUL // for 64-bit multiplies, signedness doesn't matter.

	case gc.OMUL<<16 | gc.TFLOAT32:
		a = arm64.AFMULS

	case gc.OMUL<<16 | gc.TFLOAT64:
		a = arm64.AFMULD

	case gc.ODIV<<16 | gc.TINT8,
		gc.ODIV<<16 | gc.TINT16,
		gc.ODIV<<16 | gc.TINT32,
		gc.ODIV<<16 | gc.TINT64:
		a = arm64.ASDIV

	case gc.ODIV<<16 | gc.TUINT8,
		gc.ODIV<<16 | gc.TUINT16,
		gc.ODIV<<16 | gc.TUINT32,
		gc.ODIV<<16 | gc.TPTR32,
		gc.ODIV<<16 | gc.TUINT64,
		gc.ODIV<<16 | gc.TPTR64:
		a = arm64.AUDIV

	case gc.ODIV<<16 | gc.TFLOAT32:
		a = arm64.AFDIVS

	case gc.ODIV<<16 | gc.TFLOAT64:
		a = arm64.AFDIVD

	case gc.OSQRT<<16 | gc.TFLOAT64:
		a = arm64.AFSQRTD
	}

	return a
}

const (
	ODynam   = 1 << 0
	OAddable = 1 << 1
)

func xgen(n *gc.Node, a *gc.Node, o int) bool {
	// TODO(minux)

	return -1 != 0 /*TypeKind(100016)*/
}

func sudoclean() {
	return
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
	// TODO(minux)

	*a = obj.Addr{}
	return false
}
