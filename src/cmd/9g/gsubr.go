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
	"cmd/internal/obj/ppc64"
	"fmt"
)

var resvd = []int{
	ppc64.REGZERO,
	ppc64.REGSP, // reserved for SP
	// We need to preserve the C ABI TLS pointer because sigtramp
	// may happen during C code and needs to access the g.  C
	// clobbers REGG, so if Go were to clobber REGTLS, sigtramp
	// won't know which convention to use.  By preserving REGTLS,
	// we can just retrieve g from TLS when we aren't sure.
	ppc64.REGTLS,

	// TODO(austin): Consolidate REGTLS and REGG?
	ppc64.REGG,
	ppc64.REGTMP, // REGTMP
	ppc64.FREGCVI,
	ppc64.FREGZERO,
	ppc64.FREGHALF,
	ppc64.FREGONE,
	ppc64.FREGTWO,
}

/*
 * generate
 *	as $c, n
 */
func ginscon(as int, c int64, n2 *gc.Node) {
	var n1 gc.Node

	gc.Nodconst(&n1, gc.Types[gc.TINT64], c)

	if as != ppc64.AMOVD && (c < -ppc64.BIG || c > ppc64.BIG) || n2.Op != gc.OREGISTER || as == ppc64.AMULLD {
		// cannot have more than 16-bit of immediate in ADD, etc.
		// instead, MOV into register first.
		var ntmp gc.Node
		gc.Regalloc(&ntmp, gc.Types[gc.TINT64], nil)

		rawgins(ppc64.AMOVD, &n1, &ntmp)
		rawgins(as, &ntmp, n2)
		gc.Regfree(&ntmp)
		return
	}

	rawgins(as, &n1, n2)
}

/*
 * generate
 *	as n, $c (CMP/CMPU)
 */
func ginscon2(as int, n2 *gc.Node, c int64) {
	var n1 gc.Node

	gc.Nodconst(&n1, gc.Types[gc.TINT64], c)

	switch as {
	default:
		gc.Fatal("ginscon2")

	case ppc64.ACMP:
		if -ppc64.BIG <= c && c <= ppc64.BIG {
			rawgins(as, n2, &n1)
			return
		}

	case ppc64.ACMPU:
		if 0 <= c && c <= 2*ppc64.BIG {
			rawgins(as, n2, &n1)
			return
		}
	}

	// MOV n1 into register first
	var ntmp gc.Node
	gc.Regalloc(&ntmp, gc.Types[gc.TINT64], nil)

	rawgins(ppc64.AMOVD, &n1, &ntmp)
	rawgins(as, n2, &ntmp)
	gc.Regfree(&ntmp)
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

	ft := int(gc.Simsimtype(f.Type))
	tt := int(gc.Simsimtype(t.Type))
	cvt := (*gc.Type)(t.Type)

	if gc.Iscomplex[ft] || gc.Iscomplex[tt] {
		gc.Complexmove(f, t)
		return
	}

	// cannot have two memory operands
	var r2 gc.Node
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
			gins(ppc64.AMOVD, &con, &r1)
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
			gins(ppc64.AMOVD, &con, &r1)
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

	// float constants come from memory.
	//if(isfloat[tt])
	//	goto hard;

	// 64-bit immediates are also from memory.
	//if(isint[tt])
	//	goto hard;
	//// 64-bit immediates are really 32-bit sign-extended
	//// unless moving into a register.
	//if(isint[tt]) {
	//	if(mpcmpfixfix(con.val.u.xval, minintval[TINT32]) < 0)
	//		goto hard;
	//	if(mpcmpfixfix(con.val.u.xval, maxintval[TINT32]) > 0)
	//		goto hard;
	//}

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
		a = ppc64.AMOVB

	case gc.TINT8<<16 | gc.TUINT8, // same size
		gc.TUINT8<<16 | gc.TUINT8,
		gc.TINT16<<16 | gc.TUINT8,
		// truncate
		gc.TUINT16<<16 | gc.TUINT8,
		gc.TINT32<<16 | gc.TUINT8,
		gc.TUINT32<<16 | gc.TUINT8,
		gc.TINT64<<16 | gc.TUINT8,
		gc.TUINT64<<16 | gc.TUINT8:
		a = ppc64.AMOVBZ

	case gc.TINT16<<16 | gc.TINT16, // same size
		gc.TUINT16<<16 | gc.TINT16,
		gc.TINT32<<16 | gc.TINT16,
		// truncate
		gc.TUINT32<<16 | gc.TINT16,
		gc.TINT64<<16 | gc.TINT16,
		gc.TUINT64<<16 | gc.TINT16:
		a = ppc64.AMOVH

	case gc.TINT16<<16 | gc.TUINT16, // same size
		gc.TUINT16<<16 | gc.TUINT16,
		gc.TINT32<<16 | gc.TUINT16,
		// truncate
		gc.TUINT32<<16 | gc.TUINT16,
		gc.TINT64<<16 | gc.TUINT16,
		gc.TUINT64<<16 | gc.TUINT16:
		a = ppc64.AMOVHZ

	case gc.TINT32<<16 | gc.TINT32, // same size
		gc.TUINT32<<16 | gc.TINT32,
		gc.TINT64<<16 | gc.TINT32,
		// truncate
		gc.TUINT64<<16 | gc.TINT32:
		a = ppc64.AMOVW

	case gc.TINT32<<16 | gc.TUINT32, // same size
		gc.TUINT32<<16 | gc.TUINT32,
		gc.TINT64<<16 | gc.TUINT32,
		gc.TUINT64<<16 | gc.TUINT32:
		a = ppc64.AMOVWZ

	case gc.TINT64<<16 | gc.TINT64, // same size
		gc.TINT64<<16 | gc.TUINT64,
		gc.TUINT64<<16 | gc.TINT64,
		gc.TUINT64<<16 | gc.TUINT64:
		a = ppc64.AMOVD

		/*
		 * integer up-conversions
		 */
	case gc.TINT8<<16 | gc.TINT16, // sign extend int8
		gc.TINT8<<16 | gc.TUINT16,
		gc.TINT8<<16 | gc.TINT32,
		gc.TINT8<<16 | gc.TUINT32,
		gc.TINT8<<16 | gc.TINT64,
		gc.TINT8<<16 | gc.TUINT64:
		a = ppc64.AMOVB

		goto rdst

	case gc.TUINT8<<16 | gc.TINT16, // zero extend uint8
		gc.TUINT8<<16 | gc.TUINT16,
		gc.TUINT8<<16 | gc.TINT32,
		gc.TUINT8<<16 | gc.TUINT32,
		gc.TUINT8<<16 | gc.TINT64,
		gc.TUINT8<<16 | gc.TUINT64:
		a = ppc64.AMOVBZ

		goto rdst

	case gc.TINT16<<16 | gc.TINT32, // sign extend int16
		gc.TINT16<<16 | gc.TUINT32,
		gc.TINT16<<16 | gc.TINT64,
		gc.TINT16<<16 | gc.TUINT64:
		a = ppc64.AMOVH

		goto rdst

	case gc.TUINT16<<16 | gc.TINT32, // zero extend uint16
		gc.TUINT16<<16 | gc.TUINT32,
		gc.TUINT16<<16 | gc.TINT64,
		gc.TUINT16<<16 | gc.TUINT64:
		a = ppc64.AMOVHZ

		goto rdst

	case gc.TINT32<<16 | gc.TINT64, // sign extend int32
		gc.TINT32<<16 | gc.TUINT64:
		a = ppc64.AMOVW

		goto rdst

	case gc.TUINT32<<16 | gc.TINT64, // zero extend uint32
		gc.TUINT32<<16 | gc.TUINT64:
		a = ppc64.AMOVWZ

		goto rdst

		//warn("gmove: convert float to int not implemented: %N -> %N\n", f, t);
	//return;
	// algorithm is:
	//	if small enough, use native float64 -> int64 conversion.
	//	otherwise, subtract 2^63, convert, and add it back.
	/*
	* float to integer
	 */
	case gc.TFLOAT32<<16 | gc.TINT32,
		gc.TFLOAT64<<16 | gc.TINT32,
		gc.TFLOAT32<<16 | gc.TINT64,
		gc.TFLOAT64<<16 | gc.TINT64,
		gc.TFLOAT32<<16 | gc.TINT16,
		gc.TFLOAT32<<16 | gc.TINT8,
		gc.TFLOAT32<<16 | gc.TUINT16,
		gc.TFLOAT32<<16 | gc.TUINT8,
		gc.TFLOAT64<<16 | gc.TINT16,
		gc.TFLOAT64<<16 | gc.TINT8,
		gc.TFLOAT64<<16 | gc.TUINT16,
		gc.TFLOAT64<<16 | gc.TUINT8,
		gc.TFLOAT32<<16 | gc.TUINT32,
		gc.TFLOAT64<<16 | gc.TUINT32,
		gc.TFLOAT32<<16 | gc.TUINT64,
		gc.TFLOAT64<<16 | gc.TUINT64:
		bignodes()

		var r1 gc.Node
		gc.Regalloc(&r1, gc.Types[ft], f)
		gmove(f, &r1)
		if tt == gc.TUINT64 {
			gc.Regalloc(&r2, gc.Types[gc.TFLOAT64], nil)
			gmove(&bigf, &r2)
			gins(ppc64.AFCMPU, &r1, &r2)
			p1 := (*obj.Prog)(gc.Gbranch(optoas(gc.OLT, gc.Types[gc.TFLOAT64]), nil, +1))
			gins(ppc64.AFSUB, &r2, &r1)
			gc.Patch(p1, gc.Pc)
			gc.Regfree(&r2)
		}

		gc.Regalloc(&r2, gc.Types[gc.TFLOAT64], nil)
		var r3 gc.Node
		gc.Regalloc(&r3, gc.Types[gc.TINT64], t)
		gins(ppc64.AFCTIDZ, &r1, &r2)
		p1 := (*obj.Prog)(gins(ppc64.AFMOVD, &r2, nil))
		p1.To.Type = obj.TYPE_MEM
		p1.To.Reg = ppc64.REGSP
		p1.To.Offset = -8
		p1 = gins(ppc64.AMOVD, nil, &r3)
		p1.From.Type = obj.TYPE_MEM
		p1.From.Reg = ppc64.REGSP
		p1.From.Offset = -8
		gc.Regfree(&r2)
		gc.Regfree(&r1)
		if tt == gc.TUINT64 {
			p1 := (*obj.Prog)(gc.Gbranch(optoas(gc.OLT, gc.Types[gc.TFLOAT64]), nil, +1)) // use CR0 here again
			gc.Nodreg(&r1, gc.Types[gc.TINT64], ppc64.REGTMP)
			gins(ppc64.AMOVD, &bigi, &r1)
			gins(ppc64.AADD, &r1, &r3)
			gc.Patch(p1, gc.Pc)
		}

		gmove(&r3, t)
		gc.Regfree(&r3)
		return

		//warn("gmove: convert int to float not implemented: %N -> %N\n", f, t);
	//return;
	// algorithm is:
	//	if small enough, use native int64 -> uint64 conversion.
	//	otherwise, halve (rounding to odd?), convert, and double.
	/*
	 * integer to float
	 */
	case gc.TINT32<<16 | gc.TFLOAT32,
		gc.TINT32<<16 | gc.TFLOAT64,
		gc.TINT64<<16 | gc.TFLOAT32,
		gc.TINT64<<16 | gc.TFLOAT64,
		gc.TINT16<<16 | gc.TFLOAT32,
		gc.TINT16<<16 | gc.TFLOAT64,
		gc.TINT8<<16 | gc.TFLOAT32,
		gc.TINT8<<16 | gc.TFLOAT64,
		gc.TUINT16<<16 | gc.TFLOAT32,
		gc.TUINT16<<16 | gc.TFLOAT64,
		gc.TUINT8<<16 | gc.TFLOAT32,
		gc.TUINT8<<16 | gc.TFLOAT64,
		gc.TUINT32<<16 | gc.TFLOAT32,
		gc.TUINT32<<16 | gc.TFLOAT64,
		gc.TUINT64<<16 | gc.TFLOAT32,
		gc.TUINT64<<16 | gc.TFLOAT64:
		bignodes()

		var r1 gc.Node
		gc.Regalloc(&r1, gc.Types[gc.TINT64], nil)
		gmove(f, &r1)
		if ft == gc.TUINT64 {
			gc.Nodreg(&r2, gc.Types[gc.TUINT64], ppc64.REGTMP)
			gmove(&bigi, &r2)
			gins(ppc64.ACMPU, &r1, &r2)
			p1 := (*obj.Prog)(gc.Gbranch(optoas(gc.OLT, gc.Types[gc.TUINT64]), nil, +1))
			p2 := (*obj.Prog)(gins(ppc64.ASRD, nil, &r1))
			p2.From.Type = obj.TYPE_CONST
			p2.From.Offset = 1
			gc.Patch(p1, gc.Pc)
		}

		gc.Regalloc(&r2, gc.Types[gc.TFLOAT64], t)
		p1 := (*obj.Prog)(gins(ppc64.AMOVD, &r1, nil))
		p1.To.Type = obj.TYPE_MEM
		p1.To.Reg = ppc64.REGSP
		p1.To.Offset = -8
		p1 = gins(ppc64.AFMOVD, nil, &r2)
		p1.From.Type = obj.TYPE_MEM
		p1.From.Reg = ppc64.REGSP
		p1.From.Offset = -8
		gins(ppc64.AFCFID, &r2, &r2)
		gc.Regfree(&r1)
		if ft == gc.TUINT64 {
			p1 := (*obj.Prog)(gc.Gbranch(optoas(gc.OLT, gc.Types[gc.TUINT64]), nil, +1)) // use CR0 here again
			gc.Nodreg(&r1, gc.Types[gc.TFLOAT64], ppc64.FREGTWO)
			gins(ppc64.AFMUL, &r1, &r2)
			gc.Patch(p1, gc.Pc)
		}

		gmove(&r2, t)
		gc.Regfree(&r2)
		return

		/*
		 * float to float
		 */
	case gc.TFLOAT32<<16 | gc.TFLOAT32:
		a = ppc64.AFMOVS

	case gc.TFLOAT64<<16 | gc.TFLOAT64:
		a = ppc64.AFMOVD

	case gc.TFLOAT32<<16 | gc.TFLOAT64:
		a = ppc64.AFMOVS
		goto rdst

	case gc.TFLOAT64<<16 | gc.TFLOAT32:
		a = ppc64.AFRSP
		goto rdst
	}

	gins(a, f, t)
	return

	// requires register destination
rdst:
	{
		gc.Regalloc(&r1, t.Type, t)

		gins(a, f, &r1)
		gmove(&r1, t)
		gc.Regfree(&r1)
		return
	}

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
	if as == ppc64.ACMP || as == ppc64.ACMPU {
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
	case obj.ACALL:
		if p.To.Type == obj.TYPE_REG && p.To.Reg != ppc64.REG_CTR {
			// Allow front end to emit CALL REG, and rewrite into MOV REG, CTR; CALL CTR.
			pp := gc.Prog(as)
			pp.From = p.From
			pp.To.Type = obj.TYPE_REG
			pp.To.Reg = ppc64.REG_CTR

			p.As = ppc64.AMOVD
			p.From = p.To
			p.To.Type = obj.TYPE_REG
			p.To.Reg = ppc64.REG_CTR

			if gc.Debug['g'] != 0 {
				fmt.Printf("%v\n", p)
				fmt.Printf("%v\n", pp)
			}

			return pp
		}

	// Bad things the front end has done to us. Crash to find call stack.
	case ppc64.AAND, ppc64.AMULLD:
		if p.From.Type == obj.TYPE_CONST {
			gc.Debug['h'] = 1
			gc.Fatal("bad inst: %v", p)
		}
	case ppc64.ACMP, ppc64.ACMPU:
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
	case ppc64.AMOVB,
		ppc64.AMOVBU,
		ppc64.AMOVBZ,
		ppc64.AMOVBZU:
		w = 1

	case ppc64.AMOVH,
		ppc64.AMOVHU,
		ppc64.AMOVHZ,
		ppc64.AMOVHZU:
		w = 2

	case ppc64.AMOVW,
		ppc64.AMOVWU,
		ppc64.AMOVWZ,
		ppc64.AMOVWZU:
		w = 4

	case ppc64.AMOVD,
		ppc64.AMOVDU:
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
	if n.Reg == ppc64.REGSP { // stack offset cannot be large
		return
	}
	if n.Xoffset != int64(int32(n.Xoffset)) {
		// TODO(minux): offset too large, move into R31 and add to R31 instead.
		// this is used only in test/fixedbugs/issue6036.go.
		gc.Fatal("offset too large: %v", gc.Nconv(n, 0))

		a := gc.Node(*n)
		a.Op = gc.OREGISTER
		a.Type = gc.Types[gc.Tptr]
		a.Xoffset = 0
		gc.Cgen_checknil(&a)
		ginscon(optoas(gc.OADD, gc.Types[gc.Tptr]), n.Xoffset, &a)
		n.Xoffset = 0
	}
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
		a = ppc64.ABEQ

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
		a = ppc64.ABNE

	case gc.OLT<<16 | gc.TINT8, // ACMP
		gc.OLT<<16 | gc.TINT16,
		gc.OLT<<16 | gc.TINT32,
		gc.OLT<<16 | gc.TINT64,
		gc.OLT<<16 | gc.TUINT8,
		// ACMPU
		gc.OLT<<16 | gc.TUINT16,
		gc.OLT<<16 | gc.TUINT32,
		gc.OLT<<16 | gc.TUINT64,
		gc.OLT<<16 | gc.TFLOAT32,
		// AFCMPU
		gc.OLT<<16 | gc.TFLOAT64:
		a = ppc64.ABLT

	case gc.OLE<<16 | gc.TINT8, // ACMP
		gc.OLE<<16 | gc.TINT16,
		gc.OLE<<16 | gc.TINT32,
		gc.OLE<<16 | gc.TINT64,
		gc.OLE<<16 | gc.TUINT8,
		// ACMPU
		gc.OLE<<16 | gc.TUINT16,
		gc.OLE<<16 | gc.TUINT32,
		gc.OLE<<16 | gc.TUINT64:
		// No OLE for floats, because it mishandles NaN.
		// Front end must reverse comparison or use OLT and OEQ together.
		a = ppc64.ABLE

	case gc.OGT<<16 | gc.TINT8,
		gc.OGT<<16 | gc.TINT16,
		gc.OGT<<16 | gc.TINT32,
		gc.OGT<<16 | gc.TINT64,
		gc.OGT<<16 | gc.TUINT8,
		gc.OGT<<16 | gc.TUINT16,
		gc.OGT<<16 | gc.TUINT32,
		gc.OGT<<16 | gc.TUINT64,
		gc.OGT<<16 | gc.TFLOAT32,
		gc.OGT<<16 | gc.TFLOAT64:
		a = ppc64.ABGT

	case gc.OGE<<16 | gc.TINT8,
		gc.OGE<<16 | gc.TINT16,
		gc.OGE<<16 | gc.TINT32,
		gc.OGE<<16 | gc.TINT64,
		gc.OGE<<16 | gc.TUINT8,
		gc.OGE<<16 | gc.TUINT16,
		gc.OGE<<16 | gc.TUINT32,
		gc.OGE<<16 | gc.TUINT64:
		// No OGE for floats, because it mishandles NaN.
		// Front end must reverse comparison or use OLT and OEQ together.
		a = ppc64.ABGE

	case gc.OCMP<<16 | gc.TBOOL,
		gc.OCMP<<16 | gc.TINT8,
		gc.OCMP<<16 | gc.TINT16,
		gc.OCMP<<16 | gc.TINT32,
		gc.OCMP<<16 | gc.TPTR32,
		gc.OCMP<<16 | gc.TINT64:
		a = ppc64.ACMP

	case gc.OCMP<<16 | gc.TUINT8,
		gc.OCMP<<16 | gc.TUINT16,
		gc.OCMP<<16 | gc.TUINT32,
		gc.OCMP<<16 | gc.TUINT64,
		gc.OCMP<<16 | gc.TPTR64:
		a = ppc64.ACMPU

	case gc.OCMP<<16 | gc.TFLOAT32,
		gc.OCMP<<16 | gc.TFLOAT64:
		a = ppc64.AFCMPU

	case gc.OAS<<16 | gc.TBOOL,
		gc.OAS<<16 | gc.TINT8:
		a = ppc64.AMOVB

	case gc.OAS<<16 | gc.TUINT8:
		a = ppc64.AMOVBZ

	case gc.OAS<<16 | gc.TINT16:
		a = ppc64.AMOVH

	case gc.OAS<<16 | gc.TUINT16:
		a = ppc64.AMOVHZ

	case gc.OAS<<16 | gc.TINT32:
		a = ppc64.AMOVW

	case gc.OAS<<16 | gc.TUINT32,
		gc.OAS<<16 | gc.TPTR32:
		a = ppc64.AMOVWZ

	case gc.OAS<<16 | gc.TINT64,
		gc.OAS<<16 | gc.TUINT64,
		gc.OAS<<16 | gc.TPTR64:
		a = ppc64.AMOVD

	case gc.OAS<<16 | gc.TFLOAT32:
		a = ppc64.AFMOVS

	case gc.OAS<<16 | gc.TFLOAT64:
		a = ppc64.AFMOVD

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
		a = ppc64.AADD

	case gc.OADD<<16 | gc.TFLOAT32:
		a = ppc64.AFADDS

	case gc.OADD<<16 | gc.TFLOAT64:
		a = ppc64.AFADD

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
		a = ppc64.ASUB

	case gc.OSUB<<16 | gc.TFLOAT32:
		a = ppc64.AFSUBS

	case gc.OSUB<<16 | gc.TFLOAT64:
		a = ppc64.AFSUB

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
		a = ppc64.ANEG

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
		a = ppc64.AAND

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
		a = ppc64.AOR

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
		a = ppc64.AXOR

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
		a = ppc64.ASLD

	case gc.ORSH<<16 | gc.TUINT8,
		gc.ORSH<<16 | gc.TUINT16,
		gc.ORSH<<16 | gc.TUINT32,
		gc.ORSH<<16 | gc.TPTR32,
		gc.ORSH<<16 | gc.TUINT64,
		gc.ORSH<<16 | gc.TPTR64:
		a = ppc64.ASRD

	case gc.ORSH<<16 | gc.TINT8,
		gc.ORSH<<16 | gc.TINT16,
		gc.ORSH<<16 | gc.TINT32,
		gc.ORSH<<16 | gc.TINT64:
		a = ppc64.ASRAD

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
		a = ppc64.AMULHD

	case gc.OHMUL<<16 | gc.TUINT64,
		gc.OHMUL<<16 | gc.TPTR64:
		a = ppc64.AMULHDU

	case gc.OMUL<<16 | gc.TINT8,
		gc.OMUL<<16 | gc.TINT16,
		gc.OMUL<<16 | gc.TINT32,
		gc.OMUL<<16 | gc.TINT64:
		a = ppc64.AMULLD

	case gc.OMUL<<16 | gc.TUINT8,
		gc.OMUL<<16 | gc.TUINT16,
		gc.OMUL<<16 | gc.TUINT32,
		gc.OMUL<<16 | gc.TPTR32,
		// don't use word multiply, the high 32-bit are undefined.
		// fallthrough
		gc.OMUL<<16 | gc.TUINT64,
		gc.OMUL<<16 | gc.TPTR64:
		a = ppc64.AMULLD
		// for 64-bit multiplies, signedness doesn't matter.

	case gc.OMUL<<16 | gc.TFLOAT32:
		a = ppc64.AFMULS

	case gc.OMUL<<16 | gc.TFLOAT64:
		a = ppc64.AFMUL

	case gc.ODIV<<16 | gc.TINT8,
		gc.ODIV<<16 | gc.TINT16,
		gc.ODIV<<16 | gc.TINT32,
		gc.ODIV<<16 | gc.TINT64:
		a = ppc64.ADIVD

	case gc.ODIV<<16 | gc.TUINT8,
		gc.ODIV<<16 | gc.TUINT16,
		gc.ODIV<<16 | gc.TUINT32,
		gc.ODIV<<16 | gc.TPTR32,
		gc.ODIV<<16 | gc.TUINT64,
		gc.ODIV<<16 | gc.TPTR64:
		a = ppc64.ADIVDU

	case gc.ODIV<<16 | gc.TFLOAT32:
		a = ppc64.AFDIVS

	case gc.ODIV<<16 | gc.TFLOAT64:
		a = ppc64.AFDIV
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
