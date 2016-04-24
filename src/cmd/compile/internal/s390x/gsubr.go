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

package s390x

import (
	"cmd/compile/internal/gc"
	"cmd/internal/obj"
	"cmd/internal/obj/s390x"
	"fmt"
)

var resvd = []int{
	s390x.REGZERO, // R0
	s390x.REGTMP,  // R10
	s390x.REGTMP2, // R11
	s390x.REGCTXT, // R12
	s390x.REGG,    // R13
	s390x.REG_LR,  // R14
	s390x.REGSP,   // R15
}

// generate
//	as $c, n
func ginscon(as obj.As, c int64, n2 *gc.Node) {
	var n1 gc.Node

	gc.Nodconst(&n1, gc.Types[gc.TINT64], c)

	if as != s390x.AMOVD && (c < -s390x.BIG || c > s390x.BIG) || n2.Op != gc.OREGISTER {
		// cannot have more than 16-bit of immediate in ADD, etc.
		// instead, MOV into register first.
		var ntmp gc.Node
		gc.Regalloc(&ntmp, gc.Types[gc.TINT64], nil)

		rawgins(s390x.AMOVD, &n1, &ntmp)
		rawgins(as, &ntmp, n2)
		gc.Regfree(&ntmp)
		return
	}

	rawgins(as, &n1, n2)
}

// generate
//	as n, $c (CMP/CMPU)
func ginscon2(as obj.As, n2 *gc.Node, c int64) {
	var n1 gc.Node

	gc.Nodconst(&n1, gc.Types[gc.TINT64], c)

	switch as {
	default:
		gc.Fatalf("ginscon2")

	case s390x.ACMP:
		if -s390x.BIG <= c && c <= s390x.BIG {
			rawgins(as, n2, &n1)
			return
		}

	case s390x.ACMPU:
		if 0 <= c && c <= 2*s390x.BIG {
			rawgins(as, n2, &n1)
			return
		}
	}

	// MOV n1 into register first
	var ntmp gc.Node
	gc.Regalloc(&ntmp, gc.Types[gc.TINT64], nil)

	rawgins(s390x.AMOVD, &n1, &ntmp)
	rawgins(as, n2, &ntmp)
	gc.Regfree(&ntmp)
}

func ginscmp(op gc.Op, t *gc.Type, n1, n2 *gc.Node, likely int) *obj.Prog {
	if t.IsInteger() && n1.Op == gc.OLITERAL && n2.Op != gc.OLITERAL {
		// Reverse comparison to place constant last.
		op = gc.Brrev(op)
		n1, n2 = n2, n1
	}

	var r1, r2, g1, g2 gc.Node
	gc.Regalloc(&r1, t, n1)
	gc.Regalloc(&g1, n1.Type, &r1)
	gc.Cgen(n1, &g1)
	gmove(&g1, &r1)
	if t.IsInteger() && gc.Isconst(n2, gc.CTINT) {
		ginscon2(optoas(gc.OCMP, t), &r1, n2.Int64())
	} else {
		gc.Regalloc(&r2, t, n2)
		gc.Regalloc(&g2, n1.Type, &r2)
		gc.Cgen(n2, &g2)
		gmove(&g2, &r2)
		rawgins(optoas(gc.OCMP, t), &r1, &r2)
		gc.Regfree(&g2)
		gc.Regfree(&r2)
	}
	gc.Regfree(&g1)
	gc.Regfree(&r1)
	return gc.Gbranch(optoas(op, t), nil, likely)
}

// gmvc tries to move f to t using a mvc instruction.
// If successful it returns true, otherwise it returns false.
func gmvc(f, t *gc.Node) bool {
	ft := int(gc.Simsimtype(f.Type))
	tt := int(gc.Simsimtype(t.Type))

	if ft != tt {
		return false
	}

	if f.Op != gc.OINDREG || t.Op != gc.OINDREG {
		return false
	}

	if f.Xoffset < 0 || f.Xoffset >= 4096-8 {
		return false
	}

	if t.Xoffset < 0 || t.Xoffset >= 4096-8 {
		return false
	}

	var len int64
	switch ft {
	case gc.TUINT8, gc.TINT8, gc.TBOOL:
		len = 1
	case gc.TUINT16, gc.TINT16:
		len = 2
	case gc.TUINT32, gc.TINT32, gc.TFLOAT32:
		len = 4
	case gc.TUINT64, gc.TINT64, gc.TFLOAT64, gc.TPTR64:
		len = 8
	case gc.TUNSAFEPTR:
		len = int64(gc.Widthptr)
	default:
		return false
	}

	p := gc.Prog(s390x.AMVC)
	gc.Naddr(&p.From, f)
	gc.Naddr(&p.To, t)
	p.From3 = new(obj.Addr)
	p.From3.Offset = len
	p.From3.Type = obj.TYPE_CONST
	return true
}

// generate move:
//	t = f
// hard part is conversions.
func gmove(f *gc.Node, t *gc.Node) {
	if gc.Debug['M'] != 0 {
		fmt.Printf("gmove %v -> %v\n", gc.Nconv(f, gc.FmtLong), gc.Nconv(t, gc.FmtLong))
	}

	ft := int(gc.Simsimtype(f.Type))
	tt := int(gc.Simsimtype(t.Type))
	cvt := t.Type

	if gc.Iscomplex[ft] || gc.Iscomplex[tt] {
		gc.Complexmove(f, t)
		return
	}

	var a obj.As

	// cannot have two memory operands
	if gc.Ismem(f) && gc.Ismem(t) {
		if gmvc(f, t) {
			return
		}
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
			if t.Type.IsFloat() {
				goto hard
			}

			// all immediates are 16-bit sign-extended
			// unless moving into a register.
			if t.Type.IsInteger() {
				if i := con.Int64(); int64(int16(i)) != i {
					goto hard
				}
			}

			// immediate moves to memory have a 12-bit unsigned displacement
			if t.Xoffset < 0 || t.Xoffset >= 4096-8 {
				goto hard
			}
		}
	}

	// a float-to-int or int-to-float conversion requires the source operand in a register
	if gc.Ismem(f) && ((f.Type.IsFloat() && t.Type.IsInteger()) || (f.Type.IsInteger() && t.Type.IsFloat())) {
		cvt = f.Type
		goto hard
	}

	// a float32-to-float64 or float64-to-float32 conversion requires the source operand in a register
	if gc.Ismem(f) && f.Type.IsFloat() && t.Type.IsFloat() && (ft != tt) {
		cvt = f.Type
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
		gc.Fatalf("gmove %v -> %v", gc.Tconv(f.Type, gc.FmtLong), gc.Tconv(t.Type, gc.FmtLong))

	// integer copy and truncate
	case gc.TINT8<<16 | gc.TINT8,
		gc.TUINT8<<16 | gc.TINT8,
		gc.TINT16<<16 | gc.TINT8,
		gc.TUINT16<<16 | gc.TINT8,
		gc.TINT32<<16 | gc.TINT8,
		gc.TUINT32<<16 | gc.TINT8,
		gc.TINT64<<16 | gc.TINT8,
		gc.TUINT64<<16 | gc.TINT8:
		a = s390x.AMOVB

	case gc.TINT8<<16 | gc.TUINT8,
		gc.TUINT8<<16 | gc.TUINT8,
		gc.TINT16<<16 | gc.TUINT8,
		gc.TUINT16<<16 | gc.TUINT8,
		gc.TINT32<<16 | gc.TUINT8,
		gc.TUINT32<<16 | gc.TUINT8,
		gc.TINT64<<16 | gc.TUINT8,
		gc.TUINT64<<16 | gc.TUINT8:
		a = s390x.AMOVBZ

	case gc.TINT16<<16 | gc.TINT16,
		gc.TUINT16<<16 | gc.TINT16,
		gc.TINT32<<16 | gc.TINT16,
		gc.TUINT32<<16 | gc.TINT16,
		gc.TINT64<<16 | gc.TINT16,
		gc.TUINT64<<16 | gc.TINT16:
		a = s390x.AMOVH

	case gc.TINT16<<16 | gc.TUINT16,
		gc.TUINT16<<16 | gc.TUINT16,
		gc.TINT32<<16 | gc.TUINT16,
		gc.TUINT32<<16 | gc.TUINT16,
		gc.TINT64<<16 | gc.TUINT16,
		gc.TUINT64<<16 | gc.TUINT16:
		a = s390x.AMOVHZ

	case gc.TINT32<<16 | gc.TINT32,
		gc.TUINT32<<16 | gc.TINT32,
		gc.TINT64<<16 | gc.TINT32,
		gc.TUINT64<<16 | gc.TINT32:
		a = s390x.AMOVW

	case gc.TINT32<<16 | gc.TUINT32,
		gc.TUINT32<<16 | gc.TUINT32,
		gc.TINT64<<16 | gc.TUINT32,
		gc.TUINT64<<16 | gc.TUINT32:
		a = s390x.AMOVWZ

	case gc.TINT64<<16 | gc.TINT64,
		gc.TINT64<<16 | gc.TUINT64,
		gc.TUINT64<<16 | gc.TINT64,
		gc.TUINT64<<16 | gc.TUINT64:
		a = s390x.AMOVD

	// sign extend int8
	case gc.TINT8<<16 | gc.TINT16,
		gc.TINT8<<16 | gc.TUINT16,
		gc.TINT8<<16 | gc.TINT32,
		gc.TINT8<<16 | gc.TUINT32,
		gc.TINT8<<16 | gc.TINT64,
		gc.TINT8<<16 | gc.TUINT64:
		a = s390x.AMOVB
		goto rdst

	// sign extend uint8
	case gc.TUINT8<<16 | gc.TINT16,
		gc.TUINT8<<16 | gc.TUINT16,
		gc.TUINT8<<16 | gc.TINT32,
		gc.TUINT8<<16 | gc.TUINT32,
		gc.TUINT8<<16 | gc.TINT64,
		gc.TUINT8<<16 | gc.TUINT64:
		a = s390x.AMOVBZ
		goto rdst

	// sign extend int16
	case gc.TINT16<<16 | gc.TINT32,
		gc.TINT16<<16 | gc.TUINT32,
		gc.TINT16<<16 | gc.TINT64,
		gc.TINT16<<16 | gc.TUINT64:
		a = s390x.AMOVH
		goto rdst

	// zero extend uint16
	case gc.TUINT16<<16 | gc.TINT32,
		gc.TUINT16<<16 | gc.TUINT32,
		gc.TUINT16<<16 | gc.TINT64,
		gc.TUINT16<<16 | gc.TUINT64:
		a = s390x.AMOVHZ
		goto rdst

	// sign extend int32
	case gc.TINT32<<16 | gc.TINT64,
		gc.TINT32<<16 | gc.TUINT64:
		a = s390x.AMOVW
		goto rdst

	// zero extend uint32
	case gc.TUINT32<<16 | gc.TINT64,
		gc.TUINT32<<16 | gc.TUINT64:
		a = s390x.AMOVWZ
		goto rdst

	// float to integer
	case gc.TFLOAT32<<16 | gc.TUINT8,
		gc.TFLOAT32<<16 | gc.TUINT16:
		cvt = gc.Types[gc.TUINT32]
		goto hard

	case gc.TFLOAT32<<16 | gc.TUINT32:
		a = s390x.ACLFEBR
		goto rdst

	case gc.TFLOAT32<<16 | gc.TUINT64:
		a = s390x.ACLGEBR
		goto rdst

	case gc.TFLOAT64<<16 | gc.TUINT8,
		gc.TFLOAT64<<16 | gc.TUINT16:
		cvt = gc.Types[gc.TUINT32]
		goto hard

	case gc.TFLOAT64<<16 | gc.TUINT32:
		a = s390x.ACLFDBR
		goto rdst

	case gc.TFLOAT64<<16 | gc.TUINT64:
		a = s390x.ACLGDBR
		goto rdst

	case gc.TFLOAT32<<16 | gc.TINT8,
		gc.TFLOAT32<<16 | gc.TINT16:
		cvt = gc.Types[gc.TINT32]
		goto hard

	case gc.TFLOAT32<<16 | gc.TINT32:
		a = s390x.ACFEBRA
		goto rdst

	case gc.TFLOAT32<<16 | gc.TINT64:
		a = s390x.ACGEBRA
		goto rdst

	case gc.TFLOAT64<<16 | gc.TINT8,
		gc.TFLOAT64<<16 | gc.TINT16:
		cvt = gc.Types[gc.TINT32]
		goto hard

	case gc.TFLOAT64<<16 | gc.TINT32:
		a = s390x.ACFDBRA
		goto rdst

	case gc.TFLOAT64<<16 | gc.TINT64:
		a = s390x.ACGDBRA
		goto rdst

	// integer to float
	case gc.TUINT8<<16 | gc.TFLOAT32,
		gc.TUINT16<<16 | gc.TFLOAT32:
		cvt = gc.Types[gc.TUINT32]
		goto hard

	case gc.TUINT32<<16 | gc.TFLOAT32:
		a = s390x.ACELFBR
		goto rdst

	case gc.TUINT64<<16 | gc.TFLOAT32:
		a = s390x.ACELGBR
		goto rdst

	case gc.TUINT8<<16 | gc.TFLOAT64,
		gc.TUINT16<<16 | gc.TFLOAT64:
		cvt = gc.Types[gc.TUINT32]
		goto hard

	case gc.TUINT32<<16 | gc.TFLOAT64:
		a = s390x.ACDLFBR
		goto rdst

	case gc.TUINT64<<16 | gc.TFLOAT64:
		a = s390x.ACDLGBR
		goto rdst

	case gc.TINT8<<16 | gc.TFLOAT32,
		gc.TINT16<<16 | gc.TFLOAT32:
		cvt = gc.Types[gc.TINT32]
		goto hard

	case gc.TINT32<<16 | gc.TFLOAT32:
		a = s390x.ACEFBRA
		goto rdst

	case gc.TINT64<<16 | gc.TFLOAT32:
		a = s390x.ACEGBRA
		goto rdst

	case gc.TINT8<<16 | gc.TFLOAT64,
		gc.TINT16<<16 | gc.TFLOAT64:
		cvt = gc.Types[gc.TINT32]
		goto hard

	case gc.TINT32<<16 | gc.TFLOAT64:
		a = s390x.ACDFBRA
		goto rdst

	case gc.TINT64<<16 | gc.TFLOAT64:
		a = s390x.ACDGBRA
		goto rdst

	// float to float
	case gc.TFLOAT32<<16 | gc.TFLOAT32:
		a = s390x.AFMOVS

	case gc.TFLOAT64<<16 | gc.TFLOAT64:
		a = s390x.AFMOVD

	case gc.TFLOAT32<<16 | gc.TFLOAT64:
		a = s390x.ALDEBR
		goto rdst

	case gc.TFLOAT64<<16 | gc.TFLOAT32:
		a = s390x.ALEDBR
		goto rdst
	}

	gins(a, f, t)
	return

	// requires register destination
rdst:
	if t != nil && t.Op == gc.OREGISTER {
		gins(a, f, t)
		return
	} else {
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

func intLiteral(n *gc.Node) (x int64, ok bool) {
	switch {
	case n == nil:
		return
	case gc.Isconst(n, gc.CTINT):
		return n.Int64(), true
	case gc.Isconst(n, gc.CTBOOL):
		return int64(obj.Bool2int(n.Bool())), true
	}
	return
}

// gins is called by the front end.
// It synthesizes some multiple-instruction sequences
// so the front end can stay simpler.
func gins(as obj.As, f, t *gc.Node) *obj.Prog {
	if t != nil {
		if as >= obj.A_ARCHSPECIFIC {
			if x, ok := intLiteral(f); ok {
				ginscon(as, x, t)
				return nil // caller must not use
			}
		}
		if as == s390x.ACMP || as == s390x.ACMPU {
			if x, ok := intLiteral(t); ok {
				ginscon2(as, f, x)
				return nil // caller must not use
			}
		}
	}
	return rawgins(as, f, t)
}

// generate one instruction:
//	as f, t
func rawgins(as obj.As, f *gc.Node, t *gc.Node) *obj.Prog {
	// self move check
	// TODO(mundaym): use sized math and extend to MOVB, MOVWZ etc.
	switch as {
	case s390x.AMOVD, s390x.AFMOVS, s390x.AFMOVD:
		if f != nil && t != nil &&
			f.Op == gc.OREGISTER && t.Op == gc.OREGISTER &&
			f.Reg == t.Reg {
			return nil
		}
	}

	p := gc.Prog(as)
	gc.Naddr(&p.From, f)
	gc.Naddr(&p.To, t)

	switch as {
	// Bad things the front end has done to us. Crash to find call stack.
	case s390x.ACMP, s390x.ACMPU:
		if p.From.Type == obj.TYPE_MEM || p.To.Type == obj.TYPE_MEM {
			gc.Debug['h'] = 1
			gc.Fatalf("bad inst: %v", p)
		}
	}

	if gc.Debug['g'] != 0 {
		fmt.Printf("%v\n", p)
	}

	w := int32(0)
	switch as {
	case s390x.AMOVB, s390x.AMOVBZ:
		w = 1

	case s390x.AMOVH, s390x.AMOVHZ:
		w = 2

	case s390x.AMOVW, s390x.AMOVWZ:
		w = 4

	case s390x.AMOVD:
		if p.From.Type == obj.TYPE_CONST || p.From.Type == obj.TYPE_ADDR {
			break
		}
		w = 8
	}

	if w != 0 && ((f != nil && p.From.Width < int64(w)) || (t != nil && p.To.Type != obj.TYPE_REG && p.To.Width > int64(w))) {
		gc.Dump("f", f)
		gc.Dump("t", t)
		gc.Fatalf("bad width: %v (%d, %d)\n", p, p.From.Width, p.To.Width)
	}

	return p
}

// optoas returns the Axxx equivalent of Oxxx for type t
func optoas(op gc.Op, t *gc.Type) obj.As {
	if t == nil {
		gc.Fatalf("optoas: t is nil")
	}

	// avoid constant conversions in switches below
	const (
		OMINUS_ = uint32(gc.OMINUS) << 16
		OLSH_   = uint32(gc.OLSH) << 16
		ORSH_   = uint32(gc.ORSH) << 16
		OADD_   = uint32(gc.OADD) << 16
		OSUB_   = uint32(gc.OSUB) << 16
		OMUL_   = uint32(gc.OMUL) << 16
		ODIV_   = uint32(gc.ODIV) << 16
		OOR_    = uint32(gc.OOR) << 16
		OAND_   = uint32(gc.OAND) << 16
		OXOR_   = uint32(gc.OXOR) << 16
		OEQ_    = uint32(gc.OEQ) << 16
		ONE_    = uint32(gc.ONE) << 16
		OLT_    = uint32(gc.OLT) << 16
		OLE_    = uint32(gc.OLE) << 16
		OGE_    = uint32(gc.OGE) << 16
		OGT_    = uint32(gc.OGT) << 16
		OCMP_   = uint32(gc.OCMP) << 16
		OAS_    = uint32(gc.OAS) << 16
		OHMUL_  = uint32(gc.OHMUL) << 16
		OSQRT_  = uint32(gc.OSQRT) << 16
		OLROT_  = uint32(gc.OLROT) << 16
	)

	a := obj.AXXX
	switch uint32(op)<<16 | uint32(gc.Simtype[t.Etype]) {
	default:
		gc.Fatalf("optoas: no entry for op=%v type=%v", op, t)

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
		a = s390x.ABEQ

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
		a = s390x.ABNE

	case OLT_ | gc.TINT8, // ACMP
		OLT_ | gc.TINT16,
		OLT_ | gc.TINT32,
		OLT_ | gc.TINT64,
		OLT_ | gc.TUINT8,
		// ACMPU
		OLT_ | gc.TUINT16,
		OLT_ | gc.TUINT32,
		OLT_ | gc.TUINT64,
		OLT_ | gc.TFLOAT32,
		// AFCMPU
		OLT_ | gc.TFLOAT64:
		a = s390x.ABLT

	case OLE_ | gc.TINT8, // ACMP
		OLE_ | gc.TINT16,
		OLE_ | gc.TINT32,
		OLE_ | gc.TINT64,
		OLE_ | gc.TUINT8,
		// ACMPU
		OLE_ | gc.TUINT16,
		OLE_ | gc.TUINT32,
		OLE_ | gc.TUINT64,
		OLE_ | gc.TFLOAT32,
		OLE_ | gc.TFLOAT64:
		a = s390x.ABLE

	case OGT_ | gc.TINT8,
		OGT_ | gc.TINT16,
		OGT_ | gc.TINT32,
		OGT_ | gc.TINT64,
		OGT_ | gc.TUINT8,
		OGT_ | gc.TUINT16,
		OGT_ | gc.TUINT32,
		OGT_ | gc.TUINT64,
		OGT_ | gc.TFLOAT32,
		OGT_ | gc.TFLOAT64:
		a = s390x.ABGT

	case OGE_ | gc.TINT8,
		OGE_ | gc.TINT16,
		OGE_ | gc.TINT32,
		OGE_ | gc.TINT64,
		OGE_ | gc.TUINT8,
		OGE_ | gc.TUINT16,
		OGE_ | gc.TUINT32,
		OGE_ | gc.TUINT64,
		OGE_ | gc.TFLOAT32,
		OGE_ | gc.TFLOAT64:
		a = s390x.ABGE

	case OCMP_ | gc.TBOOL,
		OCMP_ | gc.TINT8,
		OCMP_ | gc.TINT16,
		OCMP_ | gc.TINT32,
		OCMP_ | gc.TPTR32,
		OCMP_ | gc.TINT64:
		a = s390x.ACMP

	case OCMP_ | gc.TUINT8,
		OCMP_ | gc.TUINT16,
		OCMP_ | gc.TUINT32,
		OCMP_ | gc.TUINT64,
		OCMP_ | gc.TPTR64:
		a = s390x.ACMPU

	case OCMP_ | gc.TFLOAT32:
		a = s390x.ACEBR

	case OCMP_ | gc.TFLOAT64:
		a = s390x.AFCMPU

	case OAS_ | gc.TBOOL,
		OAS_ | gc.TINT8:
		a = s390x.AMOVB

	case OAS_ | gc.TUINT8:
		a = s390x.AMOVBZ

	case OAS_ | gc.TINT16:
		a = s390x.AMOVH

	case OAS_ | gc.TUINT16:
		a = s390x.AMOVHZ

	case OAS_ | gc.TINT32:
		a = s390x.AMOVW

	case OAS_ | gc.TUINT32,
		OAS_ | gc.TPTR32:
		a = s390x.AMOVWZ

	case OAS_ | gc.TINT64,
		OAS_ | gc.TUINT64,
		OAS_ | gc.TPTR64:
		a = s390x.AMOVD

	case OAS_ | gc.TFLOAT32:
		a = s390x.AFMOVS

	case OAS_ | gc.TFLOAT64:
		a = s390x.AFMOVD

	case OADD_ | gc.TINT8,
		OADD_ | gc.TUINT8,
		OADD_ | gc.TINT16,
		OADD_ | gc.TUINT16,
		OADD_ | gc.TINT32,
		OADD_ | gc.TUINT32,
		OADD_ | gc.TPTR32,
		OADD_ | gc.TINT64,
		OADD_ | gc.TUINT64,
		OADD_ | gc.TPTR64:
		a = s390x.AADD

	case OADD_ | gc.TFLOAT32:
		a = s390x.AFADDS

	case OADD_ | gc.TFLOAT64:
		a = s390x.AFADD

	case OSUB_ | gc.TINT8,
		OSUB_ | gc.TUINT8,
		OSUB_ | gc.TINT16,
		OSUB_ | gc.TUINT16,
		OSUB_ | gc.TINT32,
		OSUB_ | gc.TUINT32,
		OSUB_ | gc.TPTR32,
		OSUB_ | gc.TINT64,
		OSUB_ | gc.TUINT64,
		OSUB_ | gc.TPTR64:
		a = s390x.ASUB

	case OSUB_ | gc.TFLOAT32:
		a = s390x.AFSUBS

	case OSUB_ | gc.TFLOAT64:
		a = s390x.AFSUB

	case OMINUS_ | gc.TINT8,
		OMINUS_ | gc.TUINT8,
		OMINUS_ | gc.TINT16,
		OMINUS_ | gc.TUINT16,
		OMINUS_ | gc.TINT32,
		OMINUS_ | gc.TUINT32,
		OMINUS_ | gc.TPTR32,
		OMINUS_ | gc.TINT64,
		OMINUS_ | gc.TUINT64,
		OMINUS_ | gc.TPTR64:
		a = s390x.ANEG

	case OAND_ | gc.TINT8,
		OAND_ | gc.TUINT8,
		OAND_ | gc.TINT16,
		OAND_ | gc.TUINT16,
		OAND_ | gc.TINT32,
		OAND_ | gc.TUINT32,
		OAND_ | gc.TPTR32,
		OAND_ | gc.TINT64,
		OAND_ | gc.TUINT64,
		OAND_ | gc.TPTR64:
		a = s390x.AAND

	case OOR_ | gc.TINT8,
		OOR_ | gc.TUINT8,
		OOR_ | gc.TINT16,
		OOR_ | gc.TUINT16,
		OOR_ | gc.TINT32,
		OOR_ | gc.TUINT32,
		OOR_ | gc.TPTR32,
		OOR_ | gc.TINT64,
		OOR_ | gc.TUINT64,
		OOR_ | gc.TPTR64:
		a = s390x.AOR

	case OXOR_ | gc.TINT8,
		OXOR_ | gc.TUINT8,
		OXOR_ | gc.TINT16,
		OXOR_ | gc.TUINT16,
		OXOR_ | gc.TINT32,
		OXOR_ | gc.TUINT32,
		OXOR_ | gc.TPTR32,
		OXOR_ | gc.TINT64,
		OXOR_ | gc.TUINT64,
		OXOR_ | gc.TPTR64:
		a = s390x.AXOR

	case OLSH_ | gc.TINT8,
		OLSH_ | gc.TUINT8,
		OLSH_ | gc.TINT16,
		OLSH_ | gc.TUINT16,
		OLSH_ | gc.TINT32,
		OLSH_ | gc.TUINT32,
		OLSH_ | gc.TPTR32,
		OLSH_ | gc.TINT64,
		OLSH_ | gc.TUINT64,
		OLSH_ | gc.TPTR64:
		a = s390x.ASLD

	case ORSH_ | gc.TUINT8,
		ORSH_ | gc.TUINT16,
		ORSH_ | gc.TUINT32,
		ORSH_ | gc.TPTR32,
		ORSH_ | gc.TUINT64,
		ORSH_ | gc.TPTR64:
		a = s390x.ASRD

	case ORSH_ | gc.TINT8,
		ORSH_ | gc.TINT16,
		ORSH_ | gc.TINT32,
		ORSH_ | gc.TINT64:
		a = s390x.ASRAD

	case OHMUL_ | gc.TINT64:
		a = s390x.AMULHD

	case OHMUL_ | gc.TUINT64,
		OHMUL_ | gc.TPTR64:
		a = s390x.AMULHDU

	case OMUL_ | gc.TINT8,
		OMUL_ | gc.TINT16,
		OMUL_ | gc.TINT32,
		OMUL_ | gc.TINT64:
		a = s390x.AMULLD

	case OMUL_ | gc.TUINT8,
		OMUL_ | gc.TUINT16,
		OMUL_ | gc.TUINT32,
		OMUL_ | gc.TPTR32,
		// don't use word multiply, the high 32-bit are undefined.
		OMUL_ | gc.TUINT64,
		OMUL_ | gc.TPTR64:
		// for 64-bit multiplies, signedness doesn't matter.
		a = s390x.AMULLD

	case OMUL_ | gc.TFLOAT32:
		a = s390x.AFMULS

	case OMUL_ | gc.TFLOAT64:
		a = s390x.AFMUL

	case ODIV_ | gc.TINT8,
		ODIV_ | gc.TINT16,
		ODIV_ | gc.TINT32,
		ODIV_ | gc.TINT64:
		a = s390x.ADIVD

	case ODIV_ | gc.TUINT8,
		ODIV_ | gc.TUINT16,
		ODIV_ | gc.TUINT32,
		ODIV_ | gc.TPTR32,
		ODIV_ | gc.TUINT64,
		ODIV_ | gc.TPTR64:
		a = s390x.ADIVDU

	case ODIV_ | gc.TFLOAT32:
		a = s390x.AFDIVS

	case ODIV_ | gc.TFLOAT64:
		a = s390x.AFDIV

	case OSQRT_ | gc.TFLOAT64:
		a = s390x.AFSQRT

	case OLROT_ | gc.TUINT32,
		OLROT_ | gc.TPTR32,
		OLROT_ | gc.TINT32:
		a = s390x.ARLL

	case OLROT_ | gc.TUINT64,
		OLROT_ | gc.TPTR64,
		OLROT_ | gc.TINT64:
		a = s390x.ARLLG
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
func sudoaddable(as obj.As, n *gc.Node, a *obj.Addr) bool {
	if n.Type == nil {
		return false
	}

	*a = obj.Addr{}

	switch n.Op {
	case gc.OLITERAL:
		if !gc.Isconst(n, gc.CTINT) {
			return false
		}
		v := n.Int64()
		switch as {
		default:
			return false

		// operations that can cope with a 32-bit immediate
		// TODO(mundaym): logical operations can work on high bits
		case s390x.AADD,
			s390x.AADDC,
			s390x.ASUB,
			s390x.AMULLW,
			s390x.AAND,
			s390x.AOR,
			s390x.AXOR,
			s390x.ASLD,
			s390x.ASLW,
			s390x.ASRAW,
			s390x.ASRAD,
			s390x.ASRW,
			s390x.ASRD,
			s390x.AMOVB,
			s390x.AMOVBZ,
			s390x.AMOVH,
			s390x.AMOVHZ,
			s390x.AMOVW,
			s390x.AMOVWZ,
			s390x.AMOVD:
			if int64(int32(v)) != v {
				return false
			}

		// for comparisons avoid immediates unless they can
		// fit into a int8/uint8
		// this favours combined compare and branch instructions
		case s390x.ACMP:
			if int64(int8(v)) != v {
				return false
			}
		case s390x.ACMPU:
			if int64(uint8(v)) != v {
				return false
			}
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
			// check that the offset fits into a 12-bit displacement
			if n1.Xoffset < 0 || n1.Xoffset >= (1<<12)-8 {
				sudoclean()
				return false
			}
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
			gins(s390x.AMOVD, &n1, reg)
			gc.Cgen_checknil(reg)
			n1.Xoffset = -(oary[i] + 1)
		}

		a.Type = obj.TYPE_NONE
		a.Index = 0
		// check that the offset fits into a 12-bit displacement
		if n1.Xoffset < 0 || n1.Xoffset >= (1<<12)-8 {
			tmp := n1
			tmp.Op = gc.OREGISTER
			tmp.Type = gc.Types[gc.Tptr]
			tmp.Xoffset = 0
			gc.Cgen_checknil(&tmp)
			ginscon(s390x.AADD, n1.Xoffset, &tmp)
			n1.Xoffset = 0
		}
		gc.Naddr(a, &n1)
		return true
	}

	return false
}
