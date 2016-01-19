// Derived from Inferno utils/8c/txt.c
// http://code.google.com/p/inferno-os/source/browse/utils/8c/txt.c
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

package x86

import (
	"cmd/compile/internal/big"
	"cmd/compile/internal/gc"
	"cmd/internal/obj"
	"cmd/internal/obj/x86"
	"fmt"
)

// TODO(rsc): Can make this bigger if we move
// the text segment up higher in 8l for all GOOS.
// At the same time, can raise StackBig in ../../runtime/stack.h.
var unmappedzero uint32 = 4096

// foptoas flags
const (
	Frev  = 1 << 0
	Fpop  = 1 << 1
	Fpop2 = 1 << 2
)

/*
 * return Axxx for Oxxx on type t.
 */
func optoas(op gc.Op, t *gc.Type) int {
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
		OAS_     = uint32(gc.OAS) << 16
		OHMUL_   = uint32(gc.OHMUL) << 16
		OADDR_   = uint32(gc.OADDR) << 16
		OINC_    = uint32(gc.OINC) << 16
		ODEC_    = uint32(gc.ODEC) << 16
		OLROT_   = uint32(gc.OLROT) << 16
		OEXTEND_ = uint32(gc.OEXTEND) << 16
		OCOM_    = uint32(gc.OCOM) << 16
	)

	a := obj.AXXX
	switch uint32(op)<<16 | uint32(gc.Simtype[t.Etype]) {
	default:
		gc.Fatalf("optoas: no entry %v-%v", gc.Oconv(int(op), 0), t)

	case OADDR_ | gc.TPTR32:
		a = x86.ALEAL

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

	case OCOM_ | gc.TINT8,
		OCOM_ | gc.TUINT8:
		a = x86.ANOTB

	case OCOM_ | gc.TINT16,
		OCOM_ | gc.TUINT16:
		a = x86.ANOTW

	case OCOM_ | gc.TINT32,
		OCOM_ | gc.TUINT32,
		OCOM_ | gc.TPTR32:
		a = x86.ANOTL

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

	case OAND_ | gc.TINT8,
		OAND_ | gc.TUINT8:
		a = x86.AANDB

	case OAND_ | gc.TINT16,
		OAND_ | gc.TUINT16:
		a = x86.AANDW

	case OAND_ | gc.TINT32,
		OAND_ | gc.TUINT32,
		OAND_ | gc.TPTR32:
		a = x86.AANDL

	case OOR_ | gc.TINT8,
		OOR_ | gc.TUINT8:
		a = x86.AORB

	case OOR_ | gc.TINT16,
		OOR_ | gc.TUINT16:
		a = x86.AORW

	case OOR_ | gc.TINT32,
		OOR_ | gc.TUINT32,
		OOR_ | gc.TPTR32:
		a = x86.AORL

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

	case ORSH_ | gc.TUINT8:
		a = x86.ASHRB

	case ORSH_ | gc.TUINT16:
		a = x86.ASHRW

	case ORSH_ | gc.TUINT32,
		ORSH_ | gc.TPTR32:
		a = x86.ASHRL

	case ORSH_ | gc.TINT8:
		a = x86.ASARB

	case ORSH_ | gc.TINT16:
		a = x86.ASARW

	case ORSH_ | gc.TINT32:
		a = x86.ASARL

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

	case OHMUL_ | gc.TUINT8:
		a = x86.AMULB

	case OHMUL_ | gc.TUINT16:
		a = x86.AMULW

	case OHMUL_ | gc.TUINT32,
		OHMUL_ | gc.TPTR32:
		a = x86.AMULL

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

	case OEXTEND_ | gc.TINT16:
		a = x86.ACWD

	case OEXTEND_ | gc.TINT32:
		a = x86.ACDQ
	}

	return a
}

func foptoas(op gc.Op, t *gc.Type, flg int) int {
	a := obj.AXXX
	et := gc.Simtype[t.Etype]

	// avoid constant conversions in switches below
	const (
		OCMP_   = uint32(gc.OCMP) << 16
		OAS_    = uint32(gc.OAS) << 16
		OADD_   = uint32(gc.OADD) << 16
		OSUB_   = uint32(gc.OSUB) << 16
		OMUL_   = uint32(gc.OMUL) << 16
		ODIV_   = uint32(gc.ODIV) << 16
		OMINUS_ = uint32(gc.OMINUS) << 16
	)

	if !gc.Thearch.Use387 {
		switch uint32(op)<<16 | uint32(et) {
		default:
			gc.Fatalf("foptoas-sse: no entry %v-%v", gc.Oconv(int(op), 0), t)

		case OCMP_ | gc.TFLOAT32:
			a = x86.AUCOMISS

		case OCMP_ | gc.TFLOAT64:
			a = x86.AUCOMISD

		case OAS_ | gc.TFLOAT32:
			a = x86.AMOVSS

		case OAS_ | gc.TFLOAT64:
			a = x86.AMOVSD

		case OADD_ | gc.TFLOAT32:
			a = x86.AADDSS

		case OADD_ | gc.TFLOAT64:
			a = x86.AADDSD

		case OSUB_ | gc.TFLOAT32:
			a = x86.ASUBSS

		case OSUB_ | gc.TFLOAT64:
			a = x86.ASUBSD

		case OMUL_ | gc.TFLOAT32:
			a = x86.AMULSS

		case OMUL_ | gc.TFLOAT64:
			a = x86.AMULSD

		case ODIV_ | gc.TFLOAT32:
			a = x86.ADIVSS

		case ODIV_ | gc.TFLOAT64:
			a = x86.ADIVSD
		}

		return a
	}

	// If we need Fpop, it means we're working on
	// two different floating-point registers, not memory.
	// There the instruction only has a float64 form.
	if flg&Fpop != 0 {
		et = gc.TFLOAT64
	}

	// clear Frev if unneeded
	switch op {
	case gc.OADD,
		gc.OMUL:
		flg &^= Frev
	}

	switch uint32(op)<<16 | (uint32(et)<<8 | uint32(flg)) {
	case OADD_ | (gc.TFLOAT32<<8 | 0):
		return x86.AFADDF

	case OADD_ | (gc.TFLOAT64<<8 | 0):
		return x86.AFADDD

	case OADD_ | (gc.TFLOAT64<<8 | Fpop):
		return x86.AFADDDP

	case OSUB_ | (gc.TFLOAT32<<8 | 0):
		return x86.AFSUBF

	case OSUB_ | (gc.TFLOAT32<<8 | Frev):
		return x86.AFSUBRF

	case OSUB_ | (gc.TFLOAT64<<8 | 0):
		return x86.AFSUBD

	case OSUB_ | (gc.TFLOAT64<<8 | Frev):
		return x86.AFSUBRD

	case OSUB_ | (gc.TFLOAT64<<8 | Fpop):
		return x86.AFSUBDP

	case OSUB_ | (gc.TFLOAT64<<8 | (Fpop | Frev)):
		return x86.AFSUBRDP

	case OMUL_ | (gc.TFLOAT32<<8 | 0):
		return x86.AFMULF

	case OMUL_ | (gc.TFLOAT64<<8 | 0):
		return x86.AFMULD

	case OMUL_ | (gc.TFLOAT64<<8 | Fpop):
		return x86.AFMULDP

	case ODIV_ | (gc.TFLOAT32<<8 | 0):
		return x86.AFDIVF

	case ODIV_ | (gc.TFLOAT32<<8 | Frev):
		return x86.AFDIVRF

	case ODIV_ | (gc.TFLOAT64<<8 | 0):
		return x86.AFDIVD

	case ODIV_ | (gc.TFLOAT64<<8 | Frev):
		return x86.AFDIVRD

	case ODIV_ | (gc.TFLOAT64<<8 | Fpop):
		return x86.AFDIVDP

	case ODIV_ | (gc.TFLOAT64<<8 | (Fpop | Frev)):
		return x86.AFDIVRDP

	case OCMP_ | (gc.TFLOAT32<<8 | 0):
		return x86.AFCOMF

	case OCMP_ | (gc.TFLOAT32<<8 | Fpop):
		return x86.AFCOMFP

	case OCMP_ | (gc.TFLOAT64<<8 | 0):
		return x86.AFCOMD

	case OCMP_ | (gc.TFLOAT64<<8 | Fpop):
		return x86.AFCOMDP

	case OCMP_ | (gc.TFLOAT64<<8 | Fpop2):
		return x86.AFCOMDPP

	case OMINUS_ | (gc.TFLOAT32<<8 | 0):
		return x86.AFCHS

	case OMINUS_ | (gc.TFLOAT64<<8 | 0):
		return x86.AFCHS
	}

	gc.Fatalf("foptoas %v %v %#x", gc.Oconv(int(op), 0), t, flg)
	return 0
}

var resvd = []int{
	//	REG_DI,	// for movstring
	//	REG_SI,	// for movstring

	x86.REG_AX, // for divide
	x86.REG_CX, // for shift
	x86.REG_DX, // for divide, context
	x86.REG_SP, // for stack
}

/*
 * generate
 *	as $c, reg
 */
func gconreg(as int, c int64, reg int) {
	var n1 gc.Node
	var n2 gc.Node

	gc.Nodconst(&n1, gc.Types[gc.TINT64], c)
	gc.Nodreg(&n2, gc.Types[gc.TINT64], reg)
	gins(as, &n1, &n2)
}

/*
 * generate
 *	as $c, n
 */
func ginscon(as int, c int64, n2 *gc.Node) {
	var n1 gc.Node
	gc.Nodconst(&n1, gc.Types[gc.TINT32], c)
	gins(as, &n1, n2)
}

func ginscmp(op gc.Op, t *gc.Type, n1, n2 *gc.Node, likely int) *obj.Prog {
	if gc.Isint[t.Etype] || t.Etype == gc.Tptr {
		if (n1.Op == gc.OLITERAL || n1.Op == gc.OADDR && n1.Left.Op == gc.ONAME) && n2.Op != gc.OLITERAL {
			// Reverse comparison to place constant (including address constant) last.
			op = gc.Brrev(op)
			n1, n2 = n2, n1
		}
	}

	// General case.
	var r1, r2, g1, g2 gc.Node

	// A special case to make write barriers more efficient.
	// Comparing the first field of a named struct can be done directly.
	base := n1
	if n1.Op == gc.ODOT && n1.Left.Type.Etype == gc.TSTRUCT && n1.Left.Type.Type.Sym == n1.Right.Sym {
		base = n1.Left
	}

	if base.Op == gc.ONAME && base.Class&gc.PHEAP == 0 || n1.Op == gc.OINDREG {
		r1 = *n1
	} else {
		gc.Regalloc(&r1, t, n1)
		gc.Regalloc(&g1, n1.Type, &r1)
		gc.Cgen(n1, &g1)
		gmove(&g1, &r1)
	}
	if n2.Op == gc.OLITERAL && gc.Isint[t.Etype] || n2.Op == gc.OADDR && n2.Left.Op == gc.ONAME && n2.Left.Class == gc.PEXTERN {
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

/*
 * swap node contents
 */
func nswap(a *gc.Node, b *gc.Node) {
	t := *a
	*a = *b
	*b = t
}

/*
 * return constant i node.
 * overwritten by next call, but useful in calls to gins.
 */

var ncon_n gc.Node

func ncon(i uint32) *gc.Node {
	if ncon_n.Type == nil {
		gc.Nodconst(&ncon_n, gc.Types[gc.TUINT32], 0)
	}
	ncon_n.SetInt(int64(i))
	return &ncon_n
}

var sclean [10]gc.Node

var nsclean int

/*
 * n is a 64-bit value.  fill in lo and hi to refer to its 32-bit halves.
 */
func split64(n *gc.Node, lo *gc.Node, hi *gc.Node) {
	if !gc.Is64(n.Type) {
		gc.Fatalf("split64 %v", n.Type)
	}

	if nsclean >= len(sclean) {
		gc.Fatalf("split64 clean")
	}
	sclean[nsclean].Op = gc.OEMPTY
	nsclean++
	switch n.Op {
	default:
		switch n.Op {
		default:
			var n1 gc.Node
			if !dotaddable(n, &n1) {
				gc.Igen(n, &n1, nil)
				sclean[nsclean-1] = n1
			}

			n = &n1

		case gc.ONAME:
			if n.Class == gc.PPARAMREF {
				var n1 gc.Node
				gc.Cgen(n.Name.Heapaddr, &n1)
				sclean[nsclean-1] = n1
				n = &n1
			}

			// nothing
		case gc.OINDREG:
			break
		}

		*lo = *n
		*hi = *n
		lo.Type = gc.Types[gc.TUINT32]
		if n.Type.Etype == gc.TINT64 {
			hi.Type = gc.Types[gc.TINT32]
		} else {
			hi.Type = gc.Types[gc.TUINT32]
		}
		hi.Xoffset += 4

	case gc.OLITERAL:
		var n1 gc.Node
		n.Convconst(&n1, n.Type)
		i := n1.Int()
		gc.Nodconst(lo, gc.Types[gc.TUINT32], int64(uint32(i)))
		i >>= 32
		if n.Type.Etype == gc.TINT64 {
			gc.Nodconst(hi, gc.Types[gc.TINT32], int64(int32(i)))
		} else {
			gc.Nodconst(hi, gc.Types[gc.TUINT32], int64(uint32(i)))
		}
	}
}

func splitclean() {
	if nsclean <= 0 {
		gc.Fatalf("splitclean")
	}
	nsclean--
	if sclean[nsclean].Op != gc.OEMPTY {
		gc.Regfree(&sclean[nsclean])
	}
}

// set up nodes representing fp constants
var (
	zerof        gc.Node
	two63f       gc.Node
	two64f       gc.Node
	bignodes_did bool
)

func bignodes() {
	if bignodes_did {
		return
	}
	bignodes_did = true

	gc.Nodconst(&zerof, gc.Types[gc.TINT64], 0)
	zerof.Convconst(&zerof, gc.Types[gc.TFLOAT64])

	var i big.Int
	i.SetInt64(1)
	i.Lsh(&i, 63)
	var bigi gc.Node

	gc.Nodconst(&bigi, gc.Types[gc.TUINT64], 0)
	bigi.SetBigInt(&i)
	bigi.Convconst(&two63f, gc.Types[gc.TFLOAT64])

	gc.Nodconst(&bigi, gc.Types[gc.TUINT64], 0)
	i.Lsh(&i, 1)
	bigi.SetBigInt(&i)
	bigi.Convconst(&two64f, gc.Types[gc.TFLOAT64])
}

func memname(n *gc.Node, t *gc.Type) {
	gc.Tempname(n, t)
	n.Sym = gc.Lookup("." + n.Sym.Name[1:]) // keep optimizer from registerizing
	n.Orig.Sym = n.Sym
}

func gmove(f *gc.Node, t *gc.Node) {
	if gc.Debug['M'] != 0 {
		fmt.Printf("gmove %v -> %v\n", f, t)
	}

	ft := gc.Simsimtype(f.Type)
	tt := gc.Simsimtype(t.Type)
	cvt := t.Type

	if gc.Iscomplex[ft] || gc.Iscomplex[tt] {
		gc.Complexmove(f, t)
		return
	}

	if gc.Isfloat[ft] || gc.Isfloat[tt] {
		floatmove(f, t)
		return
	}

	// cannot have two integer memory operands;
	// except 64-bit, which always copies via registers anyway.
	var r1 gc.Node
	var a int
	if gc.Isint[ft] && gc.Isint[tt] && !gc.Is64(f.Type) && !gc.Is64(t.Type) && gc.Ismem(f) && gc.Ismem(t) {
		goto hard
	}

	// convert constant to desired type
	if f.Op == gc.OLITERAL {
		var con gc.Node
		f.Convconst(&con, t.Type)
		f = &con
		ft = gc.Simsimtype(con.Type)
	}

	// value -> value copy, only one memory operand.
	// figure out the instruction to use.
	// break out of switch for one-instruction gins.
	// goto rdst for "destination must be register".
	// goto hard for "convert to cvt type first".
	// otherwise handle and return.

	switch uint32(ft)<<16 | uint32(tt) {
	default:
		// should not happen
		gc.Fatalf("gmove %v -> %v", f, t)
		return

		/*
		 * integer copy and truncate
		 */
	case gc.TINT8<<16 | gc.TINT8, // same size
		gc.TINT8<<16 | gc.TUINT8,
		gc.TUINT8<<16 | gc.TINT8,
		gc.TUINT8<<16 | gc.TUINT8:
		a = x86.AMOVB

	case gc.TINT16<<16 | gc.TINT8, // truncate
		gc.TUINT16<<16 | gc.TINT8,
		gc.TINT32<<16 | gc.TINT8,
		gc.TUINT32<<16 | gc.TINT8,
		gc.TINT16<<16 | gc.TUINT8,
		gc.TUINT16<<16 | gc.TUINT8,
		gc.TINT32<<16 | gc.TUINT8,
		gc.TUINT32<<16 | gc.TUINT8:
		a = x86.AMOVB

		goto rsrc

	case gc.TINT64<<16 | gc.TINT8, // truncate low word
		gc.TUINT64<<16 | gc.TINT8,
		gc.TINT64<<16 | gc.TUINT8,
		gc.TUINT64<<16 | gc.TUINT8:
		var flo gc.Node
		var fhi gc.Node
		split64(f, &flo, &fhi)

		var r1 gc.Node
		gc.Nodreg(&r1, t.Type, x86.REG_AX)
		gmove(&flo, &r1)
		gins(x86.AMOVB, &r1, t)
		splitclean()
		return

	case gc.TINT16<<16 | gc.TINT16, // same size
		gc.TINT16<<16 | gc.TUINT16,
		gc.TUINT16<<16 | gc.TINT16,
		gc.TUINT16<<16 | gc.TUINT16:
		a = x86.AMOVW

	case gc.TINT32<<16 | gc.TINT16, // truncate
		gc.TUINT32<<16 | gc.TINT16,
		gc.TINT32<<16 | gc.TUINT16,
		gc.TUINT32<<16 | gc.TUINT16:
		a = x86.AMOVW

		goto rsrc

	case gc.TINT64<<16 | gc.TINT16, // truncate low word
		gc.TUINT64<<16 | gc.TINT16,
		gc.TINT64<<16 | gc.TUINT16,
		gc.TUINT64<<16 | gc.TUINT16:
		var flo gc.Node
		var fhi gc.Node
		split64(f, &flo, &fhi)

		var r1 gc.Node
		gc.Nodreg(&r1, t.Type, x86.REG_AX)
		gmove(&flo, &r1)
		gins(x86.AMOVW, &r1, t)
		splitclean()
		return

	case gc.TINT32<<16 | gc.TINT32, // same size
		gc.TINT32<<16 | gc.TUINT32,
		gc.TUINT32<<16 | gc.TINT32,
		gc.TUINT32<<16 | gc.TUINT32:
		a = x86.AMOVL

	case gc.TINT64<<16 | gc.TINT32, // truncate
		gc.TUINT64<<16 | gc.TINT32,
		gc.TINT64<<16 | gc.TUINT32,
		gc.TUINT64<<16 | gc.TUINT32:
		var fhi gc.Node
		var flo gc.Node
		split64(f, &flo, &fhi)

		var r1 gc.Node
		gc.Nodreg(&r1, t.Type, x86.REG_AX)
		gmove(&flo, &r1)
		gins(x86.AMOVL, &r1, t)
		splitclean()
		return

	case gc.TINT64<<16 | gc.TINT64, // same size
		gc.TINT64<<16 | gc.TUINT64,
		gc.TUINT64<<16 | gc.TINT64,
		gc.TUINT64<<16 | gc.TUINT64:
		var fhi gc.Node
		var flo gc.Node
		split64(f, &flo, &fhi)

		var tlo gc.Node
		var thi gc.Node
		split64(t, &tlo, &thi)
		if f.Op == gc.OLITERAL {
			gins(x86.AMOVL, &flo, &tlo)
			gins(x86.AMOVL, &fhi, &thi)
		} else {
			// Implementation of conversion-free x = y for int64 or uint64 x.
			// This is generated by the code that copies small values out of closures,
			// and that code has DX live, so avoid DX and just use AX twice.
			var r1 gc.Node
			gc.Nodreg(&r1, gc.Types[gc.TUINT32], x86.REG_AX)
			gins(x86.AMOVL, &flo, &r1)
			gins(x86.AMOVL, &r1, &tlo)
			gins(x86.AMOVL, &fhi, &r1)
			gins(x86.AMOVL, &r1, &thi)
		}

		splitclean()
		splitclean()
		return

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

	case gc.TINT8<<16 | gc.TINT64, // convert via int32
		gc.TINT8<<16 | gc.TUINT64:
		cvt = gc.Types[gc.TINT32]

		goto hard

	case gc.TUINT8<<16 | gc.TINT16, // zero extend uint8
		gc.TUINT8<<16 | gc.TUINT16:
		a = x86.AMOVBWZX

		goto rdst

	case gc.TUINT8<<16 | gc.TINT32,
		gc.TUINT8<<16 | gc.TUINT32:
		a = x86.AMOVBLZX
		goto rdst

	case gc.TUINT8<<16 | gc.TINT64, // convert via uint32
		gc.TUINT8<<16 | gc.TUINT64:
		cvt = gc.Types[gc.TUINT32]

		goto hard

	case gc.TINT16<<16 | gc.TINT32, // sign extend int16
		gc.TINT16<<16 | gc.TUINT32:
		a = x86.AMOVWLSX

		goto rdst

	case gc.TINT16<<16 | gc.TINT64, // convert via int32
		gc.TINT16<<16 | gc.TUINT64:
		cvt = gc.Types[gc.TINT32]

		goto hard

	case gc.TUINT16<<16 | gc.TINT32, // zero extend uint16
		gc.TUINT16<<16 | gc.TUINT32:
		a = x86.AMOVWLZX

		goto rdst

	case gc.TUINT16<<16 | gc.TINT64, // convert via uint32
		gc.TUINT16<<16 | gc.TUINT64:
		cvt = gc.Types[gc.TUINT32]

		goto hard

	case gc.TINT32<<16 | gc.TINT64, // sign extend int32
		gc.TINT32<<16 | gc.TUINT64:
		var thi gc.Node
		var tlo gc.Node
		split64(t, &tlo, &thi)

		var flo gc.Node
		gc.Nodreg(&flo, tlo.Type, x86.REG_AX)
		var fhi gc.Node
		gc.Nodreg(&fhi, thi.Type, x86.REG_DX)
		gmove(f, &flo)
		gins(x86.ACDQ, nil, nil)
		gins(x86.AMOVL, &flo, &tlo)
		gins(x86.AMOVL, &fhi, &thi)
		splitclean()
		return

	case gc.TUINT32<<16 | gc.TINT64, // zero extend uint32
		gc.TUINT32<<16 | gc.TUINT64:
		var tlo gc.Node
		var thi gc.Node
		split64(t, &tlo, &thi)

		gmove(f, &tlo)
		gins(x86.AMOVL, ncon(0), &thi)
		splitclean()
		return
	}

	gins(a, f, t)
	return

	// requires register source
rsrc:
	gc.Regalloc(&r1, f.Type, t)

	gmove(f, &r1)
	gins(a, &r1, t)
	gc.Regfree(&r1)
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

func floatmove(f *gc.Node, t *gc.Node) {
	var r1 gc.Node

	ft := gc.Simsimtype(f.Type)
	tt := gc.Simsimtype(t.Type)
	cvt := t.Type

	// cannot have two floating point memory operands.
	if gc.Isfloat[ft] && gc.Isfloat[tt] && gc.Ismem(f) && gc.Ismem(t) {
		goto hard
	}

	// convert constant to desired type
	if f.Op == gc.OLITERAL {
		var con gc.Node
		f.Convconst(&con, t.Type)
		f = &con
		ft = gc.Simsimtype(con.Type)

		// some constants can't move directly to memory.
		if gc.Ismem(t) {
			// float constants come from memory.
			if gc.Isfloat[tt] {
				goto hard
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
		if gc.Thearch.Use387 {
			floatmove_387(f, t)
		} else {
			floatmove_sse(f, t)
		}
		return

		// float to very long integer.
	case gc.TFLOAT32<<16 | gc.TINT64,
		gc.TFLOAT64<<16 | gc.TINT64:
		if f.Op == gc.OREGISTER {
			cvt = f.Type
			goto hardmem
		}

		var r1 gc.Node
		gc.Nodreg(&r1, gc.Types[ft], x86.REG_F0)
		if ft == gc.TFLOAT32 {
			gins(x86.AFMOVF, f, &r1)
		} else {
			gins(x86.AFMOVD, f, &r1)
		}

		// set round to zero mode during conversion
		var t1 gc.Node
		memname(&t1, gc.Types[gc.TUINT16])

		var t2 gc.Node
		memname(&t2, gc.Types[gc.TUINT16])
		gins(x86.AFSTCW, nil, &t1)
		gins(x86.AMOVW, ncon(0xf7f), &t2)
		gins(x86.AFLDCW, &t2, nil)
		if tt == gc.TINT16 {
			gins(x86.AFMOVWP, &r1, t)
		} else if tt == gc.TINT32 {
			gins(x86.AFMOVLP, &r1, t)
		} else {
			gins(x86.AFMOVVP, &r1, t)
		}
		gins(x86.AFLDCW, &t1, nil)
		return

	case gc.TFLOAT32<<16 | gc.TUINT64,
		gc.TFLOAT64<<16 | gc.TUINT64:
		if !gc.Ismem(f) {
			cvt = f.Type
			goto hardmem
		}

		bignodes()
		var f0 gc.Node
		gc.Nodreg(&f0, gc.Types[ft], x86.REG_F0)
		var f1 gc.Node
		gc.Nodreg(&f1, gc.Types[ft], x86.REG_F0+1)
		var ax gc.Node
		gc.Nodreg(&ax, gc.Types[gc.TUINT16], x86.REG_AX)

		if ft == gc.TFLOAT32 {
			gins(x86.AFMOVF, f, &f0)
		} else {
			gins(x86.AFMOVD, f, &f0)
		}

		// if 0 > v { answer = 0 }
		gins(x86.AFMOVD, &zerof, &f0)
		gins(x86.AFUCOMP, &f0, &f1)
		gins(x86.AFSTSW, nil, &ax)
		gins(x86.ASAHF, nil, nil)
		p1 := gc.Gbranch(optoas(gc.OGT, gc.Types[tt]), nil, 0)

		// if 1<<64 <= v { answer = 0 too }
		gins(x86.AFMOVD, &two64f, &f0)

		gins(x86.AFUCOMP, &f0, &f1)
		gins(x86.AFSTSW, nil, &ax)
		gins(x86.ASAHF, nil, nil)
		p2 := gc.Gbranch(optoas(gc.OGT, gc.Types[tt]), nil, 0)
		gc.Patch(p1, gc.Pc)
		gins(x86.AFMOVVP, &f0, t) // don't care about t, but will pop the stack
		var thi gc.Node
		var tlo gc.Node
		split64(t, &tlo, &thi)
		gins(x86.AMOVL, ncon(0), &tlo)
		gins(x86.AMOVL, ncon(0), &thi)
		splitclean()
		p1 = gc.Gbranch(obj.AJMP, nil, 0)
		gc.Patch(p2, gc.Pc)

		// in range; algorithm is:
		//	if small enough, use native float64 -> int64 conversion.
		//	otherwise, subtract 2^63, convert, and add it back.

		// set round to zero mode during conversion
		var t1 gc.Node
		memname(&t1, gc.Types[gc.TUINT16])

		var t2 gc.Node
		memname(&t2, gc.Types[gc.TUINT16])
		gins(x86.AFSTCW, nil, &t1)
		gins(x86.AMOVW, ncon(0xf7f), &t2)
		gins(x86.AFLDCW, &t2, nil)

		// actual work
		gins(x86.AFMOVD, &two63f, &f0)

		gins(x86.AFUCOMP, &f0, &f1)
		gins(x86.AFSTSW, nil, &ax)
		gins(x86.ASAHF, nil, nil)
		p2 = gc.Gbranch(optoas(gc.OLE, gc.Types[tt]), nil, 0)
		gins(x86.AFMOVVP, &f0, t)
		p3 := gc.Gbranch(obj.AJMP, nil, 0)
		gc.Patch(p2, gc.Pc)
		gins(x86.AFMOVD, &two63f, &f0)
		gins(x86.AFSUBDP, &f0, &f1)
		gins(x86.AFMOVVP, &f0, t)
		split64(t, &tlo, &thi)
		gins(x86.AXORL, ncon(0x80000000), &thi) // + 2^63
		gc.Patch(p3, gc.Pc)
		splitclean()

		// restore rounding mode
		gins(x86.AFLDCW, &t1, nil)

		gc.Patch(p1, gc.Pc)
		return

		/*
		 * integer to float
		 */
	case gc.TINT64<<16 | gc.TFLOAT32,
		gc.TINT64<<16 | gc.TFLOAT64:
		if t.Op == gc.OREGISTER {
			goto hardmem
		}
		var f0 gc.Node
		gc.Nodreg(&f0, t.Type, x86.REG_F0)
		gins(x86.AFMOVV, f, &f0)
		if tt == gc.TFLOAT32 {
			gins(x86.AFMOVFP, &f0, t)
		} else {
			gins(x86.AFMOVDP, &f0, t)
		}
		return

		// algorithm is:
	//	if small enough, use native int64 -> float64 conversion.
	//	otherwise, halve (rounding to odd?), convert, and double.
	case gc.TUINT64<<16 | gc.TFLOAT32,
		gc.TUINT64<<16 | gc.TFLOAT64:
		var ax gc.Node
		gc.Nodreg(&ax, gc.Types[gc.TUINT32], x86.REG_AX)

		var dx gc.Node
		gc.Nodreg(&dx, gc.Types[gc.TUINT32], x86.REG_DX)
		var cx gc.Node
		gc.Nodreg(&cx, gc.Types[gc.TUINT32], x86.REG_CX)
		var t1 gc.Node
		gc.Tempname(&t1, f.Type)
		var tlo gc.Node
		var thi gc.Node
		split64(&t1, &tlo, &thi)
		gmove(f, &t1)
		gins(x86.ACMPL, &thi, ncon(0))
		p1 := gc.Gbranch(x86.AJLT, nil, 0)

		// native
		var r1 gc.Node
		gc.Nodreg(&r1, gc.Types[tt], x86.REG_F0)

		gins(x86.AFMOVV, &t1, &r1)
		if tt == gc.TFLOAT32 {
			gins(x86.AFMOVFP, &r1, t)
		} else {
			gins(x86.AFMOVDP, &r1, t)
		}
		p2 := gc.Gbranch(obj.AJMP, nil, 0)

		// simulated
		gc.Patch(p1, gc.Pc)

		gmove(&tlo, &ax)
		gmove(&thi, &dx)
		p1 = gins(x86.ASHRL, ncon(1), &ax)
		p1.From.Index = x86.REG_DX // double-width shift DX -> AX
		p1.From.Scale = 0
		gins(x86.AMOVL, ncon(0), &cx)
		gins(x86.ASETCC, nil, &cx)
		gins(x86.AORL, &cx, &ax)
		gins(x86.ASHRL, ncon(1), &dx)
		gmove(&dx, &thi)
		gmove(&ax, &tlo)
		gc.Nodreg(&r1, gc.Types[tt], x86.REG_F0)
		var r2 gc.Node
		gc.Nodreg(&r2, gc.Types[tt], x86.REG_F0+1)
		gins(x86.AFMOVV, &t1, &r1)
		gins(x86.AFMOVD, &r1, &r1)
		gins(x86.AFADDDP, &r1, &r2)
		if tt == gc.TFLOAT32 {
			gins(x86.AFMOVFP, &r1, t)
		} else {
			gins(x86.AFMOVDP, &r1, t)
		}
		gc.Patch(p2, gc.Pc)
		splitclean()
		return
	}

	// requires register intermediate
hard:
	gc.Regalloc(&r1, cvt, t)

	gmove(f, &r1)
	gmove(&r1, t)
	gc.Regfree(&r1)
	return

	// requires memory intermediate
hardmem:
	gc.Tempname(&r1, cvt)

	gmove(f, &r1)
	gmove(&r1, t)
	return
}

func floatmove_387(f *gc.Node, t *gc.Node) {
	var r1 gc.Node
	var a int

	ft := gc.Simsimtype(f.Type)
	tt := gc.Simsimtype(t.Type)
	cvt := t.Type

	switch uint32(ft)<<16 | uint32(tt) {
	default:
		goto fatal

		/*
		* float to integer
		 */
	case gc.TFLOAT32<<16 | gc.TINT16,
		gc.TFLOAT32<<16 | gc.TINT32,
		gc.TFLOAT32<<16 | gc.TINT64,
		gc.TFLOAT64<<16 | gc.TINT16,
		gc.TFLOAT64<<16 | gc.TINT32,
		gc.TFLOAT64<<16 | gc.TINT64:
		if t.Op == gc.OREGISTER {
			goto hardmem
		}
		var r1 gc.Node
		gc.Nodreg(&r1, gc.Types[ft], x86.REG_F0)
		if f.Op != gc.OREGISTER {
			if ft == gc.TFLOAT32 {
				gins(x86.AFMOVF, f, &r1)
			} else {
				gins(x86.AFMOVD, f, &r1)
			}
		}

		// set round to zero mode during conversion
		var t1 gc.Node
		memname(&t1, gc.Types[gc.TUINT16])

		var t2 gc.Node
		memname(&t2, gc.Types[gc.TUINT16])
		gins(x86.AFSTCW, nil, &t1)
		gins(x86.AMOVW, ncon(0xf7f), &t2)
		gins(x86.AFLDCW, &t2, nil)
		if tt == gc.TINT16 {
			gins(x86.AFMOVWP, &r1, t)
		} else if tt == gc.TINT32 {
			gins(x86.AFMOVLP, &r1, t)
		} else {
			gins(x86.AFMOVVP, &r1, t)
		}
		gins(x86.AFLDCW, &t1, nil)
		return

		// convert via int32.
	case gc.TFLOAT32<<16 | gc.TINT8,
		gc.TFLOAT32<<16 | gc.TUINT16,
		gc.TFLOAT32<<16 | gc.TUINT8,
		gc.TFLOAT64<<16 | gc.TINT8,
		gc.TFLOAT64<<16 | gc.TUINT16,
		gc.TFLOAT64<<16 | gc.TUINT8:
		var t1 gc.Node
		gc.Tempname(&t1, gc.Types[gc.TINT32])

		gmove(f, &t1)
		switch tt {
		default:
			gc.Fatalf("gmove %v", t)

		case gc.TINT8:
			gins(x86.ACMPL, &t1, ncon(-0x80&(1<<32-1)))
			p1 := gc.Gbranch(optoas(gc.OLT, gc.Types[gc.TINT32]), nil, -1)
			gins(x86.ACMPL, &t1, ncon(0x7f))
			p2 := gc.Gbranch(optoas(gc.OGT, gc.Types[gc.TINT32]), nil, -1)
			p3 := gc.Gbranch(obj.AJMP, nil, 0)
			gc.Patch(p1, gc.Pc)
			gc.Patch(p2, gc.Pc)
			gmove(ncon(-0x80&(1<<32-1)), &t1)
			gc.Patch(p3, gc.Pc)
			gmove(&t1, t)

		case gc.TUINT8:
			gins(x86.ATESTL, ncon(0xffffff00), &t1)
			p1 := gc.Gbranch(x86.AJEQ, nil, +1)
			gins(x86.AMOVL, ncon(0), &t1)
			gc.Patch(p1, gc.Pc)
			gmove(&t1, t)

		case gc.TUINT16:
			gins(x86.ATESTL, ncon(0xffff0000), &t1)
			p1 := gc.Gbranch(x86.AJEQ, nil, +1)
			gins(x86.AMOVL, ncon(0), &t1)
			gc.Patch(p1, gc.Pc)
			gmove(&t1, t)
		}

		return

		// convert via int64.
	case gc.TFLOAT32<<16 | gc.TUINT32,
		gc.TFLOAT64<<16 | gc.TUINT32:
		cvt = gc.Types[gc.TINT64]

		goto hardmem

		/*
		 * integer to float
		 */
	case gc.TINT16<<16 | gc.TFLOAT32,
		gc.TINT16<<16 | gc.TFLOAT64,
		gc.TINT32<<16 | gc.TFLOAT32,
		gc.TINT32<<16 | gc.TFLOAT64,
		gc.TINT64<<16 | gc.TFLOAT32,
		gc.TINT64<<16 | gc.TFLOAT64:
		if t.Op != gc.OREGISTER {
			goto hard
		}
		if f.Op == gc.OREGISTER {
			cvt = f.Type
			goto hardmem
		}

		switch ft {
		case gc.TINT16:
			a = x86.AFMOVW

		case gc.TINT32:
			a = x86.AFMOVL

		default:
			a = x86.AFMOVV
		}

		// convert via int32 memory
	case gc.TINT8<<16 | gc.TFLOAT32,
		gc.TINT8<<16 | gc.TFLOAT64,
		gc.TUINT16<<16 | gc.TFLOAT32,
		gc.TUINT16<<16 | gc.TFLOAT64,
		gc.TUINT8<<16 | gc.TFLOAT32,
		gc.TUINT8<<16 | gc.TFLOAT64:
		cvt = gc.Types[gc.TINT32]

		goto hardmem

		// convert via int64 memory
	case gc.TUINT32<<16 | gc.TFLOAT32,
		gc.TUINT32<<16 | gc.TFLOAT64:
		cvt = gc.Types[gc.TINT64]

		goto hardmem

		// The way the code generator uses floating-point
	// registers, a move from F0 to F0 is intended as a no-op.
	// On the x86, it's not: it pushes a second copy of F0
	// on the floating point stack.  So toss it away here.
	// Also, F0 is the *only* register we ever evaluate
	// into, so we should only see register/register as F0/F0.
	/*
	 * float to float
	 */
	case gc.TFLOAT32<<16 | gc.TFLOAT32,
		gc.TFLOAT64<<16 | gc.TFLOAT64:
		if gc.Ismem(f) && gc.Ismem(t) {
			goto hard
		}
		if f.Op == gc.OREGISTER && t.Op == gc.OREGISTER {
			if f.Reg != x86.REG_F0 || t.Reg != x86.REG_F0 {
				goto fatal
			}
			return
		}

		a = x86.AFMOVF
		if ft == gc.TFLOAT64 {
			a = x86.AFMOVD
		}
		if gc.Ismem(t) {
			if f.Op != gc.OREGISTER || f.Reg != x86.REG_F0 {
				gc.Fatalf("gmove %v", f)
			}
			a = x86.AFMOVFP
			if ft == gc.TFLOAT64 {
				a = x86.AFMOVDP
			}
		}

	case gc.TFLOAT32<<16 | gc.TFLOAT64:
		if gc.Ismem(f) && gc.Ismem(t) {
			goto hard
		}
		if f.Op == gc.OREGISTER && t.Op == gc.OREGISTER {
			if f.Reg != x86.REG_F0 || t.Reg != x86.REG_F0 {
				goto fatal
			}
			return
		}

		if f.Op == gc.OREGISTER {
			gins(x86.AFMOVDP, f, t)
		} else {
			gins(x86.AFMOVF, f, t)
		}
		return

	case gc.TFLOAT64<<16 | gc.TFLOAT32:
		if gc.Ismem(f) && gc.Ismem(t) {
			goto hard
		}
		if f.Op == gc.OREGISTER && t.Op == gc.OREGISTER {
			var r1 gc.Node
			gc.Tempname(&r1, gc.Types[gc.TFLOAT32])
			gins(x86.AFMOVFP, f, &r1)
			gins(x86.AFMOVF, &r1, t)
			return
		}

		if f.Op == gc.OREGISTER {
			gins(x86.AFMOVFP, f, t)
		} else {
			gins(x86.AFMOVD, f, t)
		}
		return
	}

	gins(a, f, t)
	return

	// requires register intermediate
hard:
	gc.Regalloc(&r1, cvt, t)

	gmove(f, &r1)
	gmove(&r1, t)
	gc.Regfree(&r1)
	return

	// requires memory intermediate
hardmem:
	gc.Tempname(&r1, cvt)

	gmove(f, &r1)
	gmove(&r1, t)
	return

	// should not happen
fatal:
	gc.Fatalf("gmove %v -> %v", gc.Nconv(f, obj.FmtLong), gc.Nconv(t, obj.FmtLong))

	return
}

func floatmove_sse(f *gc.Node, t *gc.Node) {
	var r1 gc.Node
	var cvt *gc.Type
	var a int

	ft := gc.Simsimtype(f.Type)
	tt := gc.Simsimtype(t.Type)

	switch uint32(ft)<<16 | uint32(tt) {
	// should not happen
	default:
		gc.Fatalf("gmove %v -> %v", f, t)

		return

		// convert via int32.
	/*
	* float to integer
	 */
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

		goto hardmem

	case gc.TFLOAT32<<16 | gc.TINT32:
		a = x86.ACVTTSS2SL
		goto rdst

	case gc.TFLOAT64<<16 | gc.TINT32:
		a = x86.ACVTTSD2SL
		goto rdst

		// convert via int32 memory
	/*
	 * integer to float
	 */
	case gc.TINT8<<16 | gc.TFLOAT32,
		gc.TINT8<<16 | gc.TFLOAT64,
		gc.TINT16<<16 | gc.TFLOAT32,
		gc.TINT16<<16 | gc.TFLOAT64,
		gc.TUINT16<<16 | gc.TFLOAT32,
		gc.TUINT16<<16 | gc.TFLOAT64,
		gc.TUINT8<<16 | gc.TFLOAT32,
		gc.TUINT8<<16 | gc.TFLOAT64:
		cvt = gc.Types[gc.TINT32]

		goto hard

		// convert via int64 memory
	case gc.TUINT32<<16 | gc.TFLOAT32,
		gc.TUINT32<<16 | gc.TFLOAT64:
		cvt = gc.Types[gc.TINT64]

		goto hardmem

	case gc.TINT32<<16 | gc.TFLOAT32:
		a = x86.ACVTSL2SS
		goto rdst

	case gc.TINT32<<16 | gc.TFLOAT64:
		a = x86.ACVTSL2SD
		goto rdst

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

	// requires register intermediate
hard:
	gc.Regalloc(&r1, cvt, t)

	gmove(f, &r1)
	gmove(&r1, t)
	gc.Regfree(&r1)
	return

	// requires memory intermediate
hardmem:
	gc.Tempname(&r1, cvt)

	gmove(f, &r1)
	gmove(&r1, t)
	return

	// requires register destination
rdst:
	gc.Regalloc(&r1, t.Type, t)

	gins(a, f, &r1)
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
	if as == x86.AFMOVF && f != nil && f.Op == gc.OREGISTER && t != nil && t.Op == gc.OREGISTER {
		gc.Fatalf("gins MOVF reg, reg")
	}
	if as == x86.ACVTSD2SS && f != nil && f.Op == gc.OLITERAL {
		gc.Fatalf("gins CVTSD2SS const")
	}
	if as == x86.AMOVSD && t != nil && t.Op == gc.OREGISTER && t.Reg == x86.REG_F0 {
		gc.Fatalf("gins MOVSD into F0")
	}

	if as == x86.AMOVL && f != nil && f.Op == gc.OADDR && f.Left.Op == gc.ONAME && f.Left.Class != gc.PEXTERN && f.Left.Class != gc.PFUNC {
		// Turn MOVL $xxx(FP/SP) into LEAL xxx.
		// These should be equivalent but most of the backend
		// only expects to see LEAL, because that's what we had
		// historically generated. Various hidden assumptions are baked in by now.
		as = x86.ALEAL
		f = f.Left
	}

	switch as {
	case x86.AMOVB,
		x86.AMOVW,
		x86.AMOVL:
		if f != nil && t != nil && samaddr(f, t) {
			return nil
		}

	case x86.ALEAL:
		if f != nil && gc.Isconst(f, gc.CTNIL) {
			gc.Fatalf("gins LEAL nil %v", f.Type)
		}
	}

	p := gc.Prog(as)
	gc.Naddr(&p.From, f)
	gc.Naddr(&p.To, t)

	if gc.Debug['g'] != 0 {
		fmt.Printf("%v\n", p)
	}

	w := 0
	switch as {
	case x86.AMOVB:
		w = 1

	case x86.AMOVW:
		w = 2

	case x86.AMOVL:
		w = 4
	}

	if true && w != 0 && f != nil && (p.From.Width > int64(w) || p.To.Width > int64(w)) {
		gc.Dump("bad width from:", f)
		gc.Dump("bad width to:", t)
		gc.Fatalf("bad width: %v (%d, %d)\n", p, p.From.Width, p.To.Width)
	}

	if p.To.Type == obj.TYPE_ADDR && w > 0 {
		gc.Fatalf("bad use of addr: %v", p)
	}

	return p
}

func ginsnop() {
	var reg gc.Node
	gc.Nodreg(&reg, gc.Types[gc.TINT], x86.REG_AX)
	gins(x86.AXCHGL, &reg, &reg)
}

func dotaddable(n *gc.Node, n1 *gc.Node) bool {
	if n.Op != gc.ODOT {
		return false
	}

	var oary [10]int64
	var nn *gc.Node
	o := gc.Dotoffset(n, oary[:], &nn)
	if nn != nil && nn.Addable && o == 1 && oary[0] >= 0 {
		*n1 = *nn
		n1.Type = n.Type
		n1.Xoffset += oary[0]
		return true
	}

	return false
}

func sudoclean() {
}

func sudoaddable(as int, n *gc.Node, a *obj.Addr) bool {
	*a = obj.Addr{}
	return false
}
