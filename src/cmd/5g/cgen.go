// Copyright 2009 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package main

import (
	"cmd/internal/obj"
	"cmd/internal/obj/arm"
	"fmt"
)
import "cmd/internal/gc"

/*
 * peep.c
 */
/*
 * generate:
 *	res = n;
 * simplifies and calls gmove.
 */
func cgen(n *gc.Node, res *gc.Node) {
	if gc.Debug['g'] != 0 {
		gc.Dump("\ncgen-n", n)
		gc.Dump("cgen-res", res)
	}

	var n1 gc.Node
	var nr *gc.Node
	var nl *gc.Node
	var a int
	var f1 gc.Node
	var f0 gc.Node
	var n2 gc.Node
	if n == nil || n.Type == nil {
		goto ret
	}

	if res == nil || res.Type == nil {
		gc.Fatal("cgen: res nil")
	}

	switch n.Op {
	case gc.OSLICE,
		gc.OSLICEARR,
		gc.OSLICESTR,
		gc.OSLICE3,
		gc.OSLICE3ARR:
		if res.Op != gc.ONAME || res.Addable == 0 {
			var n1 gc.Node
			gc.Tempname(&n1, n.Type)
			gc.Cgen_slice(n, &n1)
			cgen(&n1, res)
		} else {
			gc.Cgen_slice(n, res)
		}
		return

	case gc.OEFACE:
		if res.Op != gc.ONAME || res.Addable == 0 {
			var n1 gc.Node
			gc.Tempname(&n1, n.Type)
			gc.Cgen_eface(n, &n1)
			cgen(&n1, res)
		} else {
			gc.Cgen_eface(n, res)
		}
		return
	}

	for n.Op == gc.OCONVNOP {
		n = n.Left
	}

	if n.Ullman >= gc.UINF {
		if n.Op == gc.OINDREG {
			gc.Fatal("cgen: this is going to misscompile")
		}
		if res.Ullman >= gc.UINF {
			var n1 gc.Node
			gc.Tempname(&n1, n.Type)
			cgen(n, &n1)
			cgen(&n1, res)
			goto ret
		}
	}

	if gc.Isfat(n.Type) {
		if n.Type.Width < 0 {
			gc.Fatal("forgot to compute width for %v", gc.Tconv(n.Type, 0))
		}
		sgen(n, res, n.Type.Width)
		goto ret
	}

	// update addressability for string, slice
	// can't do in walk because n->left->addable
	// changes if n->left is an escaping local variable.
	switch n.Op {
	case gc.OSPTR,
		gc.OLEN:
		if gc.Isslice(n.Left.Type) || gc.Istype(n.Left.Type, gc.TSTRING) {
			n.Addable = n.Left.Addable
		}

	case gc.OCAP:
		if gc.Isslice(n.Left.Type) {
			n.Addable = n.Left.Addable
		}

	case gc.OITAB:
		n.Addable = n.Left.Addable
	}

	// if both are addressable, move
	if n.Addable != 0 && res.Addable != 0 {
		if gc.Is64(n.Type) || gc.Is64(res.Type) || n.Op == gc.OREGISTER || res.Op == gc.OREGISTER || gc.Iscomplex[n.Type.Etype] != 0 || gc.Iscomplex[res.Type.Etype] != 0 {
			gmove(n, res)
		} else {
			var n1 gc.Node
			regalloc(&n1, n.Type, nil)
			gmove(n, &n1)
			cgen(&n1, res)
			regfree(&n1)
		}

		goto ret
	}

	// if both are not addressable, use a temporary.
	if n.Addable == 0 && res.Addable == 0 {
		// could use regalloc here sometimes,
		// but have to check for ullman >= UINF.
		var n1 gc.Node
		gc.Tempname(&n1, n.Type)

		cgen(n, &n1)
		cgen(&n1, res)
		return
	}

	// if result is not addressable directly but n is,
	// compute its address and then store via the address.
	if res.Addable == 0 {
		var n1 gc.Node
		igen(res, &n1, nil)
		cgen(n, &n1)
		regfree(&n1)
		return
	}

	if gc.Complexop(n, res) {
		gc.Complexgen(n, res)
		return
	}

	// if n is sudoaddable generate addr and move
	if !gc.Is64(n.Type) && !gc.Is64(res.Type) && gc.Iscomplex[n.Type.Etype] == 0 && gc.Iscomplex[res.Type.Etype] == 0 {
		a := optoas(gc.OAS, n.Type)
		var w int
		var addr obj.Addr
		if sudoaddable(a, n, &addr, &w) {
			if res.Op != gc.OREGISTER {
				var n2 gc.Node
				regalloc(&n2, res.Type, nil)
				p1 := gins(a, nil, &n2)
				p1.From = addr
				if gc.Debug['g'] != 0 {
					fmt.Printf("%v [ignore previous line]\n", p1)
				}
				gmove(&n2, res)
				regfree(&n2)
			} else {
				p1 := gins(a, nil, res)
				p1.From = addr
				if gc.Debug['g'] != 0 {
					fmt.Printf("%v [ignore previous line]\n", p1)
				}
			}

			sudoclean()
			goto ret
		}
	}

	// otherwise, the result is addressable but n is not.
	// let's do some computation.

	nl = n.Left

	nr = n.Right

	if nl != nil && nl.Ullman >= gc.UINF {
		if nr != nil && nr.Ullman >= gc.UINF {
			var n1 gc.Node
			gc.Tempname(&n1, nl.Type)
			cgen(nl, &n1)
			n2 := *n
			n2.Left = &n1
			cgen(&n2, res)
			goto ret
		}
	}

	// 64-bit ops are hard on 32-bit machine.
	if gc.Is64(n.Type) || gc.Is64(res.Type) || n.Left != nil && gc.Is64(n.Left.Type) {
		switch n.Op {
		// math goes to cgen64.
		case gc.OMINUS,
			gc.OCOM,
			gc.OADD,
			gc.OSUB,
			gc.OMUL,
			gc.OLROT,
			gc.OLSH,
			gc.ORSH,
			gc.OAND,
			gc.OOR,
			gc.OXOR:
			cgen64(n, res)

			return
		}
	}

	if nl != nil && gc.Isfloat[n.Type.Etype] != 0 && gc.Isfloat[nl.Type.Etype] != 0 {
		goto flt
	}
	switch n.Op {
	default:
		gc.Dump("cgen", n)
		gc.Fatal("cgen: unknown op %v", gc.Nconv(n, obj.FmtShort|obj.FmtSign))

	case gc.OREAL,
		gc.OIMAG,
		gc.OCOMPLEX:
		gc.Fatal("unexpected complex")

		// these call bgen to get a bool value
	case gc.OOROR,
		gc.OANDAND,
		gc.OEQ,
		gc.ONE,
		gc.OLT,
		gc.OLE,
		gc.OGE,
		gc.OGT,
		gc.ONOT:
		p1 := gc.Gbranch(arm.AB, nil, 0)

		p2 := gc.Pc
		gmove(gc.Nodbool(true), res)
		p3 := gc.Gbranch(arm.AB, nil, 0)
		gc.Patch(p1, gc.Pc)
		bgen(n, true, 0, p2)
		gmove(gc.Nodbool(false), res)
		gc.Patch(p3, gc.Pc)
		goto ret

	case gc.OPLUS:
		cgen(nl, res)
		goto ret

		// unary
	case gc.OCOM:
		a := optoas(gc.OXOR, nl.Type)

		regalloc(&n1, nl.Type, nil)
		cgen(nl, &n1)
		gc.Nodconst(&n2, nl.Type, -1)
		gins(a, &n2, &n1)
		goto norm

	case gc.OMINUS:
		regalloc(&n1, nl.Type, nil)
		cgen(nl, &n1)
		gc.Nodconst(&n2, nl.Type, 0)
		gins(optoas(gc.OMINUS, nl.Type), &n2, &n1)
		goto norm

		// symmetric binary
	case gc.OAND,
		gc.OOR,
		gc.OXOR,
		gc.OADD,
		gc.OMUL:
		a = optoas(int(n.Op), nl.Type)

		goto sbop

		// asymmetric binary
	case gc.OSUB:
		a = optoas(int(n.Op), nl.Type)

		goto abop

	case gc.OHMUL:
		cgen_hmul(nl, nr, res)

	case gc.OLROT,
		gc.OLSH,
		gc.ORSH:
		cgen_shift(int(n.Op), n.Bounded, nl, nr, res)

	case gc.OCONV:
		if gc.Eqtype(n.Type, nl.Type) || gc.Noconv(n.Type, nl.Type) {
			cgen(nl, res)
			break
		}

		var n1 gc.Node
		if nl.Addable != 0 && !gc.Is64(nl.Type) {
			regalloc(&n1, nl.Type, res)
			gmove(nl, &n1)
		} else {
			if n.Type.Width > int64(gc.Widthptr) || gc.Is64(nl.Type) || gc.Isfloat[nl.Type.Etype] != 0 {
				gc.Tempname(&n1, nl.Type)
			} else {
				regalloc(&n1, nl.Type, res)
			}
			cgen(nl, &n1)
		}

		var n2 gc.Node
		if n.Type.Width > int64(gc.Widthptr) || gc.Is64(n.Type) || gc.Isfloat[n.Type.Etype] != 0 {
			gc.Tempname(&n2, n.Type)
		} else {
			regalloc(&n2, n.Type, nil)
		}
		gmove(&n1, &n2)
		gmove(&n2, res)
		if n1.Op == gc.OREGISTER {
			regfree(&n1)
		}
		if n2.Op == gc.OREGISTER {
			regfree(&n2)
		}

	case gc.ODOT,
		gc.ODOTPTR,
		gc.OINDEX,
		gc.OIND,
		gc.ONAME: // PHEAP or PPARAMREF var
		var n1 gc.Node
		igen(n, &n1, res)

		gmove(&n1, res)
		regfree(&n1)

		// interface table is first word of interface value
	case gc.OITAB:
		var n1 gc.Node
		igen(nl, &n1, res)

		n1.Type = n.Type
		gmove(&n1, res)
		regfree(&n1)

		// pointer is the first word of string or slice.
	case gc.OSPTR:
		if gc.Isconst(nl, gc.CTSTR) {
			var n1 gc.Node
			regalloc(&n1, gc.Types[gc.Tptr], res)
			p1 := gins(arm.AMOVW, nil, &n1)
			gc.Datastring(nl.Val.U.Sval.S, &p1.From)
			gmove(&n1, res)
			regfree(&n1)
			break
		}

		var n1 gc.Node
		igen(nl, &n1, res)
		n1.Type = n.Type
		gmove(&n1, res)
		regfree(&n1)

	case gc.OLEN:
		if gc.Istype(nl.Type, gc.TMAP) || gc.Istype(nl.Type, gc.TCHAN) {
			// map has len in the first 32-bit word.
			// a zero pointer means zero length
			var n1 gc.Node
			regalloc(&n1, gc.Types[gc.Tptr], res)

			cgen(nl, &n1)

			var n2 gc.Node
			gc.Nodconst(&n2, gc.Types[gc.Tptr], 0)
			gcmp(optoas(gc.OCMP, gc.Types[gc.Tptr]), &n1, &n2)
			p1 := gc.Gbranch(optoas(gc.OEQ, gc.Types[gc.Tptr]), nil, -1)

			n2 = n1
			n2.Op = gc.OINDREG
			n2.Type = gc.Types[gc.TINT32]
			gmove(&n2, &n1)

			gc.Patch(p1, gc.Pc)

			gmove(&n1, res)
			regfree(&n1)
			break
		}

		if gc.Istype(nl.Type, gc.TSTRING) || gc.Isslice(nl.Type) {
			// both slice and string have len one pointer into the struct.
			var n1 gc.Node
			igen(nl, &n1, res)

			n1.Type = gc.Types[gc.TUINT32]
			n1.Xoffset += int64(gc.Array_nel)
			gmove(&n1, res)
			regfree(&n1)
			break
		}

		gc.Fatal("cgen: OLEN: unknown type %v", gc.Tconv(nl.Type, obj.FmtLong))

	case gc.OCAP:
		if gc.Istype(nl.Type, gc.TCHAN) {
			// chan has cap in the second 32-bit word.
			// a zero pointer means zero length
			var n1 gc.Node
			regalloc(&n1, gc.Types[gc.Tptr], res)

			cgen(nl, &n1)

			var n2 gc.Node
			gc.Nodconst(&n2, gc.Types[gc.Tptr], 0)
			gcmp(optoas(gc.OCMP, gc.Types[gc.Tptr]), &n1, &n2)
			p1 := gc.Gbranch(optoas(gc.OEQ, gc.Types[gc.Tptr]), nil, -1)

			n2 = n1
			n2.Op = gc.OINDREG
			n2.Xoffset = 4
			n2.Type = gc.Types[gc.TINT32]
			gmove(&n2, &n1)

			gc.Patch(p1, gc.Pc)

			gmove(&n1, res)
			regfree(&n1)
			break
		}

		if gc.Isslice(nl.Type) {
			var n1 gc.Node
			igen(nl, &n1, res)
			n1.Type = gc.Types[gc.TUINT32]
			n1.Xoffset += int64(gc.Array_cap)
			gmove(&n1, res)
			regfree(&n1)
			break
		}

		gc.Fatal("cgen: OCAP: unknown type %v", gc.Tconv(nl.Type, obj.FmtLong))

	case gc.OADDR:
		agen(nl, res)

		// Release res so that it is available for cgen_call.
	// Pick it up again after the call.
	case gc.OCALLMETH,
		gc.OCALLFUNC:
		rg := -1

		if n.Ullman >= gc.UINF {
			if res != nil && (res.Op == gc.OREGISTER || res.Op == gc.OINDREG) {
				rg = int(res.Val.U.Reg)
				reg[rg]--
			}
		}

		if n.Op == gc.OCALLMETH {
			gc.Cgen_callmeth(n, 0)
		} else {
			cgen_call(n, 0)
		}
		if rg >= 0 {
			reg[rg]++
		}
		cgen_callret(n, res)

	case gc.OCALLINTER:
		cgen_callinter(n, res, 0)
		cgen_callret(n, res)

	case gc.OMOD,
		gc.ODIV:
		a = optoas(int(n.Op), nl.Type)
		goto abop
	}

	goto ret

sbop: // symmetric binary
	if nl.Ullman < nr.Ullman {
		r := nl
		nl = nr
		nr = r
	}

	// TODO(kaib): use fewer registers here.
abop: // asymmetric binary
	if nl.Ullman >= nr.Ullman {
		regalloc(&n1, nl.Type, res)
		cgen(nl, &n1)
		switch n.Op {
		case gc.OADD,
			gc.OSUB,
			gc.OAND,
			gc.OOR,
			gc.OXOR:
			if gc.Smallintconst(nr) {
				n2 = *nr
				break
			}
			fallthrough

		default:
			regalloc(&n2, nr.Type, nil)
			cgen(nr, &n2)
		}
	} else {
		switch n.Op {
		case gc.OADD,
			gc.OSUB,
			gc.OAND,
			gc.OOR,
			gc.OXOR:
			if gc.Smallintconst(nr) {
				n2 = *nr
				break
			}
			fallthrough

		default:
			regalloc(&n2, nr.Type, res)
			cgen(nr, &n2)
		}

		regalloc(&n1, nl.Type, nil)
		cgen(nl, &n1)
	}

	gins(a, &n2, &n1)

	// Normalize result for types smaller than word.
norm:
	if n.Type.Width < int64(gc.Widthptr) {
		switch n.Op {
		case gc.OADD,
			gc.OSUB,
			gc.OMUL,
			gc.OCOM,
			gc.OMINUS:
			gins(optoas(gc.OAS, n.Type), &n1, &n1)
		}
	}

	gmove(&n1, res)
	regfree(&n1)
	if n2.Op != gc.OLITERAL {
		regfree(&n2)
	}
	goto ret

flt: // floating-point.
	regalloc(&f0, nl.Type, res)

	if nr != nil {
		goto flt2
	}

	if n.Op == gc.OMINUS {
		nr = gc.Nodintconst(-1)
		gc.Convlit(&nr, n.Type)
		n.Op = gc.OMUL
		goto flt2
	}

	// unary
	cgen(nl, &f0)

	if n.Op != gc.OCONV && n.Op != gc.OPLUS {
		gins(optoas(int(n.Op), n.Type), &f0, &f0)
	}
	gmove(&f0, res)
	regfree(&f0)
	goto ret

flt2: // binary
	if nl.Ullman >= nr.Ullman {
		cgen(nl, &f0)
		regalloc(&f1, n.Type, nil)
		gmove(&f0, &f1)
		cgen(nr, &f0)
		gins(optoas(int(n.Op), n.Type), &f0, &f1)
	} else {
		cgen(nr, &f0)
		regalloc(&f1, n.Type, nil)
		cgen(nl, &f1)
		gins(optoas(int(n.Op), n.Type), &f0, &f1)
	}

	gmove(&f1, res)
	regfree(&f0)
	regfree(&f1)
	goto ret

ret:
}

/*
 * generate array index into res.
 * n might be any size; res is 32-bit.
 * returns Prog* to patch to panic call.
 */
func cgenindex(n *gc.Node, res *gc.Node, bounded bool) *obj.Prog {
	if !gc.Is64(n.Type) {
		cgen(n, res)
		return nil
	}

	var tmp gc.Node
	gc.Tempname(&tmp, gc.Types[gc.TINT64])
	cgen(n, &tmp)
	var lo gc.Node
	var hi gc.Node
	split64(&tmp, &lo, &hi)
	gmove(&lo, res)
	if bounded {
		splitclean()
		return nil
	}

	var n1 gc.Node
	regalloc(&n1, gc.Types[gc.TINT32], nil)
	var n2 gc.Node
	regalloc(&n2, gc.Types[gc.TINT32], nil)
	var zero gc.Node
	gc.Nodconst(&zero, gc.Types[gc.TINT32], 0)
	gmove(&hi, &n1)
	gmove(&zero, &n2)
	gcmp(arm.ACMP, &n1, &n2)
	regfree(&n2)
	regfree(&n1)
	splitclean()
	return gc.Gbranch(arm.ABNE, nil, -1)
}

/*
 * generate:
 *	res = &n;
 * The generated code checks that the result is not nil.
 */
func agen(n *gc.Node, res *gc.Node) {
	if gc.Debug['g'] != 0 {
		gc.Dump("\nagen-res", res)
		gc.Dump("agen-r", n)
	}

	if n == nil || n.Type == nil || res == nil || res.Type == nil {
		gc.Fatal("agen")
	}

	for n.Op == gc.OCONVNOP {
		n = n.Left
	}

	var nl *gc.Node
	if gc.Isconst(n, gc.CTNIL) && n.Type.Width > int64(gc.Widthptr) {
		// Use of a nil interface or nil slice.
		// Create a temporary we can take the address of and read.
		// The generated code is just going to panic, so it need not
		// be terribly efficient. See issue 3670.
		var n1 gc.Node
		gc.Tempname(&n1, n.Type)

		gc.Gvardef(&n1)
		clearfat(&n1)
		var n2 gc.Node
		regalloc(&n2, gc.Types[gc.Tptr], res)
		gins(arm.AMOVW, &n1, &n2)
		gmove(&n2, res)
		regfree(&n2)
		goto ret
	}

	if n.Addable != 0 {
		n1 := gc.Node{}
		n1.Op = gc.OADDR
		n1.Left = n
		var n2 gc.Node
		regalloc(&n2, gc.Types[gc.Tptr], res)
		gins(arm.AMOVW, &n1, &n2)
		gmove(&n2, res)
		regfree(&n2)
		goto ret
	}

	nl = n.Left

	switch n.Op {
	default:
		gc.Fatal("agen: unknown op %v", gc.Nconv(n, obj.FmtShort|obj.FmtSign))

		// Release res so that it is available for cgen_call.
	// Pick it up again after the call.
	case gc.OCALLMETH,
		gc.OCALLFUNC:
		r := -1

		if n.Ullman >= gc.UINF {
			if res.Op == gc.OREGISTER || res.Op == gc.OINDREG {
				r = int(res.Val.U.Reg)
				reg[r]--
			}
		}

		if n.Op == gc.OCALLMETH {
			gc.Cgen_callmeth(n, 0)
		} else {
			cgen_call(n, 0)
		}
		if r >= 0 {
			reg[r]++
		}
		cgen_aret(n, res)

	case gc.OCALLINTER:
		cgen_callinter(n, res, 0)
		cgen_aret(n, res)

	case gc.OSLICE,
		gc.OSLICEARR,
		gc.OSLICESTR,
		gc.OSLICE3,
		gc.OSLICE3ARR:
		var n1 gc.Node
		gc.Tempname(&n1, n.Type)
		gc.Cgen_slice(n, &n1)
		agen(&n1, res)

	case gc.OEFACE:
		var n1 gc.Node
		gc.Tempname(&n1, n.Type)
		gc.Cgen_eface(n, &n1)
		agen(&n1, res)

	case gc.OINDEX:
		var n1 gc.Node
		agenr(n, &n1, res)
		gmove(&n1, res)
		regfree(&n1)

		// should only get here with names in this func.
	case gc.ONAME:
		if n.Funcdepth > 0 && n.Funcdepth != gc.Funcdepth {
			gc.Dump("bad agen", n)
			gc.Fatal("agen: bad ONAME funcdepth %d != %d", n.Funcdepth, gc.Funcdepth)
		}

		// should only get here for heap vars or paramref
		if n.Class&gc.PHEAP == 0 && n.Class != gc.PPARAMREF {
			gc.Dump("bad agen", n)
			gc.Fatal("agen: bad ONAME class %#x", n.Class)
		}

		cgen(n.Heapaddr, res)
		if n.Xoffset != 0 {
			var n1 gc.Node
			gc.Nodconst(&n1, gc.Types[gc.TINT32], n.Xoffset)
			var n2 gc.Node
			regalloc(&n2, n1.Type, nil)
			var n3 gc.Node
			regalloc(&n3, gc.Types[gc.TINT32], nil)
			gmove(&n1, &n2)
			gmove(res, &n3)
			gins(optoas(gc.OADD, gc.Types[gc.Tptr]), &n2, &n3)
			gmove(&n3, res)
			regfree(&n2)
			regfree(&n3)
		}

	case gc.OIND:
		cgen(nl, res)
		gc.Cgen_checknil(res)

	case gc.ODOT:
		agen(nl, res)
		if n.Xoffset != 0 {
			var n1 gc.Node
			gc.Nodconst(&n1, gc.Types[gc.TINT32], n.Xoffset)
			var n2 gc.Node
			regalloc(&n2, n1.Type, nil)
			var n3 gc.Node
			regalloc(&n3, gc.Types[gc.TINT32], nil)
			gmove(&n1, &n2)
			gmove(res, &n3)
			gins(optoas(gc.OADD, gc.Types[gc.Tptr]), &n2, &n3)
			gmove(&n3, res)
			regfree(&n2)
			regfree(&n3)
		}

	case gc.ODOTPTR:
		cgen(nl, res)
		gc.Cgen_checknil(res)
		if n.Xoffset != 0 {
			var n1 gc.Node
			gc.Nodconst(&n1, gc.Types[gc.TINT32], n.Xoffset)
			var n2 gc.Node
			regalloc(&n2, n1.Type, nil)
			var n3 gc.Node
			regalloc(&n3, gc.Types[gc.Tptr], nil)
			gmove(&n1, &n2)
			gmove(res, &n3)
			gins(optoas(gc.OADD, gc.Types[gc.Tptr]), &n2, &n3)
			gmove(&n3, res)
			regfree(&n2)
			regfree(&n3)
		}
	}

ret:
}

/*
 * generate:
 *	newreg = &n;
 *	res = newreg
 *
 * on exit, a has been changed to be *newreg.
 * caller must regfree(a).
 * The generated code checks that the result is not *nil.
 */
func igen(n *gc.Node, a *gc.Node, res *gc.Node) {
	if gc.Debug['g'] != 0 {
		gc.Dump("\nigen-n", n)
	}

	switch n.Op {
	case gc.ONAME:
		if (n.Class&gc.PHEAP != 0) || n.Class == gc.PPARAMREF {
			break
		}
		*a = *n
		return

		// Increase the refcount of the register so that igen's caller
	// has to call regfree.
	case gc.OINDREG:
		if n.Val.U.Reg != arm.REGSP {
			reg[n.Val.U.Reg]++
		}
		*a = *n
		return

	case gc.ODOT:
		igen(n.Left, a, res)
		a.Xoffset += n.Xoffset
		a.Type = n.Type
		return

	case gc.ODOTPTR:
		if n.Left.Addable != 0 || n.Left.Op == gc.OCALLFUNC || n.Left.Op == gc.OCALLMETH || n.Left.Op == gc.OCALLINTER {
			// igen-able nodes.
			var n1 gc.Node
			igen(n.Left, &n1, res)

			regalloc(a, gc.Types[gc.Tptr], &n1)
			gmove(&n1, a)
			regfree(&n1)
		} else {
			regalloc(a, gc.Types[gc.Tptr], res)
			cgen(n.Left, a)
		}

		gc.Cgen_checknil(a)
		a.Op = gc.OINDREG
		a.Xoffset = n.Xoffset
		a.Type = n.Type
		return

		// Release res so that it is available for cgen_call.
	// Pick it up again after the call.
	case gc.OCALLMETH,
		gc.OCALLFUNC,
		gc.OCALLINTER:
		r := -1

		if n.Ullman >= gc.UINF {
			if res != nil && (res.Op == gc.OREGISTER || res.Op == gc.OINDREG) {
				r = int(res.Val.U.Reg)
				reg[r]--
			}
		}

		switch n.Op {
		case gc.OCALLMETH:
			gc.Cgen_callmeth(n, 0)

		case gc.OCALLFUNC:
			cgen_call(n, 0)

		case gc.OCALLINTER:
			cgen_callinter(n, nil, 0)
		}

		if r >= 0 {
			reg[r]++
		}
		regalloc(a, gc.Types[gc.Tptr], res)
		cgen_aret(n, a)
		a.Op = gc.OINDREG
		a.Type = n.Type
		return
	}

	agenr(n, a, res)
	a.Op = gc.OINDREG
	a.Type = n.Type
}

/*
 * allocate a register in res and generate
 *  newreg = &n
 * The caller must call regfree(a).
 */
func cgenr(n *gc.Node, a *gc.Node, res *gc.Node) {
	if gc.Debug['g'] != 0 {
		gc.Dump("cgenr-n", n)
	}

	if gc.Isfat(n.Type) {
		gc.Fatal("cgenr on fat node")
	}

	if n.Addable != 0 {
		regalloc(a, gc.Types[gc.Tptr], res)
		gmove(n, a)
		return
	}

	switch n.Op {
	case gc.ONAME,
		gc.ODOT,
		gc.ODOTPTR,
		gc.OINDEX,
		gc.OCALLFUNC,
		gc.OCALLMETH,
		gc.OCALLINTER:
		var n1 gc.Node
		igen(n, &n1, res)
		regalloc(a, gc.Types[gc.Tptr], &n1)
		gmove(&n1, a)
		regfree(&n1)

	default:
		regalloc(a, n.Type, res)
		cgen(n, a)
	}
}

/*
 * generate:
 *	newreg = &n;
 *
 * caller must regfree(a).
 * The generated code checks that the result is not nil.
 */
func agenr(n *gc.Node, a *gc.Node, res *gc.Node) {
	if gc.Debug['g'] != 0 {
		gc.Dump("agenr-n", n)
	}

	nl := n.Left
	nr := n.Right

	switch n.Op {
	case gc.ODOT,
		gc.ODOTPTR,
		gc.OCALLFUNC,
		gc.OCALLMETH,
		gc.OCALLINTER:
		var n1 gc.Node
		igen(n, &n1, res)
		regalloc(a, gc.Types[gc.Tptr], &n1)
		agen(&n1, a)
		regfree(&n1)

	case gc.OIND:
		cgenr(n.Left, a, res)
		gc.Cgen_checknil(a)

	case gc.OINDEX:
		p2 := (*obj.Prog)(nil) // to be patched to panicindex.
		w := uint32(n.Type.Width)
		bounded := gc.Debug['B'] != 0 || n.Bounded
		var n1 gc.Node
		var n3 gc.Node
		if nr.Addable != 0 {
			var tmp gc.Node
			if !gc.Isconst(nr, gc.CTINT) {
				gc.Tempname(&tmp, gc.Types[gc.TINT32])
			}
			if !gc.Isconst(nl, gc.CTSTR) {
				agenr(nl, &n3, res)
			}
			if !gc.Isconst(nr, gc.CTINT) {
				p2 = cgenindex(nr, &tmp, bounded)
				regalloc(&n1, tmp.Type, nil)
				gmove(&tmp, &n1)
			}
		} else if nl.Addable != 0 {
			if !gc.Isconst(nr, gc.CTINT) {
				var tmp gc.Node
				gc.Tempname(&tmp, gc.Types[gc.TINT32])
				p2 = cgenindex(nr, &tmp, bounded)
				regalloc(&n1, tmp.Type, nil)
				gmove(&tmp, &n1)
			}

			if !gc.Isconst(nl, gc.CTSTR) {
				agenr(nl, &n3, res)
			}
		} else {
			var tmp gc.Node
			gc.Tempname(&tmp, gc.Types[gc.TINT32])
			p2 = cgenindex(nr, &tmp, bounded)
			nr = &tmp
			if !gc.Isconst(nl, gc.CTSTR) {
				agenr(nl, &n3, res)
			}
			regalloc(&n1, tmp.Type, nil)
			gins(optoas(gc.OAS, tmp.Type), &tmp, &n1)
		}

		// &a is in &n3 (allocated in res)
		// i is in &n1 (if not constant)
		// w is width

		// constant index
		if gc.Isconst(nr, gc.CTINT) {
			if gc.Isconst(nl, gc.CTSTR) {
				gc.Fatal("constant string constant index")
			}
			v := uint64(gc.Mpgetfix(nr.Val.U.Xval))
			var n2 gc.Node
			if gc.Isslice(nl.Type) || nl.Type.Etype == gc.TSTRING {
				if gc.Debug['B'] == 0 && !n.Bounded {
					n1 = n3
					n1.Op = gc.OINDREG
					n1.Type = gc.Types[gc.Tptr]
					n1.Xoffset = int64(gc.Array_nel)
					var n4 gc.Node
					regalloc(&n4, n1.Type, nil)
					gmove(&n1, &n4)
					gc.Nodconst(&n2, gc.Types[gc.TUINT32], int64(v))
					gcmp(optoas(gc.OCMP, gc.Types[gc.TUINT32]), &n4, &n2)
					regfree(&n4)
					p1 := gc.Gbranch(optoas(gc.OGT, gc.Types[gc.TUINT32]), nil, +1)
					ginscall(gc.Panicindex, 0)
					gc.Patch(p1, gc.Pc)
				}

				n1 = n3
				n1.Op = gc.OINDREG
				n1.Type = gc.Types[gc.Tptr]
				n1.Xoffset = int64(gc.Array_array)
				gmove(&n1, &n3)
			}

			gc.Nodconst(&n2, gc.Types[gc.Tptr], int64(v*uint64(w)))
			gins(optoas(gc.OADD, gc.Types[gc.Tptr]), &n2, &n3)
			*a = n3
			break
		}

		var n2 gc.Node
		regalloc(&n2, gc.Types[gc.TINT32], &n1) // i
		gmove(&n1, &n2)
		regfree(&n1)

		var n4 gc.Node
		if gc.Debug['B'] == 0 && !n.Bounded {
			// check bounds
			if gc.Isconst(nl, gc.CTSTR) {
				gc.Nodconst(&n4, gc.Types[gc.TUINT32], int64(len(nl.Val.U.Sval.S)))
			} else if gc.Isslice(nl.Type) || nl.Type.Etype == gc.TSTRING {
				n1 = n3
				n1.Op = gc.OINDREG
				n1.Type = gc.Types[gc.Tptr]
				n1.Xoffset = int64(gc.Array_nel)
				regalloc(&n4, gc.Types[gc.TUINT32], nil)
				gmove(&n1, &n4)
			} else {
				gc.Nodconst(&n4, gc.Types[gc.TUINT32], nl.Type.Bound)
			}

			gcmp(optoas(gc.OCMP, gc.Types[gc.TUINT32]), &n2, &n4)
			if n4.Op == gc.OREGISTER {
				regfree(&n4)
			}
			p1 := gc.Gbranch(optoas(gc.OLT, gc.Types[gc.TUINT32]), nil, +1)
			if p2 != nil {
				gc.Patch(p2, gc.Pc)
			}
			ginscall(gc.Panicindex, 0)
			gc.Patch(p1, gc.Pc)
		}

		if gc.Isconst(nl, gc.CTSTR) {
			regalloc(&n3, gc.Types[gc.Tptr], res)
			p1 := gins(arm.AMOVW, nil, &n3)
			gc.Datastring(nl.Val.U.Sval.S, &p1.From)
			p1.From.Type = obj.TYPE_ADDR
		} else if gc.Isslice(nl.Type) || nl.Type.Etype == gc.TSTRING {
			n1 = n3
			n1.Op = gc.OINDREG
			n1.Type = gc.Types[gc.Tptr]
			n1.Xoffset = int64(gc.Array_array)
			gmove(&n1, &n3)
		}

		if w == 0 {
		} else // nothing to do
		if w == 1 || w == 2 || w == 4 || w == 8 {
			n4 = gc.Node{}
			n4.Op = gc.OADDR
			n4.Left = &n2
			cgen(&n4, &n3)
			if w == 1 {
				gins(arm.AADD, &n2, &n3)
			} else if w == 2 {
				gshift(arm.AADD, &n2, arm.SHIFT_LL, 1, &n3)
			} else if w == 4 {
				gshift(arm.AADD, &n2, arm.SHIFT_LL, 2, &n3)
			} else if w == 8 {
				gshift(arm.AADD, &n2, arm.SHIFT_LL, 3, &n3)
			}
		} else {
			regalloc(&n4, gc.Types[gc.TUINT32], nil)
			gc.Nodconst(&n1, gc.Types[gc.TUINT32], int64(w))
			gmove(&n1, &n4)
			gins(optoas(gc.OMUL, gc.Types[gc.TUINT32]), &n4, &n2)
			gins(optoas(gc.OADD, gc.Types[gc.Tptr]), &n2, &n3)
			regfree(&n4)
		}

		*a = n3
		regfree(&n2)

	default:
		regalloc(a, gc.Types[gc.Tptr], res)
		agen(n, a)
	}
}

func gencmp0(n *gc.Node, t *gc.Type, o int, likely int, to *obj.Prog) {
	var n1 gc.Node

	regalloc(&n1, t, nil)
	cgen(n, &n1)
	a := optoas(gc.OCMP, t)
	if a != arm.ACMP {
		var n2 gc.Node
		gc.Nodconst(&n2, t, 0)
		var n3 gc.Node
		regalloc(&n3, t, nil)
		gmove(&n2, &n3)
		gcmp(a, &n1, &n3)
		regfree(&n3)
	} else {
		gins(arm.ATST, &n1, nil)
	}
	a = optoas(o, t)
	gc.Patch(gc.Gbranch(a, t, likely), to)
	regfree(&n1)
}

/*
 * generate:
 *	if(n == true) goto to;
 */
func bgen(n *gc.Node, true_ bool, likely int, to *obj.Prog) {
	if gc.Debug['g'] != 0 {
		gc.Dump("\nbgen", n)
	}

	if n == nil {
		n = gc.Nodbool(true)
	}

	if n.Ninit != nil {
		gc.Genlist(n.Ninit)
	}

	var et int
	var nl *gc.Node
	var nr *gc.Node
	if n.Type == nil {
		gc.Convlit(&n, gc.Types[gc.TBOOL])
		if n.Type == nil {
			goto ret
		}
	}

	et = int(n.Type.Etype)
	if et != gc.TBOOL {
		gc.Yyerror("cgen: bad type %v for %v", gc.Tconv(n.Type, 0), gc.Oconv(int(n.Op), 0))
		gc.Patch(gins(obj.AEND, nil, nil), to)
		goto ret
	}

	nr = nil

	switch n.Op {
	default:
		a := gc.ONE
		if !true_ {
			a = gc.OEQ
		}
		gencmp0(n, n.Type, a, likely, to)
		goto ret

		// need to ask if it is bool?
	case gc.OLITERAL:
		if !true_ == (n.Val.U.Bval == 0) {
			gc.Patch(gc.Gbranch(arm.AB, nil, 0), to)
		}
		goto ret

	case gc.OANDAND,
		gc.OOROR:
		if (n.Op == gc.OANDAND) == true_ {
			p1 := gc.Gbranch(obj.AJMP, nil, 0)
			p2 := gc.Gbranch(obj.AJMP, nil, 0)
			gc.Patch(p1, gc.Pc)
			bgen(n.Left, !true_, -likely, p2)
			bgen(n.Right, !true_, -likely, p2)
			p1 = gc.Gbranch(obj.AJMP, nil, 0)
			gc.Patch(p1, to)
			gc.Patch(p2, gc.Pc)
		} else {
			bgen(n.Left, true_, likely, to)
			bgen(n.Right, true_, likely, to)
		}

		goto ret

	case gc.OEQ,
		gc.ONE,
		gc.OLT,
		gc.OGT,
		gc.OLE,
		gc.OGE:
		nr = n.Right
		if nr == nil || nr.Type == nil {
			goto ret
		}
		fallthrough

	case gc.ONOT: // unary
		nl = n.Left

		if nl == nil || nl.Type == nil {
			goto ret
		}
	}

	switch n.Op {
	case gc.ONOT:
		bgen(nl, !true_, likely, to)
		goto ret

	case gc.OEQ,
		gc.ONE,
		gc.OLT,
		gc.OGT,
		gc.OLE,
		gc.OGE:
		a := int(n.Op)
		if !true_ {
			if gc.Isfloat[nl.Type.Etype] != 0 {
				// brcom is not valid on floats when NaN is involved.
				p1 := gc.Gbranch(arm.AB, nil, 0)

				p2 := gc.Gbranch(arm.AB, nil, 0)
				gc.Patch(p1, gc.Pc)
				ll := n.Ninit
				n.Ninit = nil
				bgen(n, true, -likely, p2)
				n.Ninit = ll
				gc.Patch(gc.Gbranch(arm.AB, nil, 0), to)
				gc.Patch(p2, gc.Pc)
				goto ret
			}

			a = gc.Brcom(a)
			true_ = !true_
		}

		// make simplest on right
		if nl.Op == gc.OLITERAL || (nl.Ullman < gc.UINF && nl.Ullman < nr.Ullman) {
			a = gc.Brrev(a)
			r := nl
			nl = nr
			nr = r
		}

		if gc.Isslice(nl.Type) {
			// only valid to cmp darray to literal nil
			if (a != gc.OEQ && a != gc.ONE) || nr.Op != gc.OLITERAL {
				gc.Yyerror("illegal array comparison")
				break
			}

			var n1 gc.Node
			igen(nl, &n1, nil)
			n1.Xoffset += int64(gc.Array_array)
			n1.Type = gc.Types[gc.Tptr]
			gencmp0(&n1, gc.Types[gc.Tptr], a, likely, to)
			regfree(&n1)
			break
		}

		if gc.Isinter(nl.Type) {
			// front end shold only leave cmp to literal nil
			if (a != gc.OEQ && a != gc.ONE) || nr.Op != gc.OLITERAL {
				gc.Yyerror("illegal interface comparison")
				break
			}

			var n1 gc.Node
			igen(nl, &n1, nil)
			n1.Type = gc.Types[gc.Tptr]
			n1.Xoffset += 0
			gencmp0(&n1, gc.Types[gc.Tptr], a, likely, to)
			regfree(&n1)
			break
		}

		if gc.Iscomplex[nl.Type.Etype] != 0 {
			gc.Complexbool(a, nl, nr, true_, likely, to)
			break
		}

		if gc.Is64(nr.Type) {
			if nl.Addable == 0 {
				var n1 gc.Node
				gc.Tempname(&n1, nl.Type)
				cgen(nl, &n1)
				nl = &n1
			}

			if nr.Addable == 0 {
				var n2 gc.Node
				gc.Tempname(&n2, nr.Type)
				cgen(nr, &n2)
				nr = &n2
			}

			cmp64(nl, nr, a, likely, to)
			break
		}

		if nr.Op == gc.OLITERAL {
			if gc.Isconst(nr, gc.CTINT) && gc.Mpgetfix(nr.Val.U.Xval) == 0 {
				gencmp0(nl, nl.Type, a, likely, to)
				break
			}

			if nr.Val.Ctype == gc.CTNIL {
				gencmp0(nl, nl.Type, a, likely, to)
				break
			}
		}

		a = optoas(a, nr.Type)

		if nr.Ullman >= gc.UINF {
			var n1 gc.Node
			regalloc(&n1, nl.Type, nil)
			cgen(nl, &n1)

			var tmp gc.Node
			gc.Tempname(&tmp, nl.Type)
			gmove(&n1, &tmp)
			regfree(&n1)

			var n2 gc.Node
			regalloc(&n2, nr.Type, nil)
			cgen(nr, &n2)

			regalloc(&n1, nl.Type, nil)
			cgen(&tmp, &n1)

			gcmp(optoas(gc.OCMP, nr.Type), &n1, &n2)
			gc.Patch(gc.Gbranch(a, nr.Type, likely), to)

			regfree(&n1)
			regfree(&n2)
			break
		}

		var n3 gc.Node
		gc.Tempname(&n3, nl.Type)
		cgen(nl, &n3)

		var tmp gc.Node
		gc.Tempname(&tmp, nr.Type)
		cgen(nr, &tmp)

		var n1 gc.Node
		regalloc(&n1, nl.Type, nil)
		gmove(&n3, &n1)

		var n2 gc.Node
		regalloc(&n2, nr.Type, nil)
		gmove(&tmp, &n2)

		gcmp(optoas(gc.OCMP, nr.Type), &n1, &n2)
		if gc.Isfloat[nl.Type.Etype] != 0 {
			if n.Op == gc.ONE {
				p1 := gc.Gbranch(arm.ABVS, nr.Type, likely)
				gc.Patch(gc.Gbranch(a, nr.Type, likely), to)
				gc.Patch(p1, to)
			} else {
				p1 := gc.Gbranch(arm.ABVS, nr.Type, -likely)
				gc.Patch(gc.Gbranch(a, nr.Type, likely), to)
				gc.Patch(p1, gc.Pc)
			}
		} else {
			gc.Patch(gc.Gbranch(a, nr.Type, likely), to)
		}

		regfree(&n1)
		regfree(&n2)
	}

	goto ret

ret:
}

/*
 * n is on stack, either local variable
 * or return value from function call.
 * return n's offset from SP.
 */
func stkof(n *gc.Node) int32 {
	switch n.Op {
	case gc.OINDREG:
		return int32(n.Xoffset)

	case gc.ODOT:
		t := n.Left.Type
		if gc.Isptr[t.Etype] != 0 {
			break
		}
		off := stkof(n.Left)
		if off == -1000 || off == 1000 {
			return off
		}
		return int32(int64(off) + n.Xoffset)

	case gc.OINDEX:
		t := n.Left.Type
		if !gc.Isfixedarray(t) {
			break
		}
		off := stkof(n.Left)
		if off == -1000 || off == 1000 {
			return off
		}
		if gc.Isconst(n.Right, gc.CTINT) {
			return int32(int64(off) + t.Type.Width*gc.Mpgetfix(n.Right.Val.U.Xval))
		}
		return 1000

	case gc.OCALLMETH,
		gc.OCALLINTER,
		gc.OCALLFUNC:
		t := n.Left.Type
		if gc.Isptr[t.Etype] != 0 {
			t = t.Type
		}

		var flist gc.Iter
		t = gc.Structfirst(&flist, gc.Getoutarg(t))
		if t != nil {
			return int32(t.Width + 4) // correct for LR
		}
	}

	// botch - probably failing to recognize address
	// arithmetic on the above. eg INDEX and DOT
	return -1000
}

/*
 * block copy:
 *	memmove(&res, &n, w);
 * NB: character copy assumed little endian architecture
 */
func sgen(n *gc.Node, res *gc.Node, w int64) {
	if gc.Debug['g'] != 0 {
		fmt.Printf("\nsgen w=%d\n", w)
		gc.Dump("r", n)
		gc.Dump("res", res)
	}

	if n.Ullman >= gc.UINF && res.Ullman >= gc.UINF {
		gc.Fatal("sgen UINF")
	}

	if w < 0 || int64(int32(w)) != w {
		gc.Fatal("sgen copy %d", w)
	}

	if n.Type == nil {
		gc.Fatal("sgen: missing type")
	}

	if w == 0 {
		// evaluate side effects only.
		var dst gc.Node
		regalloc(&dst, gc.Types[gc.Tptr], nil)

		agen(res, &dst)
		agen(n, &dst)
		regfree(&dst)
		return
	}

	// If copying .args, that's all the results, so record definition sites
	// for them for the liveness analysis.
	if res.Op == gc.ONAME && res.Sym.Name == ".args" {
		for l := gc.Curfn.Dcl; l != nil; l = l.Next {
			if l.N.Class == gc.PPARAMOUT {
				gc.Gvardef(l.N)
			}
		}
	}

	// Avoid taking the address for simple enough types.
	if componentgen(n, res) {
		return
	}

	// determine alignment.
	// want to avoid unaligned access, so have to use
	// smaller operations for less aligned types.
	// for example moving [4]byte must use 4 MOVB not 1 MOVW.
	align := int(n.Type.Align)

	var op int
	switch align {
	default:
		gc.Fatal("sgen: invalid alignment %d for %v", align, gc.Tconv(n.Type, 0))

	case 1:
		op = arm.AMOVB

	case 2:
		op = arm.AMOVH

	case 4:
		op = arm.AMOVW
	}

	if w%int64(align) != 0 {
		gc.Fatal("sgen: unaligned size %d (align=%d) for %v", w, align, gc.Tconv(n.Type, 0))
	}
	c := int32(w / int64(align))

	// offset on the stack
	osrc := stkof(n)

	odst := stkof(res)
	if osrc != -1000 && odst != -1000 && (osrc == 1000 || odst == 1000) {
		// osrc and odst both on stack, and at least one is in
		// an unknown position.  Could generate code to test
		// for forward/backward copy, but instead just copy
		// to a temporary location first.
		var tmp gc.Node
		gc.Tempname(&tmp, n.Type)

		sgen(n, &tmp, w)
		sgen(&tmp, res, w)
		return
	}

	if osrc%int32(align) != 0 || odst%int32(align) != 0 {
		gc.Fatal("sgen: unaligned offset src %d or dst %d (align %d)", osrc, odst, align)
	}

	// if we are copying forward on the stack and
	// the src and dst overlap, then reverse direction
	dir := align

	if osrc < odst && int64(odst) < int64(osrc)+w {
		dir = -dir
	}

	if op == arm.AMOVW && !gc.Nacl && dir > 0 && c >= 4 && c <= 128 {
		var r0 gc.Node
		r0.Op = gc.OREGISTER
		r0.Val.U.Reg = REGALLOC_R0
		var r1 gc.Node
		r1.Op = gc.OREGISTER
		r1.Val.U.Reg = REGALLOC_R0 + 1
		var r2 gc.Node
		r2.Op = gc.OREGISTER
		r2.Val.U.Reg = REGALLOC_R0 + 2

		var src gc.Node
		regalloc(&src, gc.Types[gc.Tptr], &r1)
		var dst gc.Node
		regalloc(&dst, gc.Types[gc.Tptr], &r2)
		if n.Ullman >= res.Ullman {
			// eval n first
			agen(n, &src)

			if res.Op == gc.ONAME {
				gc.Gvardef(res)
			}
			agen(res, &dst)
		} else {
			// eval res first
			if res.Op == gc.ONAME {
				gc.Gvardef(res)
			}
			agen(res, &dst)
			agen(n, &src)
		}

		var tmp gc.Node
		regalloc(&tmp, gc.Types[gc.Tptr], &r0)
		f := gc.Sysfunc("duffcopy")
		p := gins(obj.ADUFFCOPY, nil, f)
		gc.Afunclit(&p.To, f)

		// 8 and 128 = magic constants: see ../../runtime/asm_arm.s
		p.To.Offset = 8 * (128 - int64(c))

		regfree(&tmp)
		regfree(&src)
		regfree(&dst)
		return
	}

	var dst gc.Node
	var src gc.Node
	if n.Ullman >= res.Ullman {
		agenr(n, &dst, res) // temporarily use dst
		regalloc(&src, gc.Types[gc.Tptr], nil)
		gins(arm.AMOVW, &dst, &src)
		if res.Op == gc.ONAME {
			gc.Gvardef(res)
		}
		agen(res, &dst)
	} else {
		if res.Op == gc.ONAME {
			gc.Gvardef(res)
		}
		agenr(res, &dst, res)
		agenr(n, &src, nil)
	}

	var tmp gc.Node
	regalloc(&tmp, gc.Types[gc.TUINT32], nil)

	// set up end marker
	nend := gc.Node{}

	if c >= 4 {
		regalloc(&nend, gc.Types[gc.TUINT32], nil)

		p := gins(arm.AMOVW, &src, &nend)
		p.From.Type = obj.TYPE_ADDR
		if dir < 0 {
			p.From.Offset = int64(dir)
		} else {
			p.From.Offset = w
		}
	}

	// move src and dest to the end of block if necessary
	if dir < 0 {
		p := gins(arm.AMOVW, &src, &src)
		p.From.Type = obj.TYPE_ADDR
		p.From.Offset = w + int64(dir)

		p = gins(arm.AMOVW, &dst, &dst)
		p.From.Type = obj.TYPE_ADDR
		p.From.Offset = w + int64(dir)
	}

	// move
	if c >= 4 {
		p := gins(op, &src, &tmp)
		p.From.Type = obj.TYPE_MEM
		p.From.Offset = int64(dir)
		p.Scond |= arm.C_PBIT
		ploop := p

		p = gins(op, &tmp, &dst)
		p.To.Type = obj.TYPE_MEM
		p.To.Offset = int64(dir)
		p.Scond |= arm.C_PBIT

		p = gins(arm.ACMP, &src, nil)
		raddr(&nend, p)

		gc.Patch(gc.Gbranch(arm.ABNE, nil, 0), ploop)
		regfree(&nend)
	} else {
		var p *obj.Prog
		for {
			tmp14 := c
			c--
			if tmp14 <= 0 {
				break
			}
			p = gins(op, &src, &tmp)
			p.From.Type = obj.TYPE_MEM
			p.From.Offset = int64(dir)
			p.Scond |= arm.C_PBIT

			p = gins(op, &tmp, &dst)
			p.To.Type = obj.TYPE_MEM
			p.To.Offset = int64(dir)
			p.Scond |= arm.C_PBIT
		}
	}

	regfree(&dst)
	regfree(&src)
	regfree(&tmp)
}

func cadable(n *gc.Node) bool {
	if n.Addable == 0 {
		// dont know how it happens,
		// but it does
		return false
	}

	switch n.Op {
	case gc.ONAME:
		return true
	}

	return false
}

/*
 * copy a composite value by moving its individual components.
 * Slices, strings and interfaces are supported.
 * Small structs or arrays with elements of basic type are
 * also supported.
 * nr is N when assigning a zero value.
 * return 1 if can do, 0 if cant.
 */
func componentgen(nr *gc.Node, nl *gc.Node) bool {
	var nodl gc.Node
	var nodr gc.Node

	freel := 0
	freer := 0

	switch nl.Type.Etype {
	default:
		goto no

	case gc.TARRAY:
		t := nl.Type

		// Slices are ok.
		if gc.Isslice(t) {
			break
		}

		// Small arrays are ok.
		if t.Bound > 0 && t.Bound <= 3 && !gc.Isfat(t.Type) {
			break
		}

		goto no

		// Small structs with non-fat types are ok.
	// Zero-sized structs are treated separately elsewhere.
	case gc.TSTRUCT:
		fldcount := int64(0)

		for t := nl.Type.Type; t != nil; t = t.Down {
			if gc.Isfat(t.Type) {
				goto no
			}
			if t.Etype != gc.TFIELD {
				gc.Fatal("componentgen: not a TFIELD: %v", gc.Tconv(t, obj.FmtLong))
			}
			fldcount++
		}

		if fldcount == 0 || fldcount > 4 {
			goto no
		}

	case gc.TSTRING,
		gc.TINTER:
		break
	}

	nodl = *nl
	if !cadable(nl) {
		if nr != nil && !cadable(nr) {
			goto no
		}
		igen(nl, &nodl, nil)
		freel = 1
	}

	if nr != nil {
		nodr = *nr
		if !cadable(nr) {
			igen(nr, &nodr, nil)
			freer = 1
		}
	} else {
		// When zeroing, prepare a register containing zero.
		var tmp gc.Node
		gc.Nodconst(&tmp, nl.Type, 0)

		regalloc(&nodr, gc.Types[gc.TUINT], nil)
		gmove(&tmp, &nodr)
		freer = 1
	}

	// nl and nr are 'cadable' which basically means they are names (variables) now.
	// If they are the same variable, don't generate any code, because the
	// VARDEF we generate will mark the old value as dead incorrectly.
	// (And also the assignments are useless.)
	if nr != nil && nl.Op == gc.ONAME && nr.Op == gc.ONAME && nl == nr {
		goto yes
	}

	switch nl.Type.Etype {
	// componentgen for arrays.
	case gc.TARRAY:
		if nl.Op == gc.ONAME {
			gc.Gvardef(nl)
		}
		t := nl.Type
		if !gc.Isslice(t) {
			nodl.Type = t.Type
			nodr.Type = nodl.Type
			for fldcount := int64(0); fldcount < t.Bound; fldcount++ {
				if nr == nil {
					gc.Clearslim(&nodl)
				} else {
					gmove(&nodr, &nodl)
				}
				nodl.Xoffset += t.Type.Width
				nodr.Xoffset += t.Type.Width
			}

			goto yes
		}

		// componentgen for slices.
		nodl.Xoffset += int64(gc.Array_array)

		nodl.Type = gc.Ptrto(nl.Type.Type)

		if nr != nil {
			nodr.Xoffset += int64(gc.Array_array)
			nodr.Type = nodl.Type
		}

		gmove(&nodr, &nodl)

		nodl.Xoffset += int64(gc.Array_nel) - int64(gc.Array_array)
		nodl.Type = gc.Types[gc.Simtype[gc.TUINT]]

		if nr != nil {
			nodr.Xoffset += int64(gc.Array_nel) - int64(gc.Array_array)
			nodr.Type = nodl.Type
		}

		gmove(&nodr, &nodl)

		nodl.Xoffset += int64(gc.Array_cap) - int64(gc.Array_nel)
		nodl.Type = gc.Types[gc.Simtype[gc.TUINT]]

		if nr != nil {
			nodr.Xoffset += int64(gc.Array_cap) - int64(gc.Array_nel)
			nodr.Type = nodl.Type
		}

		gmove(&nodr, &nodl)

		goto yes

	case gc.TSTRING:
		if nl.Op == gc.ONAME {
			gc.Gvardef(nl)
		}
		nodl.Xoffset += int64(gc.Array_array)
		nodl.Type = gc.Ptrto(gc.Types[gc.TUINT8])

		if nr != nil {
			nodr.Xoffset += int64(gc.Array_array)
			nodr.Type = nodl.Type
		}

		gmove(&nodr, &nodl)

		nodl.Xoffset += int64(gc.Array_nel) - int64(gc.Array_array)
		nodl.Type = gc.Types[gc.Simtype[gc.TUINT]]

		if nr != nil {
			nodr.Xoffset += int64(gc.Array_nel) - int64(gc.Array_array)
			nodr.Type = nodl.Type
		}

		gmove(&nodr, &nodl)

		goto yes

	case gc.TINTER:
		if nl.Op == gc.ONAME {
			gc.Gvardef(nl)
		}
		nodl.Xoffset += int64(gc.Array_array)
		nodl.Type = gc.Ptrto(gc.Types[gc.TUINT8])

		if nr != nil {
			nodr.Xoffset += int64(gc.Array_array)
			nodr.Type = nodl.Type
		}

		gmove(&nodr, &nodl)

		nodl.Xoffset += int64(gc.Array_nel) - int64(gc.Array_array)
		nodl.Type = gc.Ptrto(gc.Types[gc.TUINT8])

		if nr != nil {
			nodr.Xoffset += int64(gc.Array_nel) - int64(gc.Array_array)
			nodr.Type = nodl.Type
		}

		gmove(&nodr, &nodl)

		goto yes

	case gc.TSTRUCT:
		if nl.Op == gc.ONAME {
			gc.Gvardef(nl)
		}
		loffset := nodl.Xoffset
		roffset := nodr.Xoffset

		// funarg structs may not begin at offset zero.
		if nl.Type.Etype == gc.TSTRUCT && nl.Type.Funarg != 0 && nl.Type.Type != nil {
			loffset -= nl.Type.Type.Width
		}
		if nr != nil && nr.Type.Etype == gc.TSTRUCT && nr.Type.Funarg != 0 && nr.Type.Type != nil {
			roffset -= nr.Type.Type.Width
		}

		for t := nl.Type.Type; t != nil; t = t.Down {
			nodl.Xoffset = loffset + t.Width
			nodl.Type = t.Type

			if nr == nil {
				gc.Clearslim(&nodl)
			} else {
				nodr.Xoffset = roffset + t.Width
				nodr.Type = nodl.Type
				gmove(&nodr, &nodl)
			}
		}

		goto yes
	}

no:
	if freer != 0 {
		regfree(&nodr)
	}
	if freel != 0 {
		regfree(&nodl)
	}
	return false

yes:
	if freer != 0 {
		regfree(&nodr)
	}
	if freel != 0 {
		regfree(&nodl)
	}
	return true
}
