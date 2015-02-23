// Copyright 2009 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package main

import (
	"cmd/internal/obj"
	"cmd/internal/obj/i386"
	"fmt"
)
import "cmd/internal/gc"

/*
 * reg.c
 */

/*
 * peep.c
 */
func mgen(n *gc.Node, n1 *gc.Node, rg *gc.Node) {
	n1.Op = gc.OEMPTY

	if n.Addable != 0 {
		*n1 = *n
		if n1.Op == gc.OREGISTER || n1.Op == gc.OINDREG {
			reg[n.Val.U.Reg]++
		}
		return
	}

	gc.Tempname(n1, n.Type)
	cgen(n, n1)
	if n.Type.Width <= int64(gc.Widthptr) || gc.Isfloat[n.Type.Etype] != 0 {
		n2 := *n1
		regalloc(n1, n.Type, rg)
		gmove(&n2, n1)
	}
}

func mfree(n *gc.Node) {
	if n.Op == gc.OREGISTER {
		regfree(n)
	}
}

/*
 * generate:
 *	res = n;
 * simplifies and calls gmove.
 *
 * TODO:
 *	sudoaddable
 */
func cgen(n *gc.Node, res *gc.Node) {
	if gc.Debug['g'] != 0 {
		gc.Dump("\ncgen-n", n)
		gc.Dump("cgen-res", res)
	}

	if n == nil || n.Type == nil {
		gc.Fatal("cgen: n nil")
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

	// function calls on both sides?  introduce temporary
	if n.Ullman >= gc.UINF && res.Ullman >= gc.UINF {
		var n1 gc.Node
		gc.Tempname(&n1, n.Type)
		cgen(n, &n1)
		cgen(&n1, res)
		return
	}

	// structs etc get handled specially
	if gc.Isfat(n.Type) {
		if n.Type.Width < 0 {
			gc.Fatal("forgot to compute width for %v", gc.Tconv(n.Type, 0))
		}
		sgen(n, res, n.Type.Width)
		return
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
		gmove(n, res)
		return
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

	// complex types
	if gc.Complexop(n, res) {
		gc.Complexgen(n, res)
		return
	}

	// otherwise, the result is addressable but n is not.
	// let's do some computation.

	// use ullman to pick operand to eval first.
	nl := n.Left

	nr := n.Right
	if nl != nil && nl.Ullman >= gc.UINF {
		if nr != nil && nr.Ullman >= gc.UINF {
			// both are hard
			var n1 gc.Node
			gc.Tempname(&n1, nl.Type)

			cgen(nl, &n1)
			n2 := *n
			n2.Left = &n1
			cgen(&n2, res)
			return
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
		cgen_float(n, res)
		return
	}

	var a int
	switch n.Op {
	default:
		gc.Dump("cgen", n)
		gc.Fatal("cgen %v", gc.Oconv(int(n.Op), 0))

	case gc.OREAL,
		gc.OIMAG,
		gc.OCOMPLEX:
		gc.Fatal("unexpected complex")
		return

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

	case gc.OMINUS,
		gc.OCOM:
		a = optoas(int(n.Op), nl.Type)
		goto uop

		// symmetric binary
	case gc.OAND,
		gc.OOR,
		gc.OXOR,
		gc.OADD,
		gc.OMUL:
		a = optoas(int(n.Op), nl.Type)

		if a == i386.AIMULB {
			cgen_bmul(int(n.Op), nl, nr, res)
			break
		}

		goto sbop

		// asymmetric binary
	case gc.OSUB:
		a = optoas(int(n.Op), nl.Type)

		goto abop

	case gc.OHMUL:
		cgen_hmul(nl, nr, res)

	case gc.OCONV:
		if gc.Eqtype(n.Type, nl.Type) || gc.Noconv(n.Type, nl.Type) {
			cgen(nl, res)
			break
		}

		var n2 gc.Node
		gc.Tempname(&n2, n.Type)
		var n1 gc.Node
		mgen(nl, &n1, res)
		gmove(&n1, &n2)
		gmove(&n2, res)
		mfree(&n1)

	case gc.ODOT,
		gc.ODOTPTR,
		gc.OINDEX,
		gc.OIND,
		gc.ONAME: // PHEAP or PPARAMREF var
		var n1 gc.Node
		igen(n, &n1, res)

		gmove(&n1, res)
		regfree(&n1)

	case gc.OITAB:
		var n1 gc.Node
		igen(nl, &n1, res)
		n1.Type = gc.Ptrto(gc.Types[gc.TUINTPTR])
		gmove(&n1, res)
		regfree(&n1)

		// pointer is the first word of string or slice.
	case gc.OSPTR:
		if gc.Isconst(nl, gc.CTSTR) {
			var n1 gc.Node
			regalloc(&n1, gc.Types[gc.Tptr], res)
			p1 := gins(i386.ALEAL, nil, &n1)
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
			gc.Tempname(&n1, gc.Types[gc.Tptr])

			cgen(nl, &n1)
			var n2 gc.Node
			regalloc(&n2, gc.Types[gc.Tptr], nil)
			gmove(&n1, &n2)
			n1 = n2

			gc.Nodconst(&n2, gc.Types[gc.Tptr], 0)
			gins(optoas(gc.OCMP, gc.Types[gc.Tptr]), &n1, &n2)
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
			gc.Tempname(&n1, gc.Types[gc.Tptr])

			cgen(nl, &n1)
			var n2 gc.Node
			regalloc(&n2, gc.Types[gc.Tptr], nil)
			gmove(&n1, &n2)
			n1 = n2

			gc.Nodconst(&n2, gc.Types[gc.Tptr], 0)
			gins(optoas(gc.OCMP, gc.Types[gc.Tptr]), &n1, &n2)
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

	case gc.OCALLMETH:
		gc.Cgen_callmeth(n, 0)
		cgen_callret(n, res)

	case gc.OCALLINTER:
		cgen_callinter(n, res, 0)
		cgen_callret(n, res)

	case gc.OCALLFUNC:
		cgen_call(n, 0)
		cgen_callret(n, res)

	case gc.OMOD,
		gc.ODIV:
		cgen_div(int(n.Op), nl, nr, res)

	case gc.OLSH,
		gc.ORSH,
		gc.OLROT:
		cgen_shift(int(n.Op), n.Bounded, nl, nr, res)
	}

	return

sbop: // symmetric binary
	if nl.Ullman < nr.Ullman || nl.Op == gc.OLITERAL {
		r := nl
		nl = nr
		nr = r
	}

abop: // asymmetric binary
	if gc.Smallintconst(nr) {
		var n1 gc.Node
		mgen(nl, &n1, res)
		var n2 gc.Node
		regalloc(&n2, nl.Type, &n1)
		gmove(&n1, &n2)
		gins(a, nr, &n2)
		gmove(&n2, res)
		regfree(&n2)
		mfree(&n1)
	} else if nl.Ullman >= nr.Ullman {
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

uop: // unary
	var n1 gc.Node
	gc.Tempname(&n1, nl.Type)

	cgen(nl, &n1)
	gins(a, nil, &n1)
	gmove(&n1, res)
	return
}

/*
 * generate an addressable node in res, containing the value of n.
 * n is an array index, and might be any size; res width is <= 32-bit.
 * returns Prog* to patch to panic call.
 */
func igenindex(n *gc.Node, res *gc.Node, bounded int) *obj.Prog {
	if !gc.Is64(n.Type) {
		if n.Addable != 0 {
			// nothing to do.
			*res = *n
		} else {
			gc.Tempname(res, gc.Types[gc.TUINT32])
			cgen(n, res)
		}

		return nil
	}

	var tmp gc.Node
	gc.Tempname(&tmp, gc.Types[gc.TINT64])
	cgen(n, &tmp)
	var lo gc.Node
	var hi gc.Node
	split64(&tmp, &lo, &hi)
	gc.Tempname(res, gc.Types[gc.TUINT32])
	gmove(&lo, res)
	if bounded != 0 {
		splitclean()
		return nil
	}

	var zero gc.Node
	gc.Nodconst(&zero, gc.Types[gc.TINT32], 0)
	gins(i386.ACMPL, &hi, &zero)
	splitclean()
	return gc.Gbranch(i386.AJNE, nil, +1)
}

/*
 * address gen
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
		gins(i386.ALEAL, &n1, &n2)
		gmove(&n2, res)
		regfree(&n2)
		return
	}

	// addressable var is easy
	if n.Addable != 0 {
		if n.Op == gc.OREGISTER {
			gc.Fatal("agen OREGISTER")
		}
		var n1 gc.Node
		regalloc(&n1, gc.Types[gc.Tptr], res)
		gins(i386.ALEAL, n, &n1)
		gmove(&n1, res)
		regfree(&n1)
		return
	}

	// let's compute
	nl := n.Left

	nr := n.Right

	switch n.Op {
	default:
		gc.Fatal("agen %v", gc.Oconv(int(n.Op), 0))

	case gc.OCALLMETH:
		gc.Cgen_callmeth(n, 0)
		cgen_aret(n, res)

	case gc.OCALLINTER:
		cgen_callinter(n, res, 0)
		cgen_aret(n, res)

	case gc.OCALLFUNC:
		cgen_call(n, 0)
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
		p2 := (*obj.Prog)(nil) // to be patched to panicindex.
		w := uint32(n.Type.Width)
		bounded := gc.Debug['B'] != 0 || n.Bounded
		var n3 gc.Node
		var tmp gc.Node
		var n1 gc.Node
		if nr.Addable != 0 {
			// Generate &nl first, and move nr into register.
			if !gc.Isconst(nl, gc.CTSTR) {
				igen(nl, &n3, res)
			}
			if !gc.Isconst(nr, gc.CTINT) {
				p2 = igenindex(nr, &tmp, bool2int(bounded))
				regalloc(&n1, tmp.Type, nil)
				gmove(&tmp, &n1)
			}
		} else if nl.Addable != 0 {
			// Generate nr first, and move &nl into register.
			if !gc.Isconst(nr, gc.CTINT) {
				p2 = igenindex(nr, &tmp, bool2int(bounded))
				regalloc(&n1, tmp.Type, nil)
				gmove(&tmp, &n1)
			}

			if !gc.Isconst(nl, gc.CTSTR) {
				igen(nl, &n3, res)
			}
		} else {
			p2 = igenindex(nr, &tmp, bool2int(bounded))
			nr = &tmp
			if !gc.Isconst(nl, gc.CTSTR) {
				igen(nl, &n3, res)
			}
			regalloc(&n1, tmp.Type, nil)
			gins(optoas(gc.OAS, tmp.Type), &tmp, &n1)
		}

		// For fixed array we really want the pointer in n3.
		var n2 gc.Node
		if gc.Isfixedarray(nl.Type) {
			regalloc(&n2, gc.Types[gc.Tptr], &n3)
			agen(&n3, &n2)
			regfree(&n3)
			n3 = n2
		}

		// &a[0] is in n3 (allocated in res)
		// i is in n1 (if not constant)
		// len(a) is in nlen (if needed)
		// w is width

		// constant index
		if gc.Isconst(nr, gc.CTINT) {
			if gc.Isconst(nl, gc.CTSTR) {
				gc.Fatal("constant string constant index") // front end should handle
			}
			v := uint64(gc.Mpgetfix(nr.Val.U.Xval))
			if gc.Isslice(nl.Type) || nl.Type.Etype == gc.TSTRING {
				if gc.Debug['B'] == 0 && !n.Bounded {
					nlen := n3
					nlen.Type = gc.Types[gc.TUINT32]
					nlen.Xoffset += int64(gc.Array_nel)
					gc.Nodconst(&n2, gc.Types[gc.TUINT32], int64(v))
					gins(optoas(gc.OCMP, gc.Types[gc.TUINT32]), &nlen, &n2)
					p1 := gc.Gbranch(optoas(gc.OGT, gc.Types[gc.TUINT32]), nil, +1)
					ginscall(gc.Panicindex, -1)
					gc.Patch(p1, gc.Pc)
				}
			}

			// Load base pointer in n2 = n3.
			regalloc(&n2, gc.Types[gc.Tptr], &n3)

			n3.Type = gc.Types[gc.Tptr]
			n3.Xoffset += int64(gc.Array_array)
			gmove(&n3, &n2)
			regfree(&n3)
			if v*uint64(w) != 0 {
				gc.Nodconst(&n1, gc.Types[gc.Tptr], int64(v*uint64(w)))
				gins(optoas(gc.OADD, gc.Types[gc.Tptr]), &n1, &n2)
			}

			gmove(&n2, res)
			regfree(&n2)
			break
		}

		// i is in register n1, extend to 32 bits.
		t := gc.Types[gc.TUINT32]

		if gc.Issigned[n1.Type.Etype] != 0 {
			t = gc.Types[gc.TINT32]
		}

		regalloc(&n2, t, &n1) // i
		gmove(&n1, &n2)
		regfree(&n1)

		if gc.Debug['B'] == 0 && !n.Bounded {
			// check bounds
			t := gc.Types[gc.TUINT32]

			var nlen gc.Node
			if gc.Isconst(nl, gc.CTSTR) {
				gc.Nodconst(&nlen, t, int64(len(nl.Val.U.Sval.S)))
			} else if gc.Isslice(nl.Type) || nl.Type.Etype == gc.TSTRING {
				nlen = n3
				nlen.Type = t
				nlen.Xoffset += int64(gc.Array_nel)
			} else {
				gc.Nodconst(&nlen, t, nl.Type.Bound)
			}

			gins(optoas(gc.OCMP, t), &n2, &nlen)
			p1 := gc.Gbranch(optoas(gc.OLT, t), nil, +1)
			if p2 != nil {
				gc.Patch(p2, gc.Pc)
			}
			ginscall(gc.Panicindex, -1)
			gc.Patch(p1, gc.Pc)
		}

		if gc.Isconst(nl, gc.CTSTR) {
			regalloc(&n3, gc.Types[gc.Tptr], res)
			p1 := gins(i386.ALEAL, nil, &n3)
			gc.Datastring(nl.Val.U.Sval.S, &p1.From)
			p1.From.Scale = 1
			p1.From.Index = n2.Val.U.Reg
			goto indexdone
		}

		// Load base pointer in n3.
		regalloc(&tmp, gc.Types[gc.Tptr], &n3)

		if gc.Isslice(nl.Type) || nl.Type.Etype == gc.TSTRING {
			n3.Type = gc.Types[gc.Tptr]
			n3.Xoffset += int64(gc.Array_array)
			gmove(&n3, &tmp)
		}

		regfree(&n3)
		n3 = tmp

		if w == 0 {
		} else // nothing to do
		if w == 1 || w == 2 || w == 4 || w == 8 {
			// LEAL (n3)(n2*w), n3
			p1 := gins(i386.ALEAL, &n2, &n3)

			p1.From.Scale = int8(w)
			p1.From.Type = obj.TYPE_MEM
			p1.From.Index = p1.From.Reg
			p1.From.Reg = p1.To.Reg
		} else {
			gc.Nodconst(&tmp, gc.Types[gc.TUINT32], int64(w))
			gins(optoas(gc.OMUL, gc.Types[gc.TUINT32]), &tmp, &n2)
			gins(optoas(gc.OADD, gc.Types[gc.Tptr]), &n2, &n3)
		}

	indexdone:
		gmove(&n3, res)
		regfree(&n2)
		regfree(&n3)

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
			gc.Nodconst(&n1, gc.Types[gc.Tptr], n.Xoffset)
			gins(optoas(gc.OADD, gc.Types[gc.Tptr]), &n1, res)
		}

	case gc.OIND:
		cgen(nl, res)
		gc.Cgen_checknil(res)

	case gc.ODOT:
		agen(nl, res)
		if n.Xoffset != 0 {
			var n1 gc.Node
			gc.Nodconst(&n1, gc.Types[gc.Tptr], n.Xoffset)
			gins(optoas(gc.OADD, gc.Types[gc.Tptr]), &n1, res)
		}

	case gc.ODOTPTR:
		t := nl.Type
		if gc.Isptr[t.Etype] == 0 {
			gc.Fatal("agen: not ptr %v", gc.Nconv(n, 0))
		}
		cgen(nl, res)
		gc.Cgen_checknil(res)
		if n.Xoffset != 0 {
			var n1 gc.Node
			gc.Nodconst(&n1, gc.Types[gc.Tptr], n.Xoffset)
			gins(optoas(gc.OADD, gc.Types[gc.Tptr]), &n1, res)
		}
	}
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
		if n.Val.U.Reg != i386.REG_SP {
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
		switch n.Left.Op {
		// igen-able nodes.
		case gc.ODOT,
			gc.ODOTPTR,
			gc.OCALLFUNC,
			gc.OCALLMETH,
			gc.OCALLINTER:
			var n1 gc.Node
			igen(n.Left, &n1, res)

			regalloc(a, gc.Types[gc.Tptr], &n1)
			gmove(&n1, a)
			regfree(&n1)

		default:
			regalloc(a, gc.Types[gc.Tptr], res)
			cgen(n.Left, a)
		}

		gc.Cgen_checknil(a)
		a.Op = gc.OINDREG
		a.Xoffset += n.Xoffset
		a.Type = n.Type
		return

	case gc.OCALLFUNC,
		gc.OCALLMETH,
		gc.OCALLINTER:
		switch n.Op {
		case gc.OCALLFUNC:
			cgen_call(n, 0)

		case gc.OCALLMETH:
			gc.Cgen_callmeth(n, 0)

		case gc.OCALLINTER:
			cgen_callinter(n, nil, 0)
		}

		var flist gc.Iter
		fp := gc.Structfirst(&flist, gc.Getoutarg(n.Left.Type))
		*a = gc.Node{}
		a.Op = gc.OINDREG
		a.Val.U.Reg = i386.REG_SP
		a.Addable = 1
		a.Xoffset = fp.Width
		a.Type = n.Type
		return

		// Index of fixed-size array by constant can
	// put the offset in the addressing.
	// Could do the same for slice except that we need
	// to use the real index for the bounds checking.
	case gc.OINDEX:
		if gc.Isfixedarray(n.Left.Type) || (gc.Isptr[n.Left.Type.Etype] != 0 && gc.Isfixedarray(n.Left.Left.Type)) {
			if gc.Isconst(n.Right, gc.CTINT) {
				// Compute &a.
				if gc.Isptr[n.Left.Type.Etype] == 0 {
					igen(n.Left, a, res)
				} else {
					var n1 gc.Node
					igen(n.Left, &n1, res)
					gc.Cgen_checknil(&n1)
					regalloc(a, gc.Types[gc.Tptr], res)
					gmove(&n1, a)
					regfree(&n1)
					a.Op = gc.OINDREG
				}

				// Compute &a[i] as &a + i*width.
				a.Type = n.Type

				a.Xoffset += gc.Mpgetfix(n.Right.Val.U.Xval) * n.Type.Width
				return
			}
		}
	}

	// release register for now, to avoid
	// confusing tempname.
	if res != nil && res.Op == gc.OREGISTER {
		reg[res.Val.U.Reg]--
	}
	var n1 gc.Node
	gc.Tempname(&n1, gc.Types[gc.Tptr])
	agen(n, &n1)
	if res != nil && res.Op == gc.OREGISTER {
		reg[res.Val.U.Reg]++
	}
	regalloc(a, gc.Types[gc.Tptr], res)
	gmove(&n1, a)
	a.Op = gc.OINDREG
	a.Type = n.Type
}

/*
 * branch gen
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

	if n.Type == nil {
		gc.Convlit(&n, gc.Types[gc.TBOOL])
		if n.Type == nil {
			return
		}
	}

	et := int(n.Type.Etype)
	if et != gc.TBOOL {
		gc.Yyerror("cgen: bad type %v for %v", gc.Tconv(n.Type, 0), gc.Oconv(int(n.Op), 0))
		gc.Patch(gins(obj.AEND, nil, nil), to)
		return
	}

	for n.Op == gc.OCONVNOP {
		n = n.Left
		if n.Ninit != nil {
			gc.Genlist(n.Ninit)
		}
	}

	nl := n.Left
	nr := (*gc.Node)(nil)

	if nl != nil && gc.Isfloat[nl.Type.Etype] != 0 {
		bgen_float(n, bool2int(true_), likely, to)
		return
	}

	switch n.Op {
	default:
		goto def

		// need to ask if it is bool?
	case gc.OLITERAL:
		if !true_ == (n.Val.U.Bval == 0) {
			gc.Patch(gc.Gbranch(obj.AJMP, nil, 0), to)
		}
		return

	case gc.ONAME:
		if n.Addable == 0 {
			goto def
		}
		var n1 gc.Node
		gc.Nodconst(&n1, n.Type, 0)
		gins(optoas(gc.OCMP, n.Type), n, &n1)
		a := i386.AJNE
		if !true_ {
			a = i386.AJEQ
		}
		gc.Patch(gc.Gbranch(a, n.Type, likely), to)
		return

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

		return

	case gc.OEQ,
		gc.ONE,
		gc.OLT,
		gc.OGT,
		gc.OLE,
		gc.OGE:
		nr = n.Right
		if nr == nil || nr.Type == nil {
			return
		}
		fallthrough

	case gc.ONOT: // unary
		nl = n.Left

		if nl == nil || nl.Type == nil {
			return
		}
	}

	switch n.Op {
	case gc.ONOT:
		bgen(nl, !true_, likely, to)

	case gc.OEQ,
		gc.ONE,
		gc.OLT,
		gc.OGT,
		gc.OLE,
		gc.OGE:
		a := int(n.Op)
		if !true_ {
			a = gc.Brcom(a)
			true_ = !true_
		}

		// make simplest on right
		if nl.Op == gc.OLITERAL || (nl.Ullman < nr.Ullman && nl.Ullman < gc.UINF) {
			a = gc.Brrev(a)
			r := nl
			nl = nr
			nr = r
		}

		if gc.Isslice(nl.Type) {
			// front end should only leave cmp to literal nil
			if (a != gc.OEQ && a != gc.ONE) || nr.Op != gc.OLITERAL {
				gc.Yyerror("illegal slice comparison")
				break
			}

			a = optoas(a, gc.Types[gc.Tptr])
			var n1 gc.Node
			igen(nl, &n1, nil)
			n1.Xoffset += int64(gc.Array_array)
			n1.Type = gc.Types[gc.Tptr]
			var tmp gc.Node
			gc.Nodconst(&tmp, gc.Types[gc.Tptr], 0)
			gins(optoas(gc.OCMP, gc.Types[gc.Tptr]), &n1, &tmp)
			gc.Patch(gc.Gbranch(a, gc.Types[gc.Tptr], likely), to)
			regfree(&n1)
			break
		}

		if gc.Isinter(nl.Type) {
			// front end should only leave cmp to literal nil
			if (a != gc.OEQ && a != gc.ONE) || nr.Op != gc.OLITERAL {
				gc.Yyerror("illegal interface comparison")
				break
			}

			a = optoas(a, gc.Types[gc.Tptr])
			var n1 gc.Node
			igen(nl, &n1, nil)
			n1.Type = gc.Types[gc.Tptr]
			var tmp gc.Node
			gc.Nodconst(&tmp, gc.Types[gc.Tptr], 0)
			gins(optoas(gc.OCMP, gc.Types[gc.Tptr]), &n1, &tmp)
			gc.Patch(gc.Gbranch(a, gc.Types[gc.Tptr], likely), to)
			regfree(&n1)
			break
		}

		if gc.Iscomplex[nl.Type.Etype] != 0 {
			gc.Complexbool(a, nl, nr, true_, likely, to)
			break
		}

		if gc.Is64(nr.Type) {
			if nl.Addable == 0 || gc.Isconst(nl, gc.CTINT) {
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

		var n2 gc.Node
		if nr.Ullman >= gc.UINF {
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

			var n2 gc.Node
			regalloc(&n2, nr.Type, nil)
			cgen(nr, &n2)
			nr = &n2
			goto cmp
		}

		if nl.Addable == 0 {
			var n1 gc.Node
			gc.Tempname(&n1, nl.Type)
			cgen(nl, &n1)
			nl = &n1
		}

		if gc.Smallintconst(nr) {
			gins(optoas(gc.OCMP, nr.Type), nl, nr)
			gc.Patch(gc.Gbranch(optoas(a, nr.Type), nr.Type, likely), to)
			break
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

	cmp:
		gins(optoas(gc.OCMP, nr.Type), nl, nr)
		gc.Patch(gc.Gbranch(optoas(a, nr.Type), nr.Type, likely), to)

		if nl.Op == gc.OREGISTER {
			regfree(nl)
		}
		regfree(nr)
	}

	return

def:
	var n1 gc.Node
	regalloc(&n1, n.Type, nil)
	cgen(n, &n1)
	var n2 gc.Node
	gc.Nodconst(&n2, n.Type, 0)
	gins(optoas(gc.OCMP, n.Type), &n1, &n2)
	a := i386.AJNE
	if !true_ {
		a = i386.AJEQ
	}
	gc.Patch(gc.Gbranch(a, n.Type, likely), to)
	regfree(&n1)
	return
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
			return int32(t.Width)
		}
	}

	// botch - probably failing to recognize address
	// arithmetic on the above. eg INDEX and DOT
	return -1000
}

/*
 * struct gen
 *	memmove(&res, &n, w);
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

	if w == 0 {
		// evaluate side effects only.
		var tdst gc.Node
		gc.Tempname(&tdst, gc.Types[gc.Tptr])

		agen(res, &tdst)
		agen(n, &tdst)
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

	// offset on the stack
	osrc := stkof(n)

	odst := stkof(res)

	if osrc != -1000 && odst != -1000 && (osrc == 1000 || odst == 1000) {
		// osrc and odst both on stack, and at least one is in
		// an unknown position.  Could generate code to test
		// for forward/backward copy, but instead just copy
		// to a temporary location first.
		var tsrc gc.Node
		gc.Tempname(&tsrc, n.Type)

		sgen(n, &tsrc, w)
		sgen(&tsrc, res, w)
		return
	}

	var dst gc.Node
	gc.Nodreg(&dst, gc.Types[gc.Tptr], i386.REG_DI)
	var src gc.Node
	gc.Nodreg(&src, gc.Types[gc.Tptr], i386.REG_SI)

	var tsrc gc.Node
	gc.Tempname(&tsrc, gc.Types[gc.Tptr])
	var tdst gc.Node
	gc.Tempname(&tdst, gc.Types[gc.Tptr])
	if n.Addable == 0 {
		agen(n, &tsrc)
	}
	if res.Addable == 0 {
		agen(res, &tdst)
	}
	if n.Addable != 0 {
		agen(n, &src)
	} else {
		gmove(&tsrc, &src)
	}

	if res.Op == gc.ONAME {
		gc.Gvardef(res)
	}

	if res.Addable != 0 {
		agen(res, &dst)
	} else {
		gmove(&tdst, &dst)
	}

	c := int32(w % 4) // bytes
	q := int32(w / 4) // doublewords

	// if we are copying forward on the stack and
	// the src and dst overlap, then reverse direction
	if osrc < odst && int64(odst) < int64(osrc)+w {
		// reverse direction
		gins(i386.ASTD, nil, nil) // set direction flag
		if c > 0 {
			gconreg(i386.AADDL, w-1, i386.REG_SI)
			gconreg(i386.AADDL, w-1, i386.REG_DI)

			gconreg(i386.AMOVL, int64(c), i386.REG_CX)
			gins(i386.AREP, nil, nil)   // repeat
			gins(i386.AMOVSB, nil, nil) // MOVB *(SI)-,*(DI)-
		}

		if q > 0 {
			if c > 0 {
				gconreg(i386.AADDL, -3, i386.REG_SI)
				gconreg(i386.AADDL, -3, i386.REG_DI)
			} else {
				gconreg(i386.AADDL, w-4, i386.REG_SI)
				gconreg(i386.AADDL, w-4, i386.REG_DI)
			}

			gconreg(i386.AMOVL, int64(q), i386.REG_CX)
			gins(i386.AREP, nil, nil)   // repeat
			gins(i386.AMOVSL, nil, nil) // MOVL *(SI)-,*(DI)-
		}

		// we leave with the flag clear
		gins(i386.ACLD, nil, nil)
	} else {
		gins(i386.ACLD, nil, nil) // paranoia.  TODO(rsc): remove?

		// normal direction
		if q > 128 || (q >= 4 && gc.Nacl) {
			gconreg(i386.AMOVL, int64(q), i386.REG_CX)
			gins(i386.AREP, nil, nil)   // repeat
			gins(i386.AMOVSL, nil, nil) // MOVL *(SI)+,*(DI)+
		} else if q >= 4 {
			p := gins(obj.ADUFFCOPY, nil, nil)
			p.To.Type = obj.TYPE_ADDR
			p.To.Sym = gc.Linksym(gc.Pkglookup("duffcopy", gc.Runtimepkg))

			// 10 and 128 = magic constants: see ../../runtime/asm_386.s
			p.To.Offset = 10 * (128 - int64(q))
		} else if !gc.Nacl && c == 0 {
			var cx gc.Node
			gc.Nodreg(&cx, gc.Types[gc.TINT32], i386.REG_CX)

			// We don't need the MOVSL side-effect of updating SI and DI,
			// and issuing a sequence of MOVLs directly is faster.
			src.Op = gc.OINDREG

			dst.Op = gc.OINDREG
			for q > 0 {
				gmove(&src, &cx) // MOVL x+(SI),CX
				gmove(&cx, &dst) // MOVL CX,x+(DI)
				src.Xoffset += 4
				dst.Xoffset += 4
				q--
			}
		} else {
			for q > 0 {
				gins(i386.AMOVSL, nil, nil) // MOVL *(SI)+,*(DI)+
				q--
			}
		}

		for c > 0 {
			gins(i386.AMOVSB, nil, nil) // MOVB *(SI)+,*(DI)+
			c--
		}
	}
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
 * return 1 if can do, 0 if can't.
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
