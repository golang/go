// Copyright 2009 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package main

import (
	"cmd/internal/obj"
	"cmd/internal/obj/ppc64"
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
	//print("cgen %N(%d) -> %N(%d)\n", n, n->addable, res, res->addable);
	if gc.Debug['g'] != 0 {
		gc.Dump("\ncgen-n", n)
		gc.Dump("cgen-res", res)
	}

	var a int
	var nr *gc.Node
	var nl *gc.Node
	var n1 gc.Node
	var n2 gc.Node
	if n == nil || n.Type == nil {
		goto ret
	}

	if res == nil || res.Type == nil {
		gc.Fatal("cgen: res nil")
	}

	for n.Op == gc.OCONVNOP {
		n = n.Left
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
		goto ret

	case gc.OEFACE:
		if res.Op != gc.ONAME || res.Addable == 0 {
			var n1 gc.Node
			gc.Tempname(&n1, n.Type)
			gc.Cgen_eface(n, &n1)
			cgen(&n1, res)
		} else {
			gc.Cgen_eface(n, res)
		}
		goto ret
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

	if res.Addable == 0 {
		if n.Ullman > res.Ullman {
			var n1 gc.Node
			regalloc(&n1, n.Type, res)
			cgen(n, &n1)
			if n1.Ullman > res.Ullman {
				gc.Dump("n1", &n1)
				gc.Dump("res", res)
				gc.Fatal("loop in cgen")
			}

			cgen(&n1, res)
			regfree(&n1)
			goto ret
		}

		var f int
		if res.Ullman >= gc.UINF {
			goto gen
		}

		if gc.Complexop(n, res) {
			gc.Complexgen(n, res)
			goto ret
		}

		f = 1 // gen thru register
		switch n.Op {
		case gc.OLITERAL:
			if gc.Smallintconst(n) {
				f = 0
			}

		case gc.OREGISTER:
			f = 0
		}

		if gc.Iscomplex[n.Type.Etype] == 0 {
			a := optoas(gc.OAS, res.Type)
			var addr obj.Addr
			if sudoaddable(a, res, &addr) {
				var p1 *obj.Prog
				if f != 0 {
					var n2 gc.Node
					regalloc(&n2, res.Type, nil)
					cgen(n, &n2)
					p1 = gins(a, &n2, nil)
					regfree(&n2)
				} else {
					p1 = gins(a, n, nil)
				}
				p1.To = addr
				if gc.Debug['g'] != 0 {
					fmt.Printf("%v [ignore previous line]\n", p1)
				}
				sudoclean()
				goto ret
			}
		}

	gen:
		var n1 gc.Node
		igen(res, &n1, nil)
		cgen(n, &n1)
		regfree(&n1)
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

	if gc.Complexop(n, res) {
		gc.Complexgen(n, res)
		goto ret
	}

	// if both are addressable, move
	if n.Addable != 0 {
		if n.Op == gc.OREGISTER || res.Op == gc.OREGISTER {
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

	if gc.Iscomplex[n.Type.Etype] == 0 {
		a := optoas(gc.OAS, n.Type)
		var addr obj.Addr
		if sudoaddable(a, n, &addr) {
			if res.Op == gc.OREGISTER {
				p1 := gins(a, nil, res)
				p1.From = addr
			} else {
				var n2 gc.Node
				regalloc(&n2, n.Type, nil)
				p1 := gins(a, nil, &n2)
				p1.From = addr
				gins(a, &n2, res)
				regfree(&n2)
			}

			sudoclean()
			goto ret
		}
	}

	// TODO(minux): we shouldn't reverse FP comparisons, but then we need to synthesize
	// OGE, OLE, and ONE ourselves.
	// if(nl != N && isfloat[n->type->etype] && isfloat[nl->type->etype]) goto flt;

	switch n.Op {
	default:
		gc.Dump("cgen", n)
		gc.Fatal("cgen: unknown op %v", gc.Nconv(n, obj.FmtShort|obj.FmtSign))

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
		p1 := gc.Gbranch(ppc64.ABR, nil, 0)

		p2 := gc.Pc
		gmove(gc.Nodbool(true), res)
		p3 := gc.Gbranch(ppc64.ABR, nil, 0)
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

		var n1 gc.Node
		regalloc(&n1, nl.Type, nil)
		cgen(nl, &n1)
		var n2 gc.Node
		gc.Nodconst(&n2, nl.Type, -1)
		gins(a, &n2, &n1)
		gmove(&n1, res)
		regfree(&n1)
		goto ret

	case gc.OMINUS:
		if gc.Isfloat[nl.Type.Etype] != 0 {
			nr = gc.Nodintconst(-1)
			gc.Convlit(&nr, n.Type)
			a = optoas(gc.OMUL, nl.Type)
			goto sbop
		}

		a = optoas(int(n.Op), nl.Type)
		goto uop

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

	case gc.OCONV:
		if n.Type.Width > nl.Type.Width {
			// If loading from memory, do conversion during load,
			// so as to avoid use of 8-bit register in, say, int(*byteptr).
			switch nl.Op {
			case gc.ODOT,
				gc.ODOTPTR,
				gc.OINDEX,
				gc.OIND,
				gc.ONAME:
				var n1 gc.Node
				igen(nl, &n1, res)
				var n2 gc.Node
				regalloc(&n2, n.Type, res)
				gmove(&n1, &n2)
				gmove(&n2, res)
				regfree(&n2)
				regfree(&n1)
				goto ret
			}
		}

		var n1 gc.Node
		regalloc(&n1, nl.Type, res)
		var n2 gc.Node
		regalloc(&n2, n.Type, &n1)
		cgen(nl, &n1)

		// if we do the conversion n1 -> n2 here
		// reusing the register, then gmove won't
		// have to allocate its own register.
		gmove(&n1, &n2)

		gmove(&n2, res)
		regfree(&n2)
		regfree(&n1)

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
			p1 := gins(ppc64.AMOVD, nil, &n1)
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
			// map and chan have len in the first int-sized word.
			// a zero pointer means zero length
			var n1 gc.Node
			regalloc(&n1, gc.Types[gc.Tptr], res)

			cgen(nl, &n1)

			var n2 gc.Node
			gc.Nodconst(&n2, gc.Types[gc.Tptr], 0)
			gins(optoas(gc.OCMP, gc.Types[gc.Tptr]), &n1, &n2)
			p1 := gc.Gbranch(optoas(gc.OEQ, gc.Types[gc.Tptr]), nil, 0)

			n2 = n1
			n2.Op = gc.OINDREG
			n2.Type = gc.Types[gc.Simtype[gc.TINT]]
			gmove(&n2, &n1)

			gc.Patch(p1, gc.Pc)

			gmove(&n1, res)
			regfree(&n1)
			break
		}

		if gc.Istype(nl.Type, gc.TSTRING) || gc.Isslice(nl.Type) {
			// both slice and string have len one pointer into the struct.
			// a zero pointer means zero length
			var n1 gc.Node
			igen(nl, &n1, res)

			n1.Type = gc.Types[gc.Simtype[gc.TUINT]]
			n1.Xoffset += int64(gc.Array_nel)
			gmove(&n1, res)
			regfree(&n1)
			break
		}

		gc.Fatal("cgen: OLEN: unknown type %v", gc.Tconv(nl.Type, obj.FmtLong))

	case gc.OCAP:
		if gc.Istype(nl.Type, gc.TCHAN) {
			// chan has cap in the second int-sized word.
			// a zero pointer means zero length
			var n1 gc.Node
			regalloc(&n1, gc.Types[gc.Tptr], res)

			cgen(nl, &n1)

			var n2 gc.Node
			gc.Nodconst(&n2, gc.Types[gc.Tptr], 0)
			gins(optoas(gc.OCMP, gc.Types[gc.Tptr]), &n1, &n2)
			p1 := gc.Gbranch(optoas(gc.OEQ, gc.Types[gc.Tptr]), nil, 0)

			n2 = n1
			n2.Op = gc.OINDREG
			n2.Xoffset = int64(gc.Widthint)
			n2.Type = gc.Types[gc.Simtype[gc.TINT]]
			gmove(&n2, &n1)

			gc.Patch(p1, gc.Pc)

			gmove(&n1, res)
			regfree(&n1)
			break
		}

		if gc.Isslice(nl.Type) {
			var n1 gc.Node
			igen(nl, &n1, res)
			n1.Type = gc.Types[gc.Simtype[gc.TUINT]]
			n1.Xoffset += int64(gc.Array_cap)
			gmove(&n1, res)
			regfree(&n1)
			break
		}

		gc.Fatal("cgen: OCAP: unknown type %v", gc.Tconv(nl.Type, obj.FmtLong))

	case gc.OADDR:
		if n.Bounded { // let race detector avoid nil checks
			gc.Disable_checknil++
		}
		agen(nl, res)
		if n.Bounded {
			gc.Disable_checknil--
		}

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
		if gc.Isfloat[n.Type.Etype] != 0 {
			a = optoas(int(n.Op), nl.Type)
			goto abop
		}

		if nl.Ullman >= nr.Ullman {
			var n1 gc.Node
			regalloc(&n1, nl.Type, res)
			cgen(nl, &n1)
			cgen_div(int(n.Op), &n1, nr, res)
			regfree(&n1)
		} else {
			var n2 gc.Node
			if !gc.Smallintconst(nr) {
				regalloc(&n2, nr.Type, res)
				cgen(nr, &n2)
			} else {
				n2 = *nr
			}

			cgen_div(int(n.Op), nl, &n2, res)
			if n2.Op != gc.OLITERAL {
				regfree(&n2)
			}
		}

	case gc.OLSH,
		gc.ORSH,
		gc.OLROT:
		cgen_shift(int(n.Op), n.Bounded, nl, nr, res)
	}

	goto ret

	/*
	 * put simplest on right - we'll generate into left
	 * and then adjust it using the computation of right.
	 * constants and variables have the same ullman
	 * count, so look for constants specially.
	 *
	 * an integer constant we can use as an immediate
	 * is simpler than a variable - we can use the immediate
	 * in the adjustment instruction directly - so it goes
	 * on the right.
	 *
	 * other constants, like big integers or floating point
	 * constants, require a mov into a register, so those
	 * might as well go on the left, so we can reuse that
	 * register for the computation.
	 */
sbop: // symmetric binary
	if nl.Ullman < nr.Ullman || (nl.Ullman == nr.Ullman && (gc.Smallintconst(nl) || (nr.Op == gc.OLITERAL && !gc.Smallintconst(nr)))) {
		r := nl
		nl = nr
		nr = r
	}

abop: // asymmetric binary
	if nl.Ullman >= nr.Ullman {
		regalloc(&n1, nl.Type, res)
		cgen(nl, &n1)

		/*
			 * This generates smaller code - it avoids a MOV - but it's
			 * easily 10% slower due to not being able to
			 * optimize/manipulate the move.
			 * To see, run: go test -bench . crypto/md5
			 * with and without.
			 *
				if(sudoaddable(a, nr, &addr)) {
					p1 = gins(a, N, &n1);
					p1->from = addr;
					gmove(&n1, res);
					sudoclean();
					regfree(&n1);
					goto ret;
				}
			 *
		*/
		// TODO(minux): enable using constants directly in certain instructions.
		//if(smallintconst(nr))
		//	n2 = *nr;
		//else {
		regalloc(&n2, nr.Type, nil)

		cgen(nr, &n2)
	} else //}
	{
		//if(smallintconst(nr))
		//	n2 = *nr;
		//else {
		regalloc(&n2, nr.Type, res)

		cgen(nr, &n2)

		//}
		regalloc(&n1, nl.Type, nil)

		cgen(nl, &n1)
	}

	gins(a, &n2, &n1)

	// Normalize result for types smaller than word.
	if n.Type.Width < int64(gc.Widthreg) {
		switch n.Op {
		case gc.OADD,
			gc.OSUB,
			gc.OMUL,
			gc.OLSH:
			gins(optoas(gc.OAS, n.Type), &n1, &n1)
		}
	}

	gmove(&n1, res)
	regfree(&n1)
	if n2.Op != gc.OLITERAL {
		regfree(&n2)
	}
	goto ret

uop: // unary
	regalloc(&n1, nl.Type, res)

	cgen(nl, &n1)
	gins(a, nil, &n1)
	gmove(&n1, res)
	regfree(&n1)
	goto ret

ret:
}

/*
 * allocate a register (reusing res if possible) and generate
 *  a = n
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
		regalloc(a, n.Type, res)
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
 * allocate a register (reusing res if possible) and generate
 * a = &n
 * The caller must call regfree(a).
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

		//bounded = debug['B'] || n->bounded;
		var n3 gc.Node
		var n1 gc.Node
		if nr.Addable != 0 {
			var tmp gc.Node
			if !gc.Isconst(nr, gc.CTINT) {
				gc.Tempname(&tmp, gc.Types[gc.TINT64])
			}
			if !gc.Isconst(nl, gc.CTSTR) {
				agenr(nl, &n3, res)
			}
			if !gc.Isconst(nr, gc.CTINT) {
				cgen(nr, &tmp)
				regalloc(&n1, tmp.Type, nil)
				gmove(&tmp, &n1)
			}
		} else if nl.Addable != 0 {
			if !gc.Isconst(nr, gc.CTINT) {
				var tmp gc.Node
				gc.Tempname(&tmp, gc.Types[gc.TINT64])
				cgen(nr, &tmp)
				regalloc(&n1, tmp.Type, nil)
				gmove(&tmp, &n1)
			}

			if !gc.Isconst(nl, gc.CTSTR) {
				agenr(nl, &n3, res)
			}
		} else {
			var tmp gc.Node
			gc.Tempname(&tmp, gc.Types[gc.TINT64])
			cgen(nr, &tmp)
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
			if gc.Isslice(nl.Type) || nl.Type.Etype == gc.TSTRING {
				if gc.Debug['B'] == 0 && !n.Bounded {
					n1 = n3
					n1.Op = gc.OINDREG
					n1.Type = gc.Types[gc.Tptr]
					n1.Xoffset = int64(gc.Array_nel)
					var n4 gc.Node
					regalloc(&n4, n1.Type, nil)
					gmove(&n1, &n4)
					ginscon2(optoas(gc.OCMP, gc.Types[gc.TUINT64]), &n4, int64(v))
					regfree(&n4)
					p1 := gc.Gbranch(optoas(gc.OGT, gc.Types[gc.TUINT64]), nil, +1)
					ginscall(gc.Panicindex, 0)
					gc.Patch(p1, gc.Pc)
				}

				n1 = n3
				n1.Op = gc.OINDREG
				n1.Type = gc.Types[gc.Tptr]
				n1.Xoffset = int64(gc.Array_array)
				gmove(&n1, &n3)
			}

			if v*uint64(w) != 0 {
				ginscon(optoas(gc.OADD, gc.Types[gc.Tptr]), int64(v*uint64(w)), &n3)
			}

			*a = n3
			break
		}

		var n2 gc.Node
		regalloc(&n2, gc.Types[gc.TINT64], &n1) // i
		gmove(&n1, &n2)
		regfree(&n1)

		var n4 gc.Node
		if gc.Debug['B'] == 0 && !n.Bounded {
			// check bounds
			if gc.Isconst(nl, gc.CTSTR) {
				gc.Nodconst(&n4, gc.Types[gc.TUINT64], int64(len(nl.Val.U.Sval.S)))
			} else if gc.Isslice(nl.Type) || nl.Type.Etype == gc.TSTRING {
				n1 = n3
				n1.Op = gc.OINDREG
				n1.Type = gc.Types[gc.Tptr]
				n1.Xoffset = int64(gc.Array_nel)
				regalloc(&n4, gc.Types[gc.TUINT64], nil)
				gmove(&n1, &n4)
			} else {
				if nl.Type.Bound < (1<<15)-1 {
					gc.Nodconst(&n4, gc.Types[gc.TUINT64], nl.Type.Bound)
				} else {
					regalloc(&n4, gc.Types[gc.TUINT64], nil)
					p1 := gins(ppc64.AMOVD, nil, &n4)
					p1.From.Type = obj.TYPE_CONST
					p1.From.Offset = nl.Type.Bound
				}
			}

			gins(optoas(gc.OCMP, gc.Types[gc.TUINT64]), &n2, &n4)
			if n4.Op == gc.OREGISTER {
				regfree(&n4)
			}
			p1 := gc.Gbranch(optoas(gc.OLT, gc.Types[gc.TUINT64]), nil, +1)
			if p2 != nil {
				gc.Patch(p2, gc.Pc)
			}
			ginscall(gc.Panicindex, 0)
			gc.Patch(p1, gc.Pc)
		}

		if gc.Isconst(nl, gc.CTSTR) {
			regalloc(&n3, gc.Types[gc.Tptr], res)
			p1 := gins(ppc64.AMOVD, nil, &n3)
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
		if w == 1 {
			/* w already scaled */
			gins(optoas(gc.OADD, gc.Types[gc.Tptr]), &n2, &n3)
			/* else if(w == 2 || w == 4 || w == 8) {
				// TODO(minux): scale using shift
			} */
		} else {
			regalloc(&n4, gc.Types[gc.TUINT64], nil)
			gc.Nodconst(&n1, gc.Types[gc.TUINT64], int64(w))
			gmove(&n1, &n4)
			gins(optoas(gc.OMUL, gc.Types[gc.TUINT64]), &n4, &n2)
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

func ginsadd(as int, off int64, dst *gc.Node) {
	var n1 gc.Node

	regalloc(&n1, gc.Types[gc.Tptr], dst)
	gmove(dst, &n1)
	ginscon(as, off, &n1)
	gmove(&n1, dst)
	regfree(&n1)
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

	if n == nil || n.Type == nil {
		return
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
		n3 := gc.Node{}
		n3.Op = gc.OADDR
		n3.Left = &n1
		gins(ppc64.AMOVD, &n3, &n2)
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
		gins(ppc64.AMOVD, &n1, &n2)
		gmove(&n2, res)
		regfree(&n2)
		goto ret
	}

	nl = n.Left

	switch n.Op {
	default:
		gc.Fatal("agen: unknown op %v", gc.Nconv(n, obj.FmtShort|obj.FmtSign))

		// TODO(minux): 5g has this: Release res so that it is available for cgen_call.
	// Pick it up again after the call for OCALLMETH and OCALLFUNC.
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
			ginsadd(optoas(gc.OADD, gc.Types[gc.Tptr]), n.Xoffset, res)
		}

	case gc.OIND:
		cgen(nl, res)
		gc.Cgen_checknil(res)

	case gc.ODOT:
		agen(nl, res)
		if n.Xoffset != 0 {
			ginsadd(optoas(gc.OADD, gc.Types[gc.Tptr]), n.Xoffset, res)
		}

	case gc.ODOTPTR:
		cgen(nl, res)
		gc.Cgen_checknil(res)
		if n.Xoffset != 0 {
			ginsadd(optoas(gc.OADD, gc.Types[gc.Tptr]), n.Xoffset, res)
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
		if n.Val.U.Reg != ppc64.REGSP {
			reg[n.Val.U.Reg]++
		}
		*a = *n
		return

	case gc.ODOT:
		igen(n.Left, a, res)
		a.Xoffset += n.Xoffset
		a.Type = n.Type
		fixlargeoffset(a)
		return

	case gc.ODOTPTR:
		cgenr(n.Left, a, res)
		gc.Cgen_checknil(a)
		a.Op = gc.OINDREG
		a.Xoffset += n.Xoffset
		a.Type = n.Type
		fixlargeoffset(a)
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
		a.Val.U.Reg = ppc64.REGSP
		a.Addable = 1
		a.Xoffset = fp.Width + int64(gc.Widthptr) // +widthptr: saved lr at 0(SP)
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
				fixlargeoffset(a)
				return
			}
		}
	}

	agenr(n, a, res)
	a.Op = gc.OINDREG
	a.Type = n.Type
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

	for n.Op == gc.OCONVNOP {
		n = n.Left
		if n.Ninit != nil {
			gc.Genlist(n.Ninit)
		}
	}

	switch n.Op {
	default:
		var n1 gc.Node
		regalloc(&n1, n.Type, nil)
		cgen(n, &n1)
		var n2 gc.Node
		gc.Nodconst(&n2, n.Type, 0)
		gins(optoas(gc.OCMP, n.Type), &n1, &n2)
		a := ppc64.ABNE
		if !true_ {
			a = ppc64.ABEQ
		}
		gc.Patch(gc.Gbranch(a, n.Type, likely), to)
		regfree(&n1)
		goto ret

		// need to ask if it is bool?
	case gc.OLITERAL:
		if !true_ == (n.Val.U.Bval == 0) {
			gc.Patch(gc.Gbranch(ppc64.ABR, nil, likely), to)
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
			if gc.Isfloat[nr.Type.Etype] != 0 {
				// brcom is not valid on floats when NaN is involved.
				p1 := gc.Gbranch(ppc64.ABR, nil, 0)

				p2 := gc.Gbranch(ppc64.ABR, nil, 0)
				gc.Patch(p1, gc.Pc)
				ll := n.Ninit // avoid re-genning ninit
				n.Ninit = nil
				bgen(n, true, -likely, p2)
				n.Ninit = ll
				gc.Patch(gc.Gbranch(ppc64.ABR, nil, 0), to)
				gc.Patch(p2, gc.Pc)
				goto ret
			}

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
			var n2 gc.Node
			regalloc(&n2, gc.Types[gc.Tptr], &n1)
			gmove(&n1, &n2)
			gins(optoas(gc.OCMP, gc.Types[gc.Tptr]), &n2, &tmp)
			regfree(&n2)
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
			var n2 gc.Node
			regalloc(&n2, gc.Types[gc.Tptr], &n1)
			gmove(&n1, &n2)
			gins(optoas(gc.OCMP, gc.Types[gc.Tptr]), &n2, &tmp)
			regfree(&n2)
			gc.Patch(gc.Gbranch(a, gc.Types[gc.Tptr], likely), to)
			regfree(&n1)
			break
		}

		if gc.Iscomplex[nl.Type.Etype] != 0 {
			gc.Complexbool(a, nl, nr, true_, likely, to)
			break
		}

		var n1 gc.Node
		var n2 gc.Node
		if nr.Ullman >= gc.UINF {
			regalloc(&n1, nl.Type, nil)
			cgen(nl, &n1)

			var tmp gc.Node
			gc.Tempname(&tmp, nl.Type)
			gmove(&n1, &tmp)
			regfree(&n1)

			regalloc(&n2, nr.Type, nil)
			cgen(nr, &n2)

			regalloc(&n1, nl.Type, nil)
			cgen(&tmp, &n1)

			goto cmp
		}

		regalloc(&n1, nl.Type, nil)
		cgen(nl, &n1)

		// TODO(minux): cmpi does accept 16-bit signed immediate as p->to.
		// and cmpli accepts 16-bit unsigned immediate.
		//if(smallintconst(nr)) {
		//	gins(optoas(OCMP, nr->type), &n1, nr);
		//	patch(gbranch(optoas(a, nr->type), nr->type, likely), to);
		//	regfree(&n1);
		//	break;
		//}

		regalloc(&n2, nr.Type, nil)

		cgen(nr, &n2)

	cmp:
		l := &n1
		r := &n2
		gins(optoas(gc.OCMP, nr.Type), l, r)
		if gc.Isfloat[nr.Type.Etype] != 0 && (a == gc.OLE || a == gc.OGE) {
			// To get NaN right, must rewrite x <= y into separate x < y or x = y.
			switch a {
			case gc.OLE:
				a = gc.OLT

			case gc.OGE:
				a = gc.OGT
			}

			gc.Patch(gc.Gbranch(optoas(a, nr.Type), nr.Type, likely), to)
			gc.Patch(gc.Gbranch(optoas(gc.OEQ, nr.Type), nr.Type, likely), to)
		} else {
			gc.Patch(gc.Gbranch(optoas(a, nr.Type), nr.Type, likely), to)
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
func stkof(n *gc.Node) int64 {
	switch n.Op {
	case gc.OINDREG:
		return n.Xoffset

	case gc.ODOT:
		t := n.Left.Type
		if gc.Isptr[t.Etype] != 0 {
			break
		}
		off := stkof(n.Left)
		if off == -1000 || off == 1000 {
			return off
		}
		return off + n.Xoffset

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
			return off + t.Type.Width*gc.Mpgetfix(n.Right.Val.U.Xval)
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
			return t.Width + int64(gc.Widthptr) // +widthptr: correct for saved LR
		}
	}

	// botch - probably failing to recognize address
	// arithmetic on the above. eg INDEX and DOT
	return -1000
}

/*
 * block copy:
 *	memmove(&ns, &n, w);
 */
func sgen(n *gc.Node, ns *gc.Node, w int64) {
	var res *gc.Node = ns

	if gc.Debug['g'] != 0 {
		fmt.Printf("\nsgen w=%d\n", w)
		gc.Dump("r", n)
		gc.Dump("res", ns)
	}

	if n.Ullman >= gc.UINF && ns.Ullman >= gc.UINF {
		gc.Fatal("sgen UINF")
	}

	if w < 0 {
		gc.Fatal("sgen copy %d", w)
	}

	// If copying .args, that's all the results, so record definition sites
	// for them for the liveness analysis.
	if ns.Op == gc.ONAME && ns.Sym.Name == ".args" {
		for l := gc.Curfn.Dcl; l != nil; l = l.Next {
			if l.N.Class == gc.PPARAMOUT {
				gc.Gvardef(l.N)
			}
		}
	}

	// Avoid taking the address for simple enough types.
	//if(componentgen(n, ns))
	//	return;
	if w == 0 {
		// evaluate side effects only.
		var dst gc.Node
		regalloc(&dst, gc.Types[gc.Tptr], nil)

		agen(res, &dst)
		agen(n, &dst)
		regfree(&dst)
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
		op = ppc64.AMOVBU

	case 2:
		op = ppc64.AMOVHU

	case 4:
		op = ppc64.AMOVWZU // there is no lwau, only lwaux

	case 8:
		op = ppc64.AMOVDU
	}

	if w%int64(align) != 0 {
		gc.Fatal("sgen: unaligned size %d (align=%d) for %v", w, align, gc.Tconv(n.Type, 0))
	}
	c := int32(w / int64(align))

	// offset on the stack
	osrc := int32(stkof(n))

	odst := int32(stkof(res))
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

	var dst gc.Node
	var src gc.Node
	if n.Ullman >= res.Ullman {
		agenr(n, &dst, res) // temporarily use dst
		regalloc(&src, gc.Types[gc.Tptr], nil)
		gins(ppc64.AMOVD, &dst, &src)
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
	regalloc(&tmp, gc.Types[gc.Tptr], nil)

	// set up end marker
	nend := gc.Node{}

	// move src and dest to the end of block if necessary
	if dir < 0 {
		if c >= 4 {
			regalloc(&nend, gc.Types[gc.Tptr], nil)
			gins(ppc64.AMOVD, &src, &nend)
		}

		p := gins(ppc64.AADD, nil, &src)
		p.From.Type = obj.TYPE_CONST
		p.From.Offset = w

		p = gins(ppc64.AADD, nil, &dst)
		p.From.Type = obj.TYPE_CONST
		p.From.Offset = w
	} else {
		p := gins(ppc64.AADD, nil, &src)
		p.From.Type = obj.TYPE_CONST
		p.From.Offset = int64(-dir)

		p = gins(ppc64.AADD, nil, &dst)
		p.From.Type = obj.TYPE_CONST
		p.From.Offset = int64(-dir)

		if c >= 4 {
			regalloc(&nend, gc.Types[gc.Tptr], nil)
			p := gins(ppc64.AMOVD, &src, &nend)
			p.From.Type = obj.TYPE_ADDR
			p.From.Offset = w
		}
	}

	// move
	// TODO: enable duffcopy for larger copies.
	if c >= 4 {
		p := gins(op, &src, &tmp)
		p.From.Type = obj.TYPE_MEM
		p.From.Offset = int64(dir)
		ploop := p

		p = gins(op, &tmp, &dst)
		p.To.Type = obj.TYPE_MEM
		p.To.Offset = int64(dir)

		p = gins(ppc64.ACMP, &src, &nend)

		gc.Patch(gc.Gbranch(ppc64.ABNE, nil, 0), ploop)
		regfree(&nend)
	} else {
		// TODO(austin): Instead of generating ADD $-8,R8; ADD
		// $-8,R7; n*(MOVDU 8(R8),R9; MOVDU R9,8(R7);) just
		// generate the offsets directly and eliminate the
		// ADDs.  That will produce shorter, more
		// pipeline-able code.
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

			p = gins(op, &tmp, &dst)
			p.To.Type = obj.TYPE_MEM
			p.To.Offset = int64(dir)
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
