// Copyright 2009 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package main

import (
	"cmd/internal/obj"
	"cmd/internal/obj/x86"
	"fmt"
)
import "cmd/internal/gc"

/*
 * reg.c
 */

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

	var nl *gc.Node
	var n1 gc.Node
	var nr *gc.Node
	var n2 gc.Node
	var a int
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

	if n.Addable != 0 {
		gmove(n, res)
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
		p1 := gc.Gbranch(obj.AJMP, nil, 0)

		p2 := gc.Pc
		gmove(gc.Nodbool(true), res)
		p3 := gc.Gbranch(obj.AJMP, nil, 0)
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

		if a == x86.AIMULB {
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
			p1 := gins(x86.ALEAQ, nil, &n1)
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
		if gc.Smallintconst(nr) {
			n2 = *nr
		} else {
			regalloc(&n2, nr.Type, nil)
			cgen(nr, &n2)
		}
	} else {
		if gc.Smallintconst(nr) {
			n2 = *nr
		} else {
			regalloc(&n2, nr.Type, res)
			cgen(nr, &n2)
		}

		regalloc(&n1, nl.Type, nil)
		cgen(nl, &n1)
	}

	gins(a, &n2, &n1)
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
		gc.Dump("\nagenr-n", n)
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
		freelen := 0
		w := uint64(n.Type.Width)

		// Generate the non-addressable child first.
		var n3 gc.Node
		var nlen gc.Node
		var tmp gc.Node
		var n1 gc.Node
		if nr.Addable != 0 {
			goto irad
		}
		if nl.Addable != 0 {
			cgenr(nr, &n1, nil)
			if !gc.Isconst(nl, gc.CTSTR) {
				if gc.Isfixedarray(nl.Type) {
					agenr(nl, &n3, res)
				} else {
					igen(nl, &nlen, res)
					freelen = 1
					nlen.Type = gc.Types[gc.Tptr]
					nlen.Xoffset += int64(gc.Array_array)
					regalloc(&n3, gc.Types[gc.Tptr], res)
					gmove(&nlen, &n3)
					nlen.Type = gc.Types[gc.Simtype[gc.TUINT]]
					nlen.Xoffset += int64(gc.Array_nel) - int64(gc.Array_array)
				}
			}

			goto index
		}

		gc.Tempname(&tmp, nr.Type)
		cgen(nr, &tmp)
		nr = &tmp

	irad:
		if !gc.Isconst(nl, gc.CTSTR) {
			if gc.Isfixedarray(nl.Type) {
				agenr(nl, &n3, res)
			} else {
				if nl.Addable == 0 {
					// igen will need an addressable node.
					var tmp2 gc.Node
					gc.Tempname(&tmp2, nl.Type)

					cgen(nl, &tmp2)
					nl = &tmp2
				}

				igen(nl, &nlen, res)
				freelen = 1
				nlen.Type = gc.Types[gc.Tptr]
				nlen.Xoffset += int64(gc.Array_array)
				regalloc(&n3, gc.Types[gc.Tptr], res)
				gmove(&nlen, &n3)
				nlen.Type = gc.Types[gc.Simtype[gc.TUINT]]
				nlen.Xoffset += int64(gc.Array_nel) - int64(gc.Array_array)
			}
		}

		if !gc.Isconst(nr, gc.CTINT) {
			cgenr(nr, &n1, nil)
		}

		goto index

		// &a is in &n3 (allocated in res)
		// i is in &n1 (if not constant)
		// len(a) is in nlen (if needed)
		// w is width

		// constant index
	index:
		if gc.Isconst(nr, gc.CTINT) {
			if gc.Isconst(nl, gc.CTSTR) {
				gc.Fatal("constant string constant index") // front end should handle
			}
			v := uint64(gc.Mpgetfix(nr.Val.U.Xval))
			if gc.Isslice(nl.Type) || nl.Type.Etype == gc.TSTRING {
				if gc.Debug['B'] == 0 && !n.Bounded {
					var n2 gc.Node
					gc.Nodconst(&n2, gc.Types[gc.Simtype[gc.TUINT]], int64(v))
					if gc.Smallintconst(nr) {
						gins(optoas(gc.OCMP, gc.Types[gc.Simtype[gc.TUINT]]), &nlen, &n2)
					} else {
						regalloc(&tmp, gc.Types[gc.Simtype[gc.TUINT]], nil)
						gmove(&n2, &tmp)
						gins(optoas(gc.OCMP, gc.Types[gc.Simtype[gc.TUINT]]), &nlen, &tmp)
						regfree(&tmp)
					}

					p1 := gc.Gbranch(optoas(gc.OGT, gc.Types[gc.Simtype[gc.TUINT]]), nil, +1)
					ginscall(gc.Panicindex, -1)
					gc.Patch(p1, gc.Pc)
				}

				regfree(&nlen)
			}

			if v*w != 0 {
				ginscon(optoas(gc.OADD, gc.Types[gc.Tptr]), int64(v*w), &n3)
			}
			*a = n3
			break
		}

		// type of the index
		t := gc.Types[gc.TUINT64]

		if gc.Issigned[n1.Type.Etype] != 0 {
			t = gc.Types[gc.TINT64]
		}

		var n2 gc.Node
		regalloc(&n2, t, &n1) // i
		gmove(&n1, &n2)
		regfree(&n1)

		if gc.Debug['B'] == 0 && !n.Bounded {
			// check bounds
			t = gc.Types[gc.Simtype[gc.TUINT]]

			if gc.Is64(nr.Type) {
				t = gc.Types[gc.TUINT64]
			}
			if gc.Isconst(nl, gc.CTSTR) {
				gc.Nodconst(&nlen, t, int64(len(nl.Val.U.Sval.S)))
			} else if gc.Isslice(nl.Type) || nl.Type.Etype == gc.TSTRING {
				if gc.Is64(nr.Type) {
					var n5 gc.Node
					regalloc(&n5, t, nil)
					gmove(&nlen, &n5)
					regfree(&nlen)
					nlen = n5
				}
			} else {
				gc.Nodconst(&nlen, t, nl.Type.Bound)
				if !gc.Smallintconst(&nlen) {
					var n5 gc.Node
					regalloc(&n5, t, nil)
					gmove(&nlen, &n5)
					nlen = n5
					freelen = 1
				}
			}

			gins(optoas(gc.OCMP, t), &n2, &nlen)
			p1 := gc.Gbranch(optoas(gc.OLT, t), nil, +1)
			ginscall(gc.Panicindex, -1)
			gc.Patch(p1, gc.Pc)
		}

		if gc.Isconst(nl, gc.CTSTR) {
			regalloc(&n3, gc.Types[gc.Tptr], res)
			p1 := gins(x86.ALEAQ, nil, &n3)
			gc.Datastring(nl.Val.U.Sval.S, &p1.From)
			gins(x86.AADDQ, &n2, &n3)
			goto indexdone
		}

		if w == 0 {
		} else // nothing to do
		if w == 1 || w == 2 || w == 4 || w == 8 {
			p1 := gins(x86.ALEAQ, &n2, &n3)
			p1.From.Type = obj.TYPE_MEM
			p1.From.Scale = int8(w)
			p1.From.Index = p1.From.Reg
			p1.From.Reg = p1.To.Reg
		} else {
			ginscon(optoas(gc.OMUL, t), int64(w), &n2)
			gins(optoas(gc.OADD, gc.Types[gc.Tptr]), &n2, &n3)
		}

	indexdone:
		*a = n3
		regfree(&n2)
		if freelen != 0 {
			regfree(&nlen)
		}

	default:
		regalloc(a, gc.Types[gc.Tptr], res)
		agen(n, a)
	}
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
		gins(x86.ALEAQ, &n1, &n2)
		gmove(&n2, res)
		regfree(&n2)
		goto ret
	}

	if n.Addable != 0 {
		var n1 gc.Node
		regalloc(&n1, gc.Types[gc.Tptr], res)
		gins(x86.ALEAQ, n, &n1)
		gmove(&n1, res)
		regfree(&n1)
		goto ret
	}

	nl = n.Left

	switch n.Op {
	default:
		gc.Fatal("agen: unknown op %v", gc.Nconv(n, obj.FmtShort|obj.FmtSign))

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
			ginscon(optoas(gc.OADD, gc.Types[gc.Tptr]), n.Xoffset, res)
		}

	case gc.OIND:
		cgen(nl, res)
		gc.Cgen_checknil(res)

	case gc.ODOT:
		agen(nl, res)
		if n.Xoffset != 0 {
			ginscon(optoas(gc.OADD, gc.Types[gc.Tptr]), n.Xoffset, res)
		}

	case gc.ODOTPTR:
		cgen(nl, res)
		gc.Cgen_checknil(res)
		if n.Xoffset != 0 {
			ginscon(optoas(gc.OADD, gc.Types[gc.Tptr]), n.Xoffset, res)
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
		if n.Val.U.Reg != x86.REG_SP {
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
		a.Val.U.Reg = x86.REG_SP
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

	var a int
	var et int
	var nl *gc.Node
	var n1 gc.Node
	var nr *gc.Node
	var n2 gc.Node
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
		goto def

		// need to ask if it is bool?
	case gc.OLITERAL:
		if !true_ == (n.Val.U.Bval == 0) {
			gc.Patch(gc.Gbranch(obj.AJMP, nil, likely), to)
		}
		goto ret

	case gc.ONAME:
		if n.Addable == 0 {
			goto def
		}
		var n1 gc.Node
		gc.Nodconst(&n1, n.Type, 0)
		gins(optoas(gc.OCMP, n.Type), n, &n1)
		a := x86.AJNE
		if !true_ {
			a = x86.AJEQ
		}
		gc.Patch(gc.Gbranch(a, n.Type, likely), to)
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
				p1 := gc.Gbranch(obj.AJMP, nil, 0)

				p2 := gc.Gbranch(obj.AJMP, nil, 0)
				gc.Patch(p1, gc.Pc)
				ll := n.Ninit // avoid re-genning ninit
				n.Ninit = nil
				bgen(n, true, -likely, p2)
				n.Ninit = ll
				gc.Patch(gc.Gbranch(obj.AJMP, nil, 0), to)
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

		var n2 gc.Node
		var n1 gc.Node
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

		if gc.Smallintconst(nr) {
			gins(optoas(gc.OCMP, nr.Type), &n1, nr)
			gc.Patch(gc.Gbranch(optoas(a, nr.Type), nr.Type, likely), to)
			regfree(&n1)
			break
		}

		regalloc(&n2, nr.Type, nil)
		cgen(nr, &n2)

		// only < and <= work right with NaN; reverse if needed
	cmp:
		l := &n1

		r := &n2
		if gc.Isfloat[nl.Type.Etype] != 0 && (a == gc.OGT || a == gc.OGE) {
			l = &n2
			r = &n1
			a = gc.Brrev(a)
		}

		gins(optoas(gc.OCMP, nr.Type), l, r)

		if gc.Isfloat[nr.Type.Etype] != 0 && (n.Op == gc.OEQ || n.Op == gc.ONE) {
			if n.Op == gc.OEQ {
				// neither NE nor P
				p1 := gc.Gbranch(x86.AJNE, nil, -likely)

				p2 := gc.Gbranch(x86.AJPS, nil, -likely)
				gc.Patch(gc.Gbranch(obj.AJMP, nil, 0), to)
				gc.Patch(p1, gc.Pc)
				gc.Patch(p2, gc.Pc)
			} else {
				// either NE or P
				gc.Patch(gc.Gbranch(x86.AJNE, nil, likely), to)

				gc.Patch(gc.Gbranch(x86.AJPS, nil, likely), to)
			}
		} else {
			gc.Patch(gc.Gbranch(optoas(a, nr.Type), nr.Type, likely), to)
		}
		regfree(&n1)
		regfree(&n2)
	}

	goto ret

def:
	regalloc(&n1, n.Type, nil)
	cgen(n, &n1)
	gc.Nodconst(&n2, n.Type, 0)
	gins(optoas(gc.OCMP, n.Type), &n1, &n2)
	a = x86.AJNE
	if !true_ {
		a = x86.AJEQ
	}
	gc.Patch(gc.Gbranch(a, n.Type, likely), to)
	regfree(&n1)
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
			return t.Width
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
	if componentgen(n, ns) {
		return
	}

	if w == 0 {
		// evaluate side effects only
		var nodr gc.Node
		regalloc(&nodr, gc.Types[gc.Tptr], nil)

		agen(ns, &nodr)
		agen(n, &nodr)
		regfree(&nodr)
		return
	}

	// offset on the stack
	osrc := stkof(n)

	odst := stkof(ns)

	if osrc != -1000 && odst != -1000 && (osrc == 1000 || odst == 1000) {
		// osrc and odst both on stack, and at least one is in
		// an unknown position.  Could generate code to test
		// for forward/backward copy, but instead just copy
		// to a temporary location first.
		var tmp gc.Node
		gc.Tempname(&tmp, n.Type)

		sgen(n, &tmp, w)
		sgen(&tmp, ns, w)
		return
	}

	var noddi gc.Node
	gc.Nodreg(&noddi, gc.Types[gc.Tptr], x86.REG_DI)
	var nodsi gc.Node
	gc.Nodreg(&nodsi, gc.Types[gc.Tptr], x86.REG_SI)

	var nodl gc.Node
	var nodr gc.Node
	if n.Ullman >= ns.Ullman {
		agenr(n, &nodr, &nodsi)
		if ns.Op == gc.ONAME {
			gc.Gvardef(ns)
		}
		agenr(ns, &nodl, &noddi)
	} else {
		if ns.Op == gc.ONAME {
			gc.Gvardef(ns)
		}
		agenr(ns, &nodl, &noddi)
		agenr(n, &nodr, &nodsi)
	}

	if nodl.Val.U.Reg != x86.REG_DI {
		gmove(&nodl, &noddi)
	}
	if nodr.Val.U.Reg != x86.REG_SI {
		gmove(&nodr, &nodsi)
	}
	regfree(&nodl)
	regfree(&nodr)

	c := w % 8 // bytes
	q := w / 8 // quads

	var oldcx gc.Node
	var cx gc.Node
	savex(x86.REG_CX, &cx, &oldcx, nil, gc.Types[gc.TINT64])

	// if we are copying forward on the stack and
	// the src and dst overlap, then reverse direction
	if osrc < odst && odst < osrc+w {
		// reverse direction
		gins(x86.ASTD, nil, nil) // set direction flag
		if c > 0 {
			gconreg(addptr, w-1, x86.REG_SI)
			gconreg(addptr, w-1, x86.REG_DI)

			gconreg(movptr, c, x86.REG_CX)
			gins(x86.AREP, nil, nil)   // repeat
			gins(x86.AMOVSB, nil, nil) // MOVB *(SI)-,*(DI)-
		}

		if q > 0 {
			if c > 0 {
				gconreg(addptr, -7, x86.REG_SI)
				gconreg(addptr, -7, x86.REG_DI)
			} else {
				gconreg(addptr, w-8, x86.REG_SI)
				gconreg(addptr, w-8, x86.REG_DI)
			}

			gconreg(movptr, q, x86.REG_CX)
			gins(x86.AREP, nil, nil)   // repeat
			gins(x86.AMOVSQ, nil, nil) // MOVQ *(SI)-,*(DI)-
		}

		// we leave with the flag clear
		gins(x86.ACLD, nil, nil)
	} else {
		// normal direction
		if q > 128 || (gc.Nacl && q >= 4) {
			gconreg(movptr, q, x86.REG_CX)
			gins(x86.AREP, nil, nil)   // repeat
			gins(x86.AMOVSQ, nil, nil) // MOVQ *(SI)+,*(DI)+
		} else if q >= 4 {
			p := gins(obj.ADUFFCOPY, nil, nil)
			p.To.Type = obj.TYPE_ADDR
			p.To.Sym = gc.Linksym(gc.Pkglookup("duffcopy", gc.Runtimepkg))

			// 14 and 128 = magic constants: see ../../runtime/asm_amd64.s
			p.To.Offset = 14 * (128 - q)
		} else if !gc.Nacl && c == 0 {
			// We don't need the MOVSQ side-effect of updating SI and DI,
			// and issuing a sequence of MOVQs directly is faster.
			nodsi.Op = gc.OINDREG

			noddi.Op = gc.OINDREG
			for q > 0 {
				gmove(&nodsi, &cx) // MOVQ x+(SI),CX
				gmove(&cx, &noddi) // MOVQ CX,x+(DI)
				nodsi.Xoffset += 8
				noddi.Xoffset += 8
				q--
			}
		} else {
			for q > 0 {
				gins(x86.AMOVSQ, nil, nil) // MOVQ *(SI)+,*(DI)+
				q--
			}
		}

		// copy the remaining c bytes
		if w < 4 || c <= 1 || (odst < osrc && osrc < odst+w) {
			for c > 0 {
				gins(x86.AMOVSB, nil, nil) // MOVB *(SI)+,*(DI)+
				c--
			}
		} else if w < 8 || c <= 4 {
			nodsi.Op = gc.OINDREG
			noddi.Op = gc.OINDREG
			cx.Type = gc.Types[gc.TINT32]
			nodsi.Type = gc.Types[gc.TINT32]
			noddi.Type = gc.Types[gc.TINT32]
			if c > 4 {
				nodsi.Xoffset = 0
				noddi.Xoffset = 0
				gmove(&nodsi, &cx)
				gmove(&cx, &noddi)
			}

			nodsi.Xoffset = c - 4
			noddi.Xoffset = c - 4
			gmove(&nodsi, &cx)
			gmove(&cx, &noddi)
		} else {
			nodsi.Op = gc.OINDREG
			noddi.Op = gc.OINDREG
			cx.Type = gc.Types[gc.TINT64]
			nodsi.Type = gc.Types[gc.TINT64]
			noddi.Type = gc.Types[gc.TINT64]
			nodsi.Xoffset = c - 8
			noddi.Xoffset = c - 8
			gmove(&nodsi, &cx)
			gmove(&cx, &noddi)
		}
	}

	restx(&cx, &oldcx)
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
