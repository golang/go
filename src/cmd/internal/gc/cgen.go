// Copyright 2009 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package gc

import (
	"cmd/internal/obj"
	"fmt"
)

/*
 * generate:
 *	res = n;
 * simplifies and calls Thearch.Gmove.
 */
func Cgen(n *Node, res *Node) {
	if Debug['g'] != 0 {
		Dump("\ncgen-n", n)
		Dump("cgen-res", res)
	}

	if n == nil || n.Type == nil {
		return
	}

	if res == nil || res.Type == nil {
		Fatal("cgen: res nil")
	}

	for n.Op == OCONVNOP {
		n = n.Left
	}

	switch n.Op {
	case OSLICE, OSLICEARR, OSLICESTR, OSLICE3, OSLICE3ARR:
		if res.Op != ONAME || !res.Addable {
			var n1 Node
			Tempname(&n1, n.Type)
			Cgen_slice(n, &n1)
			Cgen(&n1, res)
		} else {
			Cgen_slice(n, res)
		}
		return

	case OEFACE:
		if res.Op != ONAME || !res.Addable {
			var n1 Node
			Tempname(&n1, n.Type)
			Cgen_eface(n, &n1)
			Cgen(&n1, res)
		} else {
			Cgen_eface(n, res)
		}
		return

	case ODOTTYPE:
		cgen_dottype(n, res, nil)
		return
	}

	if n.Ullman >= UINF {
		if n.Op == OINDREG {
			Fatal("cgen: this is going to miscompile")
		}
		if res.Ullman >= UINF {
			var n1 Node
			Tempname(&n1, n.Type)
			Cgen(n, &n1)
			Cgen(&n1, res)
			return
		}
	}

	if Isfat(n.Type) {
		if n.Type.Width < 0 {
			Fatal("forgot to compute width for %v", Tconv(n.Type, 0))
		}
		sgen(n, res, n.Type.Width)
		return
	}

	if !res.Addable {
		if n.Ullman > res.Ullman {
			if Ctxt.Arch.Regsize == 4 && Is64(n.Type) {
				var n1 Node
				Tempname(&n1, n.Type)
				Cgen(n, &n1)
				Cgen(&n1, res)
				return
			}

			var n1 Node
			Regalloc(&n1, n.Type, res)
			Cgen(n, &n1)
			if n1.Ullman > res.Ullman {
				Dump("n1", &n1)
				Dump("res", res)
				Fatal("loop in cgen")
			}

			Cgen(&n1, res)
			Regfree(&n1)
			return
		}

		var f int
		if res.Ullman >= UINF {
			goto gen
		}

		if Complexop(n, res) {
			Complexgen(n, res)
			return
		}

		f = 1 // gen thru register
		switch n.Op {
		case OLITERAL:
			if Smallintconst(n) {
				f = 0
			}

		case OREGISTER:
			f = 0
		}

		if !Iscomplex[n.Type.Etype] && Ctxt.Arch.Regsize == 8 {
			a := Thearch.Optoas(OAS, res.Type)
			var addr obj.Addr
			if Thearch.Sudoaddable(a, res, &addr) {
				var p1 *obj.Prog
				if f != 0 {
					var n2 Node
					Regalloc(&n2, res.Type, nil)
					Cgen(n, &n2)
					p1 = Thearch.Gins(a, &n2, nil)
					Regfree(&n2)
				} else {
					p1 = Thearch.Gins(a, n, nil)
				}
				p1.To = addr
				if Debug['g'] != 0 {
					fmt.Printf("%v [ignore previous line]\n", p1)
				}
				Thearch.Sudoclean()
				return
			}
		}

	gen:
		if Ctxt.Arch.Thechar == '8' {
			// no registers to speak of
			var n1, n2 Node
			Tempname(&n1, n.Type)
			Cgen(n, &n1)
			Igen(res, &n2, nil)
			Thearch.Gmove(&n1, &n2)
			Regfree(&n2)
			return
		}

		var n1 Node
		Igen(res, &n1, nil)
		Cgen(n, &n1)
		Regfree(&n1)
		return
	}

	// update addressability for string, slice
	// can't do in walk because n->left->addable
	// changes if n->left is an escaping local variable.
	switch n.Op {
	case OSPTR, OLEN:
		if Isslice(n.Left.Type) || Istype(n.Left.Type, TSTRING) {
			n.Addable = n.Left.Addable
		}

	case OCAP:
		if Isslice(n.Left.Type) {
			n.Addable = n.Left.Addable
		}

	case OITAB:
		n.Addable = n.Left.Addable
	}

	if Ctxt.Arch.Thechar == '5' { // TODO(rsc): Maybe more often?
		// if both are addressable, move
		if n.Addable && res.Addable {
			if Is64(n.Type) || Is64(res.Type) || n.Op == OREGISTER || res.Op == OREGISTER || Iscomplex[n.Type.Etype] || Iscomplex[res.Type.Etype] {
				Thearch.Gmove(n, res)
			} else {
				var n1 Node
				Regalloc(&n1, n.Type, nil)
				Thearch.Gmove(n, &n1)
				Cgen(&n1, res)
				Regfree(&n1)
			}

			return
		}

		// if both are not addressable, use a temporary.
		if !n.Addable && !res.Addable {
			// could use regalloc here sometimes,
			// but have to check for ullman >= UINF.
			var n1 Node
			Tempname(&n1, n.Type)
			Cgen(n, &n1)
			Cgen(&n1, res)
			return
		}

		// if result is not addressable directly but n is,
		// compute its address and then store via the address.
		if !res.Addable {
			var n1 Node
			Igen(res, &n1, nil)
			Cgen(n, &n1)
			Regfree(&n1)
			return
		}
	}

	if Complexop(n, res) {
		Complexgen(n, res)
		return
	}

	if (Ctxt.Arch.Thechar == '6' || Ctxt.Arch.Thechar == '8') && n.Addable {
		Thearch.Gmove(n, res)
		return
	}

	if Ctxt.Arch.Thechar == '7' || Ctxt.Arch.Thechar == '9' {
		// if both are addressable, move
		if n.Addable {
			if n.Op == OREGISTER || res.Op == OREGISTER {
				Thearch.Gmove(n, res)
			} else {
				var n1 Node
				Regalloc(&n1, n.Type, nil)
				Thearch.Gmove(n, &n1)
				Cgen(&n1, res)
				Regfree(&n1)
			}
			return
		}
	}

	// if n is sudoaddable generate addr and move
	if Ctxt.Arch.Thechar == '5' && !Is64(n.Type) && !Is64(res.Type) && !Iscomplex[n.Type.Etype] && !Iscomplex[res.Type.Etype] {
		a := Thearch.Optoas(OAS, n.Type)
		var addr obj.Addr
		if Thearch.Sudoaddable(a, n, &addr) {
			if res.Op != OREGISTER {
				var n2 Node
				Regalloc(&n2, res.Type, nil)
				p1 := Thearch.Gins(a, nil, &n2)
				p1.From = addr
				if Debug['g'] != 0 {
					fmt.Printf("%v [ignore previous line]\n", p1)
				}
				Thearch.Gmove(&n2, res)
				Regfree(&n2)
			} else {
				p1 := Thearch.Gins(a, nil, res)
				p1.From = addr
				if Debug['g'] != 0 {
					fmt.Printf("%v [ignore previous line]\n", p1)
				}
			}
			Thearch.Sudoclean()
			return
		}
	}

	nl := n.Left
	nr := n.Right

	if nl != nil && nl.Ullman >= UINF {
		if nr != nil && nr.Ullman >= UINF {
			var n1 Node
			Tempname(&n1, nl.Type)
			Cgen(nl, &n1)
			n2 := *n
			n2.Left = &n1
			Cgen(&n2, res)
			return
		}
	}

	// 64-bit ops are hard on 32-bit machine.
	if Ctxt.Arch.Regsize == 4 && (Is64(n.Type) || Is64(res.Type) || n.Left != nil && Is64(n.Left.Type)) {
		switch n.Op {
		// math goes to cgen64.
		case OMINUS,
			OCOM,
			OADD,
			OSUB,
			OMUL,
			OLROT,
			OLSH,
			ORSH,
			OAND,
			OOR,
			OXOR:
			Thearch.Cgen64(n, res)
			return
		}
	}

	if Thearch.Cgen_float != nil && nl != nil && Isfloat[n.Type.Etype] && Isfloat[nl.Type.Etype] {
		Thearch.Cgen_float(n, res)
		return
	}

	if !Iscomplex[n.Type.Etype] && Ctxt.Arch.Regsize == 8 {
		a := Thearch.Optoas(OAS, n.Type)
		var addr obj.Addr
		if Thearch.Sudoaddable(a, n, &addr) {
			if res.Op == OREGISTER {
				p1 := Thearch.Gins(a, nil, res)
				p1.From = addr
			} else {
				var n2 Node
				Regalloc(&n2, n.Type, nil)
				p1 := Thearch.Gins(a, nil, &n2)
				p1.From = addr
				Thearch.Gins(a, &n2, res)
				Regfree(&n2)
			}

			Thearch.Sudoclean()
			return
		}
	}

	var a int
	switch n.Op {
	default:
		Dump("cgen", n)
		Dump("cgen-res", res)
		Fatal("cgen: unknown op %v", Nconv(n, obj.FmtShort|obj.FmtSign))

		// these call bgen to get a bool value
	case OOROR,
		OANDAND,
		OEQ,
		ONE,
		OLT,
		OLE,
		OGE,
		OGT,
		ONOT:
		p1 := Gbranch(obj.AJMP, nil, 0)

		p2 := Pc
		Thearch.Gmove(Nodbool(true), res)
		p3 := Gbranch(obj.AJMP, nil, 0)
		Patch(p1, Pc)
		Bgen(n, true, 0, p2)
		Thearch.Gmove(Nodbool(false), res)
		Patch(p3, Pc)
		return

	case OPLUS:
		Cgen(nl, res)
		return

		// unary
	case OCOM:
		a := Thearch.Optoas(OXOR, nl.Type)

		var n1 Node
		Regalloc(&n1, nl.Type, nil)
		Cgen(nl, &n1)
		var n2 Node
		Nodconst(&n2, nl.Type, -1)
		Thearch.Gins(a, &n2, &n1)
		cgen_norm(n, &n1, res)
		return

	case OMINUS:
		if Isfloat[nl.Type.Etype] {
			nr = Nodintconst(-1)
			Convlit(&nr, n.Type)
			a = Thearch.Optoas(OMUL, nl.Type)
			goto sbop
		}

		a := Thearch.Optoas(int(n.Op), nl.Type)
		// unary
		var n1 Node
		Regalloc(&n1, nl.Type, res)

		Cgen(nl, &n1)
		if Ctxt.Arch.Thechar == '5' {
			var n2 Node
			Nodconst(&n2, nl.Type, 0)
			Thearch.Gins(a, &n2, &n1)
		} else if Ctxt.Arch.Thechar == '7' {
			Thearch.Gins(a, &n1, &n1)
		} else {
			Thearch.Gins(a, nil, &n1)
		}
		cgen_norm(n, &n1, res)
		return

	case OSQRT:
		var n1 Node
		Regalloc(&n1, nl.Type, res)
		Cgen(n.Left, &n1)
		Thearch.Gins(Thearch.Optoas(OSQRT, nl.Type), &n1, &n1)
		Thearch.Gmove(&n1, res)
		Regfree(&n1)
		return

	case OGETG:
		Thearch.Getg(res)
		return

		// symmetric binary
	case OAND,
		OOR,
		OXOR,
		OADD,
		OMUL:
		if n.Op == OMUL && Thearch.Cgen_bmul != nil && Thearch.Cgen_bmul(int(n.Op), nl, nr, res) {
			break
		}
		a = Thearch.Optoas(int(n.Op), nl.Type)
		goto sbop

		// asymmetric binary
	case OSUB:
		a = Thearch.Optoas(int(n.Op), nl.Type)
		goto abop

	case OHMUL:
		Thearch.Cgen_hmul(nl, nr, res)

	case OCONV:
		if Eqtype(n.Type, nl.Type) || Noconv(n.Type, nl.Type) {
			Cgen(nl, res)
			return
		}

		if Ctxt.Arch.Thechar == '8' {
			var n1 Node
			var n2 Node
			Tempname(&n2, n.Type)
			Mgen(nl, &n1, res)
			Thearch.Gmove(&n1, &n2)
			Thearch.Gmove(&n2, res)
			Mfree(&n1)
			break
		}

		var n1 Node
		var n2 Node
		if Ctxt.Arch.Thechar == '5' {
			if nl.Addable && !Is64(nl.Type) {
				Regalloc(&n1, nl.Type, res)
				Thearch.Gmove(nl, &n1)
			} else {
				if n.Type.Width > int64(Widthptr) || Is64(nl.Type) || Isfloat[nl.Type.Etype] {
					Tempname(&n1, nl.Type)
				} else {
					Regalloc(&n1, nl.Type, res)
				}
				Cgen(nl, &n1)
			}
			if n.Type.Width > int64(Widthptr) || Is64(n.Type) || Isfloat[n.Type.Etype] {
				Tempname(&n2, n.Type)
			} else {
				Regalloc(&n2, n.Type, nil)
			}
		} else {
			if n.Type.Width > nl.Type.Width {
				// If loading from memory, do conversion during load,
				// so as to avoid use of 8-bit register in, say, int(*byteptr).
				switch nl.Op {
				case ODOT, ODOTPTR, OINDEX, OIND, ONAME:
					Igen(nl, &n1, res)
					Regalloc(&n2, n.Type, res)
					Thearch.Gmove(&n1, &n2)
					Thearch.Gmove(&n2, res)
					Regfree(&n2)
					Regfree(&n1)
					return
				}
			}
			Regalloc(&n1, nl.Type, res)
			Regalloc(&n2, n.Type, &n1)
			Cgen(nl, &n1)
		}

		// if we do the conversion n1 -> n2 here
		// reusing the register, then gmove won't
		// have to allocate its own register.
		Thearch.Gmove(&n1, &n2)
		Thearch.Gmove(&n2, res)
		if n2.Op == OREGISTER {
			Regfree(&n2)
		}
		if n1.Op == OREGISTER {
			Regfree(&n1)
		}

	case ODOT,
		ODOTPTR,
		OINDEX,
		OIND,
		ONAME: // PHEAP or PPARAMREF var
		var n1 Node
		Igen(n, &n1, res)

		Thearch.Gmove(&n1, res)
		Regfree(&n1)

		// interface table is first word of interface value
	case OITAB:
		var n1 Node
		Igen(nl, &n1, res)

		n1.Type = n.Type
		Thearch.Gmove(&n1, res)
		Regfree(&n1)

	case OSPTR:
		// pointer is the first word of string or slice.
		if Isconst(nl, CTSTR) {
			var n1 Node
			Regalloc(&n1, Types[Tptr], res)
			p1 := Thearch.Gins(Thearch.Optoas(OAS, n1.Type), nil, &n1)
			Datastring(nl.Val.U.Sval, &p1.From)
			p1.From.Type = obj.TYPE_ADDR
			Thearch.Gmove(&n1, res)
			Regfree(&n1)
			break
		}

		var n1 Node
		Igen(nl, &n1, res)
		n1.Type = n.Type
		Thearch.Gmove(&n1, res)
		Regfree(&n1)

	case OLEN:
		if Istype(nl.Type, TMAP) || Istype(nl.Type, TCHAN) {
			// map and chan have len in the first int-sized word.
			// a zero pointer means zero length
			var n1 Node
			Regalloc(&n1, Types[Tptr], res)

			Cgen(nl, &n1)

			var n2 Node
			Nodconst(&n2, Types[Tptr], 0)
			Thearch.Gins(Thearch.Optoas(OCMP, Types[Tptr]), &n1, &n2)
			p1 := Gbranch(Thearch.Optoas(OEQ, Types[Tptr]), nil, 0)

			n2 = n1
			n2.Op = OINDREG
			n2.Type = Types[Simtype[TINT]]
			Thearch.Gmove(&n2, &n1)

			Patch(p1, Pc)

			Thearch.Gmove(&n1, res)
			Regfree(&n1)
			break
		}

		if Istype(nl.Type, TSTRING) || Isslice(nl.Type) {
			// both slice and string have len one pointer into the struct.
			// a zero pointer means zero length
			var n1 Node
			Igen(nl, &n1, res)

			n1.Type = Types[Simtype[TUINT]]
			n1.Xoffset += int64(Array_nel)
			Thearch.Gmove(&n1, res)
			Regfree(&n1)
			break
		}

		Fatal("cgen: OLEN: unknown type %v", Tconv(nl.Type, obj.FmtLong))

	case OCAP:
		if Istype(nl.Type, TCHAN) {
			// chan has cap in the second int-sized word.
			// a zero pointer means zero length
			var n1 Node
			Regalloc(&n1, Types[Tptr], res)

			Cgen(nl, &n1)

			var n2 Node
			Nodconst(&n2, Types[Tptr], 0)
			Thearch.Gins(Thearch.Optoas(OCMP, Types[Tptr]), &n1, &n2)
			p1 := Gbranch(Thearch.Optoas(OEQ, Types[Tptr]), nil, 0)

			n2 = n1
			n2.Op = OINDREG
			n2.Xoffset = int64(Widthint)
			n2.Type = Types[Simtype[TINT]]
			Thearch.Gmove(&n2, &n1)

			Patch(p1, Pc)

			Thearch.Gmove(&n1, res)
			Regfree(&n1)
			break
		}

		if Isslice(nl.Type) {
			var n1 Node
			Igen(nl, &n1, res)
			n1.Type = Types[Simtype[TUINT]]
			n1.Xoffset += int64(Array_cap)
			Thearch.Gmove(&n1, res)
			Regfree(&n1)
			break
		}

		Fatal("cgen: OCAP: unknown type %v", Tconv(nl.Type, obj.FmtLong))

	case OADDR:
		if n.Bounded { // let race detector avoid nil checks
			Disable_checknil++
		}
		Agen(nl, res)
		if n.Bounded {
			Disable_checknil--
		}

	case OCALLMETH:
		cgen_callmeth(n, 0)
		cgen_callret(n, res)

	case OCALLINTER:
		cgen_callinter(n, res, 0)
		cgen_callret(n, res)

	case OCALLFUNC:
		cgen_call(n, 0)
		cgen_callret(n, res)

	case OMOD, ODIV:
		if Isfloat[n.Type.Etype] || Thearch.Dodiv == nil {
			a = Thearch.Optoas(int(n.Op), nl.Type)
			goto abop
		}

		if nl.Ullman >= nr.Ullman {
			var n1 Node
			Regalloc(&n1, nl.Type, res)
			Cgen(nl, &n1)
			cgen_div(int(n.Op), &n1, nr, res)
			Regfree(&n1)
		} else {
			var n2 Node
			if !Smallintconst(nr) {
				Regalloc(&n2, nr.Type, res)
				Cgen(nr, &n2)
			} else {
				n2 = *nr
			}

			cgen_div(int(n.Op), nl, &n2, res)
			if n2.Op != OLITERAL {
				Regfree(&n2)
			}
		}

	case OLSH, ORSH, OLROT:
		Thearch.Cgen_shift(int(n.Op), n.Bounded, nl, nr, res)
	}

	return

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
	if nl.Ullman < nr.Ullman || (nl.Ullman == nr.Ullman && (Smallintconst(nl) || (nr.Op == OLITERAL && !Smallintconst(nr)))) {
		r := nl
		nl = nr
		nr = r
	}

abop: // asymmetric binary
	var n1 Node
	var n2 Node
	if Ctxt.Arch.Thechar == '8' {
		// no registers, sigh
		if Smallintconst(nr) {
			var n1 Node
			Mgen(nl, &n1, res)
			var n2 Node
			Regalloc(&n2, nl.Type, &n1)
			Thearch.Gmove(&n1, &n2)
			Thearch.Gins(a, nr, &n2)
			Thearch.Gmove(&n2, res)
			Regfree(&n2)
			Mfree(&n1)
		} else if nl.Ullman >= nr.Ullman {
			var nt Node
			Tempname(&nt, nl.Type)
			Cgen(nl, &nt)
			var n2 Node
			Mgen(nr, &n2, nil)
			var n1 Node
			Regalloc(&n1, nl.Type, res)
			Thearch.Gmove(&nt, &n1)
			Thearch.Gins(a, &n2, &n1)
			Thearch.Gmove(&n1, res)
			Regfree(&n1)
			Mfree(&n2)
		} else {
			var n2 Node
			Regalloc(&n2, nr.Type, res)
			Cgen(nr, &n2)
			var n1 Node
			Regalloc(&n1, nl.Type, nil)
			Cgen(nl, &n1)
			Thearch.Gins(a, &n2, &n1)
			Regfree(&n2)
			Thearch.Gmove(&n1, res)
			Regfree(&n1)
		}
		return
	}

	if nl.Ullman >= nr.Ullman {
		Regalloc(&n1, nl.Type, res)
		Cgen(nl, &n1)

		if Smallintconst(nr) && Ctxt.Arch.Thechar != '5' && Ctxt.Arch.Thechar != '7' && Ctxt.Arch.Thechar != '9' { // TODO(rsc): Check opcode for arm
			n2 = *nr
		} else {
			Regalloc(&n2, nr.Type, nil)
			Cgen(nr, &n2)
		}
	} else {
		if Smallintconst(nr) && Ctxt.Arch.Thechar != '5' && Ctxt.Arch.Thechar != '7' && Ctxt.Arch.Thechar != '9' { // TODO(rsc): Check opcode for arm
			n2 = *nr
		} else {
			Regalloc(&n2, nr.Type, res)
			Cgen(nr, &n2)
		}

		Regalloc(&n1, nl.Type, nil)
		Cgen(nl, &n1)
	}

	Thearch.Gins(a, &n2, &n1)
	if n2.Op != OLITERAL {
		Regfree(&n2)
	}
	cgen_norm(n, &n1, res)
}

// cgen_norm moves n1 to res, truncating to expected type if necessary.
// n1 is a register, and cgen_norm frees it.
func cgen_norm(n, n1, res *Node) {
	switch Ctxt.Arch.Thechar {
	case '6', '8':
		// We use sized math, so the result is already truncated.
	default:
		switch n.Op {
		case OADD, OSUB, OMUL, ODIV, OCOM, OMINUS:
			// TODO(rsc): What about left shift?
			Thearch.Gins(Thearch.Optoas(OAS, n.Type), n1, n1)
		}
	}

	Thearch.Gmove(n1, res)
	Regfree(n1)
}

func Mgen(n *Node, n1 *Node, rg *Node) {
	n1.Op = OEMPTY

	if n.Addable {
		*n1 = *n
		if n1.Op == OREGISTER || n1.Op == OINDREG {
			reg[n.Reg-int16(Thearch.REGMIN)]++
		}
		return
	}

	Tempname(n1, n.Type)
	Cgen(n, n1)
	if n.Type.Width <= int64(Widthptr) || Isfloat[n.Type.Etype] {
		n2 := *n1
		Regalloc(n1, n.Type, rg)
		Thearch.Gmove(&n2, n1)
	}
}

func Mfree(n *Node) {
	if n.Op == OREGISTER {
		Regfree(n)
	}
}

/*
 * allocate a register (reusing res if possible) and generate
 *  a = n
 * The caller must call Regfree(a).
 */
func Cgenr(n *Node, a *Node, res *Node) {
	if Debug['g'] != 0 {
		Dump("cgenr-n", n)
	}

	if Isfat(n.Type) {
		Fatal("cgenr on fat node")
	}

	if n.Addable {
		Regalloc(a, n.Type, res)
		Thearch.Gmove(n, a)
		return
	}

	switch n.Op {
	case ONAME,
		ODOT,
		ODOTPTR,
		OINDEX,
		OCALLFUNC,
		OCALLMETH,
		OCALLINTER:
		var n1 Node
		Igen(n, &n1, res)
		Regalloc(a, Types[Tptr], &n1)
		Thearch.Gmove(&n1, a)
		Regfree(&n1)

	default:
		Regalloc(a, n.Type, res)
		Cgen(n, a)
	}
}

/*
 * allocate a register (reusing res if possible) and generate
 * a = &n
 * The caller must call Regfree(a).
 * The generated code checks that the result is not nil.
 */
func Agenr(n *Node, a *Node, res *Node) {
	if Debug['g'] != 0 {
		Dump("\nagenr-n", n)
	}

	nl := n.Left
	nr := n.Right

	switch n.Op {
	case ODOT, ODOTPTR, OCALLFUNC, OCALLMETH, OCALLINTER:
		var n1 Node
		Igen(n, &n1, res)
		Regalloc(a, Types[Tptr], &n1)
		Agen(&n1, a)
		Regfree(&n1)

	case OIND:
		Cgenr(n.Left, a, res)
		Cgen_checknil(a)

	case OINDEX:
		if Ctxt.Arch.Thechar == '5' {
			var p2 *obj.Prog // to be patched to panicindex.
			w := uint32(n.Type.Width)
			bounded := Debug['B'] != 0 || n.Bounded
			var n1 Node
			var n3 Node
			if nr.Addable {
				var tmp Node
				if !Isconst(nr, CTINT) {
					Tempname(&tmp, Types[TINT32])
				}
				if !Isconst(nl, CTSTR) {
					Agenr(nl, &n3, res)
				}
				if !Isconst(nr, CTINT) {
					p2 = Thearch.Cgenindex(nr, &tmp, bounded)
					Regalloc(&n1, tmp.Type, nil)
					Thearch.Gmove(&tmp, &n1)
				}
			} else if nl.Addable {
				if !Isconst(nr, CTINT) {
					var tmp Node
					Tempname(&tmp, Types[TINT32])
					p2 = Thearch.Cgenindex(nr, &tmp, bounded)
					Regalloc(&n1, tmp.Type, nil)
					Thearch.Gmove(&tmp, &n1)
				}

				if !Isconst(nl, CTSTR) {
					Agenr(nl, &n3, res)
				}
			} else {
				var tmp Node
				Tempname(&tmp, Types[TINT32])
				p2 = Thearch.Cgenindex(nr, &tmp, bounded)
				nr = &tmp
				if !Isconst(nl, CTSTR) {
					Agenr(nl, &n3, res)
				}
				Regalloc(&n1, tmp.Type, nil)
				Thearch.Gins(Thearch.Optoas(OAS, tmp.Type), &tmp, &n1)
			}

			// &a is in &n3 (allocated in res)
			// i is in &n1 (if not constant)
			// w is width

			// constant index
			if Isconst(nr, CTINT) {
				if Isconst(nl, CTSTR) {
					Fatal("constant string constant index")
				}
				v := uint64(Mpgetfix(nr.Val.U.Xval))
				var n2 Node
				if Isslice(nl.Type) || nl.Type.Etype == TSTRING {
					if Debug['B'] == 0 && !n.Bounded {
						n1 = n3
						n1.Op = OINDREG
						n1.Type = Types[Tptr]
						n1.Xoffset = int64(Array_nel)
						var n4 Node
						Regalloc(&n4, n1.Type, nil)
						Thearch.Gmove(&n1, &n4)
						Nodconst(&n2, Types[TUINT32], int64(v))
						Thearch.Gins(Thearch.Optoas(OCMP, Types[TUINT32]), &n4, &n2)
						Regfree(&n4)
						p1 := Gbranch(Thearch.Optoas(OGT, Types[TUINT32]), nil, +1)
						Ginscall(Panicindex, 0)
						Patch(p1, Pc)
					}

					n1 = n3
					n1.Op = OINDREG
					n1.Type = Types[Tptr]
					n1.Xoffset = int64(Array_array)
					Thearch.Gmove(&n1, &n3)
				}

				Nodconst(&n2, Types[Tptr], int64(v*uint64(w)))
				Thearch.Gins(Thearch.Optoas(OADD, Types[Tptr]), &n2, &n3)
				*a = n3
				break
			}

			var n2 Node
			Regalloc(&n2, Types[TINT32], &n1) // i
			Thearch.Gmove(&n1, &n2)
			Regfree(&n1)

			var n4 Node
			if Debug['B'] == 0 && !n.Bounded {
				// check bounds
				if Isconst(nl, CTSTR) {
					Nodconst(&n4, Types[TUINT32], int64(len(nl.Val.U.Sval)))
				} else if Isslice(nl.Type) || nl.Type.Etype == TSTRING {
					n1 = n3
					n1.Op = OINDREG
					n1.Type = Types[Tptr]
					n1.Xoffset = int64(Array_nel)
					Regalloc(&n4, Types[TUINT32], nil)
					Thearch.Gmove(&n1, &n4)
				} else {
					Nodconst(&n4, Types[TUINT32], nl.Type.Bound)
				}

				Thearch.Gins(Thearch.Optoas(OCMP, Types[TUINT32]), &n2, &n4)
				if n4.Op == OREGISTER {
					Regfree(&n4)
				}
				p1 := Gbranch(Thearch.Optoas(OLT, Types[TUINT32]), nil, +1)
				if p2 != nil {
					Patch(p2, Pc)
				}
				Ginscall(Panicindex, 0)
				Patch(p1, Pc)
			}

			if Isconst(nl, CTSTR) {
				Regalloc(&n3, Types[Tptr], res)
				p1 := Thearch.Gins(Thearch.Optoas(OAS, Types[Tptr]), nil, &n3)
				Datastring(nl.Val.U.Sval, &p1.From)
				p1.From.Type = obj.TYPE_ADDR
			} else if Isslice(nl.Type) || nl.Type.Etype == TSTRING {
				n1 = n3
				n1.Op = OINDREG
				n1.Type = Types[Tptr]
				n1.Xoffset = int64(Array_array)
				Thearch.Gmove(&n1, &n3)
			}

			if w == 0 {
				// nothing to do
			} else if Thearch.AddIndex != nil && Thearch.AddIndex(&n2, int64(w), &n3) {
				// done by back end
			} else if w == 1 {
				Thearch.Gins(Thearch.Optoas(OADD, Types[Tptr]), &n2, &n3)
			} else {
				Regalloc(&n4, Types[TUINT32], nil)
				Nodconst(&n1, Types[TUINT32], int64(w))
				Thearch.Gmove(&n1, &n4)
				Thearch.Gins(Thearch.Optoas(OMUL, Types[TUINT32]), &n4, &n2)
				Thearch.Gins(Thearch.Optoas(OADD, Types[Tptr]), &n2, &n3)
				Regfree(&n4)
			}
			*a = n3
			Regfree(&n2)
			break
		}
		if Ctxt.Arch.Thechar == '8' {
			var p2 *obj.Prog // to be patched to panicindex.
			w := uint32(n.Type.Width)
			bounded := Debug['B'] != 0 || n.Bounded
			var n3 Node
			var tmp Node
			var n1 Node
			if nr.Addable {
				// Generate &nl first, and move nr into register.
				if !Isconst(nl, CTSTR) {
					Igen(nl, &n3, res)
				}
				if !Isconst(nr, CTINT) {
					p2 = Thearch.Igenindex(nr, &tmp, bounded)
					Regalloc(&n1, tmp.Type, nil)
					Thearch.Gmove(&tmp, &n1)
				}
			} else if nl.Addable {
				// Generate nr first, and move &nl into register.
				if !Isconst(nr, CTINT) {
					p2 = Thearch.Igenindex(nr, &tmp, bounded)
					Regalloc(&n1, tmp.Type, nil)
					Thearch.Gmove(&tmp, &n1)
				}

				if !Isconst(nl, CTSTR) {
					Igen(nl, &n3, res)
				}
			} else {
				p2 = Thearch.Igenindex(nr, &tmp, bounded)
				nr = &tmp
				if !Isconst(nl, CTSTR) {
					Igen(nl, &n3, res)
				}
				Regalloc(&n1, tmp.Type, nil)
				Thearch.Gins(Thearch.Optoas(OAS, tmp.Type), &tmp, &n1)
			}

			// For fixed array we really want the pointer in n3.
			var n2 Node
			if Isfixedarray(nl.Type) {
				Regalloc(&n2, Types[Tptr], &n3)
				Agen(&n3, &n2)
				Regfree(&n3)
				n3 = n2
			}

			// &a[0] is in n3 (allocated in res)
			// i is in n1 (if not constant)
			// len(a) is in nlen (if needed)
			// w is width

			// constant index
			if Isconst(nr, CTINT) {
				if Isconst(nl, CTSTR) {
					Fatal("constant string constant index") // front end should handle
				}
				v := uint64(Mpgetfix(nr.Val.U.Xval))
				if Isslice(nl.Type) || nl.Type.Etype == TSTRING {
					if Debug['B'] == 0 && !n.Bounded {
						nlen := n3
						nlen.Type = Types[TUINT32]
						nlen.Xoffset += int64(Array_nel)
						Nodconst(&n2, Types[TUINT32], int64(v))
						Thearch.Gins(Thearch.Optoas(OCMP, Types[TUINT32]), &nlen, &n2)
						p1 := Gbranch(Thearch.Optoas(OGT, Types[TUINT32]), nil, +1)
						Ginscall(Panicindex, -1)
						Patch(p1, Pc)
					}
				}

				// Load base pointer in n2 = n3.
				Regalloc(&n2, Types[Tptr], &n3)

				n3.Type = Types[Tptr]
				n3.Xoffset += int64(Array_array)
				Thearch.Gmove(&n3, &n2)
				Regfree(&n3)
				if v*uint64(w) != 0 {
					Nodconst(&n1, Types[Tptr], int64(v*uint64(w)))
					Thearch.Gins(Thearch.Optoas(OADD, Types[Tptr]), &n1, &n2)
				}
				*a = n2
				break
			}

			// i is in register n1, extend to 32 bits.
			t := Types[TUINT32]

			if Issigned[n1.Type.Etype] {
				t = Types[TINT32]
			}

			Regalloc(&n2, t, &n1) // i
			Thearch.Gmove(&n1, &n2)
			Regfree(&n1)

			if Debug['B'] == 0 && !n.Bounded {
				// check bounds
				t := Types[TUINT32]

				var nlen Node
				if Isconst(nl, CTSTR) {
					Nodconst(&nlen, t, int64(len(nl.Val.U.Sval)))
				} else if Isslice(nl.Type) || nl.Type.Etype == TSTRING {
					nlen = n3
					nlen.Type = t
					nlen.Xoffset += int64(Array_nel)
				} else {
					Nodconst(&nlen, t, nl.Type.Bound)
				}

				Thearch.Gins(Thearch.Optoas(OCMP, t), &n2, &nlen)
				p1 := Gbranch(Thearch.Optoas(OLT, t), nil, +1)
				if p2 != nil {
					Patch(p2, Pc)
				}
				Ginscall(Panicindex, -1)
				Patch(p1, Pc)
			}

			if Isconst(nl, CTSTR) {
				Regalloc(&n3, Types[Tptr], res)
				p1 := Thearch.Gins(Thearch.Optoas(OAS, Types[Tptr]), nil, &n3)
				Datastring(nl.Val.U.Sval, &p1.From)
				p1.From.Type = obj.TYPE_ADDR
				Thearch.Gins(Thearch.Optoas(OADD, n3.Type), &n2, &n3)
				goto indexdone1
			}

			// Load base pointer in n3.
			Regalloc(&tmp, Types[Tptr], &n3)

			if Isslice(nl.Type) || nl.Type.Etype == TSTRING {
				n3.Type = Types[Tptr]
				n3.Xoffset += int64(Array_array)
				Thearch.Gmove(&n3, &tmp)
			}

			Regfree(&n3)
			n3 = tmp

			if w == 0 {
				// nothing to do
			} else if Thearch.AddIndex != nil && Thearch.AddIndex(&n2, int64(w), &n3) {
				// done by back end
			} else if w == 1 {
				Thearch.Gins(Thearch.Optoas(OADD, Types[Tptr]), &n2, &n3)
			} else {
				Nodconst(&tmp, Types[TUINT32], int64(w))
				Thearch.Gins(Thearch.Optoas(OMUL, Types[TUINT32]), &tmp, &n2)
				Thearch.Gins(Thearch.Optoas(OADD, Types[Tptr]), &n2, &n3)
			}

		indexdone1:
			*a = n3
			Regfree(&n2)
			break
		}

		freelen := 0
		w := uint64(n.Type.Width)

		// Generate the non-addressable child first.
		var n3 Node
		var nlen Node
		var tmp Node
		var n1 Node
		if nr.Addable {
			goto irad
		}
		if nl.Addable {
			Cgenr(nr, &n1, nil)
			if !Isconst(nl, CTSTR) {
				if Isfixedarray(nl.Type) {
					Agenr(nl, &n3, res)
				} else {
					Igen(nl, &nlen, res)
					freelen = 1
					nlen.Type = Types[Tptr]
					nlen.Xoffset += int64(Array_array)
					Regalloc(&n3, Types[Tptr], res)
					Thearch.Gmove(&nlen, &n3)
					nlen.Type = Types[Simtype[TUINT]]
					nlen.Xoffset += int64(Array_nel) - int64(Array_array)
				}
			}

			goto index
		}

		Tempname(&tmp, nr.Type)
		Cgen(nr, &tmp)
		nr = &tmp

	irad:
		if !Isconst(nl, CTSTR) {
			if Isfixedarray(nl.Type) {
				Agenr(nl, &n3, res)
			} else {
				if !nl.Addable {
					if res != nil && res.Op == OREGISTER { // give up res, which we don't need yet.
						Regfree(res)
					}

					// igen will need an addressable node.
					var tmp2 Node
					Tempname(&tmp2, nl.Type)
					Cgen(nl, &tmp2)
					nl = &tmp2

					if res != nil && res.Op == OREGISTER { // reacquire res
						Regrealloc(res)
					}
				}

				Igen(nl, &nlen, res)
				freelen = 1
				nlen.Type = Types[Tptr]
				nlen.Xoffset += int64(Array_array)
				Regalloc(&n3, Types[Tptr], res)
				Thearch.Gmove(&nlen, &n3)
				nlen.Type = Types[Simtype[TUINT]]
				nlen.Xoffset += int64(Array_nel) - int64(Array_array)
			}
		}

		if !Isconst(nr, CTINT) {
			Cgenr(nr, &n1, nil)
		}

		goto index

		// &a is in &n3 (allocated in res)
		// i is in &n1 (if not constant)
		// len(a) is in nlen (if needed)
		// w is width

		// constant index
	index:
		if Isconst(nr, CTINT) {
			if Isconst(nl, CTSTR) {
				Fatal("constant string constant index") // front end should handle
			}
			v := uint64(Mpgetfix(nr.Val.U.Xval))
			if Isslice(nl.Type) || nl.Type.Etype == TSTRING {
				if Debug['B'] == 0 && !n.Bounded {
					if nlen.Op != OREGISTER && (Ctxt.Arch.Thechar == '7' || Ctxt.Arch.Thechar == '9') {
						var tmp2 Node
						Regalloc(&tmp2, Types[Simtype[TUINT]], nil)
						Thearch.Gmove(&nlen, &tmp2)
						Regfree(&nlen) // in case it is OINDREG
						nlen = tmp2
					}
					var n2 Node
					Nodconst(&n2, Types[Simtype[TUINT]], int64(v))
					if Smallintconst(nr) {
						Thearch.Gins(Thearch.Optoas(OCMP, Types[Simtype[TUINT]]), &nlen, &n2)
					} else {
						Regalloc(&tmp, Types[Simtype[TUINT]], nil)
						Thearch.Gmove(&n2, &tmp)
						Thearch.Gins(Thearch.Optoas(OCMP, Types[Simtype[TUINT]]), &nlen, &tmp)
						Regfree(&tmp)
					}

					p1 := Gbranch(Thearch.Optoas(OGT, Types[Simtype[TUINT]]), nil, +1)
					Ginscall(Panicindex, -1)
					Patch(p1, Pc)
				}

				Regfree(&nlen)
			}

			if v*w != 0 {
				Thearch.Ginscon(Thearch.Optoas(OADD, Types[Tptr]), int64(v*w), &n3)
			}
			*a = n3
			break
		}

		// type of the index
		t := Types[TUINT64]

		if Issigned[n1.Type.Etype] {
			t = Types[TINT64]
		}

		var n2 Node
		Regalloc(&n2, t, &n1) // i
		Thearch.Gmove(&n1, &n2)
		Regfree(&n1)

		if Debug['B'] == 0 && !n.Bounded {
			// check bounds
			t = Types[Simtype[TUINT]]

			if Is64(nr.Type) {
				t = Types[TUINT64]
			}
			if Isconst(nl, CTSTR) {
				Nodconst(&nlen, t, int64(len(nl.Val.U.Sval)))
			} else if Isslice(nl.Type) || nl.Type.Etype == TSTRING {
				if Is64(nr.Type) || Ctxt.Arch.Thechar == '7' || Ctxt.Arch.Thechar == '9' {
					var n5 Node
					Regalloc(&n5, t, nil)
					Thearch.Gmove(&nlen, &n5)
					Regfree(&nlen)
					nlen = n5
				}
			} else {
				Nodconst(&nlen, t, nl.Type.Bound)
				if !Smallintconst(&nlen) {
					var n5 Node
					Regalloc(&n5, t, nil)
					Thearch.Gmove(&nlen, &n5)
					nlen = n5
					freelen = 1
				}
			}

			Thearch.Gins(Thearch.Optoas(OCMP, t), &n2, &nlen)
			p1 := Gbranch(Thearch.Optoas(OLT, t), nil, +1)
			Ginscall(Panicindex, -1)
			Patch(p1, Pc)
		}

		if Isconst(nl, CTSTR) {
			Regalloc(&n3, Types[Tptr], res)
			p1 := Thearch.Gins(Thearch.Optoas(OAS, n3.Type), nil, &n3) // XXX was LEAQ!
			Datastring(nl.Val.U.Sval, &p1.From)
			p1.From.Type = obj.TYPE_ADDR
			Thearch.Gins(Thearch.Optoas(OADD, n3.Type), &n2, &n3)
			goto indexdone
		}

		if w == 0 {
			// nothing to do
		} else if Thearch.AddIndex != nil && Thearch.AddIndex(&n2, int64(w), &n3) {
			// done by back end
		} else if w == 1 {
			Thearch.Gins(Thearch.Optoas(OADD, Types[Tptr]), &n2, &n3)
		} else {
			Thearch.Ginscon(Thearch.Optoas(OMUL, t), int64(w), &n2)
			Thearch.Gins(Thearch.Optoas(OADD, Types[Tptr]), &n2, &n3)
		}

	indexdone:
		*a = n3
		Regfree(&n2)
		if freelen != 0 {
			Regfree(&nlen)
		}

	default:
		Regalloc(a, Types[Tptr], res)
		Agen(n, a)
	}
}

/*
 * generate:
 *	res = &n;
 * The generated code checks that the result is not nil.
 */
func Agen(n *Node, res *Node) {
	if Debug['g'] != 0 {
		Dump("\nagen-res", res)
		Dump("agen-r", n)
	}

	if n == nil || n.Type == nil {
		return
	}

	for n.Op == OCONVNOP {
		n = n.Left
	}

	if Isconst(n, CTNIL) && n.Type.Width > int64(Widthptr) {
		// Use of a nil interface or nil slice.
		// Create a temporary we can take the address of and read.
		// The generated code is just going to panic, so it need not
		// be terribly efficient. See issue 3670.
		var n1 Node
		Tempname(&n1, n.Type)

		Gvardef(&n1)
		Thearch.Clearfat(&n1)
		var n2 Node
		Regalloc(&n2, Types[Tptr], res)
		var n3 Node
		n3.Op = OADDR
		n3.Left = &n1
		Thearch.Gins(Thearch.Optoas(OAS, Types[Tptr]), &n3, &n2)
		Thearch.Gmove(&n2, res)
		Regfree(&n2)
		return
	}

	if n.Addable {
		if n.Op == OREGISTER {
			Fatal("agen OREGISTER")
		}
		var n1 Node
		n1.Op = OADDR
		n1.Left = n
		var n2 Node
		Regalloc(&n2, Types[Tptr], res)
		Thearch.Gins(Thearch.Optoas(OAS, Types[Tptr]), &n1, &n2)
		Thearch.Gmove(&n2, res)
		Regfree(&n2)
		return
	}

	nl := n.Left

	switch n.Op {
	default:
		Fatal("agen: unknown op %v", Nconv(n, obj.FmtShort|obj.FmtSign))

	case OCALLMETH:
		cgen_callmeth(n, 0)
		cgen_aret(n, res)

	case OCALLINTER:
		cgen_callinter(n, res, 0)
		cgen_aret(n, res)

	case OCALLFUNC:
		cgen_call(n, 0)
		cgen_aret(n, res)

	case OEFACE, ODOTTYPE, OSLICE, OSLICEARR, OSLICESTR, OSLICE3, OSLICE3ARR:
		var n1 Node
		Tempname(&n1, n.Type)
		Cgen(n, &n1)
		Agen(&n1, res)

	case OINDEX:
		var n1 Node
		Agenr(n, &n1, res)
		Thearch.Gmove(&n1, res)
		Regfree(&n1)

	case ONAME:
		// should only get here with names in this func.
		if n.Funcdepth > 0 && n.Funcdepth != Funcdepth {
			Dump("bad agen", n)
			Fatal("agen: bad ONAME funcdepth %d != %d", n.Funcdepth, Funcdepth)
		}

		// should only get here for heap vars or paramref
		if n.Class&PHEAP == 0 && n.Class != PPARAMREF {
			Dump("bad agen", n)
			Fatal("agen: bad ONAME class %#x", n.Class)
		}

		Cgen(n.Heapaddr, res)
		if n.Xoffset != 0 {
			addOffset(res, n.Xoffset)
		}

	case OIND:
		Cgen(nl, res)
		Cgen_checknil(res)

	case ODOT:
		Agen(nl, res)
		if n.Xoffset != 0 {
			addOffset(res, n.Xoffset)
		}

	case ODOTPTR:
		Cgen(nl, res)
		Cgen_checknil(res)
		if n.Xoffset != 0 {
			addOffset(res, n.Xoffset)
		}
	}
}

func addOffset(res *Node, offset int64) {
	if Ctxt.Arch.Thechar == '6' || Ctxt.Arch.Thechar == '8' {
		Thearch.Gins(Thearch.Optoas(OADD, Types[Tptr]), Nodintconst(offset), res)
		return
	}

	var n1, n2 Node
	Regalloc(&n1, Types[Tptr], nil)
	Thearch.Gmove(res, &n1)
	Regalloc(&n2, Types[Tptr], nil)
	Thearch.Gins(Thearch.Optoas(OAS, Types[Tptr]), Nodintconst(offset), &n2)
	Thearch.Gins(Thearch.Optoas(OADD, Types[Tptr]), &n2, &n1)
	Thearch.Gmove(&n1, res)
	Regfree(&n1)
	Regfree(&n2)
}

// Igen computes the address &n, stores it in a register r,
// and rewrites a to refer to *r. The chosen r may be the
// stack pointer, it may be borrowed from res, or it may
// be a newly allocated register. The caller must call Regfree(a)
// to free r when the address is no longer needed.
// The generated code ensures that &n is not nil.
func Igen(n *Node, a *Node, res *Node) {
	if Debug['g'] != 0 {
		Dump("\nigen-n", n)
	}

	switch n.Op {
	case ONAME:
		if (n.Class&PHEAP != 0) || n.Class == PPARAMREF {
			break
		}
		*a = *n
		return

	case OINDREG:
		// Increase the refcount of the register so that igen's caller
		// has to call Regfree.
		if n.Reg != int16(Thearch.REGSP) {
			reg[n.Reg-int16(Thearch.REGMIN)]++
		}
		*a = *n
		return

	case ODOT:
		Igen(n.Left, a, res)
		a.Xoffset += n.Xoffset
		a.Type = n.Type
		Fixlargeoffset(a)
		return

	case ODOTPTR:
		Cgenr(n.Left, a, res)
		Cgen_checknil(a)
		a.Op = OINDREG
		a.Xoffset += n.Xoffset
		a.Type = n.Type
		Fixlargeoffset(a)
		return

	case OCALLFUNC, OCALLMETH, OCALLINTER:
		switch n.Op {
		case OCALLFUNC:
			cgen_call(n, 0)

		case OCALLMETH:
			cgen_callmeth(n, 0)

		case OCALLINTER:
			cgen_callinter(n, nil, 0)
		}

		var flist Iter
		fp := Structfirst(&flist, Getoutarg(n.Left.Type))
		*a = Node{}
		a.Op = OINDREG
		a.Reg = int16(Thearch.REGSP)
		a.Addable = true
		a.Xoffset = fp.Width
		if HasLinkRegister() {
			a.Xoffset += int64(Ctxt.Arch.Ptrsize)
		}
		a.Type = n.Type
		return

		// Index of fixed-size array by constant can
	// put the offset in the addressing.
	// Could do the same for slice except that we need
	// to use the real index for the bounds checking.
	case OINDEX:
		if Isfixedarray(n.Left.Type) || (Isptr[n.Left.Type.Etype] && Isfixedarray(n.Left.Left.Type)) {
			if Isconst(n.Right, CTINT) {
				// Compute &a.
				if !Isptr[n.Left.Type.Etype] {
					Igen(n.Left, a, res)
				} else {
					var n1 Node
					Igen(n.Left, &n1, res)
					Cgen_checknil(&n1)
					Regalloc(a, Types[Tptr], res)
					Thearch.Gmove(&n1, a)
					Regfree(&n1)
					a.Op = OINDREG
				}

				// Compute &a[i] as &a + i*width.
				a.Type = n.Type

				a.Xoffset += Mpgetfix(n.Right.Val.U.Xval) * n.Type.Width
				Fixlargeoffset(a)
				return
			}
		}
	}

	Agenr(n, a, res)
	a.Op = OINDREG
	a.Type = n.Type
}

/*
 * generate:
 *	if(n == true) goto to;
 */
func Bgen(n *Node, true_ bool, likely int, to *obj.Prog) {
	if Debug['g'] != 0 {
		Dump("\nbgen", n)
	}

	if n == nil {
		n = Nodbool(true)
	}

	if n.Ninit != nil {
		Genlist(n.Ninit)
	}

	if n.Type == nil {
		Convlit(&n, Types[TBOOL])
		if n.Type == nil {
			return
		}
	}

	et := int(n.Type.Etype)
	if et != TBOOL {
		Yyerror("cgen: bad type %v for %v", Tconv(n.Type, 0), Oconv(int(n.Op), 0))
		Patch(Thearch.Gins(obj.AEND, nil, nil), to)
		return
	}

	for n.Op == OCONVNOP {
		n = n.Left
		if n.Ninit != nil {
			Genlist(n.Ninit)
		}
	}

	if Thearch.Bgen_float != nil && n.Left != nil && Isfloat[n.Left.Type.Etype] {
		Thearch.Bgen_float(n, bool2int(true_), likely, to)
		return
	}

	var nl *Node
	var nr *Node
	switch n.Op {
	default:
		goto def

		// need to ask if it is bool?
	case OLITERAL:
		if true_ == n.Val.U.Bval {
			Patch(Gbranch(obj.AJMP, nil, likely), to)
		}
		return

	case ONAME:
		if !n.Addable || Ctxt.Arch.Thechar == '5' || Ctxt.Arch.Thechar == '7' || Ctxt.Arch.Thechar == '9' {
			goto def
		}
		var n1 Node
		Nodconst(&n1, n.Type, 0)
		Thearch.Gins(Thearch.Optoas(OCMP, n.Type), n, &n1)
		a := Thearch.Optoas(ONE, n.Type)
		if !true_ {
			a = Thearch.Optoas(OEQ, n.Type)
		}
		Patch(Gbranch(a, n.Type, likely), to)
		return

	case OANDAND, OOROR:
		if (n.Op == OANDAND) == true_ {
			p1 := Gbranch(obj.AJMP, nil, 0)
			p2 := Gbranch(obj.AJMP, nil, 0)
			Patch(p1, Pc)
			Bgen(n.Left, !true_, -likely, p2)
			Bgen(n.Right, !true_, -likely, p2)
			p1 = Gbranch(obj.AJMP, nil, 0)
			Patch(p1, to)
			Patch(p2, Pc)
		} else {
			Bgen(n.Left, true_, likely, to)
			Bgen(n.Right, true_, likely, to)
		}

		return

	case OEQ, ONE, OLT, OGT, OLE, OGE:
		nr = n.Right
		if nr == nil || nr.Type == nil {
			return
		}
		fallthrough

	case ONOT: // unary
		nl = n.Left

		if nl == nil || nl.Type == nil {
			return
		}
	}

	switch n.Op {
	case ONOT:
		Bgen(nl, !true_, likely, to)
		return

	case OEQ, ONE, OLT, OGT, OLE, OGE:
		a := int(n.Op)
		if !true_ {
			if Isfloat[nr.Type.Etype] {
				// brcom is not valid on floats when NaN is involved.
				p1 := Gbranch(obj.AJMP, nil, 0)
				p2 := Gbranch(obj.AJMP, nil, 0)
				Patch(p1, Pc)
				ll := n.Ninit // avoid re-genning ninit
				n.Ninit = nil
				Bgen(n, true, -likely, p2)
				n.Ninit = ll
				Patch(Gbranch(obj.AJMP, nil, 0), to)
				Patch(p2, Pc)
				return
			}

			a = Brcom(a)
			true_ = !true_
		}

		// make simplest on right
		if nl.Op == OLITERAL || (nl.Ullman < nr.Ullman && nl.Ullman < UINF) {
			a = Brrev(a)
			r := nl
			nl = nr
			nr = r
		}

		if Isslice(nl.Type) {
			// front end should only leave cmp to literal nil
			if (a != OEQ && a != ONE) || nr.Op != OLITERAL {
				Yyerror("illegal slice comparison")
				break
			}

			a = Thearch.Optoas(a, Types[Tptr])
			var n1 Node
			Igen(nl, &n1, nil)
			n1.Xoffset += int64(Array_array)
			n1.Type = Types[Tptr]
			var n2 Node
			Regalloc(&n2, Types[Tptr], &n1)
			Cgen(&n1, &n2)
			Regfree(&n1)
			var tmp Node
			Nodconst(&tmp, Types[Tptr], 0)
			Thearch.Gins(Thearch.Optoas(OCMP, Types[Tptr]), &n2, &tmp)
			Patch(Gbranch(a, Types[Tptr], likely), to)
			Regfree(&n2)
			break
		}

		if Isinter(nl.Type) {
			// front end should only leave cmp to literal nil
			if (a != OEQ && a != ONE) || nr.Op != OLITERAL {
				Yyerror("illegal interface comparison")
				break
			}

			a = Thearch.Optoas(a, Types[Tptr])
			var n1 Node
			Igen(nl, &n1, nil)
			n1.Type = Types[Tptr]
			var n2 Node
			Regalloc(&n2, Types[Tptr], &n1)
			Cgen(&n1, &n2)
			Regfree(&n1)
			var tmp Node
			Nodconst(&tmp, Types[Tptr], 0)
			Thearch.Gins(Thearch.Optoas(OCMP, Types[Tptr]), &n2, &tmp)
			Patch(Gbranch(a, Types[Tptr], likely), to)
			Regfree(&n2)
			break
		}

		if Iscomplex[nl.Type.Etype] {
			Complexbool(a, nl, nr, true_, likely, to)
			break
		}

		if Ctxt.Arch.Regsize == 4 && Is64(nr.Type) {
			if !nl.Addable || Isconst(nl, CTINT) {
				var n1 Node
				Tempname(&n1, nl.Type)
				Cgen(nl, &n1)
				nl = &n1
			}

			if !nr.Addable {
				var n2 Node
				Tempname(&n2, nr.Type)
				Cgen(nr, &n2)
				nr = &n2
			}

			Thearch.Cmp64(nl, nr, a, likely, to)
			break
		}

		var n1 Node
		var n2 Node
		if nr.Ullman >= UINF {
			Regalloc(&n1, nl.Type, nil)
			Cgen(nl, &n1)

			var tmp Node
			Tempname(&tmp, nl.Type)
			Thearch.Gmove(&n1, &tmp)
			Regfree(&n1)

			Regalloc(&n2, nr.Type, nil)
			Cgen(nr, &n2)

			Regalloc(&n1, nl.Type, nil)
			Cgen(&tmp, &n1)

			goto cmp
		}

		if !nl.Addable && Ctxt.Arch.Thechar == '8' {
			Tempname(&n1, nl.Type)
		} else {
			Regalloc(&n1, nl.Type, nil)
		}
		Cgen(nl, &n1)
		nl = &n1

		if Smallintconst(nr) && Ctxt.Arch.Thechar != '9' {
			Thearch.Gins(Thearch.Optoas(OCMP, nr.Type), nl, nr)
			Patch(Gbranch(Thearch.Optoas(a, nr.Type), nr.Type, likely), to)
			if n1.Op == OREGISTER {
				Regfree(&n1)
			}
			break
		}

		if !nr.Addable && Ctxt.Arch.Thechar == '8' {
			var tmp Node
			Tempname(&tmp, nr.Type)
			Cgen(nr, &tmp)
			nr = &tmp
		}

		Regalloc(&n2, nr.Type, nil)
		Cgen(nr, &n2)
		nr = &n2

	cmp:
		l, r := nl, nr
		// On x86, only < and <= work right with NaN; reverse if needed
		if Ctxt.Arch.Thechar == '6' && Isfloat[nl.Type.Etype] && (a == OGT || a == OGE) {
			l, r = r, l
			a = Brrev(a)
		}

		Thearch.Gins(Thearch.Optoas(OCMP, nr.Type), l, r)

		if Ctxt.Arch.Thechar == '6' && Isfloat[nr.Type.Etype] && (n.Op == OEQ || n.Op == ONE) {
			if n.Op == OEQ {
				// neither NE nor P
				p1 := Gbranch(Thearch.Optoas(ONE, nr.Type), nil, -likely)
				p2 := Gbranch(Thearch.Optoas(OPS, nr.Type), nil, -likely)
				Patch(Gbranch(obj.AJMP, nil, 0), to)
				Patch(p1, Pc)
				Patch(p2, Pc)
			} else {
				// either NE or P
				Patch(Gbranch(Thearch.Optoas(ONE, nr.Type), nil, likely), to)
				Patch(Gbranch(Thearch.Optoas(OPS, nr.Type), nil, likely), to)
			}
		} else if Ctxt.Arch.Thechar == '5' && Isfloat[nl.Type.Etype] {
			if n.Op == ONE {
				Patch(Gbranch(Thearch.Optoas(OPS, nr.Type), nr.Type, likely), to)
				Patch(Gbranch(Thearch.Optoas(a, nr.Type), nr.Type, likely), to)
			} else {
				p := Gbranch(Thearch.Optoas(OPS, nr.Type), nr.Type, -likely)
				Patch(Gbranch(Thearch.Optoas(a, nr.Type), nr.Type, likely), to)
				Patch(p, Pc)
			}
		} else if (Ctxt.Arch.Thechar == '7' || Ctxt.Arch.Thechar == '9') && Isfloat[nl.Type.Etype] && (a == OLE || a == OGE) {
			// On arm64 and ppc64, <= and >= mishandle NaN. Must decompose into < or > and =.
			if a == OLE {
				a = OLT
			} else {
				a = OGT
			}
			Patch(Gbranch(Thearch.Optoas(a, nr.Type), nr.Type, likely), to)
			Patch(Gbranch(Thearch.Optoas(OEQ, nr.Type), nr.Type, likely), to)
		} else {
			Patch(Gbranch(Thearch.Optoas(a, nr.Type), nr.Type, likely), to)
		}
		if n1.Op == OREGISTER {
			Regfree(&n1)
		}
		if n2.Op == OREGISTER {
			Regfree(&n2)
		}
	}

	return

def:
	// TODO: Optimize on systems that can compare to zero easily.
	var n1 Node
	Regalloc(&n1, n.Type, nil)
	Cgen(n, &n1)
	var n2 Node
	Nodconst(&n2, n.Type, 0)
	Thearch.Gins(Thearch.Optoas(OCMP, n.Type), &n1, &n2)
	a := Thearch.Optoas(ONE, n.Type)
	if !true_ {
		a = Thearch.Optoas(OEQ, n.Type)
	}
	Patch(Gbranch(a, n.Type, likely), to)
	Regfree(&n1)
	return
}

/*
 * n is on stack, either local variable
 * or return value from function call.
 * return n's offset from SP.
 */
func stkof(n *Node) int64 {
	switch n.Op {
	case OINDREG:
		return n.Xoffset

	case ODOT:
		t := n.Left.Type
		if Isptr[t.Etype] {
			break
		}
		off := stkof(n.Left)
		if off == -1000 || off == 1000 {
			return off
		}
		return off + n.Xoffset

	case OINDEX:
		t := n.Left.Type
		if !Isfixedarray(t) {
			break
		}
		off := stkof(n.Left)
		if off == -1000 || off == 1000 {
			return off
		}
		if Isconst(n.Right, CTINT) {
			return off + t.Type.Width*Mpgetfix(n.Right.Val.U.Xval)
		}
		return 1000

	case OCALLMETH, OCALLINTER, OCALLFUNC:
		t := n.Left.Type
		if Isptr[t.Etype] {
			t = t.Type
		}

		var flist Iter
		t = Structfirst(&flist, Getoutarg(t))
		if t != nil {
			w := t.Width
			if HasLinkRegister() {
				w += int64(Ctxt.Arch.Ptrsize)
			}
			return w
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
func sgen(n *Node, ns *Node, w int64) {
	if Debug['g'] != 0 {
		fmt.Printf("\nsgen w=%d\n", w)
		Dump("r", n)
		Dump("res", ns)
	}

	if n.Ullman >= UINF && ns.Ullman >= UINF {
		Fatal("sgen UINF")
	}

	if w < 0 {
		Fatal("sgen copy %d", w)
	}

	// If copying .args, that's all the results, so record definition sites
	// for them for the liveness analysis.
	if ns.Op == ONAME && ns.Sym.Name == ".args" {
		for l := Curfn.Func.Dcl; l != nil; l = l.Next {
			if l.N.Class == PPARAMOUT {
				Gvardef(l.N)
			}
		}
	}

	// Avoid taking the address for simple enough types.
	if Componentgen(n, ns) {
		return
	}

	if w == 0 {
		// evaluate side effects only
		var nodr Node
		Regalloc(&nodr, Types[Tptr], nil)
		Agen(ns, &nodr)
		Agen(n, &nodr)
		Regfree(&nodr)
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
		var tmp Node
		Tempname(&tmp, n.Type)
		sgen(n, &tmp, w)
		sgen(&tmp, ns, w)
		return
	}

	Thearch.Stackcopy(n, ns, osrc, odst, w)
}

/*
 * generate:
 *	call f
 *	proc=-1	normal call but no return
 *	proc=0	normal call
 *	proc=1	goroutine run in new proc
 *	proc=2	defer call save away stack
  *	proc=3	normal call to C pointer (not Go func value)
*/
func Ginscall(f *Node, proc int) {
	if f.Type != nil {
		extra := int32(0)
		if proc == 1 || proc == 2 {
			extra = 2 * int32(Widthptr)
		}
		Setmaxarg(f.Type, extra)
	}

	switch proc {
	default:
		Fatal("Ginscall: bad proc %d", proc)

	case 0, // normal call
		-1: // normal call but no return
		if f.Op == ONAME && f.Class == PFUNC {
			if f == Deferreturn {
				// Deferred calls will appear to be returning to
				// the CALL deferreturn(SB) that we are about to emit.
				// However, the stack trace code will show the line
				// of the instruction byte before the return PC.
				// To avoid that being an unrelated instruction,
				// insert an actual hardware NOP that will have the right line number.
				// This is different from obj.ANOP, which is a virtual no-op
				// that doesn't make it into the instruction stream.
				Thearch.Ginsnop()
			}

			p := Thearch.Gins(obj.ACALL, nil, f)
			Afunclit(&p.To, f)
			if proc == -1 || Noreturn(p) {
				Thearch.Gins(obj.AUNDEF, nil, nil)
			}
			break
		}

		var reg Node
		Nodreg(&reg, Types[Tptr], Thearch.REGCTXT)
		var r1 Node
		Nodreg(&r1, Types[Tptr], Thearch.REGCALLX)
		Thearch.Gmove(f, &reg)
		reg.Op = OINDREG
		Thearch.Gmove(&reg, &r1)
		reg.Op = OREGISTER
		Thearch.Gins(obj.ACALL, &reg, &r1)

	case 3: // normal call of c function pointer
		Thearch.Gins(obj.ACALL, nil, f)

	case 1, // call in new proc (go)
		2: // deferred call (defer)
		var stk Node

		// size of arguments at 0(SP)
		stk.Op = OINDREG
		stk.Reg = int16(Thearch.REGSP)
		stk.Xoffset = 0
		if HasLinkRegister() {
			stk.Xoffset += int64(Ctxt.Arch.Ptrsize)
		}
		Thearch.Ginscon(Thearch.Optoas(OAS, Types[Tptr]), int64(Argsize(f.Type)), &stk)

		// FuncVal* at 8(SP)
		stk.Xoffset = int64(Widthptr)
		if HasLinkRegister() {
			stk.Xoffset += int64(Ctxt.Arch.Ptrsize)
		}

		var reg Node
		Nodreg(&reg, Types[Tptr], Thearch.REGCALLX2)
		Thearch.Gmove(f, &reg)
		Thearch.Gins(Thearch.Optoas(OAS, Types[Tptr]), &reg, &stk)

		if proc == 1 {
			Ginscall(Newproc, 0)
		} else {
			if Hasdefer == 0 {
				Fatal("hasdefer=0 but has defer")
			}
			Ginscall(Deferproc, 0)
		}

		if proc == 2 {
			Nodreg(&reg, Types[TINT32], Thearch.REGRETURN)
			Thearch.Gins(Thearch.Optoas(OCMP, Types[TINT32]), &reg, Nodintconst(0))
			p := Gbranch(Thearch.Optoas(OEQ, Types[TINT32]), nil, +1)
			cgen_ret(nil)
			Patch(p, Pc)
		}
	}
}

/*
 * n is call to interface method.
 * generate res = n.
 */
func cgen_callinter(n *Node, res *Node, proc int) {
	i := n.Left
	if i.Op != ODOTINTER {
		Fatal("cgen_callinter: not ODOTINTER %v", Oconv(int(i.Op), 0))
	}

	f := i.Right // field
	if f.Op != ONAME {
		Fatal("cgen_callinter: not ONAME %v", Oconv(int(f.Op), 0))
	}

	i = i.Left // interface

	if !i.Addable {
		var tmpi Node
		Tempname(&tmpi, i.Type)
		Cgen(i, &tmpi)
		i = &tmpi
	}

	Genlist(n.List) // assign the args

	// i is now addable, prepare an indirected
	// register to hold its address.
	var nodi Node
	Igen(i, &nodi, res) // REG = &inter

	var nodsp Node
	Nodindreg(&nodsp, Types[Tptr], Thearch.REGSP)
	nodsp.Xoffset = 0
	if HasLinkRegister() {
		nodsp.Xoffset += int64(Ctxt.Arch.Ptrsize)
	}
	if proc != 0 {
		nodsp.Xoffset += 2 * int64(Widthptr) // leave room for size & fn
	}
	nodi.Type = Types[Tptr]
	nodi.Xoffset += int64(Widthptr)
	Cgen(&nodi, &nodsp) // {0, 8(nacl), or 16}(SP) = 8(REG) -- i.data

	var nodo Node
	Regalloc(&nodo, Types[Tptr], res)

	nodi.Type = Types[Tptr]
	nodi.Xoffset -= int64(Widthptr)
	Cgen(&nodi, &nodo) // REG = 0(REG) -- i.tab
	Regfree(&nodi)

	var nodr Node
	Regalloc(&nodr, Types[Tptr], &nodo)
	if n.Left.Xoffset == BADWIDTH {
		Fatal("cgen_callinter: badwidth")
	}
	Cgen_checknil(&nodo) // in case offset is huge
	nodo.Op = OINDREG
	nodo.Xoffset = n.Left.Xoffset + 3*int64(Widthptr) + 8
	if proc == 0 {
		// plain call: use direct c function pointer - more efficient
		Cgen(&nodo, &nodr) // REG = 32+offset(REG) -- i.tab->fun[f]
		proc = 3
	} else {
		// go/defer. generate go func value.
		Agen(&nodo, &nodr) // REG = &(32+offset(REG)) -- i.tab->fun[f]
	}

	nodr.Type = n.Left.Type
	Ginscall(&nodr, proc)

	Regfree(&nodr)
	Regfree(&nodo)
}

/*
 * generate function call;
 *	proc=0	normal call
 *	proc=1	goroutine run in new proc
 *	proc=2	defer call save away stack
 */
func cgen_call(n *Node, proc int) {
	if n == nil {
		return
	}

	var afun Node
	if n.Left.Ullman >= UINF {
		// if name involves a fn call
		// precompute the address of the fn
		Tempname(&afun, Types[Tptr])

		Cgen(n.Left, &afun)
	}

	Genlist(n.List) // assign the args
	t := n.Left.Type

	// call tempname pointer
	if n.Left.Ullman >= UINF {
		var nod Node
		Regalloc(&nod, Types[Tptr], nil)
		Cgen_as(&nod, &afun)
		nod.Type = t
		Ginscall(&nod, proc)
		Regfree(&nod)
		return
	}

	// call pointer
	if n.Left.Op != ONAME || n.Left.Class != PFUNC {
		var nod Node
		Regalloc(&nod, Types[Tptr], nil)
		Cgen_as(&nod, n.Left)
		nod.Type = t
		Ginscall(&nod, proc)
		Regfree(&nod)
		return
	}

	// call direct
	n.Left.Method = true

	Ginscall(n.Left, proc)
}

func HasLinkRegister() bool {
	c := Ctxt.Arch.Thechar
	return c != '6' && c != '8'
}

/*
 * call to n has already been generated.
 * generate:
 *	res = return value from call.
 */
func cgen_callret(n *Node, res *Node) {
	t := n.Left.Type
	if t.Etype == TPTR32 || t.Etype == TPTR64 {
		t = t.Type
	}

	var flist Iter
	fp := Structfirst(&flist, Getoutarg(t))
	if fp == nil {
		Fatal("cgen_callret: nil")
	}

	var nod Node
	nod.Op = OINDREG
	nod.Reg = int16(Thearch.REGSP)
	nod.Addable = true

	nod.Xoffset = fp.Width
	if HasLinkRegister() {
		nod.Xoffset += int64(Ctxt.Arch.Ptrsize)
	}
	nod.Type = fp.Type
	Cgen_as(res, &nod)
}

/*
 * call to n has already been generated.
 * generate:
 *	res = &return value from call.
 */
func cgen_aret(n *Node, res *Node) {
	t := n.Left.Type
	if Isptr[t.Etype] {
		t = t.Type
	}

	var flist Iter
	fp := Structfirst(&flist, Getoutarg(t))
	if fp == nil {
		Fatal("cgen_aret: nil")
	}

	var nod1 Node
	nod1.Op = OINDREG
	nod1.Reg = int16(Thearch.REGSP)
	nod1.Addable = true
	nod1.Xoffset = fp.Width
	if HasLinkRegister() {
		nod1.Xoffset += int64(Ctxt.Arch.Ptrsize)
	}
	nod1.Type = fp.Type

	if res.Op != OREGISTER {
		var nod2 Node
		Regalloc(&nod2, Types[Tptr], res)
		Agen(&nod1, &nod2)
		Thearch.Gins(Thearch.Optoas(OAS, Types[Tptr]), &nod2, res)
		Regfree(&nod2)
	} else {
		Agen(&nod1, res)
	}
}

/*
 * generate return.
 * n->left is assignments to return values.
 */
func cgen_ret(n *Node) {
	if n != nil {
		Genlist(n.List) // copy out args
	}
	if Hasdefer != 0 {
		Ginscall(Deferreturn, 0)
	}
	Genlist(Curfn.Func.Exit)
	p := Thearch.Gins(obj.ARET, nil, nil)
	if n != nil && n.Op == ORETJMP {
		p.To.Type = obj.TYPE_MEM
		p.To.Name = obj.NAME_EXTERN
		p.To.Sym = Linksym(n.Left.Sym)
	}
}

/*
 * generate division according to op, one of:
 *	res = nl / nr
 *	res = nl % nr
 */
func cgen_div(op int, nl *Node, nr *Node, res *Node) {
	var w int

	// TODO(rsc): arm64 needs to support the relevant instructions
	// in peep and optoas in order to enable this.
	// TODO(rsc): ppc64 needs to support the relevant instructions
	// in peep and optoas in order to enable this.
	if nr.Op != OLITERAL || Ctxt.Arch.Thechar == '7' || Ctxt.Arch.Thechar == '9' {
		goto longdiv
	}
	w = int(nl.Type.Width * 8)

	// Front end handled 32-bit division. We only need to handle 64-bit.
	// try to do division by multiply by (2^w)/d
	// see hacker's delight chapter 10
	switch Simtype[nl.Type.Etype] {
	default:
		goto longdiv

	case TUINT64:
		var m Magic
		m.W = w
		m.Ud = uint64(Mpgetfix(nr.Val.U.Xval))
		Umagic(&m)
		if m.Bad != 0 {
			break
		}
		if op == OMOD {
			goto longmod
		}

		var n1 Node
		Cgenr(nl, &n1, nil)
		var n2 Node
		Nodconst(&n2, nl.Type, int64(m.Um))
		var n3 Node
		Regalloc(&n3, nl.Type, res)
		Thearch.Cgen_hmul(&n1, &n2, &n3)

		if m.Ua != 0 {
			// need to add numerator accounting for overflow
			Thearch.Gins(Thearch.Optoas(OADD, nl.Type), &n1, &n3)

			Nodconst(&n2, nl.Type, 1)
			Thearch.Gins(Thearch.Optoas(ORROTC, nl.Type), &n2, &n3)
			Nodconst(&n2, nl.Type, int64(m.S)-1)
			Thearch.Gins(Thearch.Optoas(ORSH, nl.Type), &n2, &n3)
		} else {
			Nodconst(&n2, nl.Type, int64(m.S))
			Thearch.Gins(Thearch.Optoas(ORSH, nl.Type), &n2, &n3) // shift dx
		}

		Thearch.Gmove(&n3, res)
		Regfree(&n1)
		Regfree(&n3)
		return

	case TINT64:
		var m Magic
		m.W = w
		m.Sd = Mpgetfix(nr.Val.U.Xval)
		Smagic(&m)
		if m.Bad != 0 {
			break
		}
		if op == OMOD {
			goto longmod
		}

		var n1 Node
		Cgenr(nl, &n1, res)
		var n2 Node
		Nodconst(&n2, nl.Type, m.Sm)
		var n3 Node
		Regalloc(&n3, nl.Type, nil)
		Thearch.Cgen_hmul(&n1, &n2, &n3)

		if m.Sm < 0 {
			// need to add numerator
			Thearch.Gins(Thearch.Optoas(OADD, nl.Type), &n1, &n3)
		}

		Nodconst(&n2, nl.Type, int64(m.S))
		Thearch.Gins(Thearch.Optoas(ORSH, nl.Type), &n2, &n3) // shift n3

		Nodconst(&n2, nl.Type, int64(w)-1)

		Thearch.Gins(Thearch.Optoas(ORSH, nl.Type), &n2, &n1) // -1 iff num is neg
		Thearch.Gins(Thearch.Optoas(OSUB, nl.Type), &n1, &n3) // added

		if m.Sd < 0 {
			// this could probably be removed
			// by factoring it into the multiplier
			Thearch.Gins(Thearch.Optoas(OMINUS, nl.Type), nil, &n3)
		}

		Thearch.Gmove(&n3, res)
		Regfree(&n1)
		Regfree(&n3)
		return
	}

	goto longdiv

	// division and mod using (slow) hardware instruction
longdiv:
	Thearch.Dodiv(op, nl, nr, res)

	return

	// mod using formula A%B = A-(A/B*B) but
	// we know that there is a fast algorithm for A/B
longmod:
	var n1 Node
	Regalloc(&n1, nl.Type, res)

	Cgen(nl, &n1)
	var n2 Node
	Regalloc(&n2, nl.Type, nil)
	cgen_div(ODIV, &n1, nr, &n2)
	a := Thearch.Optoas(OMUL, nl.Type)
	if w == 8 {
		// use 2-operand 16-bit multiply
		// because there is no 2-operand 8-bit multiply
		a = Thearch.Optoas(OMUL, Types[TINT16]) // XXX was IMULW
	}

	if !Smallintconst(nr) {
		var n3 Node
		Regalloc(&n3, nl.Type, nil)
		Cgen(nr, &n3)
		Thearch.Gins(a, &n3, &n2)
		Regfree(&n3)
	} else {
		Thearch.Gins(a, nr, &n2)
	}
	Thearch.Gins(Thearch.Optoas(OSUB, nl.Type), &n2, &n1)
	Thearch.Gmove(&n1, res)
	Regfree(&n1)
	Regfree(&n2)
}

func Fixlargeoffset(n *Node) {
	if n == nil {
		return
	}
	if n.Op != OINDREG {
		return
	}
	if n.Reg == int16(Thearch.REGSP) { // stack offset cannot be large
		return
	}
	if n.Xoffset != int64(int32(n.Xoffset)) {
		// offset too large, add to register instead.
		a := *n

		a.Op = OREGISTER
		a.Type = Types[Tptr]
		a.Xoffset = 0
		Cgen_checknil(&a)
		Thearch.Ginscon(Thearch.Optoas(OADD, Types[Tptr]), n.Xoffset, &a)
		n.Xoffset = 0
	}
}
