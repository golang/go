// Copyright 2009 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package gc

import "cmd/internal/obj"

func CASE(a int, b int) int {
	return a<<16 | b
}

func overlap_cplx(f *Node, t *Node) bool {
	// check whether f and t could be overlapping stack references.
	// not exact, because it's hard to check for the stack register
	// in portable code.  close enough: worst case we will allocate
	// an extra temporary and the registerizer will clean it up.
	return f.Op == OINDREG && t.Op == OINDREG && f.Xoffset+f.Type.Width >= t.Xoffset && t.Xoffset+t.Type.Width >= f.Xoffset
}

func Complexbool(op int, nl *Node, nr *Node, true_ bool, likely int, to *obj.Prog) {
	var tnl Node

	// make both sides addable in ullman order
	if nr != nil {
		if nl.Ullman > nr.Ullman && nl.Addable == 0 {
			Tempname(&tnl, nl.Type)
			Thearch.Cgen(nl, &tnl)
			nl = &tnl
		}

		if nr.Addable == 0 {
			var tnr Node
			Tempname(&tnr, nr.Type)
			Thearch.Cgen(nr, &tnr)
			nr = &tnr
		}
	}

	if nl.Addable == 0 {
		Tempname(&tnl, nl.Type)
		Thearch.Cgen(nl, &tnl)
		nl = &tnl
	}

	// build tree
	// real(l) == real(r) && imag(l) == imag(r)

	var n2 Node
	var n1 Node
	subnode(&n1, &n2, nl)

	var n3 Node
	var n4 Node
	subnode(&n3, &n4, nr)

	na := Node{}
	na.Op = OANDAND
	var nb Node
	na.Left = &nb
	var nc Node
	na.Right = &nc
	na.Type = Types[TBOOL]

	nb = Node{}
	nb.Op = OEQ
	nb.Left = &n1
	nb.Right = &n3
	nb.Type = Types[TBOOL]

	nc = Node{}
	nc.Op = OEQ
	nc.Left = &n2
	nc.Right = &n4
	nc.Type = Types[TBOOL]

	if op == ONE {
		true_ = !true_
	}

	Thearch.Bgen(&na, true_, likely, to)
}

// break addable nc-complex into nr-real and ni-imaginary
func subnode(nr *Node, ni *Node, nc *Node) {
	if nc.Addable == 0 {
		Fatal("subnode not addable")
	}

	tc := Simsimtype(nc.Type)
	tc = cplxsubtype(tc)
	t := Types[tc]

	if nc.Op == OLITERAL {
		nodfconst(nr, t, &nc.Val.U.Cval.Real)
		nodfconst(ni, t, &nc.Val.U.Cval.Imag)
		return
	}

	*nr = *nc
	nr.Type = t

	*ni = *nc
	ni.Type = t
	ni.Xoffset += t.Width
}

// generate code res = -nl
func minus(nl *Node, res *Node) {
	ra := Node{}
	ra.Op = OMINUS
	ra.Left = nl
	ra.Type = nl.Type
	Thearch.Cgen(&ra, res)
}

// build and execute tree
//	real(res) = -real(nl)
//	imag(res) = -imag(nl)
func complexminus(nl *Node, res *Node) {
	var n1 Node
	var n2 Node
	var n5 Node
	var n6 Node

	subnode(&n1, &n2, nl)
	subnode(&n5, &n6, res)

	minus(&n1, &n5)
	minus(&n2, &n6)
}

// build and execute tree
//	real(res) = real(nl) op real(nr)
//	imag(res) = imag(nl) op imag(nr)
func complexadd(op int, nl *Node, nr *Node, res *Node) {
	var n1 Node
	var n2 Node
	var n3 Node
	var n4 Node
	var n5 Node
	var n6 Node

	subnode(&n1, &n2, nl)
	subnode(&n3, &n4, nr)
	subnode(&n5, &n6, res)

	ra := Node{}
	ra.Op = uint8(op)
	ra.Left = &n1
	ra.Right = &n3
	ra.Type = n1.Type
	Thearch.Cgen(&ra, &n5)

	ra = Node{}
	ra.Op = uint8(op)
	ra.Left = &n2
	ra.Right = &n4
	ra.Type = n2.Type
	Thearch.Cgen(&ra, &n6)
}

// build and execute tree
//	tmp       = real(nl)*real(nr) - imag(nl)*imag(nr)
//	imag(res) = real(nl)*imag(nr) + imag(nl)*real(nr)
//	real(res) = tmp
func complexmul(nl *Node, nr *Node, res *Node) {
	var n1 Node
	var n2 Node
	var n3 Node
	var n4 Node
	var n5 Node
	var n6 Node
	var tmp Node

	subnode(&n1, &n2, nl)
	subnode(&n3, &n4, nr)
	subnode(&n5, &n6, res)
	Tempname(&tmp, n5.Type)

	// real part -> tmp
	rm1 := Node{}

	rm1.Op = OMUL
	rm1.Left = &n1
	rm1.Right = &n3
	rm1.Type = n1.Type

	rm2 := Node{}
	rm2.Op = OMUL
	rm2.Left = &n2
	rm2.Right = &n4
	rm2.Type = n2.Type

	ra := Node{}
	ra.Op = OSUB
	ra.Left = &rm1
	ra.Right = &rm2
	ra.Type = rm1.Type
	Thearch.Cgen(&ra, &tmp)

	// imag part
	rm1 = Node{}

	rm1.Op = OMUL
	rm1.Left = &n1
	rm1.Right = &n4
	rm1.Type = n1.Type

	rm2 = Node{}
	rm2.Op = OMUL
	rm2.Left = &n2
	rm2.Right = &n3
	rm2.Type = n2.Type

	ra = Node{}
	ra.Op = OADD
	ra.Left = &rm1
	ra.Right = &rm2
	ra.Type = rm1.Type
	Thearch.Cgen(&ra, &n6)

	// tmp ->real part
	Thearch.Cgen(&tmp, &n5)
}

func nodfconst(n *Node, t *Type, fval *Mpflt) {
	*n = Node{}
	n.Op = OLITERAL
	n.Addable = 1
	ullmancalc(n)
	n.Val.U.Fval = fval
	n.Val.Ctype = CTFLT
	n.Type = t

	if Isfloat[t.Etype] == 0 {
		Fatal("nodfconst: bad type %v", Tconv(t, 0))
	}
}

/*
 * cplx.c
 */
func Complexop(n *Node, res *Node) bool {
	if n != nil && n.Type != nil {
		if Iscomplex[n.Type.Etype] != 0 {
			goto maybe
		}
	}

	if res != nil && res.Type != nil {
		if Iscomplex[res.Type.Etype] != 0 {
			goto maybe
		}
	}

	if n.Op == OREAL || n.Op == OIMAG {
		goto yes
	}

	goto no

maybe:
	switch n.Op {
	case OCONV, // implemented ops
		OADD,
		OSUB,
		OMUL,
		OMINUS,
		OCOMPLEX,
		OREAL,
		OIMAG:
		goto yes

	case ODOT,
		ODOTPTR,
		OINDEX,
		OIND,
		ONAME:
		goto yes
	}

	//dump("\ncomplex-no", n);
no:
	return false

	//dump("\ncomplex-yes", n);
yes:
	return true
}

func Complexmove(f *Node, t *Node) {
	if Debug['g'] != 0 {
		Dump("\ncomplexmove-f", f)
		Dump("complexmove-t", t)
	}

	if t.Addable == 0 {
		Fatal("complexmove: to not addable")
	}

	ft := Simsimtype(f.Type)
	tt := Simsimtype(t.Type)
	switch uint32(ft)<<16 | uint32(tt) {
	default:
		Fatal("complexmove: unknown conversion: %v -> %v\n", Tconv(f.Type, 0), Tconv(t.Type, 0))

		// complex to complex move/convert.
	// make f addable.
	// also use temporary if possible stack overlap.
	case TCOMPLEX64<<16 | TCOMPLEX64,
		TCOMPLEX64<<16 | TCOMPLEX128,
		TCOMPLEX128<<16 | TCOMPLEX64,
		TCOMPLEX128<<16 | TCOMPLEX128:
		if f.Addable == 0 || overlap_cplx(f, t) {
			var tmp Node
			Tempname(&tmp, f.Type)
			Complexmove(f, &tmp)
			f = &tmp
		}

		var n1 Node
		var n2 Node
		subnode(&n1, &n2, f)
		var n4 Node
		var n3 Node
		subnode(&n3, &n4, t)

		Thearch.Cgen(&n1, &n3)
		Thearch.Cgen(&n2, &n4)
	}
}

func Complexgen(n *Node, res *Node) {
	if Debug['g'] != 0 {
		Dump("\ncomplexgen-n", n)
		Dump("complexgen-res", res)
	}

	for n.Op == OCONVNOP {
		n = n.Left
	}

	// pick off float/complex opcodes
	switch n.Op {
	case OCOMPLEX:
		if res.Addable != 0 {
			var n1 Node
			var n2 Node
			subnode(&n1, &n2, res)
			var tmp Node
			Tempname(&tmp, n1.Type)
			Thearch.Cgen(n.Left, &tmp)
			Thearch.Cgen(n.Right, &n2)
			Thearch.Cgen(&tmp, &n1)
			return
		}

	case OREAL,
		OIMAG:
		nl := n.Left
		if nl.Addable == 0 {
			var tmp Node
			Tempname(&tmp, nl.Type)
			Complexgen(nl, &tmp)
			nl = &tmp
		}

		var n1 Node
		var n2 Node
		subnode(&n1, &n2, nl)
		if n.Op == OREAL {
			Thearch.Cgen(&n1, res)
			return
		}

		Thearch.Cgen(&n2, res)
		return
	}

	// perform conversion from n to res
	tl := Simsimtype(res.Type)

	tl = cplxsubtype(tl)
	tr := Simsimtype(n.Type)
	tr = cplxsubtype(tr)
	if tl != tr {
		if n.Addable == 0 {
			var n1 Node
			Tempname(&n1, n.Type)
			Complexmove(n, &n1)
			n = &n1
		}

		Complexmove(n, res)
		return
	}

	if res.Addable == 0 {
		var n1 Node
		Thearch.Igen(res, &n1, nil)
		Thearch.Cgen(n, &n1)
		Thearch.Regfree(&n1)
		return
	}

	if n.Addable != 0 {
		Complexmove(n, res)
		return
	}

	switch n.Op {
	default:
		Dump("complexgen: unknown op", n)
		Fatal("complexgen: unknown op %v", Oconv(int(n.Op), 0))

	case ODOT,
		ODOTPTR,
		OINDEX,
		OIND,
		ONAME, // PHEAP or PPARAMREF var
		OCALLFUNC,
		OCALLMETH,
		OCALLINTER:
		var n1 Node
		Thearch.Igen(n, &n1, res)

		Complexmove(&n1, res)
		Thearch.Regfree(&n1)
		return

	case OCONV,
		OADD,
		OSUB,
		OMUL,
		OMINUS,
		OCOMPLEX,
		OREAL,
		OIMAG:
		break
	}

	nl := n.Left
	if nl == nil {
		return
	}
	nr := n.Right

	// make both sides addable in ullman order
	var tnl Node
	if nr != nil {
		if nl.Ullman > nr.Ullman && nl.Addable == 0 {
			Tempname(&tnl, nl.Type)
			Thearch.Cgen(nl, &tnl)
			nl = &tnl
		}

		if nr.Addable == 0 {
			var tnr Node
			Tempname(&tnr, nr.Type)
			Thearch.Cgen(nr, &tnr)
			nr = &tnr
		}
	}

	if nl.Addable == 0 {
		Tempname(&tnl, nl.Type)
		Thearch.Cgen(nl, &tnl)
		nl = &tnl
	}

	switch n.Op {
	default:
		Fatal("complexgen: unknown op %v", Oconv(int(n.Op), 0))

	case OCONV:
		Complexmove(nl, res)

	case OMINUS:
		complexminus(nl, res)

	case OADD,
		OSUB:
		complexadd(int(n.Op), nl, nr, res)

	case OMUL:
		complexmul(nl, nr, res)
	}
}
