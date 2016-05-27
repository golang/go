// Copyright 2009 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package gc

import "cmd/internal/obj"

func overlap_cplx(f *Node, t *Node) bool {
	// check whether f and t could be overlapping stack references.
	// not exact, because it's hard to check for the stack register
	// in portable code.  close enough: worst case we will allocate
	// an extra temporary and the registerizer will clean it up.
	return f.Op == OINDREG && t.Op == OINDREG && f.Xoffset+f.Type.Width >= t.Xoffset && t.Xoffset+t.Type.Width >= f.Xoffset
}

func complexbool(op Op, nl, nr, res *Node, wantTrue bool, likely int, to *obj.Prog) {
	// make both sides addable in ullman order
	if nr != nil {
		if nl.Ullman > nr.Ullman && !nl.Addable {
			nl = CgenTemp(nl)
		}

		if !nr.Addable {
			nr = CgenTemp(nr)
		}
	}
	if !nl.Addable {
		nl = CgenTemp(nl)
	}

	// Break nl and nr into real and imaginary components.
	var lreal, limag, rreal, rimag Node
	subnode(&lreal, &limag, nl)
	subnode(&rreal, &rimag, nr)

	// build tree
	// if branching:
	// 	real(l) == real(r) && imag(l) == imag(r)
	// if generating a value, use a branch-free version:
	// 	real(l) == real(r) & imag(l) == imag(r)
	realeq := Node{
		Op:    OEQ,
		Left:  &lreal,
		Right: &rreal,
		Type:  Types[TBOOL],
	}
	imageq := Node{
		Op:    OEQ,
		Left:  &limag,
		Right: &rimag,
		Type:  Types[TBOOL],
	}
	and := Node{
		Op:    OANDAND,
		Left:  &realeq,
		Right: &imageq,
		Type:  Types[TBOOL],
	}

	if res != nil {
		// generating a value
		and.Op = OAND
		if op == ONE {
			and.Op = OOR
			realeq.Op = ONE
			imageq.Op = ONE
		}
		Bvgen(&and, res, true)
		return
	}

	// generating a branch
	if op == ONE {
		wantTrue = !wantTrue
	}

	Bgen(&and, wantTrue, likely, to)
}

// break addable nc-complex into nr-real and ni-imaginary
func subnode(nr *Node, ni *Node, nc *Node) {
	if !nc.Addable {
		Fatalf("subnode not addable")
	}

	tc := Simsimtype(nc.Type)
	tc = cplxsubtype(tc)
	t := Types[tc]

	if nc.Op == OLITERAL {
		u := nc.Val().U.(*Mpcplx)
		nodfconst(nr, t, &u.Real)
		nodfconst(ni, t, &u.Imag)
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
	var ra Node
	ra.Op = OMINUS
	ra.Left = nl
	ra.Type = nl.Type
	Cgen(&ra, res)
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
func complexadd(op Op, nl *Node, nr *Node, res *Node) {
	var n1 Node
	var n2 Node
	var n3 Node
	var n4 Node
	var n5 Node
	var n6 Node

	subnode(&n1, &n2, nl)
	subnode(&n3, &n4, nr)
	subnode(&n5, &n6, res)

	var ra Node
	ra.Op = op
	ra.Left = &n1
	ra.Right = &n3
	ra.Type = n1.Type
	Cgen(&ra, &n5)

	ra = Node{}
	ra.Op = op
	ra.Left = &n2
	ra.Right = &n4
	ra.Type = n2.Type
	Cgen(&ra, &n6)
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
	var rm1 Node

	rm1.Op = OMUL
	rm1.Left = &n1
	rm1.Right = &n3
	rm1.Type = n1.Type

	var rm2 Node
	rm2.Op = OMUL
	rm2.Left = &n2
	rm2.Right = &n4
	rm2.Type = n2.Type

	var ra Node
	ra.Op = OSUB
	ra.Left = &rm1
	ra.Right = &rm2
	ra.Type = rm1.Type
	Cgen(&ra, &tmp)

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
	Cgen(&ra, &n6)

	// tmp ->real part
	Cgen(&tmp, &n5)
}

func nodfconst(n *Node, t *Type, fval *Mpflt) {
	*n = Node{}
	n.Op = OLITERAL
	n.Addable = true
	ullmancalc(n)
	n.SetVal(Val{fval})
	n.Type = t

	if !t.IsFloat() {
		Fatalf("nodfconst: bad type %v", t)
	}
}

func Complexop(n *Node, res *Node) bool {
	if n != nil && n.Type != nil {
		if n.Type.IsComplex() {
			goto maybe
		}
	}

	if res != nil && res.Type != nil {
		if res.Type.IsComplex() {
			goto maybe
		}
	}

	if n.Op == OREAL || n.Op == OIMAG {
		//dump("\ncomplex-yes", n);
		return true
	}

	//dump("\ncomplex-no", n);
	return false

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
		//dump("\ncomplex-yes", n);
		return true

	case ODOT,
		ODOTPTR,
		OINDEX,
		OIND,
		ONAME:
		//dump("\ncomplex-yes", n);
		return true
	}

	//dump("\ncomplex-no", n);
	return false
}

func Complexmove(f *Node, t *Node) {
	if Debug['g'] != 0 {
		Dump("\ncomplexmove-f", f)
		Dump("complexmove-t", t)
	}

	if !t.Addable {
		Fatalf("complexmove: to not addable")
	}

	ft := Simsimtype(f.Type)
	tt := Simsimtype(t.Type)
	// complex to complex move/convert.
	// make f addable.
	// also use temporary if possible stack overlap.
	if (ft == TCOMPLEX64 || ft == TCOMPLEX128) && (tt == TCOMPLEX64 || tt == TCOMPLEX128) {
		if !f.Addable || overlap_cplx(f, t) {
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

		Cgen(&n1, &n3)
		Cgen(&n2, &n4)
	} else {
		Fatalf("complexmove: unknown conversion: %v -> %v\n", f.Type, t.Type)
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
		if res.Addable {
			var n1 Node
			var n2 Node
			subnode(&n1, &n2, res)
			var tmp Node
			Tempname(&tmp, n1.Type)
			Cgen(n.Left, &tmp)
			Cgen(n.Right, &n2)
			Cgen(&tmp, &n1)
			return
		}

	case OREAL, OIMAG:
		nl := n.Left
		if !nl.Addable {
			var tmp Node
			Tempname(&tmp, nl.Type)
			Complexgen(nl, &tmp)
			nl = &tmp
		}

		var n1 Node
		var n2 Node
		subnode(&n1, &n2, nl)
		if n.Op == OREAL {
			Cgen(&n1, res)
			return
		}

		Cgen(&n2, res)
		return
	}

	// perform conversion from n to res
	tl := Simsimtype(res.Type)

	tl = cplxsubtype(tl)
	tr := Simsimtype(n.Type)
	tr = cplxsubtype(tr)
	if tl != tr {
		if !n.Addable {
			var n1 Node
			Tempname(&n1, n.Type)
			Complexmove(n, &n1)
			n = &n1
		}

		Complexmove(n, res)
		return
	}

	if !res.Addable {
		var n1 Node
		Igen(res, &n1, nil)
		Cgen(n, &n1)
		Regfree(&n1)
		return
	}

	if n.Addable {
		Complexmove(n, res)
		return
	}

	switch n.Op {
	default:
		Dump("complexgen: unknown op", n)
		Fatalf("complexgen: unknown op %v", n.Op)

	case ODOT,
		ODOTPTR,
		OINDEX,
		OIND,
		OCALLFUNC,
		OCALLMETH,
		OCALLINTER:
		var n1 Node
		Igen(n, &n1, res)

		Complexmove(&n1, res)
		Regfree(&n1)
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
		if nl.Ullman > nr.Ullman && !nl.Addable {
			Tempname(&tnl, nl.Type)
			Cgen(nl, &tnl)
			nl = &tnl
		}

		if !nr.Addable {
			var tnr Node
			Tempname(&tnr, nr.Type)
			Cgen(nr, &tnr)
			nr = &tnr
		}
	}

	if !nl.Addable {
		Tempname(&tnl, nl.Type)
		Cgen(nl, &tnl)
		nl = &tnl
	}

	switch n.Op {
	default:
		Fatalf("complexgen: unknown op %v", n.Op)

	case OCONV:
		Complexmove(nl, res)

	case OMINUS:
		complexminus(nl, res)

	case OADD, OSUB:
		complexadd(n.Op, nl, nr, res)

	case OMUL:
		complexmul(nl, nr, res)
	}
}
