// Copyright 2009 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package gc

import (
	"cmd/internal/obj"
	"fmt"
)

const (
	Snorm = 0 + iota
	Strue
	Sfalse
	Stype
	Tdefault
	Texprconst
	Texprvar
	Ttypenil
	Ttypeconst
	Ttypevar
	Ncase = 4
)

type Case struct {
	node    *Node
	hash    uint32
	type_   uint8
	diag    uint8
	ordinal uint16
	link    *Case
}

var C *Case

func dumpcase(c0 *Case) {
	for c := c0; c != nil; c = c.link {
		switch c.type_ {
		case Tdefault:
			fmt.Printf("case-default\n")
			fmt.Printf("\tord=%d\n", c.ordinal)

		case Texprconst:
			fmt.Printf("case-exprconst\n")
			fmt.Printf("\tord=%d\n", c.ordinal)

		case Texprvar:
			fmt.Printf("case-exprvar\n")
			fmt.Printf("\tord=%d\n", c.ordinal)
			fmt.Printf("\top=%v\n", Oconv(int(c.node.Left.Op), 0))

		case Ttypenil:
			fmt.Printf("case-typenil\n")
			fmt.Printf("\tord=%d\n", c.ordinal)

		case Ttypeconst:
			fmt.Printf("case-typeconst\n")
			fmt.Printf("\tord=%d\n", c.ordinal)
			fmt.Printf("\thash=%x\n", c.hash)

		case Ttypevar:
			fmt.Printf("case-typevar\n")
			fmt.Printf("\tord=%d\n", c.ordinal)

		default:
			fmt.Printf("case-???\n")
			fmt.Printf("\tord=%d\n", c.ordinal)
			fmt.Printf("\top=%v\n", Oconv(int(c.node.Left.Op), 0))
			fmt.Printf("\thash=%x\n", c.hash)
		}
	}

	fmt.Printf("\n")
}

func ordlcmp(c1 *Case, c2 *Case) int {
	// sort default first
	if c1.type_ == Tdefault {
		return -1
	}
	if c2.type_ == Tdefault {
		return +1
	}

	// sort nil second
	if c1.type_ == Ttypenil {
		return -1
	}
	if c2.type_ == Ttypenil {
		return +1
	}

	// sort by ordinal
	if c1.ordinal > c2.ordinal {
		return +1
	}
	if c1.ordinal < c2.ordinal {
		return -1
	}
	return 0
}

func exprcmp(c1 *Case, c2 *Case) int {
	// sort non-constants last
	if c1.type_ != Texprconst {
		return +1
	}
	if c2.type_ != Texprconst {
		return -1
	}

	n1 := c1.node.Left
	n2 := c2.node.Left

	// sort by type (for switches on interface)
	ct := int(n1.Val.Ctype)

	if ct != int(n2.Val.Ctype) {
		return ct - int(n2.Val.Ctype)
	}
	if !Eqtype(n1.Type, n2.Type) {
		if n1.Type.Vargen > n2.Type.Vargen {
			return +1
		} else {
			return -1
		}
	}

	// sort by constant value
	n := 0

	switch ct {
	case CTFLT:
		n = mpcmpfltflt(n1.Val.U.Fval, n2.Val.U.Fval)

	case CTINT,
		CTRUNE:
		n = Mpcmpfixfix(n1.Val.U.Xval, n2.Val.U.Xval)

	case CTSTR:
		n = cmpslit(n1, n2)
	}

	return n
}

func typecmp(c1 *Case, c2 *Case) int {
	// sort non-constants last
	if c1.type_ != Ttypeconst {
		return +1
	}
	if c2.type_ != Ttypeconst {
		return -1
	}

	// sort by hash code
	if c1.hash > c2.hash {
		return +1
	}
	if c1.hash < c2.hash {
		return -1
	}

	// sort by ordinal so duplicate error
	// happens on later case.
	if c1.ordinal > c2.ordinal {
		return +1
	}
	if c1.ordinal < c2.ordinal {
		return -1
	}
	return 0
}

func csort(l *Case, f func(*Case, *Case) int) *Case {
	if l == nil || l.link == nil {
		return l
	}

	l1 := l
	l2 := l
	for {
		l2 = l2.link
		if l2 == nil {
			break
		}
		l2 = l2.link
		if l2 == nil {
			break
		}
		l1 = l1.link
	}

	l2 = l1.link
	l1.link = nil
	l1 = csort(l, f)
	l2 = csort(l2, f)

	/* set up lead element */
	if f(l1, l2) < 0 {
		l = l1
		l1 = l1.link
	} else {
		l = l2
		l2 = l2.link
	}

	le := l

	for {
		if l1 == nil {
			for l2 != nil {
				le.link = l2
				le = l2
				l2 = l2.link
			}

			le.link = nil
			break
		}

		if l2 == nil {
			for l1 != nil {
				le.link = l1
				le = l1
				l1 = l1.link
			}

			break
		}

		if f(l1, l2) < 0 {
			le.link = l1
			le = l1
			l1 = l1.link
		} else {
			le.link = l2
			le = l2
			l2 = l2.link
		}
	}

	le.link = nil
	return l
}

var newlabel_swt_label int

func newlabel_swt() *Node {
	newlabel_swt_label++
	namebuf = fmt.Sprintf("%.6d", newlabel_swt_label)
	return newname(Lookup(namebuf))
}

/*
 * build separate list of statements and cases
 * make labels between cases and statements
 * deal with fallthrough, break, unreachable statements
 */
func casebody(sw *Node, typeswvar *Node) {
	if sw.List == nil {
		return
	}

	lno := setlineno(sw)

	cas := (*NodeList)(nil)  // cases
	stat := (*NodeList)(nil) // statements
	def := (*Node)(nil)      // defaults
	br := Nod(OBREAK, nil, nil)

	var c *Node
	var go_ *Node
	var needvar bool
	var lc *NodeList
	var last *Node
	var n *Node
	for l := sw.List; l != nil; l = l.Next {
		n = l.N
		setlineno(n)
		if n.Op != OXCASE {
			Fatal("casebody %v", Oconv(int(n.Op), 0))
		}
		n.Op = OCASE
		needvar = count(n.List) != 1 || n.List.N.Op == OLITERAL

		go_ = Nod(OGOTO, newlabel_swt(), nil)
		if n.List == nil {
			if def != nil {
				Yyerror("more than one default case")
			}

			// reuse original default case
			n.Right = go_

			def = n
		}

		if n.List != nil && n.List.Next == nil {
			// one case - reuse OCASE node.
			c = n.List.N

			n.Left = c
			n.Right = go_
			n.List = nil
			cas = list(cas, n)
		} else {
			// expand multi-valued cases
			for lc = n.List; lc != nil; lc = lc.Next {
				c = lc.N
				cas = list(cas, Nod(OCASE, c, go_))
			}
		}

		stat = list(stat, Nod(OLABEL, go_.Left, nil))
		if typeswvar != nil && needvar && n.Nname != nil {
			l := list1(Nod(ODCL, n.Nname, nil))
			l = list(l, Nod(OAS, n.Nname, typeswvar))
			typechecklist(l, Etop)
			stat = concat(stat, l)
		}

		stat = concat(stat, n.Nbody)

		// botch - shouldn't fall thru declaration
		last = stat.End.N

		if last.Xoffset == n.Xoffset && last.Op == OXFALL {
			if typeswvar != nil {
				setlineno(last)
				Yyerror("cannot fallthrough in type switch")
			}

			if l.Next == nil {
				setlineno(last)
				Yyerror("cannot fallthrough final case in switch")
			}

			last.Op = OFALL
		} else {
			stat = list(stat, br)
		}
	}

	stat = list(stat, br)
	if def != nil {
		cas = list(cas, def)
	}

	sw.List = cas
	sw.Nbody = stat
	lineno = lno
}

func mkcaselist(sw *Node, arg int) *Case {
	var n *Node
	var c1 *Case

	c := (*Case)(nil)
	ord := 0

	for l := sw.List; l != nil; l = l.Next {
		n = l.N
		c1 = new(Case)
		c1.link = c
		c = c1

		ord++
		if int(uint16(ord)) != ord {
			Fatal("too many cases in switch")
		}
		c.ordinal = uint16(ord)
		c.node = n

		if n.Left == nil {
			c.type_ = Tdefault
			continue
		}

		switch arg {
		case Stype:
			c.hash = 0
			if n.Left.Op == OLITERAL {
				c.type_ = Ttypenil
				continue
			}

			if Istype(n.Left.Type, TINTER) {
				c.type_ = Ttypevar
				continue
			}

			c.hash = typehash(n.Left.Type)
			c.type_ = Ttypeconst
			continue

		case Snorm,
			Strue,
			Sfalse:
			c.type_ = Texprvar
			c.hash = typehash(n.Left.Type)
			switch consttype(n.Left) {
			case CTFLT,
				CTINT,
				CTRUNE,
				CTSTR:
				c.type_ = Texprconst
			}

			continue
		}
	}

	if c == nil {
		return nil
	}

	// sort by value and diagnose duplicate cases
	switch arg {
	case Stype:
		c = csort(c, typecmp)
		var c2 *Case
		for c1 := c; c1 != nil; c1 = c1.link {
			for c2 = c1.link; c2 != nil && c2.hash == c1.hash; c2 = c2.link {
				if c1.type_ == Ttypenil || c1.type_ == Tdefault {
					break
				}
				if c2.type_ == Ttypenil || c2.type_ == Tdefault {
					break
				}
				if !Eqtype(c1.node.Left.Type, c2.node.Left.Type) {
					continue
				}
				yyerrorl(int(c2.node.Lineno), "duplicate case %v in type switch\n\tprevious case at %v", Tconv(c2.node.Left.Type, 0), c1.node.Line())
			}
		}

	case Snorm,
		Strue,
		Sfalse:
		c = csort(c, exprcmp)
		for c1 := c; c1.link != nil; c1 = c1.link {
			if exprcmp(c1, c1.link) != 0 {
				continue
			}
			setlineno(c1.link.node)
			Yyerror("duplicate case %v in switch\n\tprevious case at %v", Nconv(c1.node.Left, 0), c1.node.Line())
		}
	}

	// put list back in processing order
	c = csort(c, ordlcmp)

	return c
}

var exprname *Node

func exprbsw(c0 *Case, ncase int, arg int) *Node {
	cas := (*NodeList)(nil)
	if ncase < Ncase {
		var a *Node
		var n *Node
		var lno int
		for i := 0; i < ncase; i++ {
			n = c0.node
			lno = int(setlineno(n))

			if (arg != Strue && arg != Sfalse) || assignop(n.Left.Type, exprname.Type, nil) == OCONVIFACE || assignop(exprname.Type, n.Left.Type, nil) == OCONVIFACE {
				a = Nod(OIF, nil, nil)
				a.Ntest = Nod(OEQ, exprname, n.Left) // if name == val
				typecheck(&a.Ntest, Erv)
				a.Nbody = list1(n.Right) // then goto l
			} else if arg == Strue {
				a = Nod(OIF, nil, nil)
				a.Ntest = n.Left         // if val
				a.Nbody = list1(n.Right) // then goto l // arg == Sfalse
			} else {
				a = Nod(OIF, nil, nil)
				a.Ntest = Nod(ONOT, n.Left, nil) // if !val
				typecheck(&a.Ntest, Erv)
				a.Nbody = list1(n.Right) // then goto l
			}

			cas = list(cas, a)
			c0 = c0.link
			lineno = int32(lno)
		}

		return liststmt(cas)
	}

	// find the middle and recur
	c := c0

	half := ncase >> 1
	for i := 1; i < half; i++ {
		c = c.link
	}
	a := Nod(OIF, nil, nil)
	a.Ntest = Nod(OLE, exprname, c.node.Left)
	typecheck(&a.Ntest, Erv)
	a.Nbody = list1(exprbsw(c0, half, arg))
	a.Nelse = list1(exprbsw(c.link, ncase-half, arg))
	return a
}

/*
 * normal (expression) switch.
 * rebuild case statements into if .. goto
 */
func exprswitch(sw *Node) {
	casebody(sw, nil)

	arg := Snorm
	if Isconst(sw.Ntest, CTBOOL) {
		arg = Strue
		if sw.Ntest.Val.U.Bval == 0 {
			arg = Sfalse
		}
	}

	walkexpr(&sw.Ntest, &sw.Ninit)
	t := sw.Type
	if t == nil {
		return
	}

	/*
	 * convert the switch into OIF statements
	 */
	exprname = nil

	cas := (*NodeList)(nil)
	if arg == Strue || arg == Sfalse {
		exprname = Nodbool(arg == Strue)
	} else if consttype(sw.Ntest) >= 0 {
		// leave constants to enable dead code elimination (issue 9608)
		exprname = sw.Ntest
	} else {
		exprname = temp(sw.Ntest.Type)
		cas = list1(Nod(OAS, exprname, sw.Ntest))
		typechecklist(cas, Etop)
	}

	c0 := mkcaselist(sw, arg)
	var def *Node
	if c0 != nil && c0.type_ == Tdefault {
		def = c0.node.Right
		c0 = c0.link
	} else {
		def = Nod(OBREAK, nil, nil)
	}

	var c *Case
	var a *Node
	var ncase int
	var c1 *Case
loop:
	if c0 == nil {
		cas = list(cas, def)
		sw.Nbody = concat(cas, sw.Nbody)
		sw.List = nil
		walkstmtlist(sw.Nbody)
		return
	}

	// deal with the variables one-at-a-time
	if okforcmp[t.Etype] == 0 || c0.type_ != Texprconst {
		a = exprbsw(c0, 1, arg)
		cas = list(cas, a)
		c0 = c0.link
		goto loop
	}

	// do binary search on run of constants
	ncase = 1

	for c = c0; c.link != nil; c = c.link {
		if c.link.type_ != Texprconst {
			break
		}
		ncase++
	}

	// break the chain at the count
	c1 = c.link

	c.link = nil

	// sort and compile constants
	c0 = csort(c0, exprcmp)

	a = exprbsw(c0, ncase, arg)
	cas = list(cas, a)

	c0 = c1
	goto loop
}

var hashname *Node

var facename *Node

var boolname *Node

func typeone(t *Node) *Node {
	var_ := t.Nname
	init := (*NodeList)(nil)
	if var_ == nil {
		typecheck(&nblank, Erv|Easgn)
		var_ = nblank
	} else {
		init = list1(Nod(ODCL, var_, nil))
	}

	a := Nod(OAS2, nil, nil)
	a.List = list(list1(var_), boolname) // var,bool =
	b := Nod(ODOTTYPE, facename, nil)
	b.Type = t.Left.Type // interface.(type)
	a.Rlist = list1(b)
	typecheck(&a, Etop)
	init = list(init, a)

	b = Nod(OIF, nil, nil)
	b.Ntest = boolname
	b.Nbody = list1(t.Right) // if bool { goto l }
	a = liststmt(list(init, b))
	return a
}

func typebsw(c0 *Case, ncase int) *Node {
	cas := (*NodeList)(nil)

	if ncase < Ncase {
		var n *Node
		var a *Node
		for i := 0; i < ncase; i++ {
			n = c0.node
			if c0.type_ != Ttypeconst {
				Fatal("typebsw")
			}
			a = Nod(OIF, nil, nil)
			a.Ntest = Nod(OEQ, hashname, Nodintconst(int64(c0.hash)))
			typecheck(&a.Ntest, Erv)
			a.Nbody = list1(n.Right)
			cas = list(cas, a)
			c0 = c0.link
		}

		return liststmt(cas)
	}

	// find the middle and recur
	c := c0

	half := ncase >> 1
	for i := 1; i < half; i++ {
		c = c.link
	}
	a := Nod(OIF, nil, nil)
	a.Ntest = Nod(OLE, hashname, Nodintconst(int64(c.hash)))
	typecheck(&a.Ntest, Erv)
	a.Nbody = list1(typebsw(c0, half))
	a.Nelse = list1(typebsw(c.link, ncase-half))
	return a
}

/*
 * convert switch of the form
 *	switch v := i.(type) { case t1: ..; case t2: ..; }
 * into if statements
 */
func typeswitch(sw *Node) {
	if sw.Ntest == nil {
		return
	}
	if sw.Ntest.Right == nil {
		setlineno(sw)
		Yyerror("type switch must have an assignment")
		return
	}

	walkexpr(&sw.Ntest.Right, &sw.Ninit)
	if !Istype(sw.Ntest.Right.Type, TINTER) {
		Yyerror("type switch must be on an interface")
		return
	}

	cas := (*NodeList)(nil)

	/*
	 * predeclare temporary variables
	 * and the boolean var
	 */
	facename = temp(sw.Ntest.Right.Type)

	a := Nod(OAS, facename, sw.Ntest.Right)
	typecheck(&a, Etop)
	cas = list(cas, a)

	casebody(sw, facename)

	boolname = temp(Types[TBOOL])
	typecheck(&boolname, Erv)

	hashname = temp(Types[TUINT32])
	typecheck(&hashname, Erv)

	t := sw.Ntest.Right.Type
	if isnilinter(t) {
		a = syslook("efacethash", 1)
	} else {
		a = syslook("ifacethash", 1)
	}
	argtype(a, t)
	a = Nod(OCALL, a, nil)
	a.List = list1(facename)
	a = Nod(OAS, hashname, a)
	typecheck(&a, Etop)
	cas = list(cas, a)

	c0 := mkcaselist(sw, Stype)
	var def *Node
	if c0 != nil && c0.type_ == Tdefault {
		def = c0.node.Right
		c0 = c0.link
	} else {
		def = Nod(OBREAK, nil, nil)
	}

	/*
	 * insert if statement into each case block
	 */
	var v Val
	var n *Node
	for c := c0; c != nil; c = c.link {
		n = c.node
		switch c.type_ {
		case Ttypenil:
			v.Ctype = CTNIL
			a = Nod(OIF, nil, nil)
			a.Ntest = Nod(OEQ, facename, nodlit(v))
			typecheck(&a.Ntest, Erv)
			a.Nbody = list1(n.Right) // if i==nil { goto l }
			n.Right = a

		case Ttypevar,
			Ttypeconst:
			n.Right = typeone(n)
		}
	}

	/*
	 * generate list of if statements, binary search for constant sequences
	 */
	var ncase int
	var c1 *Case
	var hash *NodeList
	var c *Case
	for c0 != nil {
		if c0.type_ != Ttypeconst {
			n = c0.node
			cas = list(cas, n.Right)
			c0 = c0.link
			continue
		}

		// identify run of constants
		c = c0
		c1 = c

		for c.link != nil && c.link.type_ == Ttypeconst {
			c = c.link
		}
		c0 = c.link
		c.link = nil

		// sort by hash
		c1 = csort(c1, typecmp)

		// for debugging: linear search
		if false {
			for c = c1; c != nil; c = c.link {
				n = c.node
				cas = list(cas, n.Right)
			}

			continue
		}

		// combine adjacent cases with the same hash
		ncase = 0

		for c = c1; c != nil; c = c.link {
			ncase++
			hash = list1(c.node.Right)
			for c.link != nil && c.link.hash == c.hash {
				hash = list(hash, c.link.node.Right)
				c.link = c.link.link
			}

			c.node.Right = liststmt(hash)
		}

		// binary search among cases to narrow by hash
		cas = list(cas, typebsw(c1, ncase))
	}

	if nerrors == 0 {
		cas = list(cas, def)
		sw.Nbody = concat(cas, sw.Nbody)
		sw.List = nil
		walkstmtlist(sw.Nbody)
	}
}

func walkswitch(sw *Node) {
	/*
	 * reorder the body into (OLIST, cases, statements)
	 * cases have OGOTO into statements.
	 * both have inserted OBREAK statements
	 */
	if sw.Ntest == nil {
		sw.Ntest = Nodbool(true)
		typecheck(&sw.Ntest, Erv)
	}

	if sw.Ntest.Op == OTYPESW {
		typeswitch(sw)

		//dump("sw", sw);
		return
	}

	exprswitch(sw)

	// Discard old AST elements after a walk. They can confuse racewealk.
	sw.Ntest = nil

	sw.List = nil
}

/*
 * type check switch statement
 */
func typecheckswitch(n *Node) {
	var top int
	var t *Type

	lno := int(lineno)
	typechecklist(n.Ninit, Etop)
	nilonly := ""

	if n.Ntest != nil && n.Ntest.Op == OTYPESW {
		// type switch
		top = Etype

		typecheck(&n.Ntest.Right, Erv)
		t = n.Ntest.Right.Type
		if t != nil && t.Etype != TINTER {
			Yyerror("cannot type switch on non-interface value %v", Nconv(n.Ntest.Right, obj.FmtLong))
		}
	} else {
		// value switch
		top = Erv

		if n.Ntest != nil {
			typecheck(&n.Ntest, Erv)
			defaultlit(&n.Ntest, nil)
			t = n.Ntest.Type
		} else {
			t = Types[TBOOL]
		}
		if t != nil {
			var badtype *Type
			if okforeq[t.Etype] == 0 {
				Yyerror("cannot switch on %v", Nconv(n.Ntest, obj.FmtLong))
			} else if t.Etype == TARRAY && !Isfixedarray(t) {
				nilonly = "slice"
			} else if t.Etype == TARRAY && Isfixedarray(t) && algtype1(t, nil) == ANOEQ {
				Yyerror("cannot switch on %v", Nconv(n.Ntest, obj.FmtLong))
			} else if t.Etype == TSTRUCT && algtype1(t, &badtype) == ANOEQ {
				Yyerror("cannot switch on %v (struct containing %v cannot be compared)", Nconv(n.Ntest, obj.FmtLong), Tconv(badtype, 0))
			} else if t.Etype == TFUNC {
				nilonly = "func"
			} else if t.Etype == TMAP {
				nilonly = "map"
			}
		}
	}

	n.Type = t

	def := (*Node)(nil)
	var ptr int
	var have *Type
	var nvar *Node
	var ll *NodeList
	var missing *Type
	var ncase *Node
	for l := n.List; l != nil; l = l.Next {
		ncase = l.N
		setlineno(n)
		if ncase.List == nil {
			// default
			if def != nil {
				Yyerror("multiple defaults in switch (first at %v)", def.Line())
			} else {
				def = ncase
			}
		} else {
			for ll = ncase.List; ll != nil; ll = ll.Next {
				setlineno(ll.N)
				typecheck(&ll.N, Erv|Etype)
				if ll.N.Type == nil || t == nil {
					continue
				}
				setlineno(ncase)
				switch top {
				case Erv: // expression switch
					defaultlit(&ll.N, t)

					if ll.N.Op == OTYPE {
						Yyerror("type %v is not an expression", Tconv(ll.N.Type, 0))
					} else if ll.N.Type != nil && assignop(ll.N.Type, t, nil) == 0 && assignop(t, ll.N.Type, nil) == 0 {
						if n.Ntest != nil {
							Yyerror("invalid case %v in switch on %v (mismatched types %v and %v)", Nconv(ll.N, 0), Nconv(n.Ntest, 0), Tconv(ll.N.Type, 0), Tconv(t, 0))
						} else {
							Yyerror("invalid case %v in switch (mismatched types %v and bool)", Nconv(ll.N, 0), Tconv(ll.N.Type, 0))
						}
					} else if nilonly != "" && !Isconst(ll.N, CTNIL) {
						Yyerror("invalid case %v in switch (can only compare %s %v to nil)", Nconv(ll.N, 0), nilonly, Nconv(n.Ntest, 0))
					}

				case Etype: // type switch
					if ll.N.Op == OLITERAL && Istype(ll.N.Type, TNIL) {
					} else if ll.N.Op != OTYPE && ll.N.Type != nil { // should this be ||?
						Yyerror("%v is not a type", Nconv(ll.N, obj.FmtLong))

						// reset to original type
						ll.N = n.Ntest.Right
					} else if ll.N.Type.Etype != TINTER && t.Etype == TINTER && !implements(ll.N.Type, t, &missing, &have, &ptr) {
						if have != nil && missing.Broke == 0 && have.Broke == 0 {
							Yyerror("impossible type switch case: %v cannot have dynamic type %v"+" (wrong type for %v method)\n\thave %v%v\n\twant %v%v", Nconv(n.Ntest.Right, obj.FmtLong), Tconv(ll.N.Type, 0), Sconv(missing.Sym, 0), Sconv(have.Sym, 0), Tconv(have.Type, obj.FmtShort), Sconv(missing.Sym, 0), Tconv(missing.Type, obj.FmtShort))
						} else if missing.Broke == 0 {
							Yyerror("impossible type switch case: %v cannot have dynamic type %v"+" (missing %v method)", Nconv(n.Ntest.Right, obj.FmtLong), Tconv(ll.N.Type, 0), Sconv(missing.Sym, 0))
						}
					}
				}
			}
		}

		if top == Etype && n.Type != nil {
			ll = ncase.List
			nvar = ncase.Nname
			if nvar != nil {
				if ll != nil && ll.Next == nil && ll.N.Type != nil && !Istype(ll.N.Type, TNIL) {
					// single entry type switch
					nvar.Ntype = typenod(ll.N.Type)
				} else {
					// multiple entry type switch or default
					nvar.Ntype = typenod(n.Type)
				}

				typecheck(&nvar, Erv|Easgn)
				ncase.Nname = nvar
			}
		}

		typechecklist(ncase.Nbody, Etop)
	}

	lineno = int32(lno)
}
