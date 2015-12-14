// Copyright 2009 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package gc

import (
	"cmd/internal/obj"
	"sort"
	"strconv"
)

const (
	// expression switch
	switchKindExpr  = iota // switch a {...} or switch 5 {...}
	switchKindTrue         // switch true {...} or switch {...}
	switchKindFalse        // switch false {...}

	// type switch
	switchKindType // switch a.(type) {...}
)

const (
	caseKindDefault = iota // default:

	// expression switch
	caseKindExprConst // case 5:
	caseKindExprVar   // case x:

	// type switch
	caseKindTypeNil   // case nil:
	caseKindTypeConst // case time.Time: (concrete type, has type hash)
	caseKindTypeVar   // case io.Reader: (interface type)
)

const binarySearchMin = 4 // minimum number of cases for binary search

// An exprSwitch walks an expression switch.
type exprSwitch struct {
	exprname *Node // node for the expression being switched on
	kind     int   // kind of switch statement (switchKind*)
}

// A typeSwitch walks a type switch.
type typeSwitch struct {
	hashname *Node // node for the hash of the type of the variable being switched on
	facename *Node // node for the concrete type of the variable being switched on
	okname   *Node // boolean node used for comma-ok type assertions
}

// A caseClause is a single case clause in a switch statement.
type caseClause struct {
	node    *Node  // points at case statement
	ordinal int    // position in switch
	hash    uint32 // hash of a type switch
	typ     uint8  // type of case
}

// typecheckswitch typechecks a switch statement.
func typecheckswitch(n *Node) {
	lno := int(lineno)
	typechecklist(n.Ninit, Etop)

	var nilonly string
	var top int
	var t *Type

	if n.Left != nil && n.Left.Op == OTYPESW {
		// type switch
		top = Etype
		typecheck(&n.Left.Right, Erv)
		t = n.Left.Right.Type
		if t != nil && t.Etype != TINTER {
			Yyerror("cannot type switch on non-interface value %v", Nconv(n.Left.Right, obj.FmtLong))
		}
	} else {
		// expression switch
		top = Erv
		if n.Left != nil {
			typecheck(&n.Left, Erv)
			defaultlit(&n.Left, nil)
			t = n.Left.Type
		} else {
			t = Types[TBOOL]
		}
		if t != nil {
			var badtype *Type
			switch {
			case !okforeq[t.Etype]:
				Yyerror("cannot switch on %v", Nconv(n.Left, obj.FmtLong))
			case t.Etype == TARRAY && !Isfixedarray(t):
				nilonly = "slice"
			case t.Etype == TARRAY && Isfixedarray(t) && algtype1(t, nil) == ANOEQ:
				Yyerror("cannot switch on %v", Nconv(n.Left, obj.FmtLong))
			case t.Etype == TSTRUCT && algtype1(t, &badtype) == ANOEQ:
				Yyerror("cannot switch on %v (struct containing %v cannot be compared)", Nconv(n.Left, obj.FmtLong), badtype)
			case t.Etype == TFUNC:
				nilonly = "func"
			case t.Etype == TMAP:
				nilonly = "map"
			}
		}
	}

	n.Type = t

	var def *Node
	var ll *NodeList
	for l := n.List; l != nil; l = l.Next {
		ncase := l.N
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
				// expression switch
				case Erv:
					defaultlit(&ll.N, t)
					switch {
					case ll.N.Op == OTYPE:
						Yyerror("type %v is not an expression", ll.N.Type)
					case ll.N.Type != nil && assignop(ll.N.Type, t, nil) == 0 && assignop(t, ll.N.Type, nil) == 0:
						if n.Left != nil {
							Yyerror("invalid case %v in switch on %v (mismatched types %v and %v)", ll.N, n.Left, ll.N.Type, t)
						} else {
							Yyerror("invalid case %v in switch (mismatched types %v and bool)", ll.N, ll.N.Type)
						}
					case nilonly != "" && !isnil(ll.N):
						Yyerror("invalid case %v in switch (can only compare %s %v to nil)", ll.N, nilonly, n.Left)
					case Isinter(t) && !Isinter(ll.N.Type) && algtype1(ll.N.Type, nil) == ANOEQ:
						Yyerror("invalid case %v in switch (incomparable type)", Nconv(ll.N, obj.FmtLong))
					}

				// type switch
				case Etype:
					var missing, have *Type
					var ptr int
					switch {
					case ll.N.Op == OLITERAL && Istype(ll.N.Type, TNIL):
					case ll.N.Op != OTYPE && ll.N.Type != nil: // should this be ||?
						Yyerror("%v is not a type", Nconv(ll.N, obj.FmtLong))
						// reset to original type
						ll.N = n.Left.Right
					case ll.N.Type.Etype != TINTER && t.Etype == TINTER && !implements(ll.N.Type, t, &missing, &have, &ptr):
						if have != nil && !missing.Broke && !have.Broke {
							Yyerror("impossible type switch case: %v cannot have dynamic type %v"+" (wrong type for %v method)\n\thave %v%v\n\twant %v%v", Nconv(n.Left.Right, obj.FmtLong), ll.N.Type, missing.Sym, have.Sym, Tconv(have.Type, obj.FmtShort), missing.Sym, Tconv(missing.Type, obj.FmtShort))
						} else if !missing.Broke {
							Yyerror("impossible type switch case: %v cannot have dynamic type %v"+" (missing %v method)", Nconv(n.Left.Right, obj.FmtLong), ll.N.Type, missing.Sym)
						}
					}
				}
			}
		}

		if top == Etype && n.Type != nil {
			ll = ncase.List
			if ncase.Rlist != nil {
				nvar := ncase.Rlist.N
				if ll != nil && ll.Next == nil && ll.N.Type != nil && !Istype(ll.N.Type, TNIL) {
					// single entry type switch
					nvar.Name.Param.Ntype = typenod(ll.N.Type)
				} else {
					// multiple entry type switch or default
					nvar.Name.Param.Ntype = typenod(n.Type)
				}

				typecheck(&nvar, Erv|Easgn)
				ncase.Rlist.N = nvar
			}
		}

		typechecklist(ncase.Nbody, Etop)
	}

	lineno = int32(lno)
}

// walkswitch walks a switch statement.
func walkswitch(sw *Node) {
	// convert switch {...} to switch true {...}
	if sw.Left == nil {
		sw.Left = Nodbool(true)
		typecheck(&sw.Left, Erv)
	}

	if sw.Left.Op == OTYPESW {
		var s typeSwitch
		s.walk(sw)
	} else {
		var s exprSwitch
		s.walk(sw)
	}
}

// walk generates an AST implementing sw.
// sw is an expression switch.
// The AST is generally of the form of a linear
// search using if..goto, although binary search
// is used with long runs of constants.
func (s *exprSwitch) walk(sw *Node) {
	casebody(sw, nil)

	cond := sw.Left
	sw.Left = nil

	s.kind = switchKindExpr
	if Isconst(cond, CTBOOL) {
		s.kind = switchKindTrue
		if !cond.Val().U.(bool) {
			s.kind = switchKindFalse
		}
	}

	walkexpr(&cond, &sw.Ninit)
	t := sw.Type
	if t == nil {
		return
	}

	// convert the switch into OIF statements
	var cas *NodeList
	if s.kind == switchKindTrue || s.kind == switchKindFalse {
		s.exprname = Nodbool(s.kind == switchKindTrue)
	} else if consttype(cond) >= 0 {
		// leave constants to enable dead code elimination (issue 9608)
		s.exprname = cond
	} else {
		s.exprname = temp(cond.Type)
		cas = list1(Nod(OAS, s.exprname, cond))
		typechecklist(cas, Etop)
	}

	// enumerate the cases, and lop off the default case
	cc := caseClauses(sw, s.kind)
	sw.List = nil
	var def *Node
	if len(cc) > 0 && cc[0].typ == caseKindDefault {
		def = cc[0].node.Right
		cc = cc[1:]
	} else {
		def = Nod(OBREAK, nil, nil)
	}

	// handle the cases in order
	for len(cc) > 0 {
		// deal with expressions one at a time
		if !okforcmp[t.Etype] || cc[0].typ != caseKindExprConst {
			a := s.walkCases(cc[:1])
			cas = list(cas, a)
			cc = cc[1:]
			continue
		}

		// do binary search on runs of constants
		var run int
		for run = 1; run < len(cc) && cc[run].typ == caseKindExprConst; run++ {
		}

		// sort and compile constants
		sort.Sort(caseClauseByExpr(cc[:run]))
		a := s.walkCases(cc[:run])
		cas = list(cas, a)
		cc = cc[run:]
	}

	// handle default case
	if nerrors == 0 {
		cas = list(cas, def)
		sw.Nbody = concat(cas, sw.Nbody)
		walkstmtlist(sw.Nbody)
	}
}

// walkCases generates an AST implementing the cases in cc.
func (s *exprSwitch) walkCases(cc []*caseClause) *Node {
	if len(cc) < binarySearchMin {
		// linear search
		var cas *NodeList
		for _, c := range cc {
			n := c.node
			lno := int(setlineno(n))

			a := Nod(OIF, nil, nil)
			if (s.kind != switchKindTrue && s.kind != switchKindFalse) || assignop(n.Left.Type, s.exprname.Type, nil) == OCONVIFACE || assignop(s.exprname.Type, n.Left.Type, nil) == OCONVIFACE {
				a.Left = Nod(OEQ, s.exprname, n.Left) // if name == val
				typecheck(&a.Left, Erv)
			} else if s.kind == switchKindTrue {
				a.Left = n.Left // if val
			} else {
				// s.kind == switchKindFalse
				a.Left = Nod(ONOT, n.Left, nil) // if !val
				typecheck(&a.Left, Erv)
			}
			a.Nbody = list1(n.Right) // goto l

			cas = list(cas, a)
			lineno = int32(lno)
		}
		return liststmt(cas)
	}

	// find the middle and recur
	half := len(cc) / 2
	a := Nod(OIF, nil, nil)
	mid := cc[half-1].node.Left
	le := Nod(OLE, s.exprname, mid)
	if Isconst(mid, CTSTR) {
		// Search by length and then by value; see exprcmp.
		lenlt := Nod(OLT, Nod(OLEN, s.exprname, nil), Nod(OLEN, mid, nil))
		leneq := Nod(OEQ, Nod(OLEN, s.exprname, nil), Nod(OLEN, mid, nil))
		a.Left = Nod(OOROR, lenlt, Nod(OANDAND, leneq, le))
	} else {
		a.Left = le
	}
	typecheck(&a.Left, Erv)
	a.Nbody = list1(s.walkCases(cc[:half]))
	a.Rlist = list1(s.walkCases(cc[half:]))
	return a
}

// casebody builds separate lists of statements and cases.
// It makes labels between cases and statements
// and deals with fallthrough, break, and unreachable statements.
func casebody(sw *Node, typeswvar *Node) {
	if sw.List == nil {
		return
	}

	lno := setlineno(sw)

	var cas *NodeList  // cases
	var stat *NodeList // statements
	var def *Node      // defaults
	br := Nod(OBREAK, nil, nil)

	for l := sw.List; l != nil; l = l.Next {
		n := l.N
		setlineno(n)
		if n.Op != OXCASE {
			Fatalf("casebody %v", Oconv(int(n.Op), 0))
		}
		n.Op = OCASE
		needvar := count(n.List) != 1 || n.List.N.Op == OLITERAL

		jmp := Nod(OGOTO, newCaseLabel(), nil)
		if n.List == nil {
			if def != nil {
				Yyerror("more than one default case")
			}
			// reuse original default case
			n.Right = jmp
			def = n
		}

		if n.List != nil && n.List.Next == nil {
			// one case -- reuse OCASE node
			n.Left = n.List.N
			n.Right = jmp
			n.List = nil
			cas = list(cas, n)
		} else {
			// expand multi-valued cases
			for lc := n.List; lc != nil; lc = lc.Next {
				cas = list(cas, Nod(OCASE, lc.N, jmp))
			}
		}

		stat = list(stat, Nod(OLABEL, jmp.Left, nil))
		if typeswvar != nil && needvar && n.Rlist != nil {
			l := list1(Nod(ODCL, n.Rlist.N, nil))
			l = list(l, Nod(OAS, n.Rlist.N, typeswvar))
			typechecklist(l, Etop)
			stat = concat(stat, l)
		}
		stat = concat(stat, n.Nbody)

		// botch - shouldn't fall thru declaration
		last := stat.End.N
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

// nSwitchLabel is the number of switch labels generated.
// This should be per-function, but it is a global counter for now.
var nSwitchLabel int

func newCaseLabel() *Node {
	label := strconv.Itoa(nSwitchLabel)
	nSwitchLabel++
	return newname(Lookup(label))
}

// caseClauses generates a slice of caseClauses
// corresponding to the clauses in the switch statement sw.
// Kind is the kind of switch statement.
func caseClauses(sw *Node, kind int) []*caseClause {
	var cc []*caseClause
	for l := sw.List; l != nil; l = l.Next {
		n := l.N
		c := new(caseClause)
		cc = append(cc, c)
		c.ordinal = len(cc)
		c.node = n

		if n.Left == nil {
			c.typ = caseKindDefault
			continue
		}

		if kind == switchKindType {
			// type switch
			switch {
			case n.Left.Op == OLITERAL:
				c.typ = caseKindTypeNil
			case Istype(n.Left.Type, TINTER):
				c.typ = caseKindTypeVar
			default:
				c.typ = caseKindTypeConst
				c.hash = typehash(n.Left.Type)
			}
		} else {
			// expression switch
			switch consttype(n.Left) {
			case CTFLT, CTINT, CTRUNE, CTSTR:
				c.typ = caseKindExprConst
			default:
				c.typ = caseKindExprVar
			}
		}
	}

	if cc == nil {
		return nil
	}

	// sort by value and diagnose duplicate cases
	if kind == switchKindType {
		// type switch
		sort.Sort(caseClauseByType(cc))
		for i, c1 := range cc {
			if c1.typ == caseKindTypeNil || c1.typ == caseKindDefault {
				break
			}
			for _, c2 := range cc[i+1:] {
				if c2.typ == caseKindTypeNil || c2.typ == caseKindDefault || c1.hash != c2.hash {
					break
				}
				if Eqtype(c1.node.Left.Type, c2.node.Left.Type) {
					yyerrorl(int(c2.node.Lineno), "duplicate case %v in type switch\n\tprevious case at %v", c2.node.Left.Type, c1.node.Line())
				}
			}
		}
	} else {
		// expression switch
		sort.Sort(caseClauseByExpr(cc))
		for i, c1 := range cc {
			if i+1 == len(cc) {
				break
			}
			c2 := cc[i+1]
			if exprcmp(c1, c2) != 0 {
				continue
			}
			setlineno(c2.node)
			Yyerror("duplicate case %v in switch\n\tprevious case at %v", c1.node.Left, c1.node.Line())
		}
	}

	// put list back in processing order
	sort.Sort(caseClauseByOrd(cc))
	return cc
}

// walk generates an AST that implements sw,
// where sw is a type switch.
// The AST is generally of the form of a linear
// search using if..goto, although binary search
// is used with long runs of concrete types.
func (s *typeSwitch) walk(sw *Node) {
	cond := sw.Left
	sw.Left = nil

	if cond == nil {
		sw.List = nil
		return
	}
	if cond.Right == nil {
		setlineno(sw)
		Yyerror("type switch must have an assignment")
		return
	}

	walkexpr(&cond.Right, &sw.Ninit)
	if !Istype(cond.Right.Type, TINTER) {
		Yyerror("type switch must be on an interface")
		return
	}

	var cas *NodeList

	// predeclare temporary variables and the boolean var
	s.facename = temp(cond.Right.Type)

	a := Nod(OAS, s.facename, cond.Right)
	typecheck(&a, Etop)
	cas = list(cas, a)

	s.okname = temp(Types[TBOOL])
	typecheck(&s.okname, Erv)

	s.hashname = temp(Types[TUINT32])
	typecheck(&s.hashname, Erv)

	// set up labels and jumps
	casebody(sw, s.facename)

	// calculate type hash
	t := cond.Right.Type
	if isnilinter(t) {
		a = syslook("efacethash", 1)
	} else {
		a = syslook("ifacethash", 1)
	}
	substArgTypes(a, t)
	a = Nod(OCALL, a, nil)
	a.List = list1(s.facename)
	a = Nod(OAS, s.hashname, a)
	typecheck(&a, Etop)
	cas = list(cas, a)

	cc := caseClauses(sw, switchKindType)
	sw.List = nil
	var def *Node
	if len(cc) > 0 && cc[0].typ == caseKindDefault {
		def = cc[0].node.Right
		cc = cc[1:]
	} else {
		def = Nod(OBREAK, nil, nil)
	}

	// insert type equality check into each case block
	for _, c := range cc {
		n := c.node
		switch c.typ {
		case caseKindTypeNil:
			var v Val
			v.U = new(NilVal)
			a = Nod(OIF, nil, nil)
			a.Left = Nod(OEQ, s.facename, nodlit(v))
			typecheck(&a.Left, Erv)
			a.Nbody = list1(n.Right) // if i==nil { goto l }
			n.Right = a

		case caseKindTypeVar, caseKindTypeConst:
			n.Right = s.typeone(n)
		}
	}

	// generate list of if statements, binary search for constant sequences
	for len(cc) > 0 {
		if cc[0].typ != caseKindTypeConst {
			n := cc[0].node
			cas = list(cas, n.Right)
			cc = cc[1:]
			continue
		}

		// identify run of constants
		var run int
		for run = 1; run < len(cc) && cc[run].typ == caseKindTypeConst; run++ {
		}

		// sort by hash
		sort.Sort(caseClauseByType(cc[:run]))

		// for debugging: linear search
		if false {
			for i := 0; i < run; i++ {
				n := cc[i].node
				cas = list(cas, n.Right)
			}
			continue
		}

		// combine adjacent cases with the same hash
		ncase := 0
		for i := 0; i < run; i++ {
			ncase++
			hash := list1(cc[i].node.Right)
			for j := i + 1; j < run && cc[i].hash == cc[j].hash; j++ {
				hash = list(hash, cc[j].node.Right)
			}
			cc[i].node.Right = liststmt(hash)
		}

		// binary search among cases to narrow by hash
		cas = list(cas, s.walkCases(cc[:ncase]))
		cc = cc[ncase:]
	}

	// handle default case
	if nerrors == 0 {
		cas = list(cas, def)
		sw.Nbody = concat(cas, sw.Nbody)
		sw.List = nil
		walkstmtlist(sw.Nbody)
	}
}

// typeone generates an AST that jumps to the
// case body if the variable is of type t.
func (s *typeSwitch) typeone(t *Node) *Node {
	var name *Node
	var init *NodeList
	if t.Rlist == nil {
		name = nblank
		typecheck(&nblank, Erv|Easgn)
	} else {
		name = t.Rlist.N
		init = list1(Nod(ODCL, name, nil))
		a := Nod(OAS, name, nil)
		typecheck(&a, Etop)
		init = list(init, a)
	}

	a := Nod(OAS2, nil, nil)
	a.List = list(list1(name), s.okname) // name, ok =
	b := Nod(ODOTTYPE, s.facename, nil)
	b.Type = t.Left.Type // interface.(type)
	a.Rlist = list1(b)
	typecheck(&a, Etop)
	init = list(init, a)

	c := Nod(OIF, nil, nil)
	c.Left = s.okname
	c.Nbody = list1(t.Right) // if ok { goto l }

	return liststmt(list(init, c))
}

// walkCases generates an AST implementing the cases in cc.
func (s *typeSwitch) walkCases(cc []*caseClause) *Node {
	if len(cc) < binarySearchMin {
		var cas *NodeList
		for _, c := range cc {
			n := c.node
			if c.typ != caseKindTypeConst {
				Fatalf("typeSwitch walkCases")
			}
			a := Nod(OIF, nil, nil)
			a.Left = Nod(OEQ, s.hashname, Nodintconst(int64(c.hash)))
			typecheck(&a.Left, Erv)
			a.Nbody = list1(n.Right)
			cas = list(cas, a)
		}
		return liststmt(cas)
	}

	// find the middle and recur
	half := len(cc) / 2
	a := Nod(OIF, nil, nil)
	a.Left = Nod(OLE, s.hashname, Nodintconst(int64(cc[half-1].hash)))
	typecheck(&a.Left, Erv)
	a.Nbody = list1(s.walkCases(cc[:half]))
	a.Rlist = list1(s.walkCases(cc[half:]))
	return a
}

type caseClauseByOrd []*caseClause

func (x caseClauseByOrd) Len() int      { return len(x) }
func (x caseClauseByOrd) Swap(i, j int) { x[i], x[j] = x[j], x[i] }
func (x caseClauseByOrd) Less(i, j int) bool {
	c1, c2 := x[i], x[j]
	switch {
	// sort default first
	case c1.typ == caseKindDefault:
		return true
	case c2.typ == caseKindDefault:
		return false

	// sort nil second
	case c1.typ == caseKindTypeNil:
		return true
	case c2.typ == caseKindTypeNil:
		return false
	}

	// sort by ordinal
	return c1.ordinal < c2.ordinal
}

type caseClauseByExpr []*caseClause

func (x caseClauseByExpr) Len() int      { return len(x) }
func (x caseClauseByExpr) Swap(i, j int) { x[i], x[j] = x[j], x[i] }
func (x caseClauseByExpr) Less(i, j int) bool {
	return exprcmp(x[i], x[j]) < 0
}

func exprcmp(c1, c2 *caseClause) int {
	// sort non-constants last
	if c1.typ != caseKindExprConst {
		return +1
	}
	if c2.typ != caseKindExprConst {
		return -1
	}

	n1 := c1.node.Left
	n2 := c2.node.Left

	// sort by type (for switches on interface)
	ct := n1.Val().Ctype()
	if ct > n2.Val().Ctype() {
		return +1
	}
	if ct < n2.Val().Ctype() {
		return -1
	}
	if !Eqtype(n1.Type, n2.Type) {
		if n1.Type.Vargen > n2.Type.Vargen {
			return +1
		} else {
			return -1
		}
	}

	// sort by constant value to enable binary search
	switch ct {
	case CTFLT:
		return mpcmpfltflt(n1.Val().U.(*Mpflt), n2.Val().U.(*Mpflt))
	case CTINT, CTRUNE:
		return Mpcmpfixfix(n1.Val().U.(*Mpint), n2.Val().U.(*Mpint))
	case CTSTR:
		// Sort strings by length and then by value.
		// It is much cheaper to compare lengths than values,
		// and all we need here is consistency.
		// We respect this sorting in exprSwitch.walkCases.
		a := n1.Val().U.(string)
		b := n2.Val().U.(string)
		if len(a) < len(b) {
			return -1
		}
		if len(a) > len(b) {
			return +1
		}
		if a == b {
			return 0
		}
		if a < b {
			return -1
		}
		return +1
	}

	return 0
}

type caseClauseByType []*caseClause

func (x caseClauseByType) Len() int      { return len(x) }
func (x caseClauseByType) Swap(i, j int) { x[i], x[j] = x[j], x[i] }
func (x caseClauseByType) Less(i, j int) bool {
	c1, c2 := x[i], x[j]
	switch {
	// sort non-constants last
	case c1.typ != caseKindTypeConst:
		return false
	case c2.typ != caseKindTypeConst:
		return true

	// sort by hash code
	case c1.hash != c2.hash:
		return c1.hash < c2.hash
	}

	// sort by ordinal
	return c1.ordinal < c2.ordinal
}
