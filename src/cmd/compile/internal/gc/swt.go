// Copyright 2009 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package gc

import (
	"cmd/compile/internal/types"
	"fmt"
	"sort"
)

const (
	// expression switch
	switchKindExpr  = iota // switch a {...} or switch 5 {...}
	switchKindTrue         // switch true {...} or switch {...}
	switchKindFalse        // switch false {...}
)

const (
	binarySearchMin = 4 // minimum number of cases for binary search
	integerRangeMin = 2 // minimum size of integer ranges
)

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
	// isconst indicates whether this case clause is a constant,
	// for the purposes of the switch code generation.
	// For expression switches, that's generally literals (case 5:, not case x:).
	// For type switches, that's concrete types (case time.Time:), not interfaces (case io.Reader:).
	isconst bool
}

// caseClauses are all the case clauses in a switch statement.
type caseClauses struct {
	list   []caseClause // general cases
	defjmp *Node        // OGOTO for default case or OBREAK if no default case present
	niljmp *Node        // OGOTO for nil type case in a type switch
}

// typecheckswitch typechecks a switch statement.
func typecheckswitch(n *Node) {
	typecheckslice(n.Ninit.Slice(), Etop)

	var nilonly string
	var top int
	var t *types.Type

	if n.Left != nil && n.Left.Op == OTYPESW {
		// type switch
		top = Etype
		n.Left.Right = typecheck(n.Left.Right, Erv)
		t = n.Left.Right.Type
		if t != nil && !t.IsInterface() {
			yyerrorl(n.Pos, "cannot type switch on non-interface value %L", n.Left.Right)
		}
	} else {
		// expression switch
		top = Erv
		if n.Left != nil {
			n.Left = typecheck(n.Left, Erv)
			n.Left = defaultlit(n.Left, nil)
			t = n.Left.Type
		} else {
			t = types.Types[TBOOL]
		}
		if t != nil {
			switch {
			case !okforeq[t.Etype]:
				yyerrorl(n.Pos, "cannot switch on %L", n.Left)
			case t.IsSlice():
				nilonly = "slice"
			case t.IsArray() && !IsComparable(t):
				yyerrorl(n.Pos, "cannot switch on %L", n.Left)
			case t.IsStruct():
				if f := IncomparableField(t); f != nil {
					yyerrorl(n.Pos, "cannot switch on %L (struct containing %v cannot be compared)", n.Left, f.Type)
				}
			case t.Etype == TFUNC:
				nilonly = "func"
			case t.IsMap():
				nilonly = "map"
			}
		}
	}

	n.Type = t

	var def, niltype *Node
	for _, ncase := range n.List.Slice() {
		if ncase.List.Len() == 0 {
			// default
			if def != nil {
				setlineno(ncase)
				yyerrorl(ncase.Pos, "multiple defaults in switch (first at %v)", def.Line())
			} else {
				def = ncase
			}
		} else {
			ls := ncase.List.Slice()
			for i1, n1 := range ls {
				setlineno(n1)
				ls[i1] = typecheck(ls[i1], Erv|Etype)
				n1 = ls[i1]
				if n1.Type == nil || t == nil {
					continue
				}

				setlineno(ncase)
				switch top {
				// expression switch
				case Erv:
					ls[i1] = defaultlit(ls[i1], t)
					n1 = ls[i1]
					switch {
					case n1.Op == OTYPE:
						yyerrorl(ncase.Pos, "type %v is not an expression", n1.Type)
					case n1.Type != nil && assignop(n1.Type, t, nil) == 0 && assignop(t, n1.Type, nil) == 0:
						if n.Left != nil {
							yyerrorl(ncase.Pos, "invalid case %v in switch on %v (mismatched types %v and %v)", n1, n.Left, n1.Type, t)
						} else {
							yyerrorl(ncase.Pos, "invalid case %v in switch (mismatched types %v and bool)", n1, n1.Type)
						}
					case nilonly != "" && !isnil(n1):
						yyerrorl(ncase.Pos, "invalid case %v in switch (can only compare %s %v to nil)", n1, nilonly, n.Left)
					case t.IsInterface() && !n1.Type.IsInterface() && !IsComparable(n1.Type):
						yyerrorl(ncase.Pos, "invalid case %L in switch (incomparable type)", n1)
					}

				// type switch
				case Etype:
					var missing, have *types.Field
					var ptr int
					switch {
					case n1.Op == OLITERAL && n1.Type.IsKind(TNIL):
						// case nil:
						if niltype != nil {
							yyerrorl(ncase.Pos, "multiple nil cases in type switch (first at %v)", niltype.Line())
						} else {
							niltype = ncase
						}
					case n1.Op != OTYPE && n1.Type != nil: // should this be ||?
						yyerrorl(ncase.Pos, "%L is not a type", n1)
						// reset to original type
						n1 = n.Left.Right
						ls[i1] = n1
					case !n1.Type.IsInterface() && t.IsInterface() && !implements(n1.Type, t, &missing, &have, &ptr):
						if have != nil && !missing.Broke() && !have.Broke() {
							yyerrorl(ncase.Pos, "impossible type switch case: %L cannot have dynamic type %v"+
								" (wrong type for %v method)\n\thave %v%S\n\twant %v%S", n.Left.Right, n1.Type, missing.Sym, have.Sym, have.Type, missing.Sym, missing.Type)
						} else if !missing.Broke() {
							if ptr != 0 {
								yyerrorl(ncase.Pos, "impossible type switch case: %L cannot have dynamic type %v"+
									" (%v method has pointer receiver)", n.Left.Right, n1.Type, missing.Sym)
							} else {
								yyerrorl(ncase.Pos, "impossible type switch case: %L cannot have dynamic type %v"+
									" (missing %v method)", n.Left.Right, n1.Type, missing.Sym)
							}
						}
					}
				}
			}
		}

		if n.Type == nil || n.Type.IsUntyped() {
			// if the value we're switching on has no type or is untyped,
			// we've already printed an error and don't need to continue
			// typechecking the body
			return
		}

		if top == Etype {
			ll := ncase.List
			if ncase.Rlist.Len() != 0 {
				nvar := ncase.Rlist.First()
				if ll.Len() == 1 && ll.First().Type != nil && !ll.First().Type.IsKind(TNIL) {
					// single entry type switch
					nvar.Type = ll.First().Type
				} else {
					// multiple entry type switch or default
					nvar.Type = n.Type
				}

				nvar = typecheck(nvar, Erv|Easgn)
				ncase.Rlist.SetFirst(nvar)
			}
		}

		typecheckslice(ncase.Nbody.Slice(), Etop)
	}
	switch top {
	// expression switch
	case Erv:
		checkDupExprCases(n.Left, n.List.Slice())
	}
}

// walkswitch walks a switch statement.
func walkswitch(sw *Node) {
	// convert switch {...} to switch true {...}
	if sw.Left == nil {
		sw.Left = nodbool(true)
		sw.Left = typecheck(sw.Left, Erv)
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

	cond = walkexpr(cond, &sw.Ninit)
	t := sw.Type
	if t == nil {
		return
	}

	// convert the switch into OIF statements
	var cas []*Node
	if s.kind == switchKindTrue || s.kind == switchKindFalse {
		s.exprname = nodbool(s.kind == switchKindTrue)
	} else if consttype(cond) >= 0 {
		// leave constants to enable dead code elimination (issue 9608)
		s.exprname = cond
	} else {
		s.exprname = temp(cond.Type)
		cas = []*Node{nod(OAS, s.exprname, cond)}
		typecheckslice(cas, Etop)
	}

	// Enumerate the cases and prepare the default case.
	clauses := s.genCaseClauses(sw.List.Slice())
	sw.List.Set(nil)
	cc := clauses.list

	// handle the cases in order
	for len(cc) > 0 {
		// deal with expressions one at a time
		if !okforcmp[t.Etype] || !cc[0].isconst {
			a := s.walkCases(cc[:1])
			cas = append(cas, a)
			cc = cc[1:]
			continue
		}

		// do binary search on runs of constants
		var run int
		for run = 1; run < len(cc) && cc[run].isconst; run++ {
		}

		// sort and compile constants
		sort.Sort(caseClauseByConstVal(cc[:run]))
		a := s.walkCases(cc[:run])
		cas = append(cas, a)
		cc = cc[run:]
	}

	// handle default case
	if nerrors == 0 {
		cas = append(cas, clauses.defjmp)
		sw.Nbody.Prepend(cas...)
		walkstmtlist(sw.Nbody.Slice())
	}
}

// walkCases generates an AST implementing the cases in cc.
func (s *exprSwitch) walkCases(cc []caseClause) *Node {
	if len(cc) < binarySearchMin {
		// linear search
		var cas []*Node
		for _, c := range cc {
			n := c.node
			lno := setlineno(n)

			a := nod(OIF, nil, nil)
			if rng := n.List.Slice(); rng != nil {
				// Integer range.
				// exprname is a temp or a constant,
				// so it is safe to evaluate twice.
				// In most cases, this conjunction will be
				// rewritten by walkinrange into a single comparison.
				low := nod(OGE, s.exprname, rng[0])
				high := nod(OLE, s.exprname, rng[1])
				a.Left = nod(OANDAND, low, high)
				a.Left = typecheck(a.Left, Erv)
				a.Left = walkexpr(a.Left, nil) // give walk the opportunity to optimize the range check
			} else if (s.kind != switchKindTrue && s.kind != switchKindFalse) || assignop(n.Left.Type, s.exprname.Type, nil) == OCONVIFACE || assignop(s.exprname.Type, n.Left.Type, nil) == OCONVIFACE {
				a.Left = nod(OEQ, s.exprname, n.Left) // if name == val
				a.Left = typecheck(a.Left, Erv)
			} else if s.kind == switchKindTrue {
				a.Left = n.Left // if val
			} else {
				// s.kind == switchKindFalse
				a.Left = nod(ONOT, n.Left, nil) // if !val
				a.Left = typecheck(a.Left, Erv)
			}
			a.Nbody.Set1(n.Right) // goto l

			cas = append(cas, a)
			lineno = lno
		}
		return liststmt(cas)
	}

	// find the middle and recur
	half := len(cc) / 2
	a := nod(OIF, nil, nil)
	n := cc[half-1].node
	var mid *Node
	if rng := n.List.Slice(); rng != nil {
		mid = rng[1] // high end of range
	} else {
		mid = n.Left
	}
	le := nod(OLE, s.exprname, mid)
	if Isconst(mid, CTSTR) {
		// Search by length and then by value; see caseClauseByConstVal.
		lenlt := nod(OLT, nod(OLEN, s.exprname, nil), nod(OLEN, mid, nil))
		leneq := nod(OEQ, nod(OLEN, s.exprname, nil), nod(OLEN, mid, nil))
		a.Left = nod(OOROR, lenlt, nod(OANDAND, leneq, le))
	} else {
		a.Left = le
	}
	a.Left = typecheck(a.Left, Erv)
	a.Nbody.Set1(s.walkCases(cc[:half]))
	a.Rlist.Set1(s.walkCases(cc[half:]))
	return a
}

// casebody builds separate lists of statements and cases.
// It makes labels between cases and statements
// and deals with fallthrough, break, and unreachable statements.
func casebody(sw *Node, typeswvar *Node) {
	if sw.List.Len() == 0 {
		return
	}

	lno := setlineno(sw)

	var cas []*Node  // cases
	var stat []*Node // statements
	var def *Node    // defaults
	br := nod(OBREAK, nil, nil)

	for i, n := range sw.List.Slice() {
		setlineno(n)
		if n.Op != OXCASE {
			Fatalf("casebody %v", n.Op)
		}
		n.Op = OCASE
		needvar := n.List.Len() != 1 || n.List.First().Op == OLITERAL

		jmp := nod(OGOTO, autolabel(".s"), nil)
		switch n.List.Len() {
		case 0:
			// default
			if def != nil {
				yyerror("more than one default case")
			}
			// reuse original default case
			n.Right = jmp
			def = n
		case 1:
			// one case -- reuse OCASE node
			n.Left = n.List.First()
			n.Right = jmp
			n.List.Set(nil)
			cas = append(cas, n)
		default:
			// Expand multi-valued cases and detect ranges of integer cases.
			if typeswvar != nil || sw.Left.Type.IsInterface() || !n.List.First().Type.IsInteger() || n.List.Len() < integerRangeMin {
				// Can't use integer ranges. Expand each case into a separate node.
				for _, n1 := range n.List.Slice() {
					cas = append(cas, nod(OCASE, n1, jmp))
				}
				break
			}
			// Find integer ranges within runs of constants.
			s := n.List.Slice()
			j := 0
			for j < len(s) {
				// Find a run of constants.
				var run int
				for run = j; run < len(s) && Isconst(s[run], CTINT); run++ {
				}
				if run-j >= integerRangeMin {
					// Search for integer ranges in s[j:run].
					// Typechecking is done, so all values are already in an appropriate range.
					search := s[j:run]
					sort.Sort(constIntNodesByVal(search))
					for beg, end := 0, 1; end <= len(search); end++ {
						if end < len(search) && search[end].Int64() == search[end-1].Int64()+1 {
							continue
						}
						if end-beg >= integerRangeMin {
							// Record range in List.
							c := nod(OCASE, nil, jmp)
							c.List.Set2(search[beg], search[end-1])
							cas = append(cas, c)
						} else {
							// Not large enough for range; record separately.
							for _, n := range search[beg:end] {
								cas = append(cas, nod(OCASE, n, jmp))
							}
						}
						beg = end
					}
					j = run
				}
				// Advance to next constant, adding individual non-constant
				// or as-yet-unhandled constant cases as we go.
				for ; j < len(s) && (j < run || !Isconst(s[j], CTINT)); j++ {
					cas = append(cas, nod(OCASE, s[j], jmp))
				}
			}
		}

		stat = append(stat, nod(OLABEL, jmp.Left, nil))
		if typeswvar != nil && needvar && n.Rlist.Len() != 0 {
			l := []*Node{
				nod(ODCL, n.Rlist.First(), nil),
				nod(OAS, n.Rlist.First(), typeswvar),
			}
			typecheckslice(l, Etop)
			stat = append(stat, l...)
		}
		stat = append(stat, n.Nbody.Slice()...)

		// Search backwards for the index of the fallthrough
		// statement. Do not assume it'll be in the last
		// position, since in some cases (e.g. when the statement
		// list contains autotmp_ variables), one or more OVARKILL
		// nodes will be at the end of the list.
		fallIndex := len(stat) - 1
		for stat[fallIndex].Op == OVARKILL {
			fallIndex--
		}
		last := stat[fallIndex]

		// botch - shouldn't fall through declaration
		if last.Xoffset == n.Xoffset && last.Op == OXFALL {
			if typeswvar != nil {
				setlineno(last)
				yyerror("cannot fallthrough in type switch")
			}

			if i+1 >= sw.List.Len() {
				setlineno(last)
				yyerror("cannot fallthrough final case in switch")
			}

			last.Op = OFALL
		} else {
			stat = append(stat, br)
		}
	}

	stat = append(stat, br)
	if def != nil {
		cas = append(cas, def)
	}

	sw.List.Set(cas)
	sw.Nbody.Set(stat)
	lineno = lno
}

// genCaseClauses generates the caseClauses value for clauses.
func (s *exprSwitch) genCaseClauses(clauses []*Node) caseClauses {
	var cc caseClauses
	for _, n := range clauses {
		if n.Left == nil && n.List.Len() == 0 {
			// default case
			if cc.defjmp != nil {
				Fatalf("duplicate default case not detected during typechecking")
			}
			cc.defjmp = n.Right
			continue
		}
		c := caseClause{node: n, ordinal: len(cc.list)}
		if n.List.Len() > 0 {
			c.isconst = true
		}
		switch consttype(n.Left) {
		case CTFLT, CTINT, CTRUNE, CTSTR:
			c.isconst = true
		}
		cc.list = append(cc.list, c)
	}

	if cc.defjmp == nil {
		cc.defjmp = nod(OBREAK, nil, nil)
	}
	return cc
}

// genCaseClauses generates the caseClauses value for clauses.
func (s *typeSwitch) genCaseClauses(clauses []*Node) caseClauses {
	var cc caseClauses
	for _, n := range clauses {
		switch {
		case n.Left == nil:
			// default case
			if cc.defjmp != nil {
				Fatalf("duplicate default case not detected during typechecking")
			}
			cc.defjmp = n.Right
			continue
		case n.Left.Op == OLITERAL:
			// nil case in type switch
			if cc.niljmp != nil {
				Fatalf("duplicate nil case not detected during typechecking")
			}
			cc.niljmp = n.Right
			continue
		}

		// general case
		c := caseClause{
			node:    n,
			ordinal: len(cc.list),
			isconst: !n.Left.Type.IsInterface(),
			hash:    typehash(n.Left.Type),
		}
		cc.list = append(cc.list, c)
	}

	if cc.defjmp == nil {
		cc.defjmp = nod(OBREAK, nil, nil)
	}

	// diagnose duplicate cases
	s.checkDupCases(cc.list)
	return cc
}

func (s *typeSwitch) checkDupCases(cc []caseClause) {
	if len(cc) < 2 {
		return
	}
	// We store seen types in a map keyed by type hash.
	// It is possible, but very unlikely, for multiple distinct types to have the same hash.
	seen := make(map[uint32][]*Node)
	// To avoid many small allocations of length 1 slices,
	// also set up a single large slice to slice into.
	nn := make([]*Node, 0, len(cc))
Outer:
	for _, c := range cc {
		prev, ok := seen[c.hash]
		if !ok {
			// First entry for this hash.
			nn = append(nn, c.node)
			seen[c.hash] = nn[len(nn)-1 : len(nn):len(nn)]
			continue
		}
		for _, n := range prev {
			if eqtype(n.Left.Type, c.node.Left.Type) {
				yyerrorl(c.node.Pos, "duplicate case %v in type switch\n\tprevious case at %v", c.node.Left.Type, n.Line())
				// avoid double-reporting errors
				continue Outer
			}
		}
		seen[c.hash] = append(seen[c.hash], c.node)
	}
}

func checkDupExprCases(exprname *Node, clauses []*Node) {
	// boolean (naked) switch, nothing to do.
	if exprname == nil {
		return
	}
	// The common case is that s's expression is not an interface.
	// In that case, all constant clauses have the same type,
	// so checking for duplicates can be done solely by value.
	if !exprname.Type.IsInterface() {
		seen := make(map[interface{}]*Node)
		for _, ncase := range clauses {
			for _, n := range ncase.List.Slice() {
				// Can't check for duplicates that aren't constants, per the spec. Issue 15896.
				// Don't check for duplicate bools. Although the spec allows it,
				// (1) the compiler hasn't checked it in the past, so compatibility mandates it, and
				// (2) it would disallow useful things like
				//       case GOARCH == "arm" && GOARM == "5":
				//       case GOARCH == "arm":
				//     which would both evaluate to false for non-ARM compiles.
				if ct := consttype(n); ct < 0 || ct == CTBOOL {
					continue
				}

				val := n.Val().Interface()
				prev, dup := seen[val]
				if !dup {
					seen[val] = n
					continue
				}
				yyerrorl(ncase.Pos, "duplicate case %s in switch\n\tprevious case at %v",
					nodeAndVal(n), prev.Line())
			}
		}
		return
	}
	// s's expression is an interface. This is fairly rare, so keep this simple.
	// Duplicates are only duplicates if they have the same type and the same value.
	type typeVal struct {
		typ string
		val interface{}
	}
	seen := make(map[typeVal]*Node)
	for _, ncase := range clauses {
		for _, n := range ncase.List.Slice() {
			if ct := consttype(n); ct < 0 || ct == CTBOOL {
				continue
			}
			tv := typeVal{
				typ: n.Type.LongString(),
				val: n.Val().Interface(),
			}
			prev, dup := seen[tv]
			if !dup {
				seen[tv] = n
				continue
			}
			yyerrorl(ncase.Pos, "duplicate case %s in switch\n\tprevious case at %v",
				nodeAndVal(n), prev.Line())
		}
	}
}

func nodeAndVal(n *Node) string {
	show := n.String()
	val := n.Val().Interface()
	if s := fmt.Sprintf("%#v", val); show != s {
		show += " (value " + s + ")"
	}
	return show
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
		sw.List.Set(nil)
		return
	}
	if cond.Right == nil {
		setlineno(sw)
		yyerror("type switch must have an assignment")
		return
	}

	cond.Right = walkexpr(cond.Right, &sw.Ninit)
	if !cond.Right.Type.IsInterface() {
		yyerror("type switch must be on an interface")
		return
	}

	var cas []*Node

	// predeclare temporary variables and the boolean var
	s.facename = temp(cond.Right.Type)

	a := nod(OAS, s.facename, cond.Right)
	a = typecheck(a, Etop)
	cas = append(cas, a)

	s.okname = temp(types.Types[TBOOL])
	s.okname = typecheck(s.okname, Erv)

	s.hashname = temp(types.Types[TUINT32])
	s.hashname = typecheck(s.hashname, Erv)

	// set up labels and jumps
	casebody(sw, s.facename)

	clauses := s.genCaseClauses(sw.List.Slice())
	sw.List.Set(nil)
	def := clauses.defjmp

	// For empty interfaces, do:
	//     if e._type == nil {
	//         do nil case if it exists, otherwise default
	//     }
	//     h := e._type.hash
	// Use a similar strategy for non-empty interfaces.

	// Get interface descriptor word.
	// For empty interfaces this will be the type.
	// For non-empty interfaces this will be the itab.
	itab := nod(OITAB, s.facename, nil)

	// Check for nil first.
	i := nod(OIF, nil, nil)
	i.Left = nod(OEQ, itab, nodnil())
	if clauses.niljmp != nil {
		// Do explicit nil case right here.
		i.Nbody.Set1(clauses.niljmp)
	} else {
		// Jump to default case.
		lbl := autolabel(".s")
		i.Nbody.Set1(nod(OGOTO, lbl, nil))
		// Wrap default case with label.
		blk := nod(OBLOCK, nil, nil)
		blk.List.Set2(nod(OLABEL, lbl, nil), def)
		def = blk
	}
	i.Left = typecheck(i.Left, Erv)
	cas = append(cas, i)

	// Load hash from type or itab.
	h := nodSym(ODOTPTR, itab, nil)
	h.Type = types.Types[TUINT32]
	h.SetTypecheck(1)
	if cond.Right.Type.IsEmptyInterface() {
		h.Xoffset = int64(2 * Widthptr) // offset of hash in runtime._type
	} else {
		h.Xoffset = int64(3 * Widthptr) // offset of hash in runtime.itab
	}
	h.SetBounded(true) // guaranteed not to fault
	a = nod(OAS, s.hashname, h)
	a = typecheck(a, Etop)
	cas = append(cas, a)

	cc := clauses.list

	// insert type equality check into each case block
	for _, c := range cc {
		c.node.Right = s.typeone(c.node)
	}

	// generate list of if statements, binary search for constant sequences
	for len(cc) > 0 {
		if !cc[0].isconst {
			n := cc[0].node
			cas = append(cas, n.Right)
			cc = cc[1:]
			continue
		}

		// identify run of constants
		var run int
		for run = 1; run < len(cc) && cc[run].isconst; run++ {
		}

		// sort by hash
		sort.Sort(caseClauseByType(cc[:run]))

		// for debugging: linear search
		if false {
			for i := 0; i < run; i++ {
				n := cc[i].node
				cas = append(cas, n.Right)
			}
			continue
		}

		// combine adjacent cases with the same hash
		ncase := 0
		for i := 0; i < run; i++ {
			ncase++
			hash := []*Node{cc[i].node.Right}
			for j := i + 1; j < run && cc[i].hash == cc[j].hash; j++ {
				hash = append(hash, cc[j].node.Right)
			}
			cc[i].node.Right = liststmt(hash)
		}

		// binary search among cases to narrow by hash
		cas = append(cas, s.walkCases(cc[:ncase]))
		cc = cc[ncase:]
	}

	// handle default case
	if nerrors == 0 {
		cas = append(cas, def)
		sw.Nbody.Prepend(cas...)
		sw.List.Set(nil)
		walkstmtlist(sw.Nbody.Slice())
	}
}

// typeone generates an AST that jumps to the
// case body if the variable is of type t.
func (s *typeSwitch) typeone(t *Node) *Node {
	var name *Node
	var init Nodes
	if t.Rlist.Len() == 0 {
		name = nblank
		nblank = typecheck(nblank, Erv|Easgn)
	} else {
		name = t.Rlist.First()
		init.Append(nod(ODCL, name, nil))
		a := nod(OAS, name, nil)
		a = typecheck(a, Etop)
		init.Append(a)
	}

	a := nod(OAS2, nil, nil)
	a.List.Set2(name, s.okname) // name, ok =
	b := nod(ODOTTYPE, s.facename, nil)
	b.Type = t.Left.Type // interface.(type)
	a.Rlist.Set1(b)
	a = typecheck(a, Etop)
	a = walkexpr(a, &init)
	init.Append(a)

	c := nod(OIF, nil, nil)
	c.Left = s.okname
	c.Nbody.Set1(t.Right) // if ok { goto l }

	init.Append(c)
	return init.asblock()
}

// walkCases generates an AST implementing the cases in cc.
func (s *typeSwitch) walkCases(cc []caseClause) *Node {
	if len(cc) < binarySearchMin {
		var cas []*Node
		for _, c := range cc {
			n := c.node
			if !c.isconst {
				Fatalf("typeSwitch walkCases")
			}
			a := nod(OIF, nil, nil)
			a.Left = nod(OEQ, s.hashname, nodintconst(int64(c.hash)))
			a.Left = typecheck(a.Left, Erv)
			a.Nbody.Set1(n.Right)
			cas = append(cas, a)
		}
		return liststmt(cas)
	}

	// find the middle and recur
	half := len(cc) / 2
	a := nod(OIF, nil, nil)
	a.Left = nod(OLE, s.hashname, nodintconst(int64(cc[half-1].hash)))
	a.Left = typecheck(a.Left, Erv)
	a.Nbody.Set1(s.walkCases(cc[:half]))
	a.Rlist.Set1(s.walkCases(cc[half:]))
	return a
}

// caseClauseByConstVal sorts clauses by constant value to enable binary search.
type caseClauseByConstVal []caseClause

func (x caseClauseByConstVal) Len() int      { return len(x) }
func (x caseClauseByConstVal) Swap(i, j int) { x[i], x[j] = x[j], x[i] }
func (x caseClauseByConstVal) Less(i, j int) bool {
	// n1 and n2 might be individual constants or integer ranges.
	// We have checked for duplicates already,
	// so ranges can be safely represented by any value in the range.
	n1 := x[i].node
	var v1 interface{}
	if s := n1.List.Slice(); s != nil {
		v1 = s[0].Val().U
	} else {
		v1 = n1.Left.Val().U
	}

	n2 := x[j].node
	var v2 interface{}
	if s := n2.List.Slice(); s != nil {
		v2 = s[0].Val().U
	} else {
		v2 = n2.Left.Val().U
	}

	switch v1 := v1.(type) {
	case *Mpflt:
		return v1.Cmp(v2.(*Mpflt)) < 0
	case *Mpint:
		return v1.Cmp(v2.(*Mpint)) < 0
	case string:
		// Sort strings by length and then by value.
		// It is much cheaper to compare lengths than values,
		// and all we need here is consistency.
		// We respect this sorting in exprSwitch.walkCases.
		a := v1
		b := v2.(string)
		if len(a) != len(b) {
			return len(a) < len(b)
		}
		return a < b
	}

	Fatalf("caseClauseByConstVal passed bad clauses %v < %v", x[i].node.Left, x[j].node.Left)
	return false
}

type caseClauseByType []caseClause

func (x caseClauseByType) Len() int      { return len(x) }
func (x caseClauseByType) Swap(i, j int) { x[i], x[j] = x[j], x[i] }
func (x caseClauseByType) Less(i, j int) bool {
	c1, c2 := x[i], x[j]
	// sort by hash code, then ordinal (for the rare case of hash collisions)
	if c1.hash != c2.hash {
		return c1.hash < c2.hash
	}
	return c1.ordinal < c2.ordinal
}

type constIntNodesByVal []*Node

func (x constIntNodesByVal) Len() int      { return len(x) }
func (x constIntNodesByVal) Swap(i, j int) { x[i], x[j] = x[j], x[i] }
func (x constIntNodesByVal) Less(i, j int) bool {
	return x[i].Val().U.(*Mpint).Cmp(x[j].Val().U.(*Mpint)) < 0
}
