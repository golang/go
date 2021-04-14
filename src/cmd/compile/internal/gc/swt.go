// Copyright 2009 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package gc

import (
	"cmd/compile/internal/types"
	"cmd/internal/src"
	"sort"
)

// typecheckswitch typechecks a switch statement.
func typecheckswitch(n *Node) {
	typecheckslice(n.Ninit.Slice(), ctxStmt)
	if n.Left != nil && n.Left.Op == OTYPESW {
		typecheckTypeSwitch(n)
	} else {
		typecheckExprSwitch(n)
	}
}

func typecheckTypeSwitch(n *Node) {
	n.Left.Right = typecheck(n.Left.Right, ctxExpr)
	t := n.Left.Right.Type
	if t != nil && !t.IsInterface() {
		yyerrorl(n.Pos, "cannot type switch on non-interface value %L", n.Left.Right)
		t = nil
	}

	// We don't actually declare the type switch's guarded
	// declaration itself. So if there are no cases, we won't
	// notice that it went unused.
	if v := n.Left.Left; v != nil && !v.isBlank() && n.List.Len() == 0 {
		yyerrorl(v.Pos, "%v declared but not used", v.Sym)
	}

	var defCase, nilCase *Node
	var ts typeSet
	for _, ncase := range n.List.Slice() {
		ls := ncase.List.Slice()
		if len(ls) == 0 { // default:
			if defCase != nil {
				yyerrorl(ncase.Pos, "multiple defaults in switch (first at %v)", defCase.Line())
			} else {
				defCase = ncase
			}
		}

		for i := range ls {
			ls[i] = typecheck(ls[i], ctxExpr|ctxType)
			n1 := ls[i]
			if t == nil || n1.Type == nil {
				continue
			}

			var missing, have *types.Field
			var ptr int
			switch {
			case n1.isNil(): // case nil:
				if nilCase != nil {
					yyerrorl(ncase.Pos, "multiple nil cases in type switch (first at %v)", nilCase.Line())
				} else {
					nilCase = ncase
				}
			case n1.Op != OTYPE:
				yyerrorl(ncase.Pos, "%L is not a type", n1)
			case !n1.Type.IsInterface() && !implements(n1.Type, t, &missing, &have, &ptr) && !missing.Broke():
				if have != nil && !have.Broke() {
					yyerrorl(ncase.Pos, "impossible type switch case: %L cannot have dynamic type %v"+
						" (wrong type for %v method)\n\thave %v%S\n\twant %v%S", n.Left.Right, n1.Type, missing.Sym, have.Sym, have.Type, missing.Sym, missing.Type)
				} else if ptr != 0 {
					yyerrorl(ncase.Pos, "impossible type switch case: %L cannot have dynamic type %v"+
						" (%v method has pointer receiver)", n.Left.Right, n1.Type, missing.Sym)
				} else {
					yyerrorl(ncase.Pos, "impossible type switch case: %L cannot have dynamic type %v"+
						" (missing %v method)", n.Left.Right, n1.Type, missing.Sym)
				}
			}

			if n1.Op == OTYPE {
				ts.add(ncase.Pos, n1.Type)
			}
		}

		if ncase.Rlist.Len() != 0 {
			// Assign the clause variable's type.
			vt := t
			if len(ls) == 1 {
				if ls[0].Op == OTYPE {
					vt = ls[0].Type
				} else if ls[0].Op != OLITERAL { // TODO(mdempsky): Should be !ls[0].isNil()
					// Invalid single-type case;
					// mark variable as broken.
					vt = nil
				}
			}

			// TODO(mdempsky): It should be possible to
			// still typecheck the case body.
			if vt == nil {
				continue
			}

			nvar := ncase.Rlist.First()
			nvar.Type = vt
			nvar = typecheck(nvar, ctxExpr|ctxAssign)
			ncase.Rlist.SetFirst(nvar)
		}

		typecheckslice(ncase.Nbody.Slice(), ctxStmt)
	}
}

type typeSet struct {
	m map[string][]typeSetEntry
}

type typeSetEntry struct {
	pos src.XPos
	typ *types.Type
}

func (s *typeSet) add(pos src.XPos, typ *types.Type) {
	if s.m == nil {
		s.m = make(map[string][]typeSetEntry)
	}

	// LongString does not uniquely identify types, so we need to
	// disambiguate collisions with types.Identical.
	// TODO(mdempsky): Add a method that *is* unique.
	ls := typ.LongString()
	prevs := s.m[ls]
	for _, prev := range prevs {
		if types.Identical(typ, prev.typ) {
			yyerrorl(pos, "duplicate case %v in type switch\n\tprevious case at %s", typ, linestr(prev.pos))
			return
		}
	}
	s.m[ls] = append(prevs, typeSetEntry{pos, typ})
}

func typecheckExprSwitch(n *Node) {
	t := types.Types[TBOOL]
	if n.Left != nil {
		n.Left = typecheck(n.Left, ctxExpr)
		n.Left = defaultlit(n.Left, nil)
		t = n.Left.Type
	}

	var nilonly string
	if t != nil {
		switch {
		case t.IsMap():
			nilonly = "map"
		case t.Etype == TFUNC:
			nilonly = "func"
		case t.IsSlice():
			nilonly = "slice"

		case !IsComparable(t):
			if t.IsStruct() {
				yyerrorl(n.Pos, "cannot switch on %L (struct containing %v cannot be compared)", n.Left, IncomparableField(t).Type)
			} else {
				yyerrorl(n.Pos, "cannot switch on %L", n.Left)
			}
			t = nil
		}
	}

	var defCase *Node
	var cs constSet
	for _, ncase := range n.List.Slice() {
		ls := ncase.List.Slice()
		if len(ls) == 0 { // default:
			if defCase != nil {
				yyerrorl(ncase.Pos, "multiple defaults in switch (first at %v)", defCase.Line())
			} else {
				defCase = ncase
			}
		}

		for i := range ls {
			setlineno(ncase)
			ls[i] = typecheck(ls[i], ctxExpr)
			ls[i] = defaultlit(ls[i], t)
			n1 := ls[i]
			if t == nil || n1.Type == nil {
				continue
			}

			if nilonly != "" && !n1.isNil() {
				yyerrorl(ncase.Pos, "invalid case %v in switch (can only compare %s %v to nil)", n1, nilonly, n.Left)
			} else if t.IsInterface() && !n1.Type.IsInterface() && !IsComparable(n1.Type) {
				yyerrorl(ncase.Pos, "invalid case %L in switch (incomparable type)", n1)
			} else {
				op1, _ := assignop(n1.Type, t)
				op2, _ := assignop(t, n1.Type)
				if op1 == OXXX && op2 == OXXX {
					if n.Left != nil {
						yyerrorl(ncase.Pos, "invalid case %v in switch on %v (mismatched types %v and %v)", n1, n.Left, n1.Type, t)
					} else {
						yyerrorl(ncase.Pos, "invalid case %v in switch (mismatched types %v and bool)", n1, n1.Type)
					}
				}
			}

			// Don't check for duplicate bools. Although the spec allows it,
			// (1) the compiler hasn't checked it in the past, so compatibility mandates it, and
			// (2) it would disallow useful things like
			//       case GOARCH == "arm" && GOARM == "5":
			//       case GOARCH == "arm":
			//     which would both evaluate to false for non-ARM compiles.
			if !n1.Type.IsBoolean() {
				cs.add(ncase.Pos, n1, "case", "switch")
			}
		}

		typecheckslice(ncase.Nbody.Slice(), ctxStmt)
	}
}

// walkswitch walks a switch statement.
func walkswitch(sw *Node) {
	// Guard against double walk, see #25776.
	if sw.List.Len() == 0 && sw.Nbody.Len() > 0 {
		return // Was fatal, but eliminating every possible source of double-walking is hard
	}

	if sw.Left != nil && sw.Left.Op == OTYPESW {
		walkTypeSwitch(sw)
	} else {
		walkExprSwitch(sw)
	}
}

// walkExprSwitch generates an AST implementing sw.  sw is an
// expression switch.
func walkExprSwitch(sw *Node) {
	lno := setlineno(sw)

	cond := sw.Left
	sw.Left = nil

	// convert switch {...} to switch true {...}
	if cond == nil {
		cond = nodbool(true)
		cond = typecheck(cond, ctxExpr)
		cond = defaultlit(cond, nil)
	}

	// Given "switch string(byteslice)",
	// with all cases being side-effect free,
	// use a zero-cost alias of the byte slice.
	// Do this before calling walkexpr on cond,
	// because walkexpr will lower the string
	// conversion into a runtime call.
	// See issue 24937 for more discussion.
	if cond.Op == OBYTES2STR && allCaseExprsAreSideEffectFree(sw) {
		cond.Op = OBYTES2STRTMP
	}

	cond = walkexpr(cond, &sw.Ninit)
	if cond.Op != OLITERAL {
		cond = copyexpr(cond, cond.Type, &sw.Nbody)
	}

	lineno = lno

	s := exprSwitch{
		exprname: cond,
	}

	var defaultGoto *Node
	var body Nodes
	for _, ncase := range sw.List.Slice() {
		label := autolabel(".s")
		jmp := npos(ncase.Pos, nodSym(OGOTO, nil, label))

		// Process case dispatch.
		if ncase.List.Len() == 0 {
			if defaultGoto != nil {
				Fatalf("duplicate default case not detected during typechecking")
			}
			defaultGoto = jmp
		}

		for _, n1 := range ncase.List.Slice() {
			s.Add(ncase.Pos, n1, jmp)
		}

		// Process body.
		body.Append(npos(ncase.Pos, nodSym(OLABEL, nil, label)))
		body.Append(ncase.Nbody.Slice()...)
		if fall, pos := hasFall(ncase.Nbody.Slice()); !fall {
			br := nod(OBREAK, nil, nil)
			br.Pos = pos
			body.Append(br)
		}
	}
	sw.List.Set(nil)

	if defaultGoto == nil {
		br := nod(OBREAK, nil, nil)
		br.Pos = br.Pos.WithNotStmt()
		defaultGoto = br
	}

	s.Emit(&sw.Nbody)
	sw.Nbody.Append(defaultGoto)
	sw.Nbody.AppendNodes(&body)
	walkstmtlist(sw.Nbody.Slice())
}

// An exprSwitch walks an expression switch.
type exprSwitch struct {
	exprname *Node // value being switched on

	done    Nodes
	clauses []exprClause
}

type exprClause struct {
	pos    src.XPos
	lo, hi *Node
	jmp    *Node
}

func (s *exprSwitch) Add(pos src.XPos, expr, jmp *Node) {
	c := exprClause{pos: pos, lo: expr, hi: expr, jmp: jmp}
	if okforcmp[s.exprname.Type.Etype] && expr.Op == OLITERAL {
		s.clauses = append(s.clauses, c)
		return
	}

	s.flush()
	s.clauses = append(s.clauses, c)
	s.flush()
}

func (s *exprSwitch) Emit(out *Nodes) {
	s.flush()
	out.AppendNodes(&s.done)
}

func (s *exprSwitch) flush() {
	cc := s.clauses
	s.clauses = nil
	if len(cc) == 0 {
		return
	}

	// Caution: If len(cc) == 1, then cc[0] might not an OLITERAL.
	// The code below is structured to implicitly handle this case
	// (e.g., sort.Slice doesn't need to invoke the less function
	// when there's only a single slice element).

	if s.exprname.Type.IsString() && len(cc) >= 2 {
		// Sort strings by length and then by value. It is
		// much cheaper to compare lengths than values, and
		// all we need here is consistency. We respect this
		// sorting below.
		sort.Slice(cc, func(i, j int) bool {
			si := cc[i].lo.StringVal()
			sj := cc[j].lo.StringVal()
			if len(si) != len(sj) {
				return len(si) < len(sj)
			}
			return si < sj
		})

		// runLen returns the string length associated with a
		// particular run of exprClauses.
		runLen := func(run []exprClause) int64 { return int64(len(run[0].lo.StringVal())) }

		// Collapse runs of consecutive strings with the same length.
		var runs [][]exprClause
		start := 0
		for i := 1; i < len(cc); i++ {
			if runLen(cc[start:]) != runLen(cc[i:]) {
				runs = append(runs, cc[start:i])
				start = i
			}
		}
		runs = append(runs, cc[start:])

		// Perform two-level binary search.
		nlen := nod(OLEN, s.exprname, nil)
		binarySearch(len(runs), &s.done,
			func(i int) *Node {
				return nod(OLE, nlen, nodintconst(runLen(runs[i-1])))
			},
			func(i int, nif *Node) {
				run := runs[i]
				nif.Left = nod(OEQ, nlen, nodintconst(runLen(run)))
				s.search(run, &nif.Nbody)
			},
		)
		return
	}

	sort.Slice(cc, func(i, j int) bool {
		return compareOp(cc[i].lo.Val(), OLT, cc[j].lo.Val())
	})

	// Merge consecutive integer cases.
	if s.exprname.Type.IsInteger() {
		merged := cc[:1]
		for _, c := range cc[1:] {
			last := &merged[len(merged)-1]
			if last.jmp == c.jmp && last.hi.Int64Val()+1 == c.lo.Int64Val() {
				last.hi = c.lo
			} else {
				merged = append(merged, c)
			}
		}
		cc = merged
	}

	s.search(cc, &s.done)
}

func (s *exprSwitch) search(cc []exprClause, out *Nodes) {
	binarySearch(len(cc), out,
		func(i int) *Node {
			return nod(OLE, s.exprname, cc[i-1].hi)
		},
		func(i int, nif *Node) {
			c := &cc[i]
			nif.Left = c.test(s.exprname)
			nif.Nbody.Set1(c.jmp)
		},
	)
}

func (c *exprClause) test(exprname *Node) *Node {
	// Integer range.
	if c.hi != c.lo {
		low := nodl(c.pos, OGE, exprname, c.lo)
		high := nodl(c.pos, OLE, exprname, c.hi)
		return nodl(c.pos, OANDAND, low, high)
	}

	// Optimize "switch true { ...}" and "switch false { ... }".
	if Isconst(exprname, CTBOOL) && !c.lo.Type.IsInterface() {
		if exprname.BoolVal() {
			return c.lo
		} else {
			return nodl(c.pos, ONOT, c.lo, nil)
		}
	}

	return nodl(c.pos, OEQ, exprname, c.lo)
}

func allCaseExprsAreSideEffectFree(sw *Node) bool {
	// In theory, we could be more aggressive, allowing any
	// side-effect-free expressions in cases, but it's a bit
	// tricky because some of that information is unavailable due
	// to the introduction of temporaries during order.
	// Restricting to constants is simple and probably powerful
	// enough.

	for _, ncase := range sw.List.Slice() {
		if ncase.Op != OCASE {
			Fatalf("switch string(byteslice) bad op: %v", ncase.Op)
		}
		for _, v := range ncase.List.Slice() {
			if v.Op != OLITERAL {
				return false
			}
		}
	}
	return true
}

// hasFall reports whether stmts ends with a "fallthrough" statement.
func hasFall(stmts []*Node) (bool, src.XPos) {
	// Search backwards for the index of the fallthrough
	// statement. Do not assume it'll be in the last
	// position, since in some cases (e.g. when the statement
	// list contains autotmp_ variables), one or more OVARKILL
	// nodes will be at the end of the list.

	i := len(stmts) - 1
	for i >= 0 && stmts[i].Op == OVARKILL {
		i--
	}
	if i < 0 {
		return false, src.NoXPos
	}
	return stmts[i].Op == OFALL, stmts[i].Pos
}

// walkTypeSwitch generates an AST that implements sw, where sw is a
// type switch.
func walkTypeSwitch(sw *Node) {
	var s typeSwitch
	s.facename = sw.Left.Right
	sw.Left = nil

	s.facename = walkexpr(s.facename, &sw.Ninit)
	s.facename = copyexpr(s.facename, s.facename.Type, &sw.Nbody)
	s.okname = temp(types.Types[TBOOL])

	// Get interface descriptor word.
	// For empty interfaces this will be the type.
	// For non-empty interfaces this will be the itab.
	itab := nod(OITAB, s.facename, nil)

	// For empty interfaces, do:
	//     if e._type == nil {
	//         do nil case if it exists, otherwise default
	//     }
	//     h := e._type.hash
	// Use a similar strategy for non-empty interfaces.
	ifNil := nod(OIF, nil, nil)
	ifNil.Left = nod(OEQ, itab, nodnil())
	lineno = lineno.WithNotStmt() // disable statement marks after the first check.
	ifNil.Left = typecheck(ifNil.Left, ctxExpr)
	ifNil.Left = defaultlit(ifNil.Left, nil)
	// ifNil.Nbody assigned at end.
	sw.Nbody.Append(ifNil)

	// Load hash from type or itab.
	dotHash := nodSym(ODOTPTR, itab, nil)
	dotHash.Type = types.Types[TUINT32]
	dotHash.SetTypecheck(1)
	if s.facename.Type.IsEmptyInterface() {
		dotHash.Xoffset = int64(2 * Widthptr) // offset of hash in runtime._type
	} else {
		dotHash.Xoffset = int64(2 * Widthptr) // offset of hash in runtime.itab
	}
	dotHash.SetBounded(true) // guaranteed not to fault
	s.hashname = copyexpr(dotHash, dotHash.Type, &sw.Nbody)

	br := nod(OBREAK, nil, nil)
	var defaultGoto, nilGoto *Node
	var body Nodes
	for _, ncase := range sw.List.Slice() {
		var caseVar *Node
		if ncase.Rlist.Len() != 0 {
			caseVar = ncase.Rlist.First()
		}

		// For single-type cases with an interface type,
		// we initialize the case variable as part of the type assertion.
		// In other cases, we initialize it in the body.
		var singleType *types.Type
		if ncase.List.Len() == 1 && ncase.List.First().Op == OTYPE {
			singleType = ncase.List.First().Type
		}
		caseVarInitialized := false

		label := autolabel(".s")
		jmp := npos(ncase.Pos, nodSym(OGOTO, nil, label))

		if ncase.List.Len() == 0 { // default:
			if defaultGoto != nil {
				Fatalf("duplicate default case not detected during typechecking")
			}
			defaultGoto = jmp
		}

		for _, n1 := range ncase.List.Slice() {
			if n1.isNil() { // case nil:
				if nilGoto != nil {
					Fatalf("duplicate nil case not detected during typechecking")
				}
				nilGoto = jmp
				continue
			}

			if singleType != nil && singleType.IsInterface() {
				s.Add(ncase.Pos, n1.Type, caseVar, jmp)
				caseVarInitialized = true
			} else {
				s.Add(ncase.Pos, n1.Type, nil, jmp)
			}
		}

		body.Append(npos(ncase.Pos, nodSym(OLABEL, nil, label)))
		if caseVar != nil && !caseVarInitialized {
			val := s.facename
			if singleType != nil {
				// We have a single concrete type. Extract the data.
				if singleType.IsInterface() {
					Fatalf("singleType interface should have been handled in Add")
				}
				val = ifaceData(ncase.Pos, s.facename, singleType)
			}
			l := []*Node{
				nodl(ncase.Pos, ODCL, caseVar, nil),
				nodl(ncase.Pos, OAS, caseVar, val),
			}
			typecheckslice(l, ctxStmt)
			body.Append(l...)
		}
		body.Append(ncase.Nbody.Slice()...)
		body.Append(br)
	}
	sw.List.Set(nil)

	if defaultGoto == nil {
		defaultGoto = br
	}
	if nilGoto == nil {
		nilGoto = defaultGoto
	}
	ifNil.Nbody.Set1(nilGoto)

	s.Emit(&sw.Nbody)
	sw.Nbody.Append(defaultGoto)
	sw.Nbody.AppendNodes(&body)

	walkstmtlist(sw.Nbody.Slice())
}

// A typeSwitch walks a type switch.
type typeSwitch struct {
	// Temporary variables (i.e., ONAMEs) used by type switch dispatch logic:
	facename *Node // value being type-switched on
	hashname *Node // type hash of the value being type-switched on
	okname   *Node // boolean used for comma-ok type assertions

	done    Nodes
	clauses []typeClause
}

type typeClause struct {
	hash uint32
	body Nodes
}

func (s *typeSwitch) Add(pos src.XPos, typ *types.Type, caseVar, jmp *Node) {
	var body Nodes
	if caseVar != nil {
		l := []*Node{
			nodl(pos, ODCL, caseVar, nil),
			nodl(pos, OAS, caseVar, nil),
		}
		typecheckslice(l, ctxStmt)
		body.Append(l...)
	} else {
		caseVar = nblank
	}

	// cv, ok = iface.(type)
	as := nodl(pos, OAS2, nil, nil)
	as.List.Set2(caseVar, s.okname) // cv, ok =
	dot := nodl(pos, ODOTTYPE, s.facename, nil)
	dot.Type = typ // iface.(type)
	as.Rlist.Set1(dot)
	as = typecheck(as, ctxStmt)
	as = walkexpr(as, &body)
	body.Append(as)

	// if ok { goto label }
	nif := nodl(pos, OIF, nil, nil)
	nif.Left = s.okname
	nif.Nbody.Set1(jmp)
	body.Append(nif)

	if !typ.IsInterface() {
		s.clauses = append(s.clauses, typeClause{
			hash: typehash(typ),
			body: body,
		})
		return
	}

	s.flush()
	s.done.AppendNodes(&body)
}

func (s *typeSwitch) Emit(out *Nodes) {
	s.flush()
	out.AppendNodes(&s.done)
}

func (s *typeSwitch) flush() {
	cc := s.clauses
	s.clauses = nil
	if len(cc) == 0 {
		return
	}

	sort.Slice(cc, func(i, j int) bool { return cc[i].hash < cc[j].hash })

	// Combine adjacent cases with the same hash.
	merged := cc[:1]
	for _, c := range cc[1:] {
		last := &merged[len(merged)-1]
		if last.hash == c.hash {
			last.body.AppendNodes(&c.body)
		} else {
			merged = append(merged, c)
		}
	}
	cc = merged

	binarySearch(len(cc), &s.done,
		func(i int) *Node {
			return nod(OLE, s.hashname, nodintconst(int64(cc[i-1].hash)))
		},
		func(i int, nif *Node) {
			// TODO(mdempsky): Omit hash equality check if
			// there's only one type.
			c := cc[i]
			nif.Left = nod(OEQ, s.hashname, nodintconst(int64(c.hash)))
			nif.Nbody.AppendNodes(&c.body)
		},
	)
}

// binarySearch constructs a binary search tree for handling n cases,
// and appends it to out. It's used for efficiently implementing
// switch statements.
//
// less(i) should return a boolean expression. If it evaluates true,
// then cases before i will be tested; otherwise, cases i and later.
//
// base(i, nif) should setup nif (an OIF node) to test case i. In
// particular, it should set nif.Left and nif.Nbody.
func binarySearch(n int, out *Nodes, less func(i int) *Node, base func(i int, nif *Node)) {
	const binarySearchMin = 4 // minimum number of cases for binary search

	var do func(lo, hi int, out *Nodes)
	do = func(lo, hi int, out *Nodes) {
		n := hi - lo
		if n < binarySearchMin {
			for i := lo; i < hi; i++ {
				nif := nod(OIF, nil, nil)
				base(i, nif)
				lineno = lineno.WithNotStmt()
				nif.Left = typecheck(nif.Left, ctxExpr)
				nif.Left = defaultlit(nif.Left, nil)
				out.Append(nif)
				out = &nif.Rlist
			}
			return
		}

		half := lo + n/2
		nif := nod(OIF, nil, nil)
		nif.Left = less(half)
		lineno = lineno.WithNotStmt()
		nif.Left = typecheck(nif.Left, ctxExpr)
		nif.Left = defaultlit(nif.Left, nil)
		do(lo, half, &nif.Nbody)
		do(half, hi, &nif.Rlist)
		out.Append(nif)
	}

	do(0, n, out)
}
