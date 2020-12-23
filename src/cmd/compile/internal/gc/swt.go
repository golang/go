// Copyright 2009 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package gc

import (
	"cmd/compile/internal/base"
	"cmd/compile/internal/ir"
	"cmd/compile/internal/types"
	"cmd/internal/src"
	"go/constant"
	"go/token"
	"sort"
)

// typecheckswitch typechecks a switch statement.
func typecheckswitch(n *ir.SwitchStmt) {
	typecheckslice(n.Init(), ctxStmt)
	if n.Tag != nil && n.Tag.Op() == ir.OTYPESW {
		typecheckTypeSwitch(n)
	} else {
		typecheckExprSwitch(n)
	}
}

func typecheckTypeSwitch(n *ir.SwitchStmt) {
	guard := n.Tag.(*ir.TypeSwitchGuard)
	guard.X = typecheck(guard.X, ctxExpr)
	t := guard.X.Type()
	if t != nil && !t.IsInterface() {
		base.ErrorfAt(n.Pos(), "cannot type switch on non-interface value %L", guard.X)
		t = nil
	}

	// We don't actually declare the type switch's guarded
	// declaration itself. So if there are no cases, we won't
	// notice that it went unused.
	if v := guard.Tag; v != nil && !ir.IsBlank(v) && len(n.Cases) == 0 {
		base.ErrorfAt(v.Pos(), "%v declared but not used", v.Sym())
	}

	var defCase, nilCase ir.Node
	var ts typeSet
	for _, ncase := range n.Cases {
		ncase := ncase.(*ir.CaseStmt)
		ls := ncase.List
		if len(ls) == 0 { // default:
			if defCase != nil {
				base.ErrorfAt(ncase.Pos(), "multiple defaults in switch (first at %v)", ir.Line(defCase))
			} else {
				defCase = ncase
			}
		}

		for i := range ls {
			ls[i] = typecheck(ls[i], ctxExpr|ctxType)
			n1 := ls[i]
			if t == nil || n1.Type() == nil {
				continue
			}

			var missing, have *types.Field
			var ptr int
			if ir.IsNil(n1) { // case nil:
				if nilCase != nil {
					base.ErrorfAt(ncase.Pos(), "multiple nil cases in type switch (first at %v)", ir.Line(nilCase))
				} else {
					nilCase = ncase
				}
				continue
			}
			if n1.Op() != ir.OTYPE {
				base.ErrorfAt(ncase.Pos(), "%L is not a type", n1)
				continue
			}
			if !n1.Type().IsInterface() && !implements(n1.Type(), t, &missing, &have, &ptr) && !missing.Broke() {
				if have != nil && !have.Broke() {
					base.ErrorfAt(ncase.Pos(), "impossible type switch case: %L cannot have dynamic type %v"+
						" (wrong type for %v method)\n\thave %v%S\n\twant %v%S", guard.X, n1.Type(), missing.Sym, have.Sym, have.Type, missing.Sym, missing.Type)
				} else if ptr != 0 {
					base.ErrorfAt(ncase.Pos(), "impossible type switch case: %L cannot have dynamic type %v"+
						" (%v method has pointer receiver)", guard.X, n1.Type(), missing.Sym)
				} else {
					base.ErrorfAt(ncase.Pos(), "impossible type switch case: %L cannot have dynamic type %v"+
						" (missing %v method)", guard.X, n1.Type(), missing.Sym)
				}
				continue
			}

			ts.add(ncase.Pos(), n1.Type())
		}

		if len(ncase.Vars) != 0 {
			// Assign the clause variable's type.
			vt := t
			if len(ls) == 1 {
				if ls[0].Op() == ir.OTYPE {
					vt = ls[0].Type()
				} else if !ir.IsNil(ls[0]) {
					// Invalid single-type case;
					// mark variable as broken.
					vt = nil
				}
			}

			nvar := ncase.Vars[0]
			nvar.SetType(vt)
			if vt != nil {
				nvar = typecheck(nvar, ctxExpr|ctxAssign)
			} else {
				// Clause variable is broken; prevent typechecking.
				nvar.SetTypecheck(1)
				nvar.SetWalkdef(1)
			}
			ncase.Vars[0] = nvar
		}

		typecheckslice(ncase.Body, ctxStmt)
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
			base.ErrorfAt(pos, "duplicate case %v in type switch\n\tprevious case at %s", typ, base.FmtPos(prev.pos))
			return
		}
	}
	s.m[ls] = append(prevs, typeSetEntry{pos, typ})
}

func typecheckExprSwitch(n *ir.SwitchStmt) {
	t := types.Types[types.TBOOL]
	if n.Tag != nil {
		n.Tag = typecheck(n.Tag, ctxExpr)
		n.Tag = defaultlit(n.Tag, nil)
		t = n.Tag.Type()
	}

	var nilonly string
	if t != nil {
		switch {
		case t.IsMap():
			nilonly = "map"
		case t.Kind() == types.TFUNC:
			nilonly = "func"
		case t.IsSlice():
			nilonly = "slice"

		case !types.IsComparable(t):
			if t.IsStruct() {
				base.ErrorfAt(n.Pos(), "cannot switch on %L (struct containing %v cannot be compared)", n.Tag, types.IncomparableField(t).Type)
			} else {
				base.ErrorfAt(n.Pos(), "cannot switch on %L", n.Tag)
			}
			t = nil
		}
	}

	var defCase ir.Node
	var cs constSet
	for _, ncase := range n.Cases {
		ncase := ncase.(*ir.CaseStmt)
		ls := ncase.List
		if len(ls) == 0 { // default:
			if defCase != nil {
				base.ErrorfAt(ncase.Pos(), "multiple defaults in switch (first at %v)", ir.Line(defCase))
			} else {
				defCase = ncase
			}
		}

		for i := range ls {
			ir.SetPos(ncase)
			ls[i] = typecheck(ls[i], ctxExpr)
			ls[i] = defaultlit(ls[i], t)
			n1 := ls[i]
			if t == nil || n1.Type() == nil {
				continue
			}

			if nilonly != "" && !ir.IsNil(n1) {
				base.ErrorfAt(ncase.Pos(), "invalid case %v in switch (can only compare %s %v to nil)", n1, nilonly, n.Tag)
			} else if t.IsInterface() && !n1.Type().IsInterface() && !types.IsComparable(n1.Type()) {
				base.ErrorfAt(ncase.Pos(), "invalid case %L in switch (incomparable type)", n1)
			} else {
				op1, _ := assignop(n1.Type(), t)
				op2, _ := assignop(t, n1.Type())
				if op1 == ir.OXXX && op2 == ir.OXXX {
					if n.Tag != nil {
						base.ErrorfAt(ncase.Pos(), "invalid case %v in switch on %v (mismatched types %v and %v)", n1, n.Tag, n1.Type(), t)
					} else {
						base.ErrorfAt(ncase.Pos(), "invalid case %v in switch (mismatched types %v and bool)", n1, n1.Type())
					}
				}
			}

			// Don't check for duplicate bools. Although the spec allows it,
			// (1) the compiler hasn't checked it in the past, so compatibility mandates it, and
			// (2) it would disallow useful things like
			//       case GOARCH == "arm" && GOARM == "5":
			//       case GOARCH == "arm":
			//     which would both evaluate to false for non-ARM compiles.
			if !n1.Type().IsBoolean() {
				cs.add(ncase.Pos(), n1, "case", "switch")
			}
		}

		typecheckslice(ncase.Body, ctxStmt)
	}
}

// walkswitch walks a switch statement.
func walkswitch(sw *ir.SwitchStmt) {
	// Guard against double walk, see #25776.
	if len(sw.Cases) == 0 && len(sw.Compiled) > 0 {
		return // Was fatal, but eliminating every possible source of double-walking is hard
	}

	if sw.Tag != nil && sw.Tag.Op() == ir.OTYPESW {
		walkTypeSwitch(sw)
	} else {
		walkExprSwitch(sw)
	}
}

// walkExprSwitch generates an AST implementing sw.  sw is an
// expression switch.
func walkExprSwitch(sw *ir.SwitchStmt) {
	lno := ir.SetPos(sw)

	cond := sw.Tag
	sw.Tag = nil

	// convert switch {...} to switch true {...}
	if cond == nil {
		cond = ir.NewBool(true)
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
	if cond.Op() == ir.OBYTES2STR && allCaseExprsAreSideEffectFree(sw) {
		cond := cond.(*ir.ConvExpr)
		cond.SetOp(ir.OBYTES2STRTMP)
	}

	cond = walkexpr(cond, sw.PtrInit())
	if cond.Op() != ir.OLITERAL && cond.Op() != ir.ONIL {
		cond = copyexpr(cond, cond.Type(), &sw.Compiled)
	}

	base.Pos = lno

	s := exprSwitch{
		exprname: cond,
	}

	var defaultGoto ir.Node
	var body ir.Nodes
	for _, ncase := range sw.Cases {
		ncase := ncase.(*ir.CaseStmt)
		label := autolabel(".s")
		jmp := ir.NewBranchStmt(ncase.Pos(), ir.OGOTO, label)

		// Process case dispatch.
		if len(ncase.List) == 0 {
			if defaultGoto != nil {
				base.Fatalf("duplicate default case not detected during typechecking")
			}
			defaultGoto = jmp
		}

		for _, n1 := range ncase.List {
			s.Add(ncase.Pos(), n1, jmp)
		}

		// Process body.
		body.Append(ir.NewLabelStmt(ncase.Pos(), label))
		body.Append(ncase.Body...)
		if fall, pos := endsInFallthrough(ncase.Body); !fall {
			br := ir.NewBranchStmt(base.Pos, ir.OBREAK, nil)
			br.SetPos(pos)
			body.Append(br)
		}
	}
	sw.Cases.Set(nil)

	if defaultGoto == nil {
		br := ir.NewBranchStmt(base.Pos, ir.OBREAK, nil)
		br.SetPos(br.Pos().WithNotStmt())
		defaultGoto = br
	}

	s.Emit(&sw.Compiled)
	sw.Compiled.Append(defaultGoto)
	sw.Compiled.Append(body.Take()...)
	walkstmtlist(sw.Compiled)
}

// An exprSwitch walks an expression switch.
type exprSwitch struct {
	exprname ir.Node // value being switched on

	done    ir.Nodes
	clauses []exprClause
}

type exprClause struct {
	pos    src.XPos
	lo, hi ir.Node
	jmp    ir.Node
}

func (s *exprSwitch) Add(pos src.XPos, expr, jmp ir.Node) {
	c := exprClause{pos: pos, lo: expr, hi: expr, jmp: jmp}
	if types.IsOrdered[s.exprname.Type().Kind()] && expr.Op() == ir.OLITERAL {
		s.clauses = append(s.clauses, c)
		return
	}

	s.flush()
	s.clauses = append(s.clauses, c)
	s.flush()
}

func (s *exprSwitch) Emit(out *ir.Nodes) {
	s.flush()
	out.Append(s.done.Take()...)
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

	if s.exprname.Type().IsString() && len(cc) >= 2 {
		// Sort strings by length and then by value. It is
		// much cheaper to compare lengths than values, and
		// all we need here is consistency. We respect this
		// sorting below.
		sort.Slice(cc, func(i, j int) bool {
			si := ir.StringVal(cc[i].lo)
			sj := ir.StringVal(cc[j].lo)
			if len(si) != len(sj) {
				return len(si) < len(sj)
			}
			return si < sj
		})

		// runLen returns the string length associated with a
		// particular run of exprClauses.
		runLen := func(run []exprClause) int64 { return int64(len(ir.StringVal(run[0].lo))) }

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
		binarySearch(len(runs), &s.done,
			func(i int) ir.Node {
				return ir.NewBinaryExpr(base.Pos, ir.OLE, ir.NewUnaryExpr(base.Pos, ir.OLEN, s.exprname), ir.NewInt(runLen(runs[i-1])))
			},
			func(i int, nif *ir.IfStmt) {
				run := runs[i]
				nif.Cond = ir.NewBinaryExpr(base.Pos, ir.OEQ, ir.NewUnaryExpr(base.Pos, ir.OLEN, s.exprname), ir.NewInt(runLen(run)))
				s.search(run, &nif.Body)
			},
		)
		return
	}

	sort.Slice(cc, func(i, j int) bool {
		return constant.Compare(cc[i].lo.Val(), token.LSS, cc[j].lo.Val())
	})

	// Merge consecutive integer cases.
	if s.exprname.Type().IsInteger() {
		merged := cc[:1]
		for _, c := range cc[1:] {
			last := &merged[len(merged)-1]
			if last.jmp == c.jmp && ir.Int64Val(last.hi)+1 == ir.Int64Val(c.lo) {
				last.hi = c.lo
			} else {
				merged = append(merged, c)
			}
		}
		cc = merged
	}

	s.search(cc, &s.done)
}

func (s *exprSwitch) search(cc []exprClause, out *ir.Nodes) {
	binarySearch(len(cc), out,
		func(i int) ir.Node {
			return ir.NewBinaryExpr(base.Pos, ir.OLE, s.exprname, cc[i-1].hi)
		},
		func(i int, nif *ir.IfStmt) {
			c := &cc[i]
			nif.Cond = c.test(s.exprname)
			nif.Body = []ir.Node{c.jmp}
		},
	)
}

func (c *exprClause) test(exprname ir.Node) ir.Node {
	// Integer range.
	if c.hi != c.lo {
		low := ir.NewBinaryExpr(c.pos, ir.OGE, exprname, c.lo)
		high := ir.NewBinaryExpr(c.pos, ir.OLE, exprname, c.hi)
		return ir.NewLogicalExpr(c.pos, ir.OANDAND, low, high)
	}

	// Optimize "switch true { ...}" and "switch false { ... }".
	if ir.IsConst(exprname, constant.Bool) && !c.lo.Type().IsInterface() {
		if ir.BoolVal(exprname) {
			return c.lo
		} else {
			return ir.NewUnaryExpr(c.pos, ir.ONOT, c.lo)
		}
	}

	return ir.NewBinaryExpr(c.pos, ir.OEQ, exprname, c.lo)
}

func allCaseExprsAreSideEffectFree(sw *ir.SwitchStmt) bool {
	// In theory, we could be more aggressive, allowing any
	// side-effect-free expressions in cases, but it's a bit
	// tricky because some of that information is unavailable due
	// to the introduction of temporaries during order.
	// Restricting to constants is simple and probably powerful
	// enough.

	for _, ncase := range sw.Cases {
		ncase := ncase.(*ir.CaseStmt)
		for _, v := range ncase.List {
			if v.Op() != ir.OLITERAL {
				return false
			}
		}
	}
	return true
}

// endsInFallthrough reports whether stmts ends with a "fallthrough" statement.
func endsInFallthrough(stmts []ir.Node) (bool, src.XPos) {
	// Search backwards for the index of the fallthrough
	// statement. Do not assume it'll be in the last
	// position, since in some cases (e.g. when the statement
	// list contains autotmp_ variables), one or more OVARKILL
	// nodes will be at the end of the list.

	i := len(stmts) - 1
	for i >= 0 && stmts[i].Op() == ir.OVARKILL {
		i--
	}
	if i < 0 {
		return false, src.NoXPos
	}
	return stmts[i].Op() == ir.OFALL, stmts[i].Pos()
}

// walkTypeSwitch generates an AST that implements sw, where sw is a
// type switch.
func walkTypeSwitch(sw *ir.SwitchStmt) {
	var s typeSwitch
	s.facename = sw.Tag.(*ir.TypeSwitchGuard).X
	sw.Tag = nil

	s.facename = walkexpr(s.facename, sw.PtrInit())
	s.facename = copyexpr(s.facename, s.facename.Type(), &sw.Compiled)
	s.okname = temp(types.Types[types.TBOOL])

	// Get interface descriptor word.
	// For empty interfaces this will be the type.
	// For non-empty interfaces this will be the itab.
	itab := ir.NewUnaryExpr(base.Pos, ir.OITAB, s.facename)

	// For empty interfaces, do:
	//     if e._type == nil {
	//         do nil case if it exists, otherwise default
	//     }
	//     h := e._type.hash
	// Use a similar strategy for non-empty interfaces.
	ifNil := ir.NewIfStmt(base.Pos, nil, nil, nil)
	ifNil.Cond = ir.NewBinaryExpr(base.Pos, ir.OEQ, itab, nodnil())
	base.Pos = base.Pos.WithNotStmt() // disable statement marks after the first check.
	ifNil.Cond = typecheck(ifNil.Cond, ctxExpr)
	ifNil.Cond = defaultlit(ifNil.Cond, nil)
	// ifNil.Nbody assigned at end.
	sw.Compiled.Append(ifNil)

	// Load hash from type or itab.
	dotHash := ir.NewSelectorExpr(base.Pos, ir.ODOTPTR, itab, nil)
	dotHash.SetType(types.Types[types.TUINT32])
	dotHash.SetTypecheck(1)
	if s.facename.Type().IsEmptyInterface() {
		dotHash.Offset = int64(2 * Widthptr) // offset of hash in runtime._type
	} else {
		dotHash.Offset = int64(2 * Widthptr) // offset of hash in runtime.itab
	}
	dotHash.SetBounded(true) // guaranteed not to fault
	s.hashname = copyexpr(dotHash, dotHash.Type(), &sw.Compiled)

	br := ir.NewBranchStmt(base.Pos, ir.OBREAK, nil)
	var defaultGoto, nilGoto ir.Node
	var body ir.Nodes
	for _, ncase := range sw.Cases {
		ncase := ncase.(*ir.CaseStmt)
		var caseVar ir.Node
		if len(ncase.Vars) != 0 {
			caseVar = ncase.Vars[0]
		}

		// For single-type cases with an interface type,
		// we initialize the case variable as part of the type assertion.
		// In other cases, we initialize it in the body.
		var singleType *types.Type
		if len(ncase.List) == 1 && ncase.List[0].Op() == ir.OTYPE {
			singleType = ncase.List[0].Type()
		}
		caseVarInitialized := false

		label := autolabel(".s")
		jmp := ir.NewBranchStmt(ncase.Pos(), ir.OGOTO, label)

		if len(ncase.List) == 0 { // default:
			if defaultGoto != nil {
				base.Fatalf("duplicate default case not detected during typechecking")
			}
			defaultGoto = jmp
		}

		for _, n1 := range ncase.List {
			if ir.IsNil(n1) { // case nil:
				if nilGoto != nil {
					base.Fatalf("duplicate nil case not detected during typechecking")
				}
				nilGoto = jmp
				continue
			}

			if singleType != nil && singleType.IsInterface() {
				s.Add(ncase.Pos(), n1.Type(), caseVar, jmp)
				caseVarInitialized = true
			} else {
				s.Add(ncase.Pos(), n1.Type(), nil, jmp)
			}
		}

		body.Append(ir.NewLabelStmt(ncase.Pos(), label))
		if caseVar != nil && !caseVarInitialized {
			val := s.facename
			if singleType != nil {
				// We have a single concrete type. Extract the data.
				if singleType.IsInterface() {
					base.Fatalf("singleType interface should have been handled in Add")
				}
				val = ifaceData(ncase.Pos(), s.facename, singleType)
			}
			l := []ir.Node{
				ir.NewDecl(ncase.Pos(), ir.ODCL, caseVar),
				ir.NewAssignStmt(ncase.Pos(), caseVar, val),
			}
			typecheckslice(l, ctxStmt)
			body.Append(l...)
		}
		body.Append(ncase.Body...)
		body.Append(br)
	}
	sw.Cases.Set(nil)

	if defaultGoto == nil {
		defaultGoto = br
	}
	if nilGoto == nil {
		nilGoto = defaultGoto
	}
	ifNil.Body = []ir.Node{nilGoto}

	s.Emit(&sw.Compiled)
	sw.Compiled.Append(defaultGoto)
	sw.Compiled.Append(body.Take()...)

	walkstmtlist(sw.Compiled)
}

// A typeSwitch walks a type switch.
type typeSwitch struct {
	// Temporary variables (i.e., ONAMEs) used by type switch dispatch logic:
	facename ir.Node // value being type-switched on
	hashname ir.Node // type hash of the value being type-switched on
	okname   ir.Node // boolean used for comma-ok type assertions

	done    ir.Nodes
	clauses []typeClause
}

type typeClause struct {
	hash uint32
	body ir.Nodes
}

func (s *typeSwitch) Add(pos src.XPos, typ *types.Type, caseVar, jmp ir.Node) {
	var body ir.Nodes
	if caseVar != nil {
		l := []ir.Node{
			ir.NewDecl(pos, ir.ODCL, caseVar),
			ir.NewAssignStmt(pos, caseVar, nil),
		}
		typecheckslice(l, ctxStmt)
		body.Append(l...)
	} else {
		caseVar = ir.BlankNode
	}

	// cv, ok = iface.(type)
	as := ir.NewAssignListStmt(pos, ir.OAS2, nil, nil)
	as.Lhs = []ir.Node{caseVar, s.okname} // cv, ok =
	dot := ir.NewTypeAssertExpr(pos, s.facename, nil)
	dot.SetType(typ) // iface.(type)
	as.Rhs = []ir.Node{dot}
	appendWalkStmt(&body, as)

	// if ok { goto label }
	nif := ir.NewIfStmt(pos, nil, nil, nil)
	nif.Cond = s.okname
	nif.Body = []ir.Node{jmp}
	body.Append(nif)

	if !typ.IsInterface() {
		s.clauses = append(s.clauses, typeClause{
			hash: types.TypeHash(typ),
			body: body,
		})
		return
	}

	s.flush()
	s.done.Append(body.Take()...)
}

func (s *typeSwitch) Emit(out *ir.Nodes) {
	s.flush()
	out.Append(s.done.Take()...)
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
			last.body.Append(c.body.Take()...)
		} else {
			merged = append(merged, c)
		}
	}
	cc = merged

	binarySearch(len(cc), &s.done,
		func(i int) ir.Node {
			return ir.NewBinaryExpr(base.Pos, ir.OLE, s.hashname, ir.NewInt(int64(cc[i-1].hash)))
		},
		func(i int, nif *ir.IfStmt) {
			// TODO(mdempsky): Omit hash equality check if
			// there's only one type.
			c := cc[i]
			nif.Cond = ir.NewBinaryExpr(base.Pos, ir.OEQ, s.hashname, ir.NewInt(int64(c.hash)))
			nif.Body.Append(c.body.Take()...)
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
// leaf(i, nif) should setup nif (an OIF node) to test case i. In
// particular, it should set nif.Left and nif.Nbody.
func binarySearch(n int, out *ir.Nodes, less func(i int) ir.Node, leaf func(i int, nif *ir.IfStmt)) {
	const binarySearchMin = 4 // minimum number of cases for binary search

	var do func(lo, hi int, out *ir.Nodes)
	do = func(lo, hi int, out *ir.Nodes) {
		n := hi - lo
		if n < binarySearchMin {
			for i := lo; i < hi; i++ {
				nif := ir.NewIfStmt(base.Pos, nil, nil, nil)
				leaf(i, nif)
				base.Pos = base.Pos.WithNotStmt()
				nif.Cond = typecheck(nif.Cond, ctxExpr)
				nif.Cond = defaultlit(nif.Cond, nil)
				out.Append(nif)
				out = &nif.Else
			}
			return
		}

		half := lo + n/2
		nif := ir.NewIfStmt(base.Pos, nil, nil, nil)
		nif.Cond = less(half)
		base.Pos = base.Pos.WithNotStmt()
		nif.Cond = typecheck(nif.Cond, ctxExpr)
		nif.Cond = defaultlit(nif.Cond, nil)
		do(lo, half, &nif.Body)
		do(half, hi, &nif.Else)
		out.Append(nif)
	}

	do(0, n, out)
}
