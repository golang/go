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
func typecheckswitch(n *ir.Node) {
	typecheckslice(n.Init().Slice(), ctxStmt)
	if n.Left() != nil && n.Left().Op() == ir.OTYPESW {
		typecheckTypeSwitch(n)
	} else {
		typecheckExprSwitch(n)
	}
}

func typecheckTypeSwitch(n *ir.Node) {
	n.Left().SetRight(typecheck(n.Left().Right(), ctxExpr))
	t := n.Left().Right().Type()
	if t != nil && !t.IsInterface() {
		base.ErrorfAt(n.Pos(), "cannot type switch on non-interface value %L", n.Left().Right())
		t = nil
	}

	// We don't actually declare the type switch's guarded
	// declaration itself. So if there are no cases, we won't
	// notice that it went unused.
	if v := n.Left().Left(); v != nil && !ir.IsBlank(v) && n.List().Len() == 0 {
		base.ErrorfAt(v.Pos(), "%v declared but not used", v.Sym())
	}

	var defCase, nilCase *ir.Node
	var ts typeSet
	for _, ncase := range n.List().Slice() {
		ls := ncase.List().Slice()
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
			switch {
			case ir.IsNil(n1): // case nil:
				if nilCase != nil {
					base.ErrorfAt(ncase.Pos(), "multiple nil cases in type switch (first at %v)", ir.Line(nilCase))
				} else {
					nilCase = ncase
				}
			case n1.Op() != ir.OTYPE:
				base.ErrorfAt(ncase.Pos(), "%L is not a type", n1)
			case !n1.Type().IsInterface() && !implements(n1.Type(), t, &missing, &have, &ptr) && !missing.Broke():
				if have != nil && !have.Broke() {
					base.ErrorfAt(ncase.Pos(), "impossible type switch case: %L cannot have dynamic type %v"+
						" (wrong type for %v method)\n\thave %v%S\n\twant %v%S", n.Left().Right(), n1.Type(), missing.Sym, have.Sym, have.Type, missing.Sym, missing.Type)
				} else if ptr != 0 {
					base.ErrorfAt(ncase.Pos(), "impossible type switch case: %L cannot have dynamic type %v"+
						" (%v method has pointer receiver)", n.Left().Right(), n1.Type(), missing.Sym)
				} else {
					base.ErrorfAt(ncase.Pos(), "impossible type switch case: %L cannot have dynamic type %v"+
						" (missing %v method)", n.Left().Right(), n1.Type(), missing.Sym)
				}
			}

			if n1.Op() == ir.OTYPE {
				ts.add(ncase.Pos(), n1.Type())
			}
		}

		if ncase.Rlist().Len() != 0 {
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

			nvar := ncase.Rlist().First()
			nvar.SetType(vt)
			if vt != nil {
				nvar = typecheck(nvar, ctxExpr|ctxAssign)
			} else {
				// Clause variable is broken; prevent typechecking.
				nvar.SetTypecheck(1)
				nvar.SetWalkdef(1)
			}
			ncase.Rlist().SetFirst(nvar)
		}

		typecheckslice(ncase.Body().Slice(), ctxStmt)
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

func typecheckExprSwitch(n *ir.Node) {
	t := types.Types[types.TBOOL]
	if n.Left() != nil {
		n.SetLeft(typecheck(n.Left(), ctxExpr))
		n.SetLeft(defaultlit(n.Left(), nil))
		t = n.Left().Type()
	}

	var nilonly string
	if t != nil {
		switch {
		case t.IsMap():
			nilonly = "map"
		case t.Etype == types.TFUNC:
			nilonly = "func"
		case t.IsSlice():
			nilonly = "slice"

		case !IsComparable(t):
			if t.IsStruct() {
				base.ErrorfAt(n.Pos(), "cannot switch on %L (struct containing %v cannot be compared)", n.Left(), IncomparableField(t).Type)
			} else {
				base.ErrorfAt(n.Pos(), "cannot switch on %L", n.Left())
			}
			t = nil
		}
	}

	var defCase *ir.Node
	var cs constSet
	for _, ncase := range n.List().Slice() {
		ls := ncase.List().Slice()
		if len(ls) == 0 { // default:
			if defCase != nil {
				base.ErrorfAt(ncase.Pos(), "multiple defaults in switch (first at %v)", ir.Line(defCase))
			} else {
				defCase = ncase
			}
		}

		for i := range ls {
			setlineno(ncase)
			ls[i] = typecheck(ls[i], ctxExpr)
			ls[i] = defaultlit(ls[i], t)
			n1 := ls[i]
			if t == nil || n1.Type() == nil {
				continue
			}

			if nilonly != "" && !ir.IsNil(n1) {
				base.ErrorfAt(ncase.Pos(), "invalid case %v in switch (can only compare %s %v to nil)", n1, nilonly, n.Left())
			} else if t.IsInterface() && !n1.Type().IsInterface() && !IsComparable(n1.Type()) {
				base.ErrorfAt(ncase.Pos(), "invalid case %L in switch (incomparable type)", n1)
			} else {
				op1, _ := assignop(n1.Type(), t)
				op2, _ := assignop(t, n1.Type())
				if op1 == ir.OXXX && op2 == ir.OXXX {
					if n.Left() != nil {
						base.ErrorfAt(ncase.Pos(), "invalid case %v in switch on %v (mismatched types %v and %v)", n1, n.Left(), n1.Type(), t)
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

		typecheckslice(ncase.Body().Slice(), ctxStmt)
	}
}

// walkswitch walks a switch statement.
func walkswitch(sw *ir.Node) {
	// Guard against double walk, see #25776.
	if sw.List().Len() == 0 && sw.Body().Len() > 0 {
		return // Was fatal, but eliminating every possible source of double-walking is hard
	}

	if sw.Left() != nil && sw.Left().Op() == ir.OTYPESW {
		walkTypeSwitch(sw)
	} else {
		walkExprSwitch(sw)
	}
}

// walkExprSwitch generates an AST implementing sw.  sw is an
// expression switch.
func walkExprSwitch(sw *ir.Node) {
	lno := setlineno(sw)

	cond := sw.Left()
	sw.SetLeft(nil)

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
	if cond.Op() == ir.OBYTES2STR && allCaseExprsAreSideEffectFree(sw) {
		cond.SetOp(ir.OBYTES2STRTMP)
	}

	cond = walkexpr(cond, sw.PtrInit())
	if cond.Op() != ir.OLITERAL && cond.Op() != ir.ONIL {
		cond = copyexpr(cond, cond.Type(), sw.PtrBody())
	}

	base.Pos = lno

	s := exprSwitch{
		exprname: cond,
	}

	var defaultGoto *ir.Node
	var body ir.Nodes
	for _, ncase := range sw.List().Slice() {
		label := autolabel(".s")
		jmp := npos(ncase.Pos(), nodSym(ir.OGOTO, nil, label))

		// Process case dispatch.
		if ncase.List().Len() == 0 {
			if defaultGoto != nil {
				base.Fatalf("duplicate default case not detected during typechecking")
			}
			defaultGoto = jmp
		}

		for _, n1 := range ncase.List().Slice() {
			s.Add(ncase.Pos(), n1, jmp)
		}

		// Process body.
		body.Append(npos(ncase.Pos(), nodSym(ir.OLABEL, nil, label)))
		body.Append(ncase.Body().Slice()...)
		if fall, pos := hasFall(ncase.Body().Slice()); !fall {
			br := ir.Nod(ir.OBREAK, nil, nil)
			br.SetPos(pos)
			body.Append(br)
		}
	}
	sw.PtrList().Set(nil)

	if defaultGoto == nil {
		br := ir.Nod(ir.OBREAK, nil, nil)
		br.SetPos(br.Pos().WithNotStmt())
		defaultGoto = br
	}

	s.Emit(sw.PtrBody())
	sw.PtrBody().Append(defaultGoto)
	sw.PtrBody().AppendNodes(&body)
	walkstmtlist(sw.Body().Slice())
}

// An exprSwitch walks an expression switch.
type exprSwitch struct {
	exprname *ir.Node // value being switched on

	done    ir.Nodes
	clauses []exprClause
}

type exprClause struct {
	pos    src.XPos
	lo, hi *ir.Node
	jmp    *ir.Node
}

func (s *exprSwitch) Add(pos src.XPos, expr, jmp *ir.Node) {
	c := exprClause{pos: pos, lo: expr, hi: expr, jmp: jmp}
	if okforcmp[s.exprname.Type().Etype] && expr.Op() == ir.OLITERAL {
		s.clauses = append(s.clauses, c)
		return
	}

	s.flush()
	s.clauses = append(s.clauses, c)
	s.flush()
}

func (s *exprSwitch) Emit(out *ir.Nodes) {
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

	if s.exprname.Type().IsString() && len(cc) >= 2 {
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
		binarySearch(len(runs), &s.done,
			func(i int) *ir.Node {
				return ir.Nod(ir.OLE, ir.Nod(ir.OLEN, s.exprname, nil), nodintconst(runLen(runs[i-1])))
			},
			func(i int, nif *ir.Node) {
				run := runs[i]
				nif.SetLeft(ir.Nod(ir.OEQ, ir.Nod(ir.OLEN, s.exprname, nil), nodintconst(runLen(run))))
				s.search(run, nif.PtrBody())
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

func (s *exprSwitch) search(cc []exprClause, out *ir.Nodes) {
	binarySearch(len(cc), out,
		func(i int) *ir.Node {
			return ir.Nod(ir.OLE, s.exprname, cc[i-1].hi)
		},
		func(i int, nif *ir.Node) {
			c := &cc[i]
			nif.SetLeft(c.test(s.exprname))
			nif.PtrBody().Set1(c.jmp)
		},
	)
}

func (c *exprClause) test(exprname *ir.Node) *ir.Node {
	// Integer range.
	if c.hi != c.lo {
		low := ir.NodAt(c.pos, ir.OGE, exprname, c.lo)
		high := ir.NodAt(c.pos, ir.OLE, exprname, c.hi)
		return ir.NodAt(c.pos, ir.OANDAND, low, high)
	}

	// Optimize "switch true { ...}" and "switch false { ... }".
	if ir.IsConst(exprname, constant.Bool) && !c.lo.Type().IsInterface() {
		if exprname.BoolVal() {
			return c.lo
		} else {
			return ir.NodAt(c.pos, ir.ONOT, c.lo, nil)
		}
	}

	return ir.NodAt(c.pos, ir.OEQ, exprname, c.lo)
}

func allCaseExprsAreSideEffectFree(sw *ir.Node) bool {
	// In theory, we could be more aggressive, allowing any
	// side-effect-free expressions in cases, but it's a bit
	// tricky because some of that information is unavailable due
	// to the introduction of temporaries during order.
	// Restricting to constants is simple and probably powerful
	// enough.

	for _, ncase := range sw.List().Slice() {
		if ncase.Op() != ir.OCASE {
			base.Fatalf("switch string(byteslice) bad op: %v", ncase.Op())
		}
		for _, v := range ncase.List().Slice() {
			if v.Op() != ir.OLITERAL {
				return false
			}
		}
	}
	return true
}

// hasFall reports whether stmts ends with a "fallthrough" statement.
func hasFall(stmts []*ir.Node) (bool, src.XPos) {
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
func walkTypeSwitch(sw *ir.Node) {
	var s typeSwitch
	s.facename = sw.Left().Right()
	sw.SetLeft(nil)

	s.facename = walkexpr(s.facename, sw.PtrInit())
	s.facename = copyexpr(s.facename, s.facename.Type(), sw.PtrBody())
	s.okname = temp(types.Types[types.TBOOL])

	// Get interface descriptor word.
	// For empty interfaces this will be the type.
	// For non-empty interfaces this will be the itab.
	itab := ir.Nod(ir.OITAB, s.facename, nil)

	// For empty interfaces, do:
	//     if e._type == nil {
	//         do nil case if it exists, otherwise default
	//     }
	//     h := e._type.hash
	// Use a similar strategy for non-empty interfaces.
	ifNil := ir.Nod(ir.OIF, nil, nil)
	ifNil.SetLeft(ir.Nod(ir.OEQ, itab, nodnil()))
	base.Pos = base.Pos.WithNotStmt() // disable statement marks after the first check.
	ifNil.SetLeft(typecheck(ifNil.Left(), ctxExpr))
	ifNil.SetLeft(defaultlit(ifNil.Left(), nil))
	// ifNil.Nbody assigned at end.
	sw.PtrBody().Append(ifNil)

	// Load hash from type or itab.
	dotHash := nodSym(ir.ODOTPTR, itab, nil)
	dotHash.SetType(types.Types[types.TUINT32])
	dotHash.SetTypecheck(1)
	if s.facename.Type().IsEmptyInterface() {
		dotHash.SetOffset(int64(2 * Widthptr)) // offset of hash in runtime._type
	} else {
		dotHash.SetOffset(int64(2 * Widthptr)) // offset of hash in runtime.itab
	}
	dotHash.SetBounded(true) // guaranteed not to fault
	s.hashname = copyexpr(dotHash, dotHash.Type(), sw.PtrBody())

	br := ir.Nod(ir.OBREAK, nil, nil)
	var defaultGoto, nilGoto *ir.Node
	var body ir.Nodes
	for _, ncase := range sw.List().Slice() {
		var caseVar *ir.Node
		if ncase.Rlist().Len() != 0 {
			caseVar = ncase.Rlist().First()
		}

		// For single-type cases with an interface type,
		// we initialize the case variable as part of the type assertion.
		// In other cases, we initialize it in the body.
		var singleType *types.Type
		if ncase.List().Len() == 1 && ncase.List().First().Op() == ir.OTYPE {
			singleType = ncase.List().First().Type()
		}
		caseVarInitialized := false

		label := autolabel(".s")
		jmp := npos(ncase.Pos(), nodSym(ir.OGOTO, nil, label))

		if ncase.List().Len() == 0 { // default:
			if defaultGoto != nil {
				base.Fatalf("duplicate default case not detected during typechecking")
			}
			defaultGoto = jmp
		}

		for _, n1 := range ncase.List().Slice() {
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

		body.Append(npos(ncase.Pos(), nodSym(ir.OLABEL, nil, label)))
		if caseVar != nil && !caseVarInitialized {
			val := s.facename
			if singleType != nil {
				// We have a single concrete type. Extract the data.
				if singleType.IsInterface() {
					base.Fatalf("singleType interface should have been handled in Add")
				}
				val = ifaceData(ncase.Pos(), s.facename, singleType)
			}
			l := []*ir.Node{
				ir.NodAt(ncase.Pos(), ir.ODCL, caseVar, nil),
				ir.NodAt(ncase.Pos(), ir.OAS, caseVar, val),
			}
			typecheckslice(l, ctxStmt)
			body.Append(l...)
		}
		body.Append(ncase.Body().Slice()...)
		body.Append(br)
	}
	sw.PtrList().Set(nil)

	if defaultGoto == nil {
		defaultGoto = br
	}
	if nilGoto == nil {
		nilGoto = defaultGoto
	}
	ifNil.PtrBody().Set1(nilGoto)

	s.Emit(sw.PtrBody())
	sw.PtrBody().Append(defaultGoto)
	sw.PtrBody().AppendNodes(&body)

	walkstmtlist(sw.Body().Slice())
}

// A typeSwitch walks a type switch.
type typeSwitch struct {
	// Temporary variables (i.e., ONAMEs) used by type switch dispatch logic:
	facename *ir.Node // value being type-switched on
	hashname *ir.Node // type hash of the value being type-switched on
	okname   *ir.Node // boolean used for comma-ok type assertions

	done    ir.Nodes
	clauses []typeClause
}

type typeClause struct {
	hash uint32
	body ir.Nodes
}

func (s *typeSwitch) Add(pos src.XPos, typ *types.Type, caseVar, jmp *ir.Node) {
	var body ir.Nodes
	if caseVar != nil {
		l := []*ir.Node{
			ir.NodAt(pos, ir.ODCL, caseVar, nil),
			ir.NodAt(pos, ir.OAS, caseVar, nil),
		}
		typecheckslice(l, ctxStmt)
		body.Append(l...)
	} else {
		caseVar = ir.BlankNode
	}

	// cv, ok = iface.(type)
	as := ir.NodAt(pos, ir.OAS2, nil, nil)
	as.PtrList().Set2(caseVar, s.okname) // cv, ok =
	dot := ir.NodAt(pos, ir.ODOTTYPE, s.facename, nil)
	dot.SetType(typ) // iface.(type)
	as.PtrRlist().Set1(dot)
	as = typecheck(as, ctxStmt)
	as = walkexpr(as, &body)
	body.Append(as)

	// if ok { goto label }
	nif := ir.NodAt(pos, ir.OIF, nil, nil)
	nif.SetLeft(s.okname)
	nif.PtrBody().Set1(jmp)
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

func (s *typeSwitch) Emit(out *ir.Nodes) {
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
		func(i int) *ir.Node {
			return ir.Nod(ir.OLE, s.hashname, nodintconst(int64(cc[i-1].hash)))
		},
		func(i int, nif *ir.Node) {
			// TODO(mdempsky): Omit hash equality check if
			// there's only one type.
			c := cc[i]
			nif.SetLeft(ir.Nod(ir.OEQ, s.hashname, nodintconst(int64(c.hash))))
			nif.PtrBody().AppendNodes(&c.body)
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
func binarySearch(n int, out *ir.Nodes, less func(i int) *ir.Node, leaf func(i int, nif *ir.Node)) {
	const binarySearchMin = 4 // minimum number of cases for binary search

	var do func(lo, hi int, out *ir.Nodes)
	do = func(lo, hi int, out *ir.Nodes) {
		n := hi - lo
		if n < binarySearchMin {
			for i := lo; i < hi; i++ {
				nif := ir.Nod(ir.OIF, nil, nil)
				leaf(i, nif)
				base.Pos = base.Pos.WithNotStmt()
				nif.SetLeft(typecheck(nif.Left(), ctxExpr))
				nif.SetLeft(defaultlit(nif.Left(), nil))
				out.Append(nif)
				out = nif.PtrRlist()
			}
			return
		}

		half := lo + n/2
		nif := ir.Nod(ir.OIF, nil, nil)
		nif.SetLeft(less(half))
		base.Pos = base.Pos.WithNotStmt()
		nif.SetLeft(typecheck(nif.Left(), ctxExpr))
		nif.SetLeft(defaultlit(nif.Left(), nil))
		do(lo, half, nif.PtrBody())
		do(half, hi, nif.PtrRlist())
		out.Append(nif)
	}

	do(0, n, out)
}
