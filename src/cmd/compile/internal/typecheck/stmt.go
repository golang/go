// Copyright 2009 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package typecheck

import (
	"cmd/compile/internal/base"
	"cmd/compile/internal/ir"
	"cmd/compile/internal/types"
	"cmd/internal/src"
)

// range
func typecheckrange(n *ir.RangeStmt) {
	// Typechecking order is important here:
	// 0. first typecheck range expression (slice/map/chan),
	//	it is evaluated only once and so logically it is not part of the loop.
	// 1. typecheck produced values,
	//	this part can declare new vars and so it must be typechecked before body,
	//	because body can contain a closure that captures the vars.
	// 2. decldepth++ to denote loop body.
	// 3. typecheck body.
	// 4. decldepth--.
	typecheckrangeExpr(n)

	// second half of dance, the first half being typecheckrangeExpr
	n.SetTypecheck(1)
	ls := n.Vars
	for i1, n1 := range ls {
		if n1.Typecheck() == 0 {
			ls[i1] = AssignExpr(ls[i1])
		}
	}

	decldepth++
	Stmts(n.Body)
	decldepth--
}

func typecheckrangeExpr(n *ir.RangeStmt) {
	n.X = Expr(n.X)

	t := n.X.Type()
	if t == nil {
		return
	}
	// delicate little dance.  see typecheckas2
	ls := n.Vars
	for i1, n1 := range ls {
		if !ir.DeclaredBy(n1, n) {
			ls[i1] = AssignExpr(ls[i1])
		}
	}

	if t.IsPtr() && t.Elem().IsArray() {
		t = t.Elem()
	}
	n.SetType(t)

	var t1, t2 *types.Type
	toomany := false
	switch t.Kind() {
	default:
		base.ErrorfAt(n.Pos(), "cannot range over %L", n.X)
		return

	case types.TARRAY, types.TSLICE:
		t1 = types.Types[types.TINT]
		t2 = t.Elem()

	case types.TMAP:
		t1 = t.Key()
		t2 = t.Elem()

	case types.TCHAN:
		if !t.ChanDir().CanRecv() {
			base.ErrorfAt(n.Pos(), "invalid operation: range %v (receive from send-only type %v)", n.X, n.X.Type())
			return
		}

		t1 = t.Elem()
		t2 = nil
		if len(n.Vars) == 2 {
			toomany = true
		}

	case types.TSTRING:
		t1 = types.Types[types.TINT]
		t2 = types.RuneType
	}

	if len(n.Vars) > 2 || toomany {
		base.ErrorfAt(n.Pos(), "too many variables in range")
	}

	var v1, v2 ir.Node
	if len(n.Vars) != 0 {
		v1 = n.Vars[0]
	}
	if len(n.Vars) > 1 {
		v2 = n.Vars[1]
	}

	// this is not only an optimization but also a requirement in the spec.
	// "if the second iteration variable is the blank identifier, the range
	// clause is equivalent to the same clause with only the first variable
	// present."
	if ir.IsBlank(v2) {
		if v1 != nil {
			n.Vars = []ir.Node{v1}
		}
		v2 = nil
	}

	if v1 != nil {
		if ir.DeclaredBy(v1, n) {
			v1.SetType(t1)
		} else if v1.Type() != nil {
			if op, why := assignop(t1, v1.Type()); op == ir.OXXX {
				base.ErrorfAt(n.Pos(), "cannot assign type %v to %L in range%s", t1, v1, why)
			}
		}
		checkassign(n, v1)
	}

	if v2 != nil {
		if ir.DeclaredBy(v2, n) {
			v2.SetType(t2)
		} else if v2.Type() != nil {
			if op, why := assignop(t2, v2.Type()); op == ir.OXXX {
				base.ErrorfAt(n.Pos(), "cannot assign type %v to %L in range%s", t2, v2, why)
			}
		}
		checkassign(n, v2)
	}
}

// select
func typecheckselect(sel *ir.SelectStmt) {
	var def ir.Node
	lno := ir.SetPos(sel)
	Stmts(sel.Init())
	for _, ncase := range sel.Cases {
		ncase := ncase.(*ir.CaseStmt)

		if len(ncase.List) == 0 {
			// default
			if def != nil {
				base.ErrorfAt(ncase.Pos(), "multiple defaults in select (first at %v)", ir.Line(def))
			} else {
				def = ncase
			}
		} else if len(ncase.List) > 1 {
			base.ErrorfAt(ncase.Pos(), "select cases cannot be lists")
		} else {
			ncase.List[0] = Stmt(ncase.List[0])
			n := ncase.List[0]
			ncase.Comm = n
			ncase.List.Set(nil)
			oselrecv2 := func(dst, recv ir.Node, colas bool) {
				n := ir.NewAssignListStmt(n.Pos(), ir.OSELRECV2, nil, nil)
				n.Lhs = []ir.Node{dst, ir.BlankNode}
				n.Rhs = []ir.Node{recv}
				n.Def = colas
				n.SetTypecheck(1)
				ncase.Comm = n
			}
			switch n.Op() {
			default:
				pos := n.Pos()
				if n.Op() == ir.ONAME {
					// We don't have the right position for ONAME nodes (see #15459 and
					// others). Using ncase.Pos for now as it will provide the correct
					// line number (assuming the expression follows the "case" keyword
					// on the same line). This matches the approach before 1.10.
					pos = ncase.Pos()
				}
				base.ErrorfAt(pos, "select case must be receive, send or assign recv")

			case ir.OAS:
				// convert x = <-c into x, _ = <-c
				// remove implicit conversions; the eventual assignment
				// will reintroduce them.
				n := n.(*ir.AssignStmt)
				if r := n.Y; r.Op() == ir.OCONVNOP || r.Op() == ir.OCONVIFACE {
					r := r.(*ir.ConvExpr)
					if r.Implicit() {
						n.Y = r.X
					}
				}
				if n.Y.Op() != ir.ORECV {
					base.ErrorfAt(n.Pos(), "select assignment must have receive on right hand side")
					break
				}
				oselrecv2(n.X, n.Y, n.Def)

			case ir.OAS2RECV:
				n := n.(*ir.AssignListStmt)
				if n.Rhs[0].Op() != ir.ORECV {
					base.ErrorfAt(n.Pos(), "select assignment must have receive on right hand side")
					break
				}
				n.SetOp(ir.OSELRECV2)

			case ir.ORECV:
				// convert <-c into _, _ = <-c
				n := n.(*ir.UnaryExpr)
				oselrecv2(ir.BlankNode, n, false)

			case ir.OSEND:
				break
			}
		}

		Stmts(ncase.Body)
	}

	base.Pos = lno
}

type typeSet struct {
	m map[string][]typeSetEntry
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

type typeSetEntry struct {
	pos src.XPos
	typ *types.Type
}

func typecheckExprSwitch(n *ir.SwitchStmt) {
	t := types.Types[types.TBOOL]
	if n.Tag != nil {
		n.Tag = Expr(n.Tag)
		n.Tag = DefaultLit(n.Tag, nil)
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
			ls[i] = Expr(ls[i])
			ls[i] = DefaultLit(ls[i], t)
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

		Stmts(ncase.Body)
	}
}

func typecheckTypeSwitch(n *ir.SwitchStmt) {
	guard := n.Tag.(*ir.TypeSwitchGuard)
	guard.X = Expr(guard.X)
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
			ls[i] = check(ls[i], ctxExpr|ctxType)
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
				nvar = AssignExpr(nvar)
			} else {
				// Clause variable is broken; prevent typechecking.
				nvar.SetTypecheck(1)
				nvar.SetWalkdef(1)
			}
			ncase.Vars[0] = nvar
		}

		Stmts(ncase.Body)
	}
}

// typecheckswitch typechecks a switch statement.
func typecheckswitch(n *ir.SwitchStmt) {
	Stmts(n.Init())
	if n.Tag != nil && n.Tag.Op() == ir.OTYPESW {
		typecheckTypeSwitch(n)
	} else {
		typecheckExprSwitch(n)
	}
}
