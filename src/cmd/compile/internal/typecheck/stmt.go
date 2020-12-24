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

func typecheckrangeExpr(n *ir.RangeStmt) {
	n.X = Expr(n.X)

	t := n.X.Type()
	if t == nil {
		return
	}
	// delicate little dance.  see typecheckas2
	if n.Key != nil && !ir.DeclaredBy(n.Key, n) {
		n.Key = AssignExpr(n.Key)
	}
	if n.Value != nil && !ir.DeclaredBy(n.Value, n) {
		n.Value = AssignExpr(n.Value)
	}
	if t.IsPtr() && t.Elem().IsArray() {
		t = t.Elem()
	}
	n.SetType(t)

	var tk, tv *types.Type
	toomany := false
	switch t.Kind() {
	default:
		base.ErrorfAt(n.Pos(), "cannot range over %L", n.X)
		return

	case types.TARRAY, types.TSLICE:
		tk = types.Types[types.TINT]
		tv = t.Elem()

	case types.TMAP:
		tk = t.Key()
		tv = t.Elem()

	case types.TCHAN:
		if !t.ChanDir().CanRecv() {
			base.ErrorfAt(n.Pos(), "invalid operation: range %v (receive from send-only type %v)", n.X, n.X.Type())
			return
		}

		tk = t.Elem()
		tv = nil
		if n.Value != nil {
			toomany = true
		}

	case types.TSTRING:
		tk = types.Types[types.TINT]
		tv = types.RuneType
	}

	if toomany {
		base.ErrorfAt(n.Pos(), "too many variables in range")
	}

	do := func(nn ir.Node, t *types.Type) {
		if nn != nil {
			if ir.DeclaredBy(nn, n) {
				nn.SetType(t)
			} else if nn.Type() != nil {
				if op, why := assignop(t, nn.Type()); op == ir.OXXX {
					base.ErrorfAt(n.Pos(), "cannot assign type %v to %L in range%s", t, nn, why)
				}
			}
			checkassign(n, nn)
		}
	}
	do(n.Key, tk)
	do(n.Value, tv)
}

// type check assignment.
// if this assignment is the definition of a var on the left side,
// fill in the var's type.
func tcAssign(n *ir.AssignStmt) {
	if base.EnableTrace && base.Flag.LowerT {
		defer tracePrint("typecheckas", n)(nil)
	}

	// delicate little dance.
	// the definition of n may refer to this assignment
	// as its definition, in which case it will call typecheckas.
	// in that case, do not call typecheck back, or it will cycle.
	// if the variable has a type (ntype) then typechecking
	// will not look at defn, so it is okay (and desirable,
	// so that the conversion below happens).
	n.X = Resolve(n.X)

	if !ir.DeclaredBy(n.X, n) || n.X.Name().Ntype != nil {
		n.X = AssignExpr(n.X)
	}

	// Use ctxMultiOK so we can emit an "N variables but M values" error
	// to be consistent with typecheckas2 (#26616).
	n.Y = typecheck(n.Y, ctxExpr|ctxMultiOK)
	checkassign(n, n.X)
	if n.Y != nil && n.Y.Type() != nil {
		if n.Y.Type().IsFuncArgStruct() {
			base.Errorf("assignment mismatch: 1 variable but %v returns %d values", n.Y.(*ir.CallExpr).X, n.Y.Type().NumFields())
			// Multi-value RHS isn't actually valid for OAS; nil out
			// to indicate failed typechecking.
			n.Y.SetType(nil)
		} else if n.X.Type() != nil {
			n.Y = AssignConv(n.Y, n.X.Type(), "assignment")
		}
	}

	if ir.DeclaredBy(n.X, n) && n.X.Name().Ntype == nil {
		n.Y = DefaultLit(n.Y, nil)
		n.X.SetType(n.Y.Type())
	}

	// second half of dance.
	// now that right is done, typecheck the left
	// just to get it over with.  see dance above.
	n.SetTypecheck(1)

	if n.X.Typecheck() == 0 {
		n.X = AssignExpr(n.X)
	}
	if !ir.IsBlank(n.X) {
		types.CheckSize(n.X.Type()) // ensure width is calculated for backend
	}
}

func tcAssignList(n *ir.AssignListStmt) {
	if base.EnableTrace && base.Flag.LowerT {
		defer tracePrint("typecheckas2", n)(nil)
	}

	ls := n.Lhs
	for i1, n1 := range ls {
		// delicate little dance.
		n1 = Resolve(n1)
		ls[i1] = n1

		if !ir.DeclaredBy(n1, n) || n1.Name().Ntype != nil {
			ls[i1] = AssignExpr(ls[i1])
		}
	}

	cl := len(n.Lhs)
	cr := len(n.Rhs)
	if cl > 1 && cr == 1 {
		n.Rhs[0] = typecheck(n.Rhs[0], ctxExpr|ctxMultiOK)
	} else {
		Exprs(n.Rhs)
	}
	checkassignlist(n, n.Lhs)

	var l ir.Node
	var r ir.Node
	if cl == cr {
		// easy
		ls := n.Lhs
		rs := n.Rhs
		for il, nl := range ls {
			nr := rs[il]
			if nl.Type() != nil && nr.Type() != nil {
				rs[il] = AssignConv(nr, nl.Type(), "assignment")
			}
			if ir.DeclaredBy(nl, n) && nl.Name().Ntype == nil {
				rs[il] = DefaultLit(rs[il], nil)
				nl.SetType(rs[il].Type())
			}
		}

		goto out
	}

	l = n.Lhs[0]
	r = n.Rhs[0]

	// x,y,z = f()
	if cr == 1 {
		if r.Type() == nil {
			goto out
		}
		switch r.Op() {
		case ir.OCALLMETH, ir.OCALLINTER, ir.OCALLFUNC:
			if !r.Type().IsFuncArgStruct() {
				break
			}
			cr = r.Type().NumFields()
			if cr != cl {
				goto mismatch
			}
			r.(*ir.CallExpr).Use = ir.CallUseList
			n.SetOp(ir.OAS2FUNC)
			for i, l := range n.Lhs {
				f := r.Type().Field(i)
				if f.Type != nil && l.Type() != nil {
					checkassignto(f.Type, l)
				}
				if ir.DeclaredBy(l, n) && l.Name().Ntype == nil {
					l.SetType(f.Type)
				}
			}
			goto out
		}
	}

	// x, ok = y
	if cl == 2 && cr == 1 {
		if r.Type() == nil {
			goto out
		}
		switch r.Op() {
		case ir.OINDEXMAP, ir.ORECV, ir.ODOTTYPE:
			switch r.Op() {
			case ir.OINDEXMAP:
				n.SetOp(ir.OAS2MAPR)
			case ir.ORECV:
				n.SetOp(ir.OAS2RECV)
			case ir.ODOTTYPE:
				r := r.(*ir.TypeAssertExpr)
				n.SetOp(ir.OAS2DOTTYPE)
				r.SetOp(ir.ODOTTYPE2)
			}
			if l.Type() != nil {
				checkassignto(r.Type(), l)
			}
			if ir.DeclaredBy(l, n) {
				l.SetType(r.Type())
			}
			l := n.Lhs[1]
			if l.Type() != nil && !l.Type().IsBoolean() {
				checkassignto(types.Types[types.TBOOL], l)
			}
			if ir.DeclaredBy(l, n) && l.Name().Ntype == nil {
				l.SetType(types.Types[types.TBOOL])
			}
			goto out
		}
	}

mismatch:
	switch r.Op() {
	default:
		base.Errorf("assignment mismatch: %d variables but %d values", cl, cr)
	case ir.OCALLFUNC, ir.OCALLMETH, ir.OCALLINTER:
		r := r.(*ir.CallExpr)
		base.Errorf("assignment mismatch: %d variables but %v returns %d values", cl, r.X, cr)
	}

	// second half of dance
out:
	n.SetTypecheck(1)
	ls = n.Lhs
	for i1, n1 := range ls {
		if n1.Typecheck() == 0 {
			ls[i1] = AssignExpr(ls[i1])
		}
	}
}

// tcFor typechecks an OFOR node.
func tcFor(n *ir.ForStmt) ir.Node {
	Stmts(n.Init())
	decldepth++
	n.Cond = Expr(n.Cond)
	n.Cond = DefaultLit(n.Cond, nil)
	if n.Cond != nil {
		t := n.Cond.Type()
		if t != nil && !t.IsBoolean() {
			base.Errorf("non-bool %L used as for condition", n.Cond)
		}
	}
	n.Post = Stmt(n.Post)
	if n.Op() == ir.OFORUNTIL {
		Stmts(n.Late)
	}
	Stmts(n.Body)
	decldepth--
	return n
}

func tcGoDefer(n *ir.GoDeferStmt) {
	what := "defer"
	if n.Op() == ir.OGO {
		what = "go"
	}

	switch n.Call.Op() {
	// ok
	case ir.OCALLINTER,
		ir.OCALLMETH,
		ir.OCALLFUNC,
		ir.OCLOSE,
		ir.OCOPY,
		ir.ODELETE,
		ir.OPANIC,
		ir.OPRINT,
		ir.OPRINTN,
		ir.ORECOVER:
		return

	case ir.OAPPEND,
		ir.OCAP,
		ir.OCOMPLEX,
		ir.OIMAG,
		ir.OLEN,
		ir.OMAKE,
		ir.OMAKESLICE,
		ir.OMAKECHAN,
		ir.OMAKEMAP,
		ir.ONEW,
		ir.OREAL,
		ir.OLITERAL: // conversion or unsafe.Alignof, Offsetof, Sizeof
		if orig := ir.Orig(n.Call); orig.Op() == ir.OCONV {
			break
		}
		base.ErrorfAt(n.Pos(), "%s discards result of %v", what, n.Call)
		return
	}

	// type is broken or missing, most likely a method call on a broken type
	// we will warn about the broken type elsewhere. no need to emit a potentially confusing error
	if n.Call.Type() == nil || n.Call.Type().Broke() {
		return
	}

	if !n.Diag() {
		// The syntax made sure it was a call, so this must be
		// a conversion.
		n.SetDiag(true)
		base.ErrorfAt(n.Pos(), "%s requires function call, not conversion", what)
	}
}

// tcIf typechecks an OIF node.
func tcIf(n *ir.IfStmt) ir.Node {
	Stmts(n.Init())
	n.Cond = Expr(n.Cond)
	n.Cond = DefaultLit(n.Cond, nil)
	if n.Cond != nil {
		t := n.Cond.Type()
		if t != nil && !t.IsBoolean() {
			base.Errorf("non-bool %L used as if condition", n.Cond)
		}
	}
	Stmts(n.Body)
	Stmts(n.Else)
	return n
}

// range
func tcRange(n *ir.RangeStmt) {
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
	if n.Key != nil && n.Key.Typecheck() == 0 {
		n.Key = AssignExpr(n.Key)
	}
	if n.Value != nil && n.Value.Typecheck() == 0 {
		n.Value = AssignExpr(n.Value)
	}

	decldepth++
	Stmts(n.Body)
	decldepth--
}

// tcReturn typechecks an ORETURN node.
func tcReturn(n *ir.ReturnStmt) ir.Node {
	typecheckargs(n)
	if ir.CurFunc == nil {
		base.Errorf("return outside function")
		n.SetType(nil)
		return n
	}

	if ir.HasNamedResults(ir.CurFunc) && len(n.Results) == 0 {
		return n
	}
	typecheckaste(ir.ORETURN, nil, false, ir.CurFunc.Type().Results(), n.Results, func() string { return "return argument" })
	return n
}

// select
func tcSelect(sel *ir.SelectStmt) {
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

// tcSend typechecks an OSEND node.
func tcSend(n *ir.SendStmt) ir.Node {
	n.Chan = Expr(n.Chan)
	n.Value = Expr(n.Value)
	n.Chan = DefaultLit(n.Chan, nil)
	t := n.Chan.Type()
	if t == nil {
		return n
	}
	if !t.IsChan() {
		base.Errorf("invalid operation: %v (send to non-chan type %v)", n, t)
		return n
	}

	if !t.ChanDir().CanSend() {
		base.Errorf("invalid operation: %v (send to receive-only type %v)", n, t)
		return n
	}

	n.Value = AssignConv(n.Value, t.Elem(), "send")
	if n.Value.Type() == nil {
		return n
	}
	return n
}

// tcSwitch typechecks a switch statement.
func tcSwitch(n *ir.SwitchStmt) {
	Stmts(n.Init())
	if n.Tag != nil && n.Tag.Op() == ir.OTYPESW {
		tcSwitchType(n)
	} else {
		tcSwitchExpr(n)
	}
}

func tcSwitchExpr(n *ir.SwitchStmt) {
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

func tcSwitchType(n *ir.SwitchStmt) {
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

		if ncase.Var != nil {
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

			nvar := ncase.Var
			nvar.SetType(vt)
			if vt != nil {
				nvar = AssignExpr(nvar)
			} else {
				// Clause variable is broken; prevent typechecking.
				nvar.SetTypecheck(1)
				nvar.SetWalkdef(1)
			}
			ncase.Var = nvar
		}

		Stmts(ncase.Body)
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
