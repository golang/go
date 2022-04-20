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

func RangeExprType(t *types.Type) *types.Type {
	if t.IsPtr() && t.Elem().IsArray() {
		return t.Elem()
	}
	return t
}

func typecheckrangeExpr(n *ir.RangeStmt) {
	n.X = Expr(n.X)
	if n.X.Type() == nil {
		return
	}

	t := RangeExprType(n.X.Type())
	// delicate little dance.  see tcAssignList
	if n.Key != nil && !ir.DeclaredBy(n.Key, n) {
		n.Key = AssignExpr(n.Key)
	}
	if n.Value != nil && !ir.DeclaredBy(n.Value, n) {
		n.Value = AssignExpr(n.Value)
	}

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
				if op, why := Assignop(t, nn.Type()); op == ir.OXXX {
					base.ErrorfAt(n.Pos(), "cannot assign type %v to %L in range%s", t, nn, why)
				}
			}
			checkassign(nn)
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
		defer tracePrint("tcAssign", n)(nil)
	}

	if n.Y == nil {
		n.X = AssignExpr(n.X)
		return
	}

	lhs, rhs := []ir.Node{n.X}, []ir.Node{n.Y}
	assign(n, lhs, rhs)
	n.X, n.Y = lhs[0], rhs[0]

	// TODO(mdempsky): This seems out of place.
	if !ir.IsBlank(n.X) {
		types.CheckSize(n.X.Type()) // ensure width is calculated for backend
	}
}

func tcAssignList(n *ir.AssignListStmt) {
	if base.EnableTrace && base.Flag.LowerT {
		defer tracePrint("tcAssignList", n)(nil)
	}

	assign(n, n.Lhs, n.Rhs)
}

func assign(stmt ir.Node, lhs, rhs []ir.Node) {
	// delicate little dance.
	// the definition of lhs may refer to this assignment
	// as its definition, in which case it will call tcAssign.
	// in that case, do not call typecheck back, or it will cycle.
	// if the variable has a type (ntype) then typechecking
	// will not look at defn, so it is okay (and desirable,
	// so that the conversion below happens).

	checkLHS := func(i int, typ *types.Type) {
		lhs[i] = Resolve(lhs[i])
		if n := lhs[i]; typ != nil && ir.DeclaredBy(n, stmt) && n.Name().Ntype == nil {
			if typ.Kind() != types.TNIL {
				n.SetType(defaultType(typ))
			} else {
				base.Errorf("use of untyped nil")
			}
		}
		if lhs[i].Typecheck() == 0 {
			lhs[i] = AssignExpr(lhs[i])
		}
		checkassign(lhs[i])
	}

	assignType := func(i int, typ *types.Type) {
		checkLHS(i, typ)
		if typ != nil {
			checkassignto(typ, lhs[i])
		}
	}

	cr := len(rhs)
	if len(rhs) == 1 {
		rhs[0] = typecheck(rhs[0], ctxExpr|ctxMultiOK)
		if rtyp := rhs[0].Type(); rtyp != nil && rtyp.IsFuncArgStruct() {
			cr = rtyp.NumFields()
		}
	} else {
		Exprs(rhs)
	}

	// x, ok = y
assignOK:
	for len(lhs) == 2 && cr == 1 {
		stmt := stmt.(*ir.AssignListStmt)
		r := rhs[0]

		switch r.Op() {
		case ir.OINDEXMAP:
			stmt.SetOp(ir.OAS2MAPR)
		case ir.ORECV:
			stmt.SetOp(ir.OAS2RECV)
		case ir.ODOTTYPE:
			r := r.(*ir.TypeAssertExpr)
			stmt.SetOp(ir.OAS2DOTTYPE)
			r.SetOp(ir.ODOTTYPE2)
		case ir.ODYNAMICDOTTYPE:
			r := r.(*ir.DynamicTypeAssertExpr)
			stmt.SetOp(ir.OAS2DOTTYPE)
			r.SetOp(ir.ODYNAMICDOTTYPE2)
		default:
			break assignOK
		}

		assignType(0, r.Type())
		assignType(1, types.UntypedBool)
		return
	}

	if len(lhs) != cr {
		if r, ok := rhs[0].(*ir.CallExpr); ok && len(rhs) == 1 {
			if r.Type() != nil {
				base.ErrorfAt(stmt.Pos(), "assignment mismatch: %d variable%s but %v returns %d value%s", len(lhs), plural(len(lhs)), r.X, cr, plural(cr))
			}
		} else {
			base.ErrorfAt(stmt.Pos(), "assignment mismatch: %d variable%s but %v value%s", len(lhs), plural(len(lhs)), len(rhs), plural(len(rhs)))
		}

		for i := range lhs {
			checkLHS(i, nil)
		}
		return
	}

	// x,y,z = f()
	if cr > len(rhs) {
		stmt := stmt.(*ir.AssignListStmt)
		stmt.SetOp(ir.OAS2FUNC)
		r := rhs[0].(*ir.CallExpr)
		rtyp := r.Type()

		mismatched := false
		failed := false
		for i := range lhs {
			result := rtyp.Field(i).Type
			assignType(i, result)

			if lhs[i].Type() == nil || result == nil {
				failed = true
			} else if lhs[i] != ir.BlankNode && !types.Identical(lhs[i].Type(), result) {
				mismatched = true
			}
		}
		if mismatched && !failed {
			RewriteMultiValueCall(stmt, r)
		}
		return
	}

	for i, r := range rhs {
		checkLHS(i, r.Type())
		if lhs[i].Type() != nil {
			rhs[i] = AssignConv(r, lhs[i].Type(), "assignment")
		}
	}
}

func plural(n int) string {
	if n == 1 {
		return ""
	}
	return "s"
}

// tcCheckNil typechecks an OCHECKNIL node.
func tcCheckNil(n *ir.UnaryExpr) ir.Node {
	n.X = Expr(n.X)
	if !n.X.Type().IsPtrShaped() {
		base.FatalfAt(n.Pos(), "%L is not pointer shaped", n.X)
	}
	return n
}

// tcFor typechecks an OFOR node.
func tcFor(n *ir.ForStmt) ir.Node {
	Stmts(n.Init())
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
	if n.Call.Type() == nil {
		return
	}

	// The syntax made sure it was a call, so this must be
	// a conversion.
	base.FatalfAt(n.Pos(), "%s requires function call, not conversion", what)
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

	Stmts(n.Body)
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
	var def *ir.CommClause
	lno := ir.SetPos(sel)
	Stmts(sel.Init())
	for _, ncase := range sel.Cases {
		if ncase.Comm == nil {
			// default
			if def != nil {
				base.ErrorfAt(ncase.Pos(), "multiple defaults in select (first at %v)", ir.Line(def))
			} else {
				def = ncase
			}
		} else {
			n := Stmt(ncase.Comm)
			ncase.Comm = n
			oselrecv2 := func(dst, recv ir.Node, def bool) {
				selrecv := ir.NewAssignListStmt(n.Pos(), ir.OSELRECV2, []ir.Node{dst, ir.BlankNode}, []ir.Node{recv})
				selrecv.Def = def
				selrecv.SetTypecheck(1)
				selrecv.SetInit(n.Init())
				ncase.Comm = selrecv
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
	for _, ncase := range n.Cases {
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
				op1, _ := Assignop(n1.Type(), t)
				op2, _ := Assignop(t, n1.Type())
				if op1 == ir.OXXX && op2 == ir.OXXX {
					if n.Tag != nil {
						base.ErrorfAt(ncase.Pos(), "invalid case %v in switch on %v (mismatched types %v and %v)", n1, n.Tag, n1.Type(), t)
					} else {
						base.ErrorfAt(ncase.Pos(), "invalid case %v in switch (mismatched types %v and bool)", n1, n1.Type())
					}
				}
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
			if n1.Op() == ir.ODYNAMICTYPE {
				continue
			}
			if n1.Op() != ir.OTYPE {
				base.ErrorfAt(ncase.Pos(), "%L is not a type", n1)
				continue
			}
			if !n1.Type().IsInterface() && !implements(n1.Type(), t, &missing, &have, &ptr) {
				if have != nil {
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
				if ls[0].Op() == ir.OTYPE || ls[0].Op() == ir.ODYNAMICTYPE {
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
				nvar = AssignExpr(nvar).(*ir.Name)
			} else {
				// Clause variable is broken; prevent typechecking.
				nvar.SetTypecheck(1)
			}
			ncase.Var = nvar
		}

		Stmts(ncase.Body)
	}
}

type typeSet struct {
	m map[string]src.XPos
}

func (s *typeSet) add(pos src.XPos, typ *types.Type) {
	if s.m == nil {
		s.m = make(map[string]src.XPos)
	}

	ls := typ.LinkString()
	if prev, ok := s.m[ls]; ok {
		base.ErrorfAt(pos, "duplicate case %v in type switch\n\tprevious case at %s", typ, base.FmtPos(prev))
		return
	}
	s.m[ls] = pos
}
