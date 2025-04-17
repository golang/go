// Copyright 2009 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package typecheck

import (
	"cmd/compile/internal/base"
	"cmd/compile/internal/ir"
	"cmd/compile/internal/types"
	"cmd/internal/src"
	"internal/types/errors"
)

func RangeExprType(t *types.Type) *types.Type {
	if t.IsPtr() && t.Elem().IsArray() {
		return t.Elem()
	}
	return t
}

func typecheckrangeExpr(n *ir.RangeStmt) {
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
		if n := lhs[i]; typ != nil && ir.DeclaredBy(n, stmt) && n.Type() == nil {
			base.Assertf(typ.Kind() == types.TNIL, "unexpected untyped nil")
			n.SetType(defaultType(typ))
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
				base.ErrorfAt(stmt.Pos(), errors.WrongAssignCount, "assignment mismatch: %d variable%s but %v returns %d value%s", len(lhs), plural(len(lhs)), r.Fun, cr, plural(cr))
			}
		} else {
			base.ErrorfAt(stmt.Pos(), errors.WrongAssignCount, "assignment mismatch: %d variable%s but %v value%s", len(lhs), plural(len(lhs)), len(rhs), plural(len(rhs)))
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
		r := rhs[0]
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
	Stmts(n.Body)
	return n
}

// tcGoDefer typechecks (normalizes) an OGO/ODEFER statement.
func tcGoDefer(n *ir.GoDeferStmt) {
	call := normalizeGoDeferCall(n.Pos(), n.Op(), n.Call, n.PtrInit())
	call.GoDefer = true
	n.Call = call
}

// normalizeGoDeferCall normalizes call into a normal function call
// with no arguments and no results, suitable for use in an OGO/ODEFER
// statement.
//
// For example, it normalizes:
//
//	f(x, y)
//
// into:
//
//	x1, y1 := x, y          // added to init
//	func() { f(x1, y1) }()  // result
func normalizeGoDeferCall(pos src.XPos, op ir.Op, call ir.Node, init *ir.Nodes) *ir.CallExpr {
	init.Append(ir.TakeInit(call)...)

	if call, ok := call.(*ir.CallExpr); ok && call.Op() == ir.OCALLFUNC {
		if sig := call.Fun.Type(); sig.NumParams()+sig.NumResults() == 0 {
			return call // already in normal form
		}
	}

	// Create a new wrapper function without parameters or results.
	wrapperFn := ir.NewClosureFunc(pos, pos, op, types.NewSignature(nil, nil, nil), ir.CurFunc, Target)
	wrapperFn.DeclareParams(true)
	wrapperFn.SetWrapper(true)

	// argps collects the list of operands within the call expression
	// that must be evaluated at the go/defer statement.
	var argps []*ir.Node

	var visit func(argp *ir.Node)
	visit = func(argp *ir.Node) {
		arg := *argp
		if arg == nil {
			return
		}

		// Recognize a few common expressions that can be evaluated within
		// the wrapper, so we don't need to allocate space for them within
		// the closure.
		switch arg.Op() {
		case ir.OLITERAL, ir.ONIL, ir.OMETHEXPR, ir.ONEW:
			return
		case ir.ONAME:
			arg := arg.(*ir.Name)
			if arg.Class == ir.PFUNC {
				return // reference to global function
			}
		case ir.OADDR:
			arg := arg.(*ir.AddrExpr)
			if arg.X.Op() == ir.OLINKSYMOFFSET {
				return // address of global symbol
			}

		case ir.OCONVNOP:
			arg := arg.(*ir.ConvExpr)

			// For unsafe.Pointer->uintptr conversion arguments, save the
			// unsafe.Pointer argument. This is necessary to handle cases
			// like fixedbugs/issue24491a.go correctly.
			//
			// TODO(mdempsky): Limit to static callees with
			// //go:uintptr{escapes,keepalive}?
			if arg.Type().IsUintptr() && arg.X.Type().IsUnsafePtr() {
				visit(&arg.X)
				return
			}

		case ir.OARRAYLIT, ir.OSLICELIT, ir.OSTRUCTLIT:
			// TODO(mdempsky): For very large slices, it may be preferable
			// to construct them at the go/defer statement instead.
			list := arg.(*ir.CompLitExpr).List
			for i, el := range list {
				switch el := el.(type) {
				case *ir.KeyExpr:
					visit(&el.Value)
				case *ir.StructKeyExpr:
					visit(&el.Value)
				default:
					visit(&list[i])
				}
			}
			return
		}

		argps = append(argps, argp)
	}

	visitList := func(list []ir.Node) {
		for i := range list {
			visit(&list[i])
		}
	}

	switch call.Op() {
	default:
		base.Fatalf("unexpected call op: %v", call.Op())

	case ir.OCALLFUNC:
		call := call.(*ir.CallExpr)

		// If the callee is a named function, link to the original callee.
		if wrapped := ir.StaticCalleeName(call.Fun); wrapped != nil {
			wrapperFn.WrappedFunc = wrapped.Func
		}

		visit(&call.Fun)
		visitList(call.Args)

	case ir.OCALLINTER:
		call := call.(*ir.CallExpr)
		argps = append(argps, &call.Fun.(*ir.SelectorExpr).X) // must be first for OCHECKNIL; see below
		visitList(call.Args)

	case ir.OAPPEND, ir.ODELETE, ir.OPRINT, ir.OPRINTLN, ir.ORECOVERFP:
		call := call.(*ir.CallExpr)
		visitList(call.Args)
		visit(&call.RType)

	case ir.OCOPY:
		call := call.(*ir.BinaryExpr)
		visit(&call.X)
		visit(&call.Y)
		visit(&call.RType)

	case ir.OCLEAR, ir.OCLOSE, ir.OPANIC:
		call := call.(*ir.UnaryExpr)
		visit(&call.X)
	}

	if len(argps) != 0 {
		// Found one or more operands that need to be evaluated upfront
		// and spilled to temporary variables, which can be captured by
		// the wrapper function.

		stmtPos := base.Pos
		callPos := base.Pos

		as := ir.NewAssignListStmt(callPos, ir.OAS2, make([]ir.Node, len(argps)), make([]ir.Node, len(argps)))
		for i, argp := range argps {
			arg := *argp

			pos := callPos
			if ir.HasUniquePos(arg) {
				pos = arg.Pos()
			}

			// tmp := arg
			tmp := TempAt(pos, ir.CurFunc, arg.Type())
			init.Append(Stmt(ir.NewDecl(pos, ir.ODCL, tmp)))
			tmp.Defn = as
			as.Lhs[i] = tmp
			as.Rhs[i] = arg

			// Rewrite original expression to use/capture tmp.
			*argp = ir.NewClosureVar(pos, wrapperFn, tmp)
		}
		init.Append(Stmt(as))

		// For "go/defer iface.M()", if iface is nil, we need to panic at
		// the point of the go/defer statement.
		if call.Op() == ir.OCALLINTER {
			iface := as.Lhs[0]
			init.Append(Stmt(ir.NewUnaryExpr(stmtPos, ir.OCHECKNIL, ir.NewUnaryExpr(iface.Pos(), ir.OITAB, iface))))
		}
	}

	// Move call into the wrapper function, now that it's safe to
	// evaluate there.
	wrapperFn.Body = []ir.Node{call}

	// Finally, construct a call to the wrapper.
	return Call(call.Pos(), wrapperFn.OClosure, nil, false).(*ir.CallExpr)
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
	n.X = Expr(n.X)

	// delicate little dance.  see tcAssignList
	if n.Key != nil {
		if !ir.DeclaredBy(n.Key, n) {
			n.Key = AssignExpr(n.Key)
		}
		checkassign(n.Key)
	}
	if n.Value != nil {
		if !ir.DeclaredBy(n.Value, n) {
			n.Value = AssignExpr(n.Value)
		}
		checkassign(n.Value)
	}

	// second half of dance
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
	if ir.CurFunc == nil {
		base.FatalfAt(n.Pos(), "return outside function")
	}

	typecheckargs(n)
	if len(n.Results) != 0 {
		typecheckaste(ir.ORETURN, nil, false, ir.CurFunc.Type().Results(), n.Results, func() string { return "return argument" })
	}
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
				base.ErrorfAt(ncase.Pos(), errors.DuplicateDefault, "multiple defaults in select (first at %v)", ir.Line(def))
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
				base.ErrorfAt(pos, errors.InvalidSelectCase, "select case must be receive, send or assign recv")

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
					base.ErrorfAt(n.Pos(), errors.InvalidSelectCase, "select assignment must have receive on right hand side")
					break
				}
				oselrecv2(n.X, n.Y, n.Def)

			case ir.OAS2RECV:
				n := n.(*ir.AssignListStmt)
				if n.Rhs[0].Op() != ir.ORECV {
					base.ErrorfAt(n.Pos(), errors.InvalidSelectCase, "select assignment must have receive on right hand side")
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
				base.ErrorfAt(n.Pos(), errors.InvalidExprSwitch, "cannot switch on %L (struct containing %v cannot be compared)", n.Tag, types.IncomparableField(t).Type)
			} else {
				base.ErrorfAt(n.Pos(), errors.InvalidExprSwitch, "cannot switch on %L", n.Tag)
			}
			t = nil
		}
	}

	var defCase ir.Node
	for _, ncase := range n.Cases {
		ls := ncase.List
		if len(ls) == 0 { // default:
			if defCase != nil {
				base.ErrorfAt(ncase.Pos(), errors.DuplicateDefault, "multiple defaults in switch (first at %v)", ir.Line(defCase))
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
				base.ErrorfAt(ncase.Pos(), errors.MismatchedTypes, "invalid case %v in switch (can only compare %s %v to nil)", n1, nilonly, n.Tag)
			} else if t.IsInterface() && !n1.Type().IsInterface() && !types.IsComparable(n1.Type()) {
				base.ErrorfAt(ncase.Pos(), errors.UndefinedOp, "invalid case %L in switch (incomparable type)", n1)
			} else {
				op1, _ := assignOp(n1.Type(), t)
				op2, _ := assignOp(t, n1.Type())
				if op1 == ir.OXXX && op2 == ir.OXXX {
					if n.Tag != nil {
						base.ErrorfAt(ncase.Pos(), errors.MismatchedTypes, "invalid case %v in switch on %v (mismatched types %v and %v)", n1, n.Tag, n1.Type(), t)
					} else {
						base.ErrorfAt(ncase.Pos(), errors.MismatchedTypes, "invalid case %v in switch (mismatched types %v and bool)", n1, n1.Type())
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
		base.ErrorfAt(n.Pos(), errors.InvalidTypeSwitch, "cannot type switch on non-interface value %L", guard.X)
		t = nil
	}

	// We don't actually declare the type switch's guarded
	// declaration itself. So if there are no cases, we won't
	// notice that it went unused.
	if v := guard.Tag; v != nil && !ir.IsBlank(v) && len(n.Cases) == 0 {
		base.ErrorfAt(v.Pos(), errors.UnusedVar, "%v declared but not used", v.Sym())
	}

	var defCase, nilCase ir.Node
	var ts typeSet
	for _, ncase := range n.Cases {
		ls := ncase.List
		if len(ls) == 0 { // default:
			if defCase != nil {
				base.ErrorfAt(ncase.Pos(), errors.DuplicateDefault, "multiple defaults in switch (first at %v)", ir.Line(defCase))
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

			if ir.IsNil(n1) { // case nil:
				if nilCase != nil {
					base.ErrorfAt(ncase.Pos(), errors.DuplicateCase, "multiple nil cases in type switch (first at %v)", ir.Line(nilCase))
				} else {
					nilCase = ncase
				}
				continue
			}
			if n1.Op() == ir.ODYNAMICTYPE {
				continue
			}
			if n1.Op() != ir.OTYPE {
				base.ErrorfAt(ncase.Pos(), errors.NotAType, "%L is not a type", n1)
				continue
			}
			if !n1.Type().IsInterface() {
				why := ImplementsExplain(n1.Type(), t)
				if why != "" {
					base.ErrorfAt(ncase.Pos(), errors.ImpossibleAssert, "impossible type switch case: %L cannot have dynamic type %v (%s)", guard.X, n1.Type(), why)
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
		base.ErrorfAt(pos, errors.DuplicateCase, "duplicate case %v in type switch\n\tprevious case at %s", typ, base.FmtPos(prev))
		return
	}
	s.m[ls] = pos
}
