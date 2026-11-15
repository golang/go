// Copyright 2026 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package types2

import (
	"cmd/compile/internal/syntax"
	. "internal/types/errors"
	"strings"
)

const logosMisplacedTry = "[correctness::misplaced_try] ? must end an assignment, expression statement, defer, or errdefer"

const logosErrdeferNeedsError = "[correctness::errdefer_needs_error_result] errdefer requires enclosing function to return error"

func logosIsReserved(name string) bool {
	return strings.HasPrefix(name, "_logos_") || strings.HasPrefix(name, "Logos_")
}

func (check *Checker) logosReservedIdent(id *syntax.Name) {
	if !check.conf.Logos || check.logosCopy || !logosIsReserved(id.Value) {
		return
	}
	check.errorf(id, BadDecl, "[correctness::reserved_identifier] identifier %s is reserved", id.Value)
}

func (check *Checker) logosEnclosingError() bool {
	sig := check.sig
	if sig == nil {
		return false
	}
	n := sig.results.Len()
	if n == 0 {
		return false
	}
	return Identical(sig.results.vars[n-1].typ, universeError)
}

func (check *Checker) logosTryResults(t *syntax.TryExpr) (results []*operand, ok bool) {
	fail := func() ([]*operand, bool) { return nil, false }

	inner := syntax.Unparen(t.X)
	call, isCall := inner.(*syntax.CallExpr)
	if !isCall {
		check.use(t.X)
		check.error(t.Pos(), InvalidCall, "? operand must be a call")
		return fail()
	}

	var cx operand
	if kind := check.rawExpr(nil, &cx, call, nil, false); kind == conversion {
		check.error(t.Pos(), InvalidCall, "? operand must be a call")
		return fail()
	}
	if !cx.isValid() {
		return fail()
	}

	var keep []Type
	switch cx.mode() {
	case value:
		if tup, isTup := cx.typ().(*Tuple); isTup {
			vars := tup.vars
			have := "no results"
			if len(vars) > 0 {
				have = TypeString(vars[len(vars)-1].typ, check.qualifier)
			}
			if len(vars) == 0 || !Identical(vars[len(vars)-1].typ, universeError) {
				check.errorf(t.Pos(), InvalidCall, "? last result must be error, have %s", have)
				return fail()
			}
			for _, v := range vars[:len(vars)-1] {
				keep = append(keep, v.typ)
			}
		} else {
			if !Identical(cx.typ(), universeError) {
				check.errorf(t.Pos(), InvalidCall, "? last result must be error, have %s", TypeString(cx.typ(), check.qualifier))
				return fail()
			}
		}
	case commaerr:
		keep = append(keep, cx.typ())
	default:
		check.error(t.Pos(), InvalidCall, "? last result must be error")
		return fail()
	}

	if !check.logosEnclosingError() {
		check.error(t.Pos(), WrongResultCount, "? requires enclosing function to return error")
		return fail()
	}

	results = make([]*operand, len(keep))
	for i, typ := range keep {
		var e syntax.Expr = syntax.NewName(syntax.StartPos(call), nth(i+1, "function result"))
		if len(keep) == 1 {
			e = call
		}
		results[i] = &operand{mode_: value, expr: e, typ_: typ}
	}

	var tx operand
	if len(results) == 0 {
		tx = operand{mode_: novalue, expr: t}
	} else {
		typ := results[0].typ()
		if len(results) > 1 {
			vars := make([]*Var, len(results))
			for i, r := range results {
				vars[i] = newVar(ResultVar, r.Pos(), check.pkg, "", r.typ())
			}
			typ = NewTuple(vars...)
		}
		tx = operand{mode_: value, expr: t, typ_: typ}
	}
	check.record(&tx)

	return results, true
}

func (check *Checker) logosTryMisplaced(t *syntax.TryExpr) {
	if inner, isCall := syntax.Unparen(t.X).(*syntax.CallExpr); isCall {
		var cx operand
		check.rawExpr(nil, &cx, inner, nil, false)
	} else {
		check.use(t.X)
	}
	check.error(t.Pos(), InvalidCall, logosMisplacedTry)
}

func (check *Checker) logosPostStmt(post syntax.Stmt) {
	if !check.logosForPost(post) {
		check.simpleStmt(post)
	}
}

func (check *Checker) logosForPost(post syntax.Stmt) bool {
	if !check.conf.Logos || post == nil {
		return false
	}
	switch s := post.(type) {
	case *syntax.AssignStmt:
		if s.Op != 0 {
			return false
		}
		return check.logosTryAssignStmt(s)
	case *syntax.ExprStmt:
		return check.logosTryExprStmt(s)
	default:
		return false
	}
}

func (check *Checker) logosTryCountError(t *syntax.TryExpr, nvals, nvars int) {
	if nvals == 0 {
		check.error(t.Pos(), WrongAssignCount, "? provides no values to assign")
		return
	}
	check.errorf(t.Pos(), WrongAssignCount, "? provides %s for %s", measure(nvals, "value"), measure(nvars, "variable"))
}

func (check *Checker) logosTryAssignStmt(s *syntax.AssignStmt) bool {
	if s.Rhs == nil || s.Op != 0 {
		return false
	}
	rhs := syntax.UnpackListExpr(s.Rhs)
	if !check.conf.Logos || len(rhs) != 1 {
		return false
	}
	t, ok := syntax.Unparen(rhs[0]).(*syntax.TryExpr)
	if !ok {
		return false
	}
	check.logosTryAssignEq(syntax.UnpackListExpr(s.Lhs), t)
	return true
}

func (check *Checker) logosTryAssignEq(lhs []syntax.Expr, t *syntax.TryExpr) {
	results, ok := check.logosTryResults(t)
	if !ok || len(results) != len(lhs) {
		if ok {
			check.logosTryCountError(t, len(results), len(lhs))
		}
		check.useLHS(lhs...)
		return
	}
	context := "assignment"
	if len(lhs) > 1 {
		context = "multiple assignment"
	}
	for i, l := range lhs {
		check.assignVar(l, nil, results[i], context)
	}
}

func (check *Checker) logosTryInitHook(lhsVars []*Var, rhs []syntax.Expr) bool {
	if !check.conf.Logos || len(rhs) != 1 {
		return false
	}
	t, ok := syntax.Unparen(rhs[0]).(*syntax.TryExpr)
	if !ok {
		return false
	}
	check.logosTryInitVars(lhsVars, t)
	return true
}

func (check *Checker) logosTryNoNewVar(lhs, rhs []syntax.Expr) bool {
	if !check.conf.Logos || len(rhs) != 1 || len(lhs) == 0 {
		return false
	}
	if _, ok := syntax.Unparen(rhs[0]).(*syntax.TryExpr); !ok {
		return false
	}
	check.softErrorf(lhs[0].Pos(), NoNewVar, "no new variables on left side of :=")
	return true
}

func (check *Checker) logosTryInitVars(lhsVars []*Var, t *syntax.TryExpr) {
	results, ok := check.logosTryResults(t)
	if !ok || len(results) != len(lhsVars) {
		if ok {
			check.logosTryCountError(t, len(results), len(lhsVars))
		}
		for _, v := range lhsVars {
			if v.typ == nil {
				v.typ = Typ[Invalid]
			}
		}
		return
	}
	for i, lhs := range lhsVars {
		check.initVar(lhs, results[i], "assignment")
	}
}

func (check *Checker) logosTryExprStmt(s *syntax.ExprStmt) bool {
	if !check.conf.Logos {
		return false
	}
	t, ok := syntax.Unparen(s.X).(*syntax.TryExpr)
	if !ok {
		return false
	}
	check.logosTryResults(t)
	return true
}

func (check *Checker) logosTryCallStmt(s *syntax.CallStmt) bool {
	if !check.conf.Logos {
		return false
	}
	t, ok := syntax.Unparen(s.Call).(*syntax.TryExpr)
	if !ok {
		return false
	}
	if s.Tok == syntax.Go {
		check.logosTryMisplaced(t)
		return true
	}
	check.logosTryResults(t)
	return true
}

func (check *Checker) logosErrdefer(s *syntax.ErrdeferStmt) {
	if !check.conf.Logos {
		check.error(s, InvalidSyntaxTree, "errdefer requires logos mode")
		return
	}
	if !check.logosEnclosingError() {
		check.error(s, InvalidDefer, logosErrdeferNeedsError)
		check.logosErrdeferUse(s.Call)
		return
	}
	switch c := syntax.Unparen(s.Call).(type) {
	case *syntax.TryExpr:
		check.logosTryResults(c)
	case *syntax.CallExpr:
		var x operand
		switch check.rawExpr(nil, &x, c, nil, false) {
		case statement:
		default:
			check.error(c, InvalidDefer, "errdefer requires a function call")
		}
	default:
		check.use(s.Call)
	}
}

func (check *Checker) logosErrdeferUse(call syntax.Expr) {
	switch c := syntax.Unparen(call).(type) {
	case *syntax.TryExpr:
		if inner, ok := syntax.Unparen(c.X).(*syntax.CallExpr); ok {
			var x operand
			check.rawExpr(nil, &x, inner, nil, false)
		} else {
			check.use(c.X)
		}
	case *syntax.CallExpr:
		var x operand
		check.rawExpr(nil, &x, c, nil, false)
	default:
		check.use(call)
	}
}

func (check *Checker) logosTryOperand(x *operand, e *syntax.TryExpr) exprKind {
	if !check.conf.Logos {
		check.error(e, InvalidSyntaxTree, "invalid use of ? outside logos mode")
		x.invalidate()
		x.expr = e
		x.typ_ = Typ[Invalid]
		return statement
	}
	check.logosTryMisplaced(e)
	x.invalidate()
	x.expr = e
	x.typ_ = Typ[Invalid]
	return statement
}
