// Copyright 2026 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package types2

import (
	"cmd/compile/internal/syntax"
	. "internal/types/errors"
	"strings"
)

// LoGo/s M1-S3 type rules for postfix `?` (SPEC Sec 1.4, T-104.1) and
// errdefer (T-104.2). Upstream files call in through hook lines only:
// expr.go, stmt.go, and recording.go.
//
// A `?` in an allowed position (single right-hand side of `=` or `:=`,
// expression statement, `defer` or `errdefer` call) is handled wholly by
// the statement hook, which types the inner call once via
// logosTryResults. Any other `?` reports misplaced_try: the expression
// fallback and the `go` and for-post statements type the operand silently
// via logosTryMisplaced and report only the position error. Each TryExpr
// node yields at most one error.
//
// logosTryResults records the TryExpr type in Info.Types: the single
// remaining type, a *Tuple of the remainders, or novalue when only
// error remains.

const logosMisplacedTry = "[correctness::misplaced_try] ? must end an assignment, expression statement, defer, or errdefer"

func logosIsReserved(name string) bool {
	return strings.HasPrefix(name, "_logos_") || strings.HasPrefix(name, "Logos_")
}

// logosReservedIdent rejects reserved identifiers in .lgo files (C6).
// Hooked from recordDef, which also covers short declarations whose
// declare call carries no name.
func (check *Checker) logosReservedIdent(id *syntax.Name) {
	if !check.conf.Logos || !logosIsReserved(id.Value) {
		return
	}
	check.errorf(id, BadDecl, "[correctness::reserved_identifier] identifier %s is reserved", id.Value)
}

// logosEnclosingError reports whether the innermost enclosing function
// returns error last, identical to the predeclared error.
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

// logosTryResults checks C1 for t: the operand must be a call whose last
// result is identical to error, and the enclosing function must return
// error last. A conversion is syntactically a call but not a call operand.
// It returns one operand per non-error result. On failure it reports
// exactly one error and returns ok == false, in which case callers mark
// the lhs used but report nothing more for t.
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
			if len(vars) == 0 || !Identical(vars[len(vars)-1].typ, universeError) {
				check.error(t.Pos(), InvalidCall, "? last result must be error")
				return fail()
			}
			for _, v := range vars[:len(vars)-1] {
				keep = append(keep, v.typ)
			}
		} else {
			if !Identical(cx.typ(), universeError) {
				check.error(t.Pos(), InvalidCall, "? last result must be error")
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

// logosTryMisplaced types the operand of t silently and reports
// misplaced_try. A call operand is typechecked so its uses are marked
// and genuine inner errors still surface, but the C1 operand and
// enclosing checks run without diagnostics; a non-call operand only has
// its uses marked. The position error is the single diagnostic for t.
func (check *Checker) logosTryMisplaced(t *syntax.TryExpr) {
	if inner, isCall := syntax.Unparen(t.X).(*syntax.CallExpr); isCall {
		var cx operand
		check.rawExpr(nil, &cx, inner, nil, false)
	} else {
		check.use(t.X)
	}
	check.error(t.Pos(), InvalidCall, logosMisplacedTry)
}

// logosPostStmt checks a for-statement post statement. A `?` in the for
// init stays valid through the assignment hooks; a `?` accepted as the
// post statement itself is misplaced_try (M1), handled here so the normal
// statement checker is skipped and no duplicate is reported. Any other
// post statement runs the upstream path, where a nested `?` reaches the
// expression fallback and reports misplaced_try itself.
func (check *Checker) logosPostStmt(post syntax.Stmt) {
	if !check.logosForPost(post) {
		check.simpleStmt(post)
	}
}

// logosForPost reports whether post is a `?` in an accepted statement
// shape. It types the operand silently, reports misplaced_try, marks the
// assigned variables used, and returns true; the caller then skips the
// normal check. Otherwise it returns false and the caller runs the
// upstream path.
func (check *Checker) logosForPost(post syntax.Stmt) bool {
	if !check.conf.Logos || post == nil {
		return false
	}
	var lhs []syntax.Expr
	var t *syntax.TryExpr
	switch s := post.(type) {
	case *syntax.AssignStmt:
		if s.Rhs == nil || s.Op != 0 {
			return false
		}
		rhs := syntax.UnpackListExpr(s.Rhs)
		if len(rhs) != 1 {
			return false
		}
		var ok bool
		t, ok = syntax.Unparen(rhs[0]).(*syntax.TryExpr)
		if !ok {
			return false
		}
		lhs = syntax.UnpackListExpr(s.Lhs)
	case *syntax.ExprStmt:
		var ok bool
		t, ok = syntax.Unparen(s.X).(*syntax.TryExpr)
		if !ok {
			return false
		}
	default:
		return false
	}
	check.logosTryMisplaced(t)
	if lhs != nil {
		check.useLHS(lhs...)
	}
	return true
}

// logosTryCountError reports a C3 arity mismatch.
func (check *Checker) logosTryCountError(t *syntax.TryExpr, nvals, nvars int) {
	if nvals == 0 {
		check.error(t.Pos(), WrongAssignCount, "? provides no values to assign")
		return
	}
	check.errorf(t.Pos(), WrongAssignCount, "? provides %s for %s", measure(nvals, "value"), measure(nvars, "variable"))
}

// logosTryAssignStmt handles `?` as the single right-hand side of `=`.
// `:=` flows through shortVarDecl, which hooks logosTryInitHook and
// logosTryNoNewVar. It reports true when it handled the statement.
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

// logosTryAssignEq handles `lhs = f()?`.
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

// logosTryInitHook initializes lhsVars from a single `?` right-hand side.
// It reports true when rhs is a `?` operand (handled, even on failure);
// otherwise it reports false and the caller runs the upstream path.
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

// logosTryNoNewVar reports the redeclare error for a `?` declaration at
// the first variable (the frozen anchor); other declarations use the
// upstream position.
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

// logosTryInitVars initializes lhsVars from the stripped `?` results,
// reporting C3 arity errors instead of the generic mismatch.
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

// logosTryExprStmt handles `f()?` as an expression statement. Any
// remaining count is allowed; the values are discarded.
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

// logosTryCallStmt handles `?` as the call of a go or defer statement.
// A `?` in a go statement is never an allowed position (SPEC Sec 1.4),
// so it always reports misplaced_try; a `defer f()?` discards any
// remaining count like an expression statement.
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

// logosErrdefer checks an errdefer statement (C5): the enclosing function
// must return error last, and the cleanup must be a call, optionally
// wrapped in `?` (which adds C1).
func (check *Checker) logosErrdefer(s *syntax.ErrdeferStmt) {
	if !check.conf.Logos {
		check.error(s, InvalidSyntaxTree, "errdefer requires logos mode")
		return
	}
	if !check.logosEnclosingError() {
		check.error(s, InvalidDefer, "[correctness::errdefer_needs_error_result] errdefer requires enclosing function to return error")
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
		// The parser already reported "errdefer requires a function
		// call"; mark uses without adding another error.
		check.use(s.Call)
	}
}

// logosErrdeferUse types a cleanup call for its uses after an
// enclosing-result error, without `?` validation repeating it.
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

// logosTryOperand checks a `?` in expression position, which no statement
// hook claimed. The position is never allowed (SPEC Sec 1.4), so it always
// reports misplaced_try; the operand is typed silently (uses marked,
// inner call errors surface) without C1 diagnostics, so each TryExpr
// yields exactly the one position error.
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
	return statement // avoid follow-up errors
}
