// Copyright 2026 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package types2

import (
	"cmd/compile/internal/syntax"
	. "internal/types/errors"
)

const logosLambdaNeedsTarget = "[correctness::lambda_needs_target] lambda needs a target function type"

func (check *Checker) logosLambdaDefer(x *operand, e syntax.Expr) bool {
	if !check.conf.Logos {
		return false
	}
	lam, ok := syntax.Unparen(e).(*syntax.LambdaExpr)
	if !ok {
		return false
	}
	x.mode_ = nilvalue
	x.expr = e
	x.typ_ = Typ[UntypedNil]
	x.val = nil
	if check.logosLambdaPending == nil {
		check.logosLambdaPending = make(map[*syntax.LambdaExpr]bool)
	}
	check.logosLambdaPending[lam] = true
	check.later(func() {
		if check.logosLambdaPending[lam] {
			delete(check.logosLambdaPending, lam)
			check.logosLambdaNoTarget(lam)
		}
	})
	return true
}

func (check *Checker) logosLambdaResolve(x *operand, T Type) {
	if x.mode_ != nilvalue {
		return
	}
	lam, ok := syntax.Unparen(x.expr).(*syntax.LambdaExpr)
	if !ok || !check.logosLambdaPending[lam] {
		return
	}
	delete(check.logosLambdaPending, lam)
	e := x.expr
	t := newTarget(T, "")
	if t == nil {
		check.logosLambdaNoTarget(lam)
		x.invalidate()
		return
	}
	check.logosLambdaTargeted(x, t.sig, lam)
	if x.isValid() {
		x.expr = e
		check.record(x)
	}
}

func (check *Checker) logosLambdaNoTarget(lam *syntax.LambdaExpr) {
	check.errorf(lam.Arrow, IncompatibleAssign, "%s", logosLambdaNeedsTarget)
}

func logosTargetType(typ Type) Type {
	if o := logosAsOption(typ); o != nil {
		return o.Elem
	}
	return typ
}

func (check *Checker) logosAppendArgs(id builtinId, argList []syntax.Expr, args *[]*operand) bool {
	if !check.conf.Logos || id != _Append || len(argList) < 2 {
		return false
	}
	list := make([]*operand, len(argList))
	for i, e := range argList {
		var x operand
		if !check.logosLambdaDefer(&x, e) {
			check.expr(nil, &x, e)
		}
		list[i] = &x
	}
	*args = list
	return true
}

func (check *Checker) logosLambdaTargeted(x *operand, sig *Signature, e *syntax.LambdaExpr) exprKind {
	fail := func() exprKind {
		x.mode_ = invalid
		x.expr = e
		x.typ_ = Typ[Invalid]
		return expression
	}
	if len(e.Params) != sig.params.Len() {
		check.errorf(e.Arrow, IncompatibleAssign, "cannot use %s as %s: parameter count mismatch (have %d, want %d)", ExprString(e), sig, len(e.Params), sig.params.Len())
		return fail()
	}
	scope := NewScope(check.scope, e.Pos(), endPos(e.Body), "lambda")
	scope.isFunc = true
	params := make([]*Var, len(e.Params))
	for i, name := range e.Params {
		p := NewParam(name.Pos(), check.pkg, name.Value, sig.params.At(i).Type())
		params[i] = p
		check.declare(scope, name, p, name.Pos())
	}
	lamsig := NewSignatureType(nil, nil, nil, NewTuple(params...), sig.results, sig.variadic)
	lamsig.scope = scope
	lamsig.scope.pos = e.Pos()
	lamsig.scope.end = endPos(e.Body)
	check.logosChainNestedTry(e.Body)
	saved := check.environment
	check.environment = environment{
		decl:    check.decl,
		scope:   scope,
		version: check.version,
		iota:    check.iota,
		sig:     lamsig,
	}
	defer func() { check.environment = saved }()
	switch body := e.Body.(type) {
	case *syntax.BlockStmt:
		check.logosEnterLit(body)
		defer check.logosExitFunc()
		check.stmtList(0, body.List)
		if check.hasLabel && !check.conf.IgnoreBranchErrors {
			check.labels(body)
		}
		if sig.results.Len() > 0 && !check.isTerminating(body, "") {
			check.error(body.Rbrace, MissingReturn, "missing return")
		}
		check.usage(scope)
	default:
		expr, ok := e.Body.(syntax.Expr)
		if !ok {
			check.error(e.Body, InvalidSyntaxTree, "invalid lambda body")
			return fail()
		}
		if sig.results.Len() == 0 {
			synth := &syntax.ExprStmt{X: expr}
			synth.SetPos(expr.Pos())
			check.stmt(0, synth)
		} else {
			synth := &syntax.ReturnStmt{Results: expr}
			synth.SetPos(expr.Pos())
			check.initVars(sig.results.vars, []syntax.Expr{expr}, synth)
		}
		check.usage(scope)
	}
	check.environment = saved
	x.mode_ = value
	x.expr = e
	x.typ_ = sig
	check.record(x)
	return expression
}

func (check *Checker) logosLambda(x *operand, t *target, e *syntax.LambdaExpr) exprKind {
	fail := func() exprKind {
		x.mode_ = invalid
		x.expr = e
		x.typ_ = Typ[Invalid]
		return expression
	}
	if !check.conf.Logos {
		check.error(e, InvalidSyntaxTree, "unexpected lambda expression")
		return fail()
	}
	if t == nil || t.sig == nil {
		check.logosLambdaNoTarget(e)
		return fail()
	}
	return check.logosLambdaTargeted(x, t.sig, e)
}
