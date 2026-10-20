// Copyright 2026 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package types2

import (
	"cmd/compile/internal/syntax"
	. "internal/types/errors"
)

// LoGo/s M6-S7 type rules for collection chains (SPEC Sec 1.4,
// T-104.6). Upstream files call in through hook lines only: call.go
// (chain calls). Every rule below applies only to a package of .lgo
// files (Config.Logos).
//
// Chain methods live on slice types: filter, map, find, any, all,
// sortBy, first, last, get, take, drop, sum, and count. Lambdas take
// const-view parameters; mutating the receiver or a parameter fails;
// `?` in an expression-lambda body propagates from the enclosing
// function. Lowering lives in logos/chains.go.

const logosChainMutation = "[correctness::const_view_mutation]"

const logosSideEffectBeforeSort = "[suspicious::side_effect_before_sort]"

// logosChainMethods is the set of chain method names.
var logosChainMethods = map[string]bool{
	"filter": true,
	"map":    true,
	"find":   true,
	"any":    true,
	"all":    true,
	"sortBy": true,
	"first":  true,
	"last":   true,
	"get":    true,
	"take":   true,
	"drop":   true,
	"sum":    true,
	"count":  true,
}

// logosChainCall checks a chain method call on a slice. It reports true
// when handled, even on failure; otherwise it reports false and the
// caller runs the upstream path.
func (check *Checker) logosChainCall(x *operand, call *syntax.CallExpr) bool {
	if !check.conf.Logos {
		return false
	}
	sel, ok := syntax.Unparen(call.Fun).(*syntax.SelectorExpr)
	if !ok {
		return false
	}
	if !logosChainMethods[sel.Sel.Value] {
		return false
	}
	var rx operand
	check.expr(nil, &rx, sel.X)
	if !rx.isValid() {
		x.mode_ = invalid
		x.expr = call
		x.typ_ = Typ[Invalid]
		return true
	}
	// A chain only reads its receiver, so a const slice view (SPEC T-201)
	// chains like the plain slice; the qualifier is erased before code
	// generation. Mutating uses stay rejected inside the lambdas.
	rt := rx.typ()
	if c := logosAsConstView(rt); c != nil {
		rt = c.Elem
	}
	if obj, _, _ := lookupFieldOrMethod(rt, rx.mode() == variable, check.pkg, sel.Sel.Value, false); obj != nil {
		return false
	}
	st, ok := Unalias(rt).(*Slice)
	if !ok {
		return false
	}
	check.logosChainMethod(x, call, sel, st.Elem())
	return true
}

// logosChainRoots reports the assignable roots of a receiver: its
// leftmost name, if any.
func logosChainRoots(e syntax.Expr) map[string]bool {
	roots := make(map[string]bool)
	cur := syntax.Unparen(e)
	for {
		if nm, ok := cur.(*syntax.Name); ok {
			roots[nm.Value] = true
			return roots
		}
		if s, ok := cur.(*syntax.SelectorExpr); ok {
			cur = syntax.Unparen(s.X)
			continue
		}
		if ix, ok := cur.(*syntax.IndexExpr); ok {
			cur = syntax.Unparen(ix.X)
			continue
		}
		return roots
	}
}

// logosChainMethod checks one chain method with element type elem.
func (check *Checker) logosChainMethod(x *operand, call *syntax.CallExpr, sel *syntax.SelectorExpr, elem Type) {
	method := sel.Sel.Value
	roots := logosChainRoots(sel.X)
	fail := func() {
		x.mode_ = invalid
		x.expr = call
		x.typ_ = Typ[Invalid]
	}
	done := func(t Type) {
		x.mode_ = value
		x.expr = call
		x.typ_ = t
		check.hasCallOrRecv = true
	}
	args := call.ArgList
	switch method {
	case "filter", "find", "any", "all":
		if !check.logosChainArity(call, args, 1) {
			fail()
			return
		}
		u, _, ok := check.logosChainPred(args[0], elem, method, roots)
		if !ok {
			fail()
			return
		}
		_ = u
		switch method {
		case "filter":
			done(NewSlice(elem))
		case "find":
			done(logosWrapOption(elem))
		default:
			done(Typ[Bool])
		}
	case "map":
		if !check.logosChainArity(call, args, 1) {
			fail()
			return
		}
		u, _, ok := check.logosChainMapArg(args[0], elem, roots)
		if !ok {
			fail()
			return
		}
		done(NewSlice(u))
	case "sortBy":
		if !check.logosChainArity(call, args, 1) {
			fail()
			return
		}
		_, _, ok := check.logosChainKeyArg(args[0], elem, roots)
		if !ok {
			fail()
			return
		}
		check.logosWarnSideEffectBeforeSort(sel, sel.X)
		done(NewSlice(elem))
	case "first", "last":
		if !check.logosChainArity(call, args, 0) {
			fail()
			return
		}
		done(logosWrapOption(elem))
	case "get":
		if !check.logosChainArity(call, args, 1) {
			fail()
			return
		}
		if !check.logosChainIntArg(args[0], method) {
			fail()
			return
		}
		done(logosWrapOption(elem))
	case "take", "drop":
		if !check.logosChainArity(call, args, 1) {
			fail()
			return
		}
		if !check.logosChainIntArg(args[0], method) {
			fail()
			return
		}
		done(NewSlice(elem))
	case "sum":
		if !check.logosChainArity(call, args, 0) {
			fail()
			return
		}
		if b, ok := Unalias(elem).(*Basic); !ok || b.Info()&IsNumeric == 0 {
			check.errorf(call, IncompatibleAssign, "cannot sum non-numeric element type %s", elem)
			fail()
			return
		}
		done(elem)
	case "count":
		if len(args) == 0 {
			done(Typ[Int])
			return
		}
		if !check.logosChainArity(call, args, 1) {
			fail()
			return
		}
		if _, _, ok := check.logosChainPred(args[0], elem, method, roots); !ok {
			fail()
			return
		}
		done(Typ[Int])
	}
}

// logosChainArity validates the argument count of a chain call.
func (check *Checker) logosChainArity(call *syntax.CallExpr, args []syntax.Expr, want int) bool {
	if hasDots(call) {
		check.errorf(call, InvalidDotDotDot, "invalid use of ... in chain call to %s", call.Fun)
		return false
	}
	if len(args) != want {
		check.errorf(call, WrongArgCount, "wrong number of arguments in chain call to %s: have %d, want %d", call.Fun, len(args), want)
		return false
	}
	return true
}

// logosChainIntArg checks an integer index argument.
func (check *Checker) logosChainIntArg(arg syntax.Expr, method string) bool {
	var op operand
	check.expr(nil, &op, arg)
	if !op.isValid() {
		return false
	}
	if b, ok := Unalias(op.typ()).(*Basic); !ok || b.Info()&IsInteger == 0 {
		check.errorf(&op, IncompatibleAssign, "chain %s index must be an integer, have %s", method, op.typ())
		return false
	}
	return true
}

// logosChainPred checks a predicate lambda or function value against
// func(elem) bool. It returns the result type for callers that need it.
func (check *Checker) logosChainPred(arg syntax.Expr, elem Type, method string, roots map[string]bool) (Type, bool, bool) {
	if lam, ok := syntax.Unparen(arg).(*syntax.LambdaExpr); ok {
		return check.logosChainLambda(lam, elem, logosPredKind, roots)
	}
	return check.logosChainFuncArg(arg, elem, Typ[Bool], method)
}

// logosChainMapArg checks a mapper, inferring the element type.
func (check *Checker) logosChainMapArg(arg syntax.Expr, elem Type, roots map[string]bool) (Type, bool, bool) {
	if lam, ok := syntax.Unparen(arg).(*syntax.LambdaExpr); ok {
		return check.logosChainLambda(lam, elem, logosMapKind, roots)
	}
	return check.logosChainFuncInfer(arg, elem)
}

// logosChainKeyArg checks a sort key, inferring the ordered key type.
func (check *Checker) logosChainKeyArg(arg syntax.Expr, elem Type, roots map[string]bool) (Type, bool, bool) {
	if lam, ok := syntax.Unparen(arg).(*syntax.LambdaExpr); ok {
		return check.logosChainLambda(lam, elem, logosKeyKind, roots)
	}
	u, isTry, ok := check.logosChainFuncInfer(arg, elem)
	if !ok {
		return nil, false, false
	}
	if !logosOrderedKey(u) {
		check.errorf(arg, IncompatibleAssign, "chain sortBy key must be ordered, have %s", u)
		return nil, false, false
	}
	return u, isTry, true
}

// logosPredKind, logosMapKind, logosKeyKind select the chain lambda result rule.
type logosChainKind int

const (
	logosPredKind logosChainKind = iota
	logosMapKind
	logosKeyKind
)

// logosIsTryChain reports whether call is a chain call with a `?`
// expression-lambda argument: lowering it needs an error hoist in the
// enclosing function. It returns the offending `?` for diagnostics.
func logosIsTryChain(call *syntax.CallExpr) *syntax.TryExpr {
	sel, ok := syntax.Unparen(call.Fun).(*syntax.SelectorExpr)
	if !ok || sel.Sel == nil || !logosChainMethods[sel.Sel.Value] {
		return nil
	}
	for _, arg := range call.ArgList {
		lam, ok := syntax.Unparen(arg).(*syntax.LambdaExpr)
		if !ok {
			continue
		}
		if lb, ok := lam.Body.(syntax.Expr); ok {
			if tb, ok := syntax.Unparen(lb).(*syntax.TryExpr); ok {
				return tb
			}
		}
	}
	return nil
}

// logosChainNestedTry rejects a `?`-chain nested in a lambda body
// outside a nested function literal: lowering hoists `?`-chains to the
// enclosing function, which would lift the nested chain out of the
// per-element closure and unbind its captured parameters.
func (check *Checker) logosChainNestedTry(body syntax.Node) {
	syntax.Inspect(body, func(n syntax.Node) bool {
		if _, ok := n.(*syntax.FuncLit); ok {
			return false
		}
		if call, ok := n.(*syntax.CallExpr); ok {
			if tb := logosIsTryChain(call); tb != nil {
				check.error(tb.Pos(), IncompatibleAssign, "[correctness::nested_try_capture] ? cannot propagate from a nested chain")
				return false
			}
		}
		return true
	})
}

// logosRejectTryChainUnder rejects a `?`-chain in a conditional option
// position (a `??` fallback) outside a nested function literal: the
// wrapper evaluates it only when the option is absent, so hoisting it
// to the enclosing function would run it eagerly and propagate an
// error the source never raises. Left operands and `?.` receivers
// evaluate unconditionally and lower through transparent wrappers.
func (check *Checker) logosRejectTryChainUnder(e syntax.Expr) {
	syntax.Inspect(e, func(n syntax.Node) bool {
		if _, ok := n.(*syntax.FuncLit); ok {
			return false
		}
		if call, ok := n.(*syntax.CallExpr); ok {
			if tb := logosIsTryChain(call); tb != nil {
				check.error(tb.Pos(), IncompatibleAssign, "[correctness::nested_try_capture] ? cannot propagate from a nested chain")
				return false
			}
		}
		return true
	})
}

// logosChainLambda checks a chain lambda with element type elem. The
// single parameter is a const view; `?` in an expression body
// propagates from the enclosing function. It returns the produced type
// and whether the body ends in `?`.
func (check *Checker) logosChainLambda(lam *syntax.LambdaExpr, elem Type, kind logosChainKind, roots map[string]bool) (Type, bool, bool) {
	if len(lam.Params) != 1 {
		check.errorf(lam.Arrow, IncompatibleAssign, "chain lambda takes exactly one parameter, have %d", len(lam.Params))
		return nil, false, false
	}
	check.logosChainNestedTry(lam.Body)
	name := lam.Params[0]
	merged := make(map[string]bool, len(roots)+1)
	for r := range roots {
		merged[r] = true
	}
	merged[name.Value] = true
	scope := NewScope(check.scope, lam.Pos(), endPos(lam.Body), "chain lambda")
	scope.isFunc = true
	p := NewParam(name.Pos(), check.pkg, name.Value, elem)
	check.declare(scope, name, p, name.Pos())
	saved := check.environment
	check.environment = environment{decl: check.decl, scope: scope, version: check.version, iota: check.iota, sig: check.sig}
	defer func() { check.environment = saved }()
	if block, ok := lam.Body.(*syntax.BlockStmt); ok {
		u, isTry, ok := check.logosChainBlock(lam, merged, scope, elem, kind, block)
		check.usage(scope)
		return u, isTry, ok
	}
	body, ok := lam.Body.(syntax.Expr)
	if !ok {
		check.error(lam.Body, InvalidSyntaxTree, "invalid chain lambda body")
		return nil, false, false
	}
	check.logosChainNoMutate(merged, body)
	if t, ok := syntax.Unparen(body).(*syntax.TryExpr); ok {
		results, ok := check.logosTryResults(t)
		if !ok {
			return nil, false, false
		}
		if len(results) != 1 {
			check.errorf(t.Pos(), WrongResultCount, "chain lambda ? provides %d values, want 1", len(results))
			return nil, false, false
		}
		u := results[0].typ()
		if kind == logosPredKind {
			check.assignment(results[0], Typ[Bool], "chain predicate")
			check.usage(check.scope)
		}
		if kind == logosKeyKind && !logosOrderedKey(u) {
			check.errorf(body, IncompatibleAssign, "chain sortBy key must be ordered, have %s", u)
			return nil, false, false
		}
		if isUntyped(u) {
			u = Default(u)
		}
		check.usage(check.scope)
		return u, true, true
	}
	var op operand
	check.expr(nil, &op, body)
	if !op.isValid() {
		return nil, false, false
	}
	u := op.typ()
	if kind == logosPredKind {
		check.assignment(&op, Typ[Bool], "chain predicate")
		check.usage(scope)
		return Typ[Bool], false, true
	}
	if isUntyped(u) {
		if u == Typ[UntypedNil] {
			check.error(&op, UntypedNilUse, "cannot infer chain element type from nil")
			return nil, false, false
		}
		u = Default(u)
	}
	if kind == logosKeyKind && !logosOrderedKey(u) {
		check.errorf(&op, IncompatibleAssign, "chain sortBy key must be ordered, have %s", u)
		return nil, false, false
	}
	check.usage(scope)
	return u, false, true
}

// logosOrderedKey reports whether u is ordered for sortBy.
func logosOrderedKey(u Type) bool {
	b, ok := Unalias(u).(*Basic)
	return ok && b.Info()&IsOrdered != 0
}

// logosChainBlock checks a block chain lambda with an already built
// scope holding the parameter. Blocks lower through an immediately
// invoked function, so any `?` fails here with misplaced_try: it can
// never propagate from a boundary. Predicates validate against bool;
// mappers and keys infer from top-level returns checked once below.
func (check *Checker) logosChainBlock(lam *syntax.LambdaExpr, roots map[string]bool, scope *Scope, elem Type, kind logosChainKind, block *syntax.BlockStmt) (Type, bool, bool) {
	check.logosChainNoMutate(roots, block)
	syntax.Inspect(block, func(n syntax.Node) bool {
		switch n.(type) {
		case *syntax.FuncLit, *syntax.LambdaExpr:
			return false
		}
		if t, ok := n.(*syntax.TryExpr); ok {
			check.error(t.Pos(), InvalidCall, logosMisplacedTry)
			return false
		}
		return true
	})
	var collected []*operand
	failed := false
	for _, s := range block.List {
		ret, isRet := s.(*syntax.ReturnStmt)
		if !isRet {
			check.stmt(0, s)
			continue
		}
		for _, res := range syntax.UnpackListExpr(ret.Results) {
			var op operand
			check.expr(nil, &op, res)
			if !op.isValid() {
				failed = true
				continue
			}
			if _, ok := op.typ().(*Tuple); ok {
				check.error(res, WrongResultCount, "cannot infer chain element type from multi-value return")
				failed = true
				continue
			}
			collected = append(collected, &op)
		}
	}
	if failed {
		return nil, false, false
	}
	if kind == logosPredKind {
		for _, op := range collected {
			check.assignment(op, Typ[Bool], "chain predicate")
		}
		if !check.isTerminating(block, "") {
			check.error(block.Rbrace, MissingReturn, "missing return")
			return nil, false, false
		}
		return Typ[Bool], false, true
	}
	var types []Type
	for _, op := range collected {
		t := op.typ()
		if isUntyped(t) {
			if t == Typ[UntypedNil] {
				check.error(op.expr, UntypedNilUse, "cannot infer chain element type from nil")
				return nil, false, false
			}
			t = Default(t)
		}
		types = append(types, t)
	}
	if len(types) == 0 {
		check.error(block, IncompatibleAssign, "cannot infer chain element type: no return")
		return nil, false, false
	}
	for _, t := range types[1:] {
		if !Identical(t, types[0]) {
			check.errorf(block, IncompatibleAssign, "inconsistent chain element types %s and %s", types[0], t)
			return nil, false, false
		}
	}
	for _, op := range collected {
		check.assignment(op, types[0], "chain element")
	}
	if !check.isTerminating(block, "") {
		check.error(block.Rbrace, MissingReturn, "missing return")
		return nil, false, false
	}
	if kind == logosKeyKind && !logosOrderedKey(types[0]) {
		check.errorf(block, IncompatibleAssign, "chain sortBy key must be ordered, have %s", types[0])
		return nil, false, false
	}
	return types[0], false, true
}

// logosChainNoMutate rejects assignments to const-view roots (the
// receiver and the lambda parameters) inside a chain lambda body.
func (check *Checker) logosChainNoMutate(roots map[string]bool, body syntax.Node) {
	syntax.Inspect(body, func(n syntax.Node) bool {
		as, ok := n.(*syntax.AssignStmt)
		if !ok {
			return true
		}
		if as.Op == syntax.Def {
			return true
		}
		for _, lhs := range syntax.UnpackListExpr(as.Lhs) {
			if logosChainMutates(lhs, roots) {
				check.errorf(lhs, IncompatibleAssign, "%s cannot assign to chain receiver or parameter", logosChainMutation)
			}
		}
		return true
	})
}

// logosChainMutates reports whether an assignment target mutates a
// const-view root: the name itself, or a selection or index through it.
func logosChainMutates(e syntax.Expr, roots map[string]bool) bool {
	cur := syntax.Unparen(e)
	for {
		if nm, ok := cur.(*syntax.Name); ok {
			return roots[nm.Value]
		}
		if s, ok := cur.(*syntax.SelectorExpr); ok {
			cur = syntax.Unparen(s.X)
			continue
		}
		if ix, ok := cur.(*syntax.IndexExpr); ok {
			cur = syntax.Unparen(ix.X)
			continue
		}
		return false
	}
}

// logosChainFuncArg checks a non-lambda chain argument against
// func(elem) want.
func (check *Checker) logosChainFuncArg(arg syntax.Expr, elem, want Type, method string) (Type, bool, bool) {
	var op operand
	check.expr(nil, &op, arg)
	if !op.isValid() {
		return nil, false, false
	}
	sig, ok := Unalias(op.typ()).(*Signature)
	if !ok || sig.Params().Len() != 1 {
		check.errorf(arg, IncompatibleAssign, "chain %s needs func with one parameter, have %s", method, op.typ())
		return nil, false, false
	}
	fn := NewSignatureType(nil, nil, nil, NewTuple(NewParam(nopos, check.pkg, "", elem)), NewTuple(NewParam(nopos, check.pkg, "", want)), false)
	check.assignment(&op, fn, "chain "+method)
	return want, false, true
}

// logosWarnSideEffectBeforeSort warns against a chain lambda with a
// call of unknown purity before `.sortBy()` (SPEC Sec 1.4, T-102): the
// fused prefix may run its effects an unexpected number of times once
// sorting materializes the run. Only the lambdas of the chain prefix
// (the sortBy receiver, transitively) are scanned; the sortBy key
// lambda itself and the receiver base expression are out of scope. The
// diagnostic is warn-severity like unapplied_struct_default: it reports
// through softErrorf and LogosWarnOnly keeps it out of firstErr.
func (check *Checker) logosWarnSideEffectBeforeSort(sel *syntax.SelectorExpr, recv syntax.Expr) {
	for _, lam := range logosPrefixLambdas(recv) {
		if logosLambdaHasImpureCall(check, lam) {
			check.softErrorf(sel.Sel, IncompatibleAssign, "%s chain lambda with a call of unknown purity before sortBy", logosSideEffectBeforeSort)
			return
		}
	}
}

// logosPrefixLambdas collects the lambda arguments of the chain calls
// forming the receiver prefix, outermost first. The walk follows
// selector calls over known chain methods; anything else ends it.
func logosPrefixLambdas(recv syntax.Expr) []*syntax.LambdaExpr {
	var lams []*syntax.LambdaExpr
	for {
		call, ok := syntax.Unparen(recv).(*syntax.CallExpr)
		if !ok {
			return lams
		}
		sel, ok := syntax.Unparen(call.Fun).(*syntax.SelectorExpr)
		if !ok || sel.Sel == nil || !logosChainMethods[sel.Sel.Value] {
			return lams
		}
		for _, arg := range call.ArgList {
			if lam, ok := syntax.Unparen(arg).(*syntax.LambdaExpr); ok {
				lams = append(lams, lam)
			}
		}
		recv = sel.X
	}
}

// logosLambdaHasImpureCall reports whether a chain lambda body holds a
// call of unknown purity. Known-pure calls are conversions (the callee
// names a type) and builtins without observable effects; every other
// call, including method and package-qualified calls, counts as unknown.
// Nested function literals are pruned: their calls run only when the
// literal is invoked, not per element.
func logosLambdaHasImpureCall(check *Checker, lam *syntax.LambdaExpr) bool {
	found := false
	syntax.Inspect(lam.Body, func(n syntax.Node) bool {
		if found {
			return false
		}
		if _, ok := n.(*syntax.FuncLit); ok {
			return false
		}
		call, ok := n.(*syntax.CallExpr)
		if !ok {
			return true
		}
		if !logosKnownPureCall(check, call) {
			found = true
			return false
		}
		return true
	})
	return found
}

// logosKnownPureCall reports whether a call is known pure: a conversion
// or an effect-free builtin. Copy, clear, delete, and append mutate
// their arguments; panic, recover, print, and println have observable
// effects; all of those count as unknown.
func logosKnownPureCall(check *Checker, call *syntax.CallExpr) bool {
	name, ok := syntax.Unparen(call.Fun).(*syntax.Name)
	if !ok {
		return false
	}
	obj := check.lookup(name.Value)
	if obj == nil {
		return false
	}
	if _, ok := obj.(*TypeName); ok {
		return true
	}
	if b, ok := obj.(*Builtin); ok {
		switch b.Name() {
		case "copy", "clear", "delete", "append", "panic", "recover", "print", "println":
			return false
		}
		return true
	}
	return false
}

// logosChainFuncInfer checks a non-lambda mapper, inferring the element
// type from its single result.
func (check *Checker) logosChainFuncInfer(arg syntax.Expr, elem Type) (Type, bool, bool) {
	var op operand
	check.expr(nil, &op, arg)
	if !op.isValid() {
		return nil, false, false
	}
	sig, ok := Unalias(op.typ()).(*Signature)
	if !ok || sig.Params().Len() != 1 || sig.Results().Len() != 1 {
		check.errorf(arg, IncompatibleAssign, "chain map needs func with one parameter and one result, have %s", op.typ())
		return nil, false, false
	}
	u := sig.Results().At(0).Type()
	fn := NewSignatureType(nil, nil, nil, NewTuple(NewParam(nopos, check.pkg, "", elem)), NewTuple(NewParam(nopos, check.pkg, "", u)), false)
	check.assignment(&op, fn, "chain map")
	return u, false, true
}
