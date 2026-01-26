// Copyright 2013 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// This file implements initialization and assignment checks.

package types2

import (
	"cmd/compile/internal/syntax"
	"fmt"
	. "internal/types/errors"
	"strings"
)

// assignment reports whether x can be assigned to a variable of type T,
// if necessary by attempting to convert untyped values to the appropriate
// type. context describes the context in which the assignment takes place.
// Use T == nil to indicate assignment to an untyped blank identifier.
// If the assignment check fails, x.mode is set to invalid.
func (check *Checker) assignment(x *operand, T Type, context string) {
	check.singleValue(x)

	switch x.mode {
	case invalid:
		return // error reported before
	case nilvalue:
		assert(isTypes2)
		// ok
	case constant_, variable, mapindex, value, commaok, commaerr:
		// ok
	default:
		// we may get here because of other problems (go.dev/issue/39634, crash 12)
		// TODO(gri) do we need a new "generic" error code here?
		check.errorf(x, IncompatibleAssign, "cannot assign %s to %s in %s", x, T, context)
		x.mode = invalid
		return
	}

	if isUntyped(x.typ) {
		target := T
		// spec: "If an untyped constant is assigned to a variable of interface
		// type or the blank identifier, the constant is first converted to type
		// bool, rune, int, float64, complex128 or string respectively, depending
		// on whether the value is a boolean, rune, integer, floating-point,
		// complex, or string constant."
		if isTypes2 {
			if x.isNil() {
				if T == nil {
					check.errorf(x, UntypedNilUse, "use of untyped nil in %s", context)
					x.mode = invalid
					return
				}
			} else if T == nil || isNonTypeParamInterface(T) {
				target = Default(x.typ)
			}
		} else { // go/types
			if T == nil || isNonTypeParamInterface(T) {
				if T == nil && x.typ == Typ[UntypedNil] {
					check.errorf(x, UntypedNilUse, "use of untyped nil in %s", context)
					x.mode = invalid
					return
				}
				target = Default(x.typ)
			}
		}
		newType, val, code := check.implicitTypeAndValue(x, target)
		if code != 0 {
			msg := check.sprintf("cannot use %s as %s value in %s", x, target, context)
			switch code {
			case TruncatedFloat:
				msg += " (truncated)"
			case NumericOverflow:
				msg += " (overflows)"
			default:
				code = IncompatibleAssign
			}
			check.error(x, code, msg)
			x.mode = invalid
			return
		}
		if val != nil {
			x.val = val
			check.updateExprVal(x.expr, val)
		}
		if newType != x.typ {
			x.typ = newType
			check.updateExprType(x.expr, newType, false)
		}
	}
	// x.typ is typed

	// A generic (non-instantiated) function value cannot be assigned to a variable.
	if sig, _ := x.typ.Underlying().(*Signature); sig != nil && sig.TypeParams().Len() > 0 {
		check.errorf(x, WrongTypeArgCount, "cannot use generic function %s without instantiation in %s", x, context)
		x.mode = invalid
		return
	}

	// spec: "If a left-hand side is the blank identifier, any typed or
	// non-constant value except for the predeclared identifier nil may
	// be assigned to it."
	if T == nil {
		return
	}

	cause := ""
	if ok, code := x.assignableTo(check, T, &cause); !ok {
		if cause != "" {
			check.errorf(x, code, "cannot use %s as %s value in %s: %s", x, T, context, cause)
		} else {
			check.errorf(x, code, "cannot use %s as %s value in %s", x, T, context)
		}
		x.mode = invalid
	}
}

func (check *Checker) initConst(lhs *Const, x *operand) {
	if x.mode == invalid || !isValid(x.typ) || !isValid(lhs.typ) {
		if lhs.typ == nil {
			lhs.typ = Typ[Invalid]
		}
		return
	}

	// rhs must be a constant
	if x.mode != constant_ {
		check.errorf(x, InvalidConstInit, "%s is not constant", x)
		if lhs.typ == nil {
			lhs.typ = Typ[Invalid]
		}
		return
	}
	assert(isConstType(x.typ))

	// If the lhs doesn't have a type yet, use the type of x.
	if lhs.typ == nil {
		lhs.typ = x.typ
	}

	check.assignment(x, lhs.typ, "constant declaration")
	if x.mode == invalid {
		return
	}

	lhs.val = x.val
}

// initVar checks the initialization lhs = x in a variable declaration.
// If lhs doesn't have a type yet, it is given the type of x,
// or Typ[Invalid] in case of an error.
// If the initialization check fails, x.mode is set to invalid.
func (check *Checker) initVar(lhs *Var, x *operand, context string) {
	if x.mode == invalid || !isValid(x.typ) || !isValid(lhs.typ) {
		if lhs.typ == nil {
			lhs.typ = Typ[Invalid]
		}
		x.mode = invalid
		return
	}

	// If lhs doesn't have a type yet, use the type of x.
	if lhs.typ == nil {
		typ := x.typ
		if isUntyped(typ) {
			// convert untyped types to default types
			if typ == Typ[UntypedNil] {
				check.errorf(x, UntypedNilUse, "use of untyped nil in %s", context)
				lhs.typ = Typ[Invalid]
				x.mode = invalid
				return
			}
			typ = Default(typ)
		}
		lhs.typ = typ
	}

	check.assignment(x, lhs.typ, context)
}

// lhsVar checks a lhs variable in an assignment and returns its type.
// lhsVar takes care of not counting a lhs identifier as a "use" of
// that identifier. The result is nil if it is the blank identifier,
// and Typ[Invalid] if it is an invalid lhs expression.
func (check *Checker) lhsVar(lhs syntax.Expr) Type {
	// Determine if the lhs is a (possibly parenthesized) identifier.
	ident, _ := syntax.Unparen(lhs).(*syntax.Name)

	// Don't evaluate lhs if it is the blank identifier.
	if ident != nil && ident.Value == "_" {
		check.recordDef(ident, nil)
		return nil
	}

	// If the lhs is an identifier denoting a variable v, this reference
	// is not a 'use' of v. Remember current value of v.used and restore
	// after evaluating the lhs via check.expr.
	var v *Var
	var v_used bool
	if ident != nil {
		if obj := check.lookup(ident.Value); obj != nil {
			// It's ok to mark non-local variables, but ignore variables
			// from other packages to avoid potential race conditions with
			// dot-imported variables.
			if w, _ := obj.(*Var); w != nil && w.pkg == check.pkg {
				v = w
				v_used = check.usedVars[v]
			}
		}
	}

	var x operand
	check.expr(nil, &x, lhs)

	if v != nil {
		check.usedVars[v] = v_used // restore v.used
	}

	if x.mode == invalid || !isValid(x.typ) {
		return Typ[Invalid]
	}

	// spec: "Each left-hand side operand must be addressable, a map index
	// expression, or the blank identifier. Operands may be parenthesized."
	switch x.mode {
	case invalid:
		return Typ[Invalid]
	case variable, mapindex:
		// ok
	default:
		if sel, ok := x.expr.(*syntax.SelectorExpr); ok {
			var op operand
			check.expr(nil, &op, sel.X)
			if op.mode == mapindex {
				check.errorf(&x, UnaddressableFieldAssign, "cannot assign to struct field %s in map", ExprString(x.expr))
				return Typ[Invalid]
			}
		}
		check.errorf(&x, UnassignableOperand, "cannot assign to %s (neither addressable nor a map index expression)", x.expr)
		return Typ[Invalid]
	}

	return x.typ
}

// assignVar checks the assignment lhs = rhs (if x == nil), or lhs = x (if x != nil).
// If x != nil, it must be the evaluation of rhs (and rhs will be ignored).
// If the assignment check fails and x != nil, x.mode is set to invalid.
func (check *Checker) assignVar(lhs, rhs syntax.Expr, x *operand, context string) {
	T := check.lhsVar(lhs) // nil if lhs is _
	if !isValid(T) {
		if x != nil {
			x.mode = invalid
		} else {
			check.use(rhs)
		}
		return
	}

	if x == nil {
		var target *target
		// avoid calling ExprString if not needed
		if T != nil {
			if _, ok := T.Underlying().(*Signature); ok {
				target = newTarget(T, ExprString(lhs))
			}
		}
		x = new(operand)
		check.expr(target, x, rhs)
	}

	if T == nil && context == "assignment" {
		context = "assignment to _ identifier"
	}
	check.assignment(x, T, context)
}

// operandTypes returns the list of types for the given operands.
func operandTypes(list []*operand) (res []Type) {
	for _, x := range list {
		res = append(res, x.typ)
	}
	return res
}

// varTypes returns the list of types for the given variables.
func varTypes(list []*Var) (res []Type) {
	for _, x := range list {
		res = append(res, x.typ)
	}
	return res
}

// typesSummary returns a string of the form "(t1, t2, ...)" where the
// ti's are user-friendly string representations for the given types.
// If variadic is set and the last type is a slice, its string is of
// the form "...E" where E is the slice's element type.
// If hasDots is set, the last argument string is of the form "T..."
// where T is the last type.
// Only one of variadic and hasDots may be set.
func (check *Checker) typesSummary(list []Type, variadic, hasDots bool) string {
	assert(!(variadic && hasDots))
	var res []string
	for i, t := range list {
		var s string
		switch {
		case t == nil:
			fallthrough // should not happen but be cautious
		case !isValid(t):
			s = "unknown type"
		case isUntyped(t): // => *Basic
			if isNumeric(t) {
				// Do not imply a specific type requirement:
				// "have number, want float64" is better than
				// "have untyped int, want float64" or
				// "have int, want float64".
				s = "number"
			} else {
				// If we don't have a number, omit the "untyped" qualifier
				// for compactness.
				s = strings.ReplaceAll(t.(*Basic).name, "untyped ", "")
			}
		default:
			s = check.sprintf("%s", t)
		}
		// handle ... parameters/arguments
		if i == len(list)-1 {
			switch {
			case variadic:
				// In correct code, the parameter type is a slice, but be careful.
				if t, _ := t.(*Slice); t != nil {
					s = check.sprintf("%s", t.elem)
				}
				s = "..." + s
			case hasDots:
				s += "..."
			}
		}
		res = append(res, s)
	}
	return "(" + strings.Join(res, ", ") + ")"
}

func measure(x int, unit string) string {
	if x != 1 {
		unit += "s"
	}
	return fmt.Sprintf("%d %s", x, unit)
}

func (check *Checker) assignError(rhs []syntax.Expr, l, r int) {
	vars := measure(l, "variable")
	vals := measure(r, "value")
	rhs0 := rhs[0]

	if len(rhs) == 1 {
		if call, _ := syntax.Unparen(rhs0).(*syntax.CallExpr); call != nil {
			check.errorf(rhs0, WrongAssignCount, "assignment mismatch: %s but %s returns %s", vars, call.Fun, vals)
			return
		}
	}
	check.errorf(rhs0, WrongAssignCount, "assignment mismatch: %s but %s", vars, vals)
}

func (check *Checker) returnError(at poser, lhs []*Var, rhs []*operand) {
	l, r := len(lhs), len(rhs)
	qualifier := "not enough"
	if r > l {
		at = rhs[l] // report at first extra value
		qualifier = "too many"
	} else if r > 0 {
		at = rhs[r-1] // report at last value
	}
	err := check.newError(WrongResultCount)
	err.addf(at, "%s return values", qualifier)
	err.addf(nopos, "have %s", check.typesSummary(operandTypes(rhs), false, false))
	err.addf(nopos, "want %s", check.typesSummary(varTypes(lhs), false, false))
	err.report()
}

// initVars type-checks assignments of initialization expressions orig_rhs
// to variables lhs.
// If returnStmt is non-nil, initVars type-checks the implicit assignment
// of result expressions orig_rhs to function result parameters lhs.
func (check *Checker) initVars(lhs []*Var, orig_rhs []syntax.Expr, returnStmt syntax.Stmt) {
	context := "assignment"
	if returnStmt != nil {
		context = "return statement"
	}

	l, r := len(lhs), len(orig_rhs)

	// If l == 1 and the rhs is a single call, for a better
	// error message don't handle it as n:n mapping below.
	isCall := false
	if r == 1 {
		_, isCall = syntax.Unparen(orig_rhs[0]).(*syntax.CallExpr)
	}

	// If we have a n:n mapping from lhs variable to rhs expression,
	// each value can be assigned to its corresponding variable.
	if l == r && !isCall {
		var x operand
		for i, lhs := range lhs {
			desc := lhs.name
			if returnStmt != nil && desc == "" {
				desc = "result variable"
			}
			check.expr(newTarget(lhs.typ, desc), &x, orig_rhs[i])
			check.initVar(lhs, &x, context)
		}
		return
	}

	// If we don't have an n:n mapping, the rhs must be a single expression
	// resulting in 2 or more values; otherwise we have an assignment mismatch.
	if r != 1 {
		// Only report a mismatch error if there are no other errors on the rhs.
		if check.use(orig_rhs...) {
			if returnStmt != nil {
				rhs := check.exprList(orig_rhs)
				check.returnError(returnStmt, lhs, rhs)
			} else {
				check.assignError(orig_rhs, l, r)
			}
		}
		// ensure that LHS variables have a type
		for _, v := range lhs {
			if v.typ == nil {
				v.typ = Typ[Invalid]
			}
		}
		return
	}

	rhs, commaOk := check.multiExpr(orig_rhs[0], l == 2 && returnStmt == nil)
	r = len(rhs)
	if l == r {
		for i, lhs := range lhs {
			check.initVar(lhs, rhs[i], context)
		}
		// Only record comma-ok expression if both initializations succeeded
		// (go.dev/issue/59371).
		if commaOk && rhs[0].mode != invalid && rhs[1].mode != invalid {
			check.recordCommaOkTypes(orig_rhs[0], rhs)
		}
		return
	}

	// In all other cases we have an assignment mismatch.
	// Only report a mismatch error if there are no other errors on the rhs.
	if rhs[0].mode != invalid {
		if returnStmt != nil {
			check.returnError(returnStmt, lhs, rhs)
		} else {
			check.assignError(orig_rhs, l, r)
		}
	}
	// ensure that LHS variables have a type
	for _, v := range lhs {
		if v.typ == nil {
			v.typ = Typ[Invalid]
		}
	}
	// orig_rhs[0] was already evaluated
}

// assignVars type-checks assignments of expressions orig_rhs to variables lhs.
func (check *Checker) assignVars(lhs, orig_rhs []syntax.Expr) {
	l, r := len(lhs), len(orig_rhs)

	// If l == 1 and the rhs is a single call, for a better
	// error message don't handle it as n:n mapping below.
	isCall := false
	if r == 1 {
		_, isCall = syntax.Unparen(orig_rhs[0]).(*syntax.CallExpr)
	}

	// If we have a n:n mapping from lhs variable to rhs expression,
	// each value can be assigned to its corresponding variable.
	if l == r && !isCall {
		for i, lhs := range lhs {
			check.assignVar(lhs, orig_rhs[i], nil, "assignment")
		}
		return
	}

	// If we don't have an n:n mapping, the rhs must be a single expression
	// resulting in 2 or more values; otherwise we have an assignment mismatch.
	if r != 1 {
		// Only report a mismatch error if there are no other errors on the lhs or rhs.
		okLHS := check.useLHS(lhs...)
		okRHS := check.use(orig_rhs...)
		if okLHS && okRHS {
			check.assignError(orig_rhs, l, r)
		}
		return
	}

	rhs, commaOk := check.multiExpr(orig_rhs[0], l == 2)
	r = len(rhs)
	if l == r {
		for i, lhs := range lhs {
			check.assignVar(lhs, nil, rhs[i], "assignment")
		}
		// Only record comma-ok expression if both assignments succeeded
		// (go.dev/issue/59371).
		if commaOk && rhs[0].mode != invalid && rhs[1].mode != invalid {
			check.recordCommaOkTypes(orig_rhs[0], rhs)
		}
		return
	}

	// In all other cases we have an assignment mismatch.
	// Only report a mismatch error if there are no other errors on the rhs.
	if rhs[0].mode != invalid {
		check.assignError(orig_rhs, l, r)
	}
	check.useLHS(lhs...)
	// orig_rhs[0] was already evaluated
}

func (check *Checker) shortVarDecl(pos poser, lhs, rhs []syntax.Expr) {
	top := len(check.delayed)
	scope := check.scope

	// collect lhs variables
	seen := make(map[string]bool, len(lhs))
	lhsVars := make([]*Var, len(lhs))
	newVars := make([]*Var, 0, len(lhs))
	hasErr := false
	for i, lhs := range lhs {
		ident, _ := lhs.(*syntax.Name)
		if ident == nil {
			check.useLHS(lhs)
			// TODO(gri) This is redundant with a go/parser error. Consider omitting in go/types?
			check.errorf(lhs, BadDecl, "non-name %s on left side of :=", lhs)
			hasErr = true
			continue
		}

		name := ident.Value
		if name != "_" {
			if seen[name] {
				check.errorf(lhs, RepeatedDecl, "%s repeated on left side of :=", lhs)
				hasErr = true
				continue
			}
			seen[name] = true
		}

		// Use the correct obj if the ident is redeclared. The
		// variable's scope starts after the declaration; so we
		// must use Scope.Lookup here and call Scope.Insert
		// (via check.declare) later.
		if alt := scope.Lookup(name); alt != nil {
			check.recordUse(ident, alt)
			// redeclared object must be a variable
			if obj, _ := alt.(*Var); obj != nil {
				lhsVars[i] = obj
			} else {
				check.errorf(lhs, UnassignableOperand, "cannot assign to %s", lhs)
				hasErr = true
			}
			continue
		}

		// declare new variable
		obj := newVar(LocalVar, ident.Pos(), check.pkg, name, nil)
		lhsVars[i] = obj
		if name != "_" {
			newVars = append(newVars, obj)
		}
		check.recordDef(ident, obj)
	}

	// create dummy variables where the lhs is invalid
	for i, obj := range lhsVars {
		if obj == nil {
			lhsVars[i] = newVar(LocalVar, lhs[i].Pos(), check.pkg, "_", nil)
		}
	}

	check.initVars(lhsVars, rhs, nil)

	// process function literals in rhs expressions before scope changes
	check.processDelayed(top)

	if len(newVars) == 0 && !hasErr {
		check.softErrorf(pos, NoNewVar, "no new variables on left side of :=")
		return
	}

	// declare new variables
	// spec: "The scope of a constant or variable identifier declared inside
	// a function begins at the end of the ConstSpec or VarSpec (ShortVarDecl
	// for short variable declarations) and ends at the end of the innermost
	// containing block."
	scopePos := endPos(rhs[len(rhs)-1])
	for _, obj := range newVars {
		check.declare(scope, nil, obj, scopePos) // id = nil: recordDef already called
	}
}
