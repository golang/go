// Copyright 2013 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// This file implements initialization and assignment checks.

package types2

import (
	"cmd/compile/internal/syntax"
	"fmt"
	"strings"
)

// assignment reports whether x can be assigned to a variable of type T,
// if necessary by attempting to convert untyped values to the appropriate
// type. context describes the context in which the assignment takes place.
// Use T == nil to indicate assignment to an untyped blank identifier.
// x.mode is set to invalid if the assignment failed.
func (check *Checker) assignment(x *operand, T Type, context string) {
	check.singleValue(x)

	switch x.mode {
	case invalid:
		return // error reported before
	case constant_, variable, mapindex, value, nilvalue, commaok, commaerr:
		// ok
	default:
		// we may get here because of other problems (issue #39634, crash 12)
		check.errorf(x, "cannot assign %s to %s in %s", x, T, context)
		return
	}

	if isUntyped(x.typ) {
		target := T
		// spec: "If an untyped constant is assigned to a variable of interface
		// type or the blank identifier, the constant is first converted to type
		// bool, rune, int, float64, complex128 or string respectively, depending
		// on whether the value is a boolean, rune, integer, floating-point,
		// complex, or string constant."
		if x.isNil() {
			if T == nil {
				check.errorf(x, "use of untyped nil in %s", context)
				x.mode = invalid
				return
			}
		} else if T == nil || isNonTypeParamInterface(T) {
			target = Default(x.typ)
		}
		newType, val, code := check.implicitTypeAndValue(x, target)
		if code != 0 {
			msg := check.sprintf("cannot use %s as %s value in %s", x, target, context)
			switch code {
			case _TruncatedFloat:
				msg += " (truncated)"
			case _NumericOverflow:
				msg += " (overflows)"
			}
			check.error(x, msg)
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
	if sig, _ := under(x.typ).(*Signature); sig != nil && sig.TypeParams().Len() > 0 {
		check.errorf(x, "cannot use generic function %s without instantiation in %s", x, context)
	}

	// spec: "If a left-hand side is the blank identifier, any typed or
	// non-constant value except for the predeclared identifier nil may
	// be assigned to it."
	if T == nil {
		return
	}

	reason := ""
	if ok, _ := x.assignableTo(check, T, &reason); !ok {
		if check.conf.CompilerErrorMessages {
			if reason != "" {
				check.errorf(x, "cannot use %s as type %s in %s:\n\t%s", x, T, context, reason)
			} else {
				check.errorf(x, "cannot use %s as type %s in %s", x, T, context)
			}
		} else {
			if reason != "" {
				check.errorf(x, "cannot use %s as %s value in %s: %s", x, T, context, reason)
			} else {
				check.errorf(x, "cannot use %s as %s value in %s", x, T, context)
			}
		}
		x.mode = invalid
	}
}

func (check *Checker) initConst(lhs *Const, x *operand) {
	if x.mode == invalid || x.typ == Typ[Invalid] || lhs.typ == Typ[Invalid] {
		if lhs.typ == nil {
			lhs.typ = Typ[Invalid]
		}
		return
	}

	// rhs must be a constant
	if x.mode != constant_ {
		check.errorf(x, "%s is not constant", x)
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

func (check *Checker) initVar(lhs *Var, x *operand, context string) Type {
	if x.mode == invalid || x.typ == Typ[Invalid] || lhs.typ == Typ[Invalid] {
		if lhs.typ == nil {
			lhs.typ = Typ[Invalid]
		}
		// Note: This was reverted in go/types (https://golang.org/cl/292751).
		// TODO(gri): decide what to do (also affects test/run.go exclusion list)
		lhs.used = true // avoid follow-on "declared but not used" errors
		return nil
	}

	// If the lhs doesn't have a type yet, use the type of x.
	if lhs.typ == nil {
		typ := x.typ
		if isUntyped(typ) {
			// convert untyped types to default types
			if typ == Typ[UntypedNil] {
				check.errorf(x, "use of untyped nil in %s", context)
				lhs.typ = Typ[Invalid]
				return nil
			}
			typ = Default(typ)
		}
		lhs.typ = typ
	}

	check.assignment(x, lhs.typ, context)
	if x.mode == invalid {
		lhs.used = true // avoid follow-on "declared but not used" errors
		return nil
	}

	return x.typ
}

func (check *Checker) assignVar(lhs syntax.Expr, x *operand) Type {
	if x.mode == invalid || x.typ == Typ[Invalid] {
		check.use(lhs)
		return nil
	}

	// Determine if the lhs is a (possibly parenthesized) identifier.
	ident, _ := unparen(lhs).(*syntax.Name)

	// Don't evaluate lhs if it is the blank identifier.
	if ident != nil && ident.Value == "_" {
		check.recordDef(ident, nil)
		check.assignment(x, nil, "assignment to _ identifier")
		if x.mode == invalid {
			return nil
		}
		return x.typ
	}

	// If the lhs is an identifier denoting a variable v, this assignment
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
				v_used = v.used
			}
		}
	}

	var z operand
	check.expr(&z, lhs)
	if v != nil {
		v.used = v_used // restore v.used
	}

	if z.mode == invalid || z.typ == Typ[Invalid] {
		return nil
	}

	// spec: "Each left-hand side operand must be addressable, a map index
	// expression, or the blank identifier. Operands may be parenthesized."
	switch z.mode {
	case invalid:
		return nil
	case variable, mapindex:
		// ok
	default:
		if sel, ok := z.expr.(*syntax.SelectorExpr); ok {
			var op operand
			check.expr(&op, sel.X)
			if op.mode == mapindex {
				check.errorf(&z, "cannot assign to struct field %s in map", syntax.String(z.expr))
				return nil
			}
		}
		check.errorf(&z, "cannot assign to %s", &z)
		return nil
	}

	check.assignment(x, z.typ, "assignment")
	if x.mode == invalid {
		return nil
	}

	return x.typ
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
func (check *Checker) typesSummary(list []Type, variadic bool) string {
	var res []string
	for i, t := range list {
		var s string
		switch {
		case t == nil:
			fallthrough // should not happen but be cautious
		case t == Typ[Invalid]:
			s = "<T>"
		case isUntyped(t):
			if isNumeric(t) {
				// Do not imply a specific type requirement:
				// "have number, want float64" is better than
				// "have untyped int, want float64" or
				// "have int, want float64".
				s = "number"
			} else {
				// If we don't have a number, omit the "untyped" qualifier
				// for compactness.
				s = strings.Replace(t.(*Basic).name, "untyped ", "", -1)
			}
		case variadic && i == len(list)-1:
			s = check.sprintf("...%s", t.(*Slice).elem)
		}
		if s == "" {
			s = check.sprintf("%s", t)
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

func (check *Checker) assignError(rhs []syntax.Expr, nvars, nvals int) {
	vars := measure(nvars, "variable")
	vals := measure(nvals, "value")
	rhs0 := rhs[0]

	if len(rhs) == 1 {
		if call, _ := unparen(rhs0).(*syntax.CallExpr); call != nil {
			check.errorf(rhs0, "assignment mismatch: %s but %s returns %s", vars, call.Fun, vals)
			return
		}
	}
	check.errorf(rhs0, "assignment mismatch: %s but %s", vars, vals)
}

// If returnStmt != nil, initVars is called to type-check the assignment
// of return expressions, and returnStmt is the return statement.
func (check *Checker) initVars(lhs []*Var, orig_rhs []syntax.Expr, returnStmt syntax.Stmt) {
	rhs, commaOk := check.exprList(orig_rhs, len(lhs) == 2 && returnStmt == nil)

	if len(lhs) != len(rhs) {
		// invalidate lhs
		for _, obj := range lhs {
			obj.used = true // avoid declared but not used errors
			if obj.typ == nil {
				obj.typ = Typ[Invalid]
			}
		}
		// don't report an error if we already reported one
		for _, x := range rhs {
			if x.mode == invalid {
				return
			}
		}
		if returnStmt != nil {
			var at poser = returnStmt
			qualifier := "not enough"
			if len(rhs) > len(lhs) {
				at = rhs[len(lhs)].expr // report at first extra value
				qualifier = "too many"
			} else if len(rhs) > 0 {
				at = rhs[len(rhs)-1].expr // report at last value
			}
			var err error_
			err.errorf(at, "%s return values", qualifier)
			err.errorf(nopos, "have %s", check.typesSummary(operandTypes(rhs), false))
			err.errorf(nopos, "want %s", check.typesSummary(varTypes(lhs), false))
			check.report(&err)
			return
		}
		if check.conf.CompilerErrorMessages {
			check.assignError(orig_rhs, len(lhs), len(rhs))
		} else {
			check.errorf(rhs[0], "cannot initialize %d variables with %d values", len(lhs), len(rhs))
		}
		return
	}

	context := "assignment"
	if returnStmt != nil {
		context = "return statement"
	}

	if commaOk {
		var a [2]Type
		for i := range a {
			a[i] = check.initVar(lhs[i], rhs[i], context)
		}
		check.recordCommaOkTypes(orig_rhs[0], a)
		return
	}

	ok := true
	for i, lhs := range lhs {
		if check.initVar(lhs, rhs[i], context) == nil {
			ok = false
		}
	}

	// avoid follow-on "declared but not used" errors if any initialization failed
	if !ok {
		for _, lhs := range lhs {
			lhs.used = true
		}
	}
}

func (check *Checker) assignVars(lhs, orig_rhs []syntax.Expr) {
	rhs, commaOk := check.exprList(orig_rhs, len(lhs) == 2)

	if len(lhs) != len(rhs) {
		check.use(lhs...)
		// don't report an error if we already reported one
		for _, x := range rhs {
			if x.mode == invalid {
				return
			}
		}
		if check.conf.CompilerErrorMessages {
			check.assignError(orig_rhs, len(lhs), len(rhs))
		} else {
			check.errorf(rhs[0], "cannot assign %d values to %d variables", len(rhs), len(lhs))
		}
		return
	}

	if commaOk {
		var a [2]Type
		for i := range a {
			a[i] = check.assignVar(lhs[i], rhs[i])
		}
		check.recordCommaOkTypes(orig_rhs[0], a)
		return
	}

	ok := true
	for i, lhs := range lhs {
		if check.assignVar(lhs, rhs[i]) == nil {
			ok = false
		}
	}

	// avoid follow-on "declared but not used" errors if any assignment failed
	if !ok {
		// don't call check.use to avoid re-evaluation of the lhs expressions
		for _, lhs := range lhs {
			if name, _ := unparen(lhs).(*syntax.Name); name != nil {
				if obj := check.lookup(name.Value); obj != nil {
					// see comment in assignVar
					if v, _ := obj.(*Var); v != nil && v.pkg == check.pkg {
						v.used = true
					}
				}
			}
		}
	}
}

// unpack unpacks a *syntax.ListExpr into a list of syntax.Expr.
// Helper introduced for the go/types -> types2 port.
// TODO(gri) Should find a more efficient solution that doesn't
// require introduction of a new slice for simple
// expressions.
func unpackExpr(x syntax.Expr) []syntax.Expr {
	if x, _ := x.(*syntax.ListExpr); x != nil {
		return x.ElemList
	}
	if x != nil {
		return []syntax.Expr{x}
	}
	return nil
}

func (check *Checker) shortVarDecl(pos syntax.Pos, lhs, rhs []syntax.Expr) {
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
			check.use(lhs)
			check.errorf(lhs, "non-name %s on left side of :=", lhs)
			hasErr = true
			continue
		}

		name := ident.Value
		if name != "_" {
			if seen[name] {
				check.errorf(lhs, "%s repeated on left side of :=", lhs)
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
				check.errorf(lhs, "cannot assign to %s", lhs)
				hasErr = true
			}
			continue
		}

		// declare new variable
		obj := NewVar(ident.Pos(), check.pkg, name, nil)
		lhsVars[i] = obj
		if name != "_" {
			newVars = append(newVars, obj)
		}
		check.recordDef(ident, obj)
	}

	// create dummy variables where the lhs is invalid
	for i, obj := range lhsVars {
		if obj == nil {
			lhsVars[i] = NewVar(lhs[i].Pos(), check.pkg, "_", nil)
		}
	}

	check.initVars(lhsVars, rhs, nil)

	// process function literals in rhs expressions before scope changes
	check.processDelayed(top)

	if len(newVars) == 0 && !hasErr {
		check.softErrorf(pos, "no new variables on left side of :=")
		return
	}

	// declare new variables
	// spec: "The scope of a constant or variable identifier declared inside
	// a function begins at the end of the ConstSpec or VarSpec (ShortVarDecl
	// for short variable declarations) and ends at the end of the innermost
	// containing block."
	scopePos := syntax.EndPos(rhs[len(rhs)-1])
	for _, obj := range newVars {
		check.declare(scope, nil, obj, scopePos) // id = nil: recordDef already called
	}
}
