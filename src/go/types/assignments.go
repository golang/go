// Copyright 2013 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// This file implements initialization and assignment checks.

package types

import (
	"go/ast"
	"go/token"
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
	case constant_, variable, mapindex, value, commaok:
		// ok
	default:
		unreachable()
	}

	if isUntyped(x.typ) {
		target := T
		// spec: "If an untyped constant is assigned to a variable of interface
		// type or the blank identifier, the constant is first converted to type
		// bool, rune, int, float64, complex128 or string respectively, depending
		// on whether the value is a boolean, rune, integer, floating-point, complex,
		// or string constant."
		if T == nil || IsInterface(T) {
			if T == nil && x.typ == Typ[UntypedNil] {
				check.errorf(x.pos(), "use of untyped nil in %s", context)
				x.mode = invalid
				return
			}
			target = Default(x.typ)
		}
		check.convertUntyped(x, target)
		if x.mode == invalid {
			return
		}
	}
	// x.typ is typed

	// spec: "If a left-hand side is the blank identifier, any typed or
	// non-constant value except for the predeclared identifier nil may
	// be assigned to it."
	if T == nil {
		return
	}

	if reason := ""; !x.assignableTo(check.conf, T, &reason) {
		if reason != "" {
			check.errorf(x.pos(), "cannot use %s as %s value in %s: %s", x, T, context, reason)
		} else {
			check.errorf(x.pos(), "cannot use %s as %s value in %s", x, T, context)
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
		check.errorf(x.pos(), "%s is not constant", x)
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
		return nil
	}

	// If the lhs doesn't have a type yet, use the type of x.
	if lhs.typ == nil {
		typ := x.typ
		if isUntyped(typ) {
			// convert untyped types to default types
			if typ == Typ[UntypedNil] {
				check.errorf(x.pos(), "use of untyped nil in %s", context)
				lhs.typ = Typ[Invalid]
				return nil
			}
			typ = Default(typ)
		}
		lhs.typ = typ
	}

	check.assignment(x, lhs.typ, context)
	if x.mode == invalid {
		return nil
	}

	return x.typ
}

func (check *Checker) assignVar(lhs ast.Expr, x *operand) Type {
	if x.mode == invalid || x.typ == Typ[Invalid] {
		return nil
	}

	// Determine if the lhs is a (possibly parenthesized) identifier.
	ident, _ := unparen(lhs).(*ast.Ident)

	// Don't evaluate lhs if it is the blank identifier.
	if ident != nil && ident.Name == "_" {
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
		if obj := check.lookup(ident.Name); obj != nil {
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
		if sel, ok := z.expr.(*ast.SelectorExpr); ok {
			var op operand
			check.expr(&op, sel.X)
			if op.mode == mapindex {
				check.errorf(z.pos(), "cannot assign to struct field %s in map", ExprString(z.expr))
				return nil
			}
		}
		check.errorf(z.pos(), "cannot assign to %s", &z)
		return nil
	}

	check.assignment(x, z.typ, "assignment")
	if x.mode == invalid {
		return nil
	}

	return x.typ
}

// If returnPos is valid, initVars is called to type-check the assignment of
// return expressions, and returnPos is the position of the return statement.
func (check *Checker) initVars(lhs []*Var, rhs []ast.Expr, returnPos token.Pos) {
	l := len(lhs)
	get, r, commaOk := unpack(func(x *operand, i int) { check.multiExpr(x, rhs[i]) }, len(rhs), l == 2 && !returnPos.IsValid())
	if get == nil || l != r {
		// invalidate lhs and use rhs
		for _, obj := range lhs {
			if obj.typ == nil {
				obj.typ = Typ[Invalid]
			}
		}
		if get == nil {
			return // error reported by unpack
		}
		check.useGetter(get, r)
		if returnPos.IsValid() {
			check.errorf(returnPos, "wrong number of return values (want %d, got %d)", l, r)
			return
		}
		check.errorf(rhs[0].Pos(), "cannot initialize %d variables with %d values", l, r)
		return
	}

	context := "assignment"
	if returnPos.IsValid() {
		context = "return statement"
	}

	var x operand
	if commaOk {
		var a [2]Type
		for i := range a {
			get(&x, i)
			a[i] = check.initVar(lhs[i], &x, context)
		}
		check.recordCommaOkTypes(rhs[0], a)
		return
	}

	for i, lhs := range lhs {
		get(&x, i)
		check.initVar(lhs, &x, context)
	}
}

func (check *Checker) assignVars(lhs, rhs []ast.Expr) {
	l := len(lhs)
	get, r, commaOk := unpack(func(x *operand, i int) { check.multiExpr(x, rhs[i]) }, len(rhs), l == 2)
	if get == nil {
		check.useLHS(lhs...)
		return // error reported by unpack
	}
	if l != r {
		check.useGetter(get, r)
		check.errorf(rhs[0].Pos(), "cannot assign %d values to %d variables", r, l)
		return
	}

	var x operand
	if commaOk {
		var a [2]Type
		for i := range a {
			get(&x, i)
			a[i] = check.assignVar(lhs[i], &x)
		}
		check.recordCommaOkTypes(rhs[0], a)
		return
	}

	for i, lhs := range lhs {
		get(&x, i)
		check.assignVar(lhs, &x)
	}
}

func (check *Checker) shortVarDecl(pos token.Pos, lhs, rhs []ast.Expr) {
	top := len(check.delayed)
	scope := check.scope

	// collect lhs variables
	var newVars []*Var
	var lhsVars = make([]*Var, len(lhs))
	for i, lhs := range lhs {
		var obj *Var
		if ident, _ := lhs.(*ast.Ident); ident != nil {
			// Use the correct obj if the ident is redeclared. The
			// variable's scope starts after the declaration; so we
			// must use Scope.Lookup here and call Scope.Insert
			// (via check.declare) later.
			name := ident.Name
			if alt := scope.Lookup(name); alt != nil {
				// redeclared object must be a variable
				if alt, _ := alt.(*Var); alt != nil {
					obj = alt
				} else {
					check.errorf(lhs.Pos(), "cannot assign to %s", lhs)
				}
				check.recordUse(ident, alt)
			} else {
				// declare new variable, possibly a blank (_) variable
				obj = NewVar(ident.Pos(), check.pkg, name, nil)
				if name != "_" {
					newVars = append(newVars, obj)
				}
				check.recordDef(ident, obj)
			}
		} else {
			check.useLHS(lhs)
			check.errorf(lhs.Pos(), "cannot declare %s", lhs)
		}
		if obj == nil {
			obj = NewVar(lhs.Pos(), check.pkg, "_", nil) // dummy variable
		}
		lhsVars[i] = obj
	}

	check.initVars(lhsVars, rhs, token.NoPos)

	// process function literals in rhs expressions before scope changes
	check.processDelayed(top)

	// declare new variables
	if len(newVars) > 0 {
		// spec: "The scope of a constant or variable identifier declared inside
		// a function begins at the end of the ConstSpec or VarSpec (ShortVarDecl
		// for short variable declarations) and ends at the end of the innermost
		// containing block."
		scopePos := rhs[len(rhs)-1].End()
		for _, obj := range newVars {
			check.declare(scope, nil, obj, scopePos) // recordObject already called
		}
	} else {
		check.softErrorf(pos, "no new variables on left side of :=")
	}
}
