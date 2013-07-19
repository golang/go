// Copyright 2013 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// This file implements initialization and assignment checks.

package types

import (
	"go/ast"

	"code.google.com/p/go.tools/go/exact"
)

// assignment reports whether x can be assigned to a variable of type 'to',
// if necessary by attempting to convert untyped values to the appropriate
// type. If x.mode == invalid upon return, then assignment has already
// issued an error message and the caller doesn't have to report another.
// TODO(gri) This latter behavior is for historic reasons and complicates
// callers. Needs to be cleaned up.
func (check *checker) assignment(x *operand, to Type) bool {
	if x.mode == invalid {
		return false
	}

	if t, ok := x.typ.(*Tuple); ok {
		assert(t.Len() > 1)
		check.errorf(x.pos(), "%d-valued expression %s used as single value", t.Len(), x)
		x.mode = invalid
		return false
	}

	check.convertUntyped(x, to)

	return x.mode != invalid && x.isAssignableTo(check.conf, to)
}

func (check *checker) initConst(lhs *Const, x *operand) {
	lhs.val = exact.MakeUnknown()

	if x.mode == invalid || x.typ == Typ[Invalid] || lhs.typ == Typ[Invalid] {
		if lhs.typ == nil {
			lhs.typ = Typ[Invalid]
		}
		return // nothing else to check
	}

	// If the lhs doesn't have a type yet, use the type of x.
	if lhs.typ == nil {
		lhs.typ = x.typ
	}

	if !check.assignment(x, lhs.typ) {
		if x.mode != invalid {
			check.errorf(x.pos(), "cannot define constant %s (type %s) as %s", lhs.Name(), lhs.typ, x)
		}
		return
	}

	// rhs must be a constant
	if x.mode != constant {
		check.errorf(x.pos(), "%s is not constant", x)
		return
	}

	// rhs type must be a valid constant type
	if !isConstType(x.typ) {
		check.errorf(x.pos(), "%s has invalid constant type", x)
		return
	}

	lhs.val = x.val
}

func (check *checker) initVar(lhs *Var, x *operand) {
	if x.mode == invalid || x.typ == Typ[Invalid] || lhs.typ == Typ[Invalid] {
		if lhs.typ == nil {
			lhs.typ = Typ[Invalid]
		}
		return // nothing else to check
	}

	// If the lhs doesn't have a type yet, use the type of x.
	if lhs.typ == nil {
		typ := x.typ
		if isUntyped(typ) {
			// convert untyped types to default types
			if typ == Typ[UntypedNil] {
				check.errorf(x.pos(), "use of untyped nil")
				lhs.typ = Typ[Invalid]
				return // nothing else to check
			}
			typ = defaultType(typ)
		}
		lhs.typ = typ
	}

	if !check.assignment(x, lhs.typ) {
		if x.mode != invalid {
			check.errorf(x.pos(), "cannot initialize variable %s (type %s) with %s", lhs.Name(), lhs.typ, x)
		}
	}
}

func (check *checker) assignVar(lhs ast.Expr, x *operand) {
	if x.mode == invalid || x.typ == Typ[Invalid] {
		return
	}

	// Don't evaluate lhs if it is the blank identifier.
	if ident, _ := lhs.(*ast.Ident); ident != nil && ident.Name == "_" {
		check.recordObject(ident, nil)
		check.updateExprType(x.expr, x.typ, true) // rhs has its final type
		return
	}

	var z operand
	check.expr(&z, lhs)
	if z.mode == invalid || z.typ == Typ[Invalid] {
		return
	}

	if z.mode == constant || z.mode == value {
		check.errorf(z.pos(), "cannot assign to non-variable %s", &z)
		return
	}

	// TODO(gri) z.mode can also be valueok which in some cases is ok (maps)
	// but in others isn't (channels). Complete the checks here.

	if !check.assignment(x, z.typ) {
		if x.mode != invalid {
			check.errorf(x.pos(), "cannot assign %s to %s", x, &z)
		}
	}
}

func (check *checker) initVars(lhs []*Var, rhs []ast.Expr, allowCommaOk bool) {
	l := len(lhs)
	r := len(rhs)
	assert(l > 0)

	// If the lhs and rhs have corresponding expressions,
	// treat each matching pair as an individual pair.
	if l == r {
		var x operand
		for i, e := range rhs {
			check.expr(&x, e)
			check.initVar(lhs[i], &x)
		}
		return
	}

	// Otherwise, the rhs must be a single expression (possibly
	// a function call returning multiple values, or a comma-ok
	// expression).
	if r == 1 {
		// l > 1
		// Start with rhs so we have expression types
		// for declarations with implicit types.
		var x operand
		check.expr(&x, rhs[0])
		if x.mode == invalid {
			invalidateVars(lhs)
			return
		}

		if t, ok := x.typ.(*Tuple); ok {
			// function result
			r = t.Len()
			if l == r {
				for i, lhs := range lhs {
					x.mode = value
					x.expr = rhs[0]
					x.typ = t.At(i).typ
					check.initVar(lhs, &x)
				}
				return
			}
		}

		if allowCommaOk && x.mode == valueok && l == 2 {
			// comma-ok expression
			x.mode = value
			check.initVar(lhs[0], &x)

			x.mode = value
			x.typ = Typ[UntypedBool]
			check.initVar(lhs[1], &x)
			return
		}
	}

	// lhs variables may be function result parameters (return statement);
	// use rhs position for properly located error messages
	check.errorf(rhs[0].Pos(), "assignment count mismatch (%d vs %d)", l, r)
	invalidateVars(lhs)
}

func (check *checker) assignVars(lhs, rhs []ast.Expr) {
	l := len(lhs)
	r := len(rhs)
	assert(l > 0)

	// If the lhs and rhs have corresponding expressions,
	// treat each matching pair as an individual pair.
	if l == r {
		var x operand
		for i, e := range rhs {
			check.expr(&x, e)
			check.assignVar(lhs[i], &x)
		}
		return
	}

	// Otherwise, the rhs must be a single expression (possibly
	// a function call returning multiple values, or a comma-ok
	// expression).
	if r == 1 {
		// l > 1
		var x operand
		check.expr(&x, rhs[0])
		if x.mode == invalid {
			return
		}

		if t, ok := x.typ.(*Tuple); ok {
			// function result
			r = t.Len()
			if l == r {
				for i, lhs := range lhs {
					x.mode = value
					x.expr = rhs[0]
					x.typ = t.At(i).typ
					check.assignVar(lhs, &x)
				}
				return
			}
		}

		if x.mode == valueok && l == 2 {
			// comma-ok expression
			x.mode = value
			check.assignVar(lhs[0], &x)

			x.mode = value
			x.typ = Typ[UntypedBool]
			check.assignVar(lhs[1], &x)
			return
		}
	}

	check.errorf(rhs[0].Pos(), "assignment count mismatch (%d vs %d)", l, r)
}

func (check *checker) shortVarDecl(lhs, rhs []ast.Expr) {
	scope := check.topScope

	// collect lhs variables
	vars := make([]*Var, len(lhs))
	for i, lhs := range lhs {
		var obj *Var
		if ident, _ := lhs.(*ast.Ident); ident != nil {
			// Use the correct obj if the ident is redeclared. The
			// variable's scope starts after the declaration; so we
			// must use Scope.Lookup here and call Scope.Insert later.
			if alt := scope.Lookup(nil, ident.Name); alt != nil {
				// redeclared object must be a variable
				if alt, _ := alt.(*Var); alt != nil {
					obj = alt
				} else {
					check.errorf(lhs.Pos(), "cannot assign to %s", lhs)
				}
			} else {
				// declare new variable
				obj = NewVar(ident.Pos(), check.pkg, ident.Name, nil)
			}
			check.recordObject(ident, obj) // obj may be nil
		} else {
			check.errorf(lhs.Pos(), "cannot declare %s", lhs)
		}
		if obj == nil {
			obj = NewVar(lhs.Pos(), check.pkg, "_", nil) // dummy variable
		}
		vars[i] = obj
	}

	check.initVars(vars, rhs, true)

	// declare variables
	n := 0 // number of new variables
	for _, obj := range vars {
		if obj.name == "_" {
			obj.setParent(scope)
			continue // blank identifiers are not visible
		}
		if scope.Insert(obj) == nil {
			n++ // new declaration
		}
	}
	if n == 0 {
		check.errorf(vars[0].Pos(), "no new variables on left side of :=")
	}
}

func invalidateVars(list []*Var) {
	for _, obj := range list {
		if obj.typ == nil {
			obj.typ = Typ[Invalid]
		}
	}
}
