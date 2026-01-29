// Copyright 2025 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// This file implements typechecking of range statements.

package types2

import (
	"cmd/compile/internal/syntax"
	"go/constant"
	. "internal/types/errors"
)

// rangeStmt type-checks a range statement of form
//
//	for sKey, sValue = range rangeVar { ... }
//
// where sKey, sValue, sExtra may be nil. isDef indicates whether these
// variables are assigned to only (=) or whether there is a short variable
// declaration (:=). If the latter and there are no variables, an error is
// reported at noNewVarPos.
func (check *Checker) rangeStmt(inner stmtContext, rangeStmt *syntax.ForStmt, noNewVarPos poser, sKey, sValue, sExtra, rangeVar syntax.Expr, isDef bool) {
	// check expression to iterate over
	var x operand

	// From the spec:
	//   The range expression x is evaluated before beginning the loop,
	//   with one exception: if at most one iteration variable is present
	//   and x or len(x) is constant, the range expression is not evaluated.
	// So we have to be careful not to evaluate the arg in the
	// described situation.

	check.hasCallOrRecv = false
	check.expr(nil, &x, rangeVar)

	if isTypes2 && x.mode != invalid && sValue == nil && !check.hasCallOrRecv {
		if t, ok := arrayPtrDeref(x.typ.Underlying()).(*Array); ok {
			for {
				// Put constant info on the thing inside parentheses.
				// That's where (*../noder/writer).expr expects it.
				// See issue 73476.
				p, ok := rangeVar.(*syntax.ParenExpr)
				if !ok {
					break
				}
				rangeVar = p.X
			}
			// Override type of rangeVar to be a constant
			// (and thus side-effects will not be computed
			// by the backend).
			check.record(&operand{
				mode: constant_,
				expr: rangeVar,
				typ:  Typ[Int],
				val:  constant.MakeInt64(t.len),
				id:   x.id,
			})
		}
	}

	// determine key/value types
	var key, val Type
	if x.mode != invalid {
		k, v, cause, ok := rangeKeyVal(check, x.typ, func(v goVersion) bool {
			return check.allowVersion(v)
		})
		switch {
		case !ok && cause != "":
			check.softErrorf(&x, InvalidRangeExpr, "cannot range over %s: %s", &x, cause)
		case !ok:
			check.softErrorf(&x, InvalidRangeExpr, "cannot range over %s", &x)
		case k == nil && sKey != nil:
			check.softErrorf(sKey, InvalidIterVar, "range over %s permits no iteration variables", &x)
		case v == nil && sValue != nil:
			check.softErrorf(sValue, InvalidIterVar, "range over %s permits only one iteration variable", &x)
		case sExtra != nil:
			check.softErrorf(sExtra, InvalidIterVar, "range clause permits at most two iteration variables")
		}
		key, val = k, v
	}

	// Open the for-statement block scope now, after the range clause.
	// Iteration variables declared with := need to go in this scope (was go.dev/issue/51437).
	check.openScope(rangeStmt, "range")
	defer check.closeScope()

	// check assignment to/declaration of iteration variables
	// (irregular assignment, cannot easily map to existing assignment checks)

	// lhs expressions and initialization value (rhs) types
	lhs := [2]syntax.Expr{sKey, sValue} // sKey, sValue may be nil
	rhs := [2]Type{key, val}            // key, val may be nil

	rangeOverInt := isInteger(x.typ)

	if isDef {
		// short variable declaration
		var vars []*Var
		for i, lhs := range lhs {
			if lhs == nil {
				continue
			}

			// determine lhs variable
			var obj *Var
			if ident, _ := lhs.(*syntax.Name); ident != nil {
				// declare new variable
				name := ident.Value
				obj = newVar(LocalVar, ident.Pos(), check.pkg, name, nil)
				check.recordDef(ident, obj)
				// _ variables don't count as new variables
				if name != "_" {
					vars = append(vars, obj)
				}
			} else {
				check.errorf(lhs, InvalidSyntaxTree, "cannot declare %s", lhs)
				obj = newVar(LocalVar, lhs.Pos(), check.pkg, "_", nil) // dummy variable
			}
			assert(obj.typ == nil)

			// initialize lhs iteration variable, if any
			typ := rhs[i]
			if typ == nil || typ == Typ[Invalid] {
				// typ == Typ[Invalid] can happen if allowVersion fails.
				obj.typ = Typ[Invalid]
				check.usedVars[obj] = true // don't complain about unused variable
				continue
			}

			if rangeOverInt {
				assert(i == 0) // at most one iteration variable (rhs[1] == nil or Typ[Invalid] for rangeOverInt)
				check.initVar(obj, &x, "range clause")
			} else {
				var y operand
				y.mode = value
				y.expr = lhs // we don't have a better rhs expression to use here
				y.typ = typ
				check.initVar(obj, &y, "assignment") // error is on variable, use "assignment" not "range clause"
			}
			assert(obj.typ != nil)
		}

		// declare variables
		if len(vars) > 0 {
			scopePos := rangeStmt.Body.Pos()
			for _, obj := range vars {
				check.declare(check.scope, nil /* recordDef already called */, obj, scopePos)
			}
		} else {
			check.error(noNewVarPos, NoNewVar, "no new variables on left side of :=")
		}
	} else if sKey != nil /* lhs[0] != nil */ {
		// ordinary assignment
		for i, lhs := range lhs {
			if lhs == nil {
				continue
			}

			// assign to lhs iteration variable, if any
			typ := rhs[i]
			if typ == nil || typ == Typ[Invalid] {
				continue
			}

			if rangeOverInt {
				assert(i == 0) // at most one iteration variable (rhs[1] == nil or Typ[Invalid] for rangeOverInt)
				check.assignVar(lhs, nil, &x, "range clause")
				// If the assignment succeeded, if x was untyped before, it now
				// has a type inferred via the assignment. It must be an integer.
				// (go.dev/issues/67027)
				if x.mode != invalid && !isInteger(x.typ) {
					check.softErrorf(lhs, InvalidRangeExpr, "cannot use iteration variable of type %s", x.typ)
				}
			} else {
				var y operand
				y.mode = value
				y.expr = lhs // we don't have a better rhs expression to use here
				y.typ = typ
				check.assignVar(lhs, nil, &y, "assignment") // error is on variable, use "assignment" not "range clause"
			}
		}
	} else if rangeOverInt {
		// If we don't have any iteration variables, we still need to
		// check that a (possibly untyped) integer range expression x
		// is valid.
		// We do this by checking the assignment _ = x. This ensures
		// that an untyped x can be converted to a value of its default
		// type (rune or int).
		check.assignment(&x, nil, "range clause")
	}

	check.stmt(inner, rangeStmt.Body)
}

// rangeKeyVal returns the key and value type produced by a range clause
// over an expression of type orig.
// If allowVersion != nil, it is used to check the required language version.
// If the range clause is not permitted, rangeKeyVal returns ok = false.
// When ok = false, rangeKeyVal may also return a reason in cause.
// The check parameter is only used in case of an error; it may be nil.
func rangeKeyVal(check *Checker, orig Type, allowVersion func(goVersion) bool) (key, val Type, cause string, ok bool) {
	bad := func(cause string) (Type, Type, string, bool) {
		return Typ[Invalid], Typ[Invalid], cause, false
	}

	rtyp, err := commonUnder(orig, func(t, u Type) *typeError {
		// A channel must permit receive operations.
		if ch, _ := u.(*Chan); ch != nil && ch.dir == SendOnly {
			return typeErrorf("receive from send-only channel %s", t)
		}
		return nil
	})
	if rtyp == nil {
		return bad(err.format(check))
	}

	switch typ := arrayPtrDeref(rtyp).(type) {
	case *Basic:
		if isString(typ) {
			return Typ[Int], universeRune, "", true // use 'rune' name
		}
		if isInteger(typ) {
			if allowVersion != nil && !allowVersion(go1_22) {
				return bad("requires go1.22 or later")
			}
			return orig, nil, "", true
		}
	case *Array:
		return Typ[Int], typ.elem, "", true
	case *Slice:
		return Typ[Int], typ.elem, "", true
	case *Map:
		return typ.key, typ.elem, "", true
	case *Chan:
		assert(typ.dir != SendOnly)
		return typ.elem, nil, "", true
	case *Signature:
		if allowVersion != nil && !allowVersion(go1_23) {
			return bad("requires go1.23 or later")
		}
		// check iterator arity
		switch {
		case typ.Params().Len() != 1:
			return bad("func must be func(yield func(...) bool): wrong argument count")
		case typ.Results().Len() != 0:
			return bad("func must be func(yield func(...) bool): unexpected results")
		}
		assert(typ.Recv() == nil)
		// check iterator argument type
		u, err := commonUnder(typ.Params().At(0).Type(), nil)
		cb, _ := u.(*Signature)
		switch {
		case cb == nil:
			if err != nil {
				return bad(check.sprintf("func must be func(yield func(...) bool): in yield type, %s", err.format(check)))
			} else {
				return bad("func must be func(yield func(...) bool): argument is not func")
			}
		case cb.Params().Len() > 2:
			return bad("func must be func(yield func(...) bool): yield func has too many parameters")
		case cb.Results().Len() != 1 || !Identical(cb.Results().At(0).Type(), universeBool):
			// see go.dev/issues/71131, go.dev/issues/71164
			if cb.Results().Len() == 1 && isBoolean(cb.Results().At(0).Type()) {
				return bad("func must be func(yield func(...) bool): yield func returns user-defined boolean, not bool")
			} else {
				return bad("func must be func(yield func(...) bool): yield func does not return bool")
			}
		}
		assert(cb.Recv() == nil)
		// determine key and value types, if any
		if cb.Params().Len() >= 1 {
			key = cb.Params().At(0).Type()
		}
		if cb.Params().Len() >= 2 {
			val = cb.Params().At(1).Type()
		}
		return key, val, "", true
	}
	return
}
