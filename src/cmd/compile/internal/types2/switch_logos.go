// Copyright 2026 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package types2

import (
	"cmd/compile/internal/syntax"
	"go/constant"
	. "internal/types/errors"
	"strings"
)

// LoGo/s M2-S5 type rules for switch expressions (SPEC Sec 2.3, T-203.1).
// exprInternal calls in through one hook line.
//
// A switch expression takes the header of a switch statement. Its arms
// carry expressions instead of statement lists, and its result type is
// the common type of the arms: the identical type of the typed arms, the
// default type of untyped constants, or the pointer, map, slice, or
// interface type of the other arms when the remaining arms are nil. A
// value switch needs a default arm; a type switch needs a default arm or
// full coverage of a closed union and nil. The checker mirrors
// switchStmt and typeSwitchStmt of stmt.go, which stay untouched.

// logosSwitchExpr type-checks the switch expression e into x.
func (check *Checker) logosSwitchExpr(x *operand, e *syntax.SwitchExpr) exprKind {
	x.invalidate()
	x.typ_ = Typ[Invalid]
	x.expr = e
	if !check.conf.Logos {
		check.error(e, InvalidSyntaxTree, "invalid use of switch expression outside logos mode")
		return statement
	}

	check.openScope(e, "switch")
	defer check.closeScope()

	// A `?` in the init would be accepted by the assignment hooks below,
	// but lowering moves the init inside the `func() T` literal, which has
	// no error result to propagate to. Type the operand silently (as for
	// any misplaced `?`, so genuine inner errors still surface and uses
	// are marked), mark an assigned-to variable used, and reject it.
	if t := logosInitTry(e.Init); t != nil {
		check.logosTryMisplaced(t)
		if s, ok := e.Init.(*syntax.AssignStmt); ok && s.Op != syntax.Def {
			check.useLHS(syntax.UnpackListExpr(s.Lhs)...)
		}
		return statement
	}

	check.simpleStmt(e.Init)
	check.logosMultipleArmDefaults(e.Arms)

	var results []*operand
	guard, _ := e.Tag.(*syntax.TypeSwitchGuard)
	if guard != nil {
		results = check.logosTypeArms(e, guard)
	} else {
		results = check.logosValueArms(e)
	}
	if len(e.Arms) == 0 {
		check.error(e, InvalidExprSwitch, "switch expression has no arms")
		return statement
	}
	if guard == nil && !logosHasDefault(e.Arms) {
		check.errorf(e, InvalidExprSwitch, "%s switch expression needs a default arm", logosNeedDefault)
	}

	typ := check.logosArmType(e, results)
	if typ == nil {
		return statement
	}
	x.mode_ = value
	x.typ_ = typ
	return expression
}

// logosInitTry returns the first `?` in the switch-expression init itself,
// if any. The init is a simple statement, so `x := f()?` would otherwise
// flow through the assignment hooks and check cleanly. A `?` inside a
// nested function literal belongs to that literal and moves with it into
// the lowered body, so the scan does not descend into one.
func logosInitTry(init syntax.SimpleStmt) *syntax.TryExpr {
	if init == nil {
		return nil
	}
	var found *syntax.TryExpr
	syntax.Inspect(init, func(n syntax.Node) bool {
		if found != nil {
			return false
		}
		switch n := n.(type) {
		case *syntax.FuncLit:
			return false
		case *syntax.TryExpr:
			found = n
			return false
		}
		return true
	})
	return found
}

// logosHasDefault reports whether one of arms is a default arm.
func logosHasDefault(arms []*syntax.ArmClause) bool {
	for _, arm := range arms {
		if arm.Cases == nil {
			return true
		}
	}
	return false
}

// logosMultipleArmDefaults reports every default arm after the first.
func (check *Checker) logosMultipleArmDefaults(arms []*syntax.ArmClause) {
	var first *syntax.ArmClause
	for _, arm := range arms {
		if arm.Cases != nil {
			continue
		}
		if first != nil {
			check.errorf(arm, DuplicateDefault, "multiple defaults (first at %s)", first.Pos())
		} else {
			first = arm
		}
	}
}

// logosArmResult type-checks the result expression of arm and returns its
// operand. The caller opens the arm scope.
func (check *Checker) logosArmResult(arm *syntax.ArmClause) *operand {
	var v operand
	check.expr(nil, &v, arm.Body)
	return &v
}

// logosValueArms checks the tag, the case values, and the result of every
// arm of a value switch expression.
func (check *Checker) logosValueArms(e *syntax.SwitchExpr) []*operand {
	var tag operand
	if e.Tag != nil {
		check.expr(nil, &tag, e.Tag)
		check.assignment(&tag, nil, "switch expression")
		if tag.isValid() && !Comparable(tag.typ()) && !hasNil(tag.typ()) {
			check.errorf(&tag, InvalidExprSwitch, "cannot switch on %s (%s is not comparable)", &tag, tag.typ())
			tag.invalidate()
		}
	} else {
		pos := e.Rbrace
		if len(e.Arms) > 0 {
			pos = e.Arms[0].Pos()
		}
		tag.mode_ = constant_
		tag.typ_ = Typ[Bool]
		tag.val = constant.MakeBool(true)
		tag.expr = syntax.NewName(pos, "true")
	}

	seen := make(valueMap)
	results := make([]*operand, 0, len(e.Arms))
	for _, arm := range e.Arms {
		check.caseValues(&tag, syntax.UnpackListExpr(arm.Cases), seen)
		check.openScope(arm, "case")
		results = append(results, check.logosArmResult(arm))
		check.closeScope()
	}
	return results
}

// logosTypeArms checks the guard, the case types, and the result of every
// arm of a type switch expression, then its coverage.
func (check *Checker) logosTypeArms(e *syntax.SwitchExpr, guard *syntax.TypeSwitchGuard) []*operand {
	lhs := guard.Lhs
	if lhs != nil {
		if lhs.Value == "_" {
			check.softErrorf(lhs, NoNewVar, "no new variable on left side of :=")
			lhs = nil
		} else {
			check.recordDef(lhs, nil)
		}
	}

	var sx *operand
	{
		var x operand
		check.expr(nil, &x, guard.X)
		if x.isValid() {
			if isTypeParam(x.typ()) {
				check.errorf(&x, InvalidTypeSwitch, "cannot use type switch on type parameter value %s", &x)
			} else if IsInterface(x.typ()) {
				sx = &x
			} else {
				check.errorf(&x, InvalidTypeSwitch, "%s is not an interface", &x)
			}
		}
	}

	var lhsVars []*Var
	seen := make(map[Type]syntax.Expr)
	results := make([]*operand, 0, len(e.Arms))
	for _, arm := range e.Arms {
		T := check.caseTypes(sx, syntax.UnpackListExpr(arm.Cases), seen)
		check.openScope(arm, "case")
		if lhs != nil {
			obj := newVar(LocalVar, lhs.Pos(), check.pkg, lhs.Value, T)
			check.declare(check.scope, nil, obj, arm.Arrow)
			check.recordImplicit(arm, obj)
			lhsVars = append(lhsVars, obj)
		}
		results = append(results, check.logosArmResult(arm))
		check.closeScope()
	}

	if lhs != nil {
		var used bool
		for _, v := range lhsVars {
			if check.usedVars[v] {
				used = true
			}
			check.usedVars[v] = true
		}
		if !used {
			check.softErrorf(lhs, UnusedVar, "%s declared and not used", lhs.Value)
		}
	}

	if sx != nil && len(e.Arms) > 0 && !logosHasDefault(e.Arms) {
		check.logosCoverage(e, sx, seen)
	}
	return results
}

// logosCoverage reports nonexhaustive_switch when the cases of a type
// switch expression without default miss a member of the closed union
// operand sx, or nil. A switch on any other interface type has no finite
// member set, so only a default arm covers it. Misses list members in
// declaration order, then nil.
func (check *Checker) logosCoverage(e *syntax.SwitchExpr, sx *operand, seen map[Type]syntax.Expr) {
	var misses []string
	if tset := logosClosedUnion(sx.typ()); tset != nil {
		for _, term := range tset.terms {
			if !check.logosCovers(term.typ, seen) {
				misses = append(misses, TypeString(term.typ, check.qualifier))
			}
		}
		if _, ok := seen[nil]; !ok {
			misses = append(misses, "nil")
		}
	} else {
		misses = append(misses, "default")
	}
	if len(misses) == 0 {
		return
	}
	check.errorf(e, InvalidTypeSwitch, "%s switch on %s misses: %s", logosNonExhaust, sx.typ(), strings.Join(misses, ", "))
}

// logosCovers reports whether some case type in seen matches the member
// type member: the identical type, or an interface that member implements
// (a sub-union covers its members, any covers every member).
func (check *Checker) logosCovers(member Type, seen map[Type]syntax.Expr) bool {
	for c := range seen {
		if c == nil {
			continue
		}
		if Identical(member, c) {
			return true
		}
		if IsInterface(c) && !isTypeParam(c) && check.implements(member, c, false, nil) {
			return true
		}
	}
	return false
}

// logosArmType computes the common type of the arm operands and converts
// every arm to it. It returns nil when an arm is invalid or the arms do
// not agree, after reporting the cause.
//
// A typed arm fixes the common type: every other arm must be assignable
// to it (an untyped constant by representability, nil iff the type is
// nilable), so untyped arms are never folded with maxType first. Only
// when no typed arm exists are the untyped arms folded and defaulted.
func (check *Checker) logosArmType(e *syntax.SwitchExpr, arms []*operand) Type {
	for _, v := range arms {
		if !v.isValid() {
			return nil
		}
	}

	var typed Type
	agree := true
	for _, v := range arms {
		if v.isNil() || isUntyped(v.typ()) {
			continue
		}
		if typed == nil {
			typed = v.typ()
			continue
		}
		if !Identical(typed, v.typ()) {
			check.errorf(v, MismatchedTypes, "%s arms have types %s and %s", logosArmMismatch, typed, v.typ())
			agree = false
		}
	}
	if !agree {
		return nil
	}

	common := typed
	if common == nil {
		var untyped Type
		for _, v := range arms {
			switch {
			case v.isNil():
			case isUntyped(v.typ()):
				if untyped == nil {
					untyped = v.typ()
				} else if m := maxType(untyped, v.typ()); m != nil {
					untyped = m
				} else {
					check.errorf(v, MismatchedTypes, "%s arms have types %s and %s", logosArmMismatch, untyped, v.typ())
					agree = false
				}
			}
		}
		if !agree {
			return nil
		}
		if untyped == nil {
			check.error(e, UntypedNilUse, "use of untyped nil in switch expression")
			return nil
		}
		common = Default(untyped)
	}
	for _, v := range arms {
		// Pre-validate, so a failing arm reports only the tagged
		// mismatch below and never the untagged assignment message.
		// This covers typed and defaulted common types alike: a nil
		// arm fails iff the common type is not nilable.
		if !check.logosArmAssignable(v, common) {
			agree = false
			continue
		}
		check.assignment(v, common, "switch expression arm")
		if !v.isValid() {
			agree = false
		}
	}
	if !agree {
		return nil
	}
	return common
}

// logosArmAssignable reports whether the arm v converts to the typed
// common type: an untyped constant by representability, nil iff the type
// is nilable. On failure it reports switch_arm_mismatch and the caller
// skips the assignment, so each arm yields at most the one tagged error.
func (check *Checker) logosArmAssignable(v *operand, common Type) bool {
	if ok, _ := v.assignableTo(check, common, nil); !ok {
		check.errorf(v, MismatchedTypes, "%s arms have types %s and %s", logosArmMismatch, common, v.typ())
		return false
	}
	return true
}
