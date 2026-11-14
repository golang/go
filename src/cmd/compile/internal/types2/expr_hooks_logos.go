// Copyright 2026 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package types2

import (
	"cmd/compile/internal/syntax"
	. "internal/types/errors"
)

func (check *Checker) logosExprAddr(x *operand) bool {
	if check.conf.Logos && logosAsReadonlyDescendants(x.typ()) != nil {
		x.typ_ = &Pointer{base: x.typ()}
		return true
	}
	return check.logosConstAddr(x)
}

func (check *Checker) logosExprOperands(x, y *operand, e, lhs, rhs syntax.Expr, op syntax.Operator) {
	if e == nil {
		check.logosLHSPush(lhs)
	}
	check.expr(nil, x, lhs)
	if e == nil {
		check.logosLHSPop()
	}
	anded := check.logosPushAndCond(op, lhs)
	top := len(check.delayed)
	check.expr(nil, y, rhs)
	if anded {
		check.processDelayed(top)
		check.logosPopNarrow()
	}
}

func (check *Checker) logosExprUnionConstMiss(x, y *operand) bool {
	if check.logosUnionConstMiss(x, y) {
		x.invalidate()
		return true
	}
	return false
}

func (check *Checker) logosExprRegionLitMark(x *operand, e *syntax.FuncLit) {
	if sig, _ := x.typ().(*Signature); sig != nil {
		check.logosRegionLitMark(e, sig)
	}
}

func (check *Checker) logosExprReadonlyPrepare(x *operand) Type {
	readonlyContainer := x.typ()
	if check.conf.Logos {
		x.typ_, _, _ = logosReadonlyBase(x.typ())
	}
	return readonlyContainer
}

func (check *Checker) logosExprReadonlyProject(x *operand, readonlyContainer Type) {
	if check.conf.Logos {
		x.typ_ = logosReadonlyProject(readonlyContainer, x.typ(), true)
	}
}

func (check *Checker) logosExprReadonlyAssert(e syntax.Expr, x *operand, T Type) bool {
	if check.conf.Logos && (logosContainsReadonlyDescendants(x.typ()) || logosContainsReadonlyDescendants(T)) {
		check.errorf(e, ImpossibleAssert, "%s cannot erase readonly descendants in a type assertion", logosConstEscape)
		x.invalidate()
		return true
	}
	return false
}
