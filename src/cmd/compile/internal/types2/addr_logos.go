// Copyright 2026 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package types2

import (
	"cmd/compile/internal/syntax"
	. "internal/types/errors"
)

func (check *Checker) logosAddrFresh(x *operand, e *syntax.Operation) bool {
	if !check.conf.Logos {
		return false
	}
	if _, ok := syntax.Unparen(e.X).(*syntax.CompositeLit); ok {
		return false
	}
	if !x.isValid() || x.isNil() {
		return false
	}
	base, viaReference, constRooted := check.logosConstBase(e.X)
	switch {
	case constRooted && viaReference || !constRooted && x.mode() != variable && logosAsConstView(x.typ()) != nil:
		check.errorf(e, IncompatibleAssign, "%s cannot take the address of %s through const view %s", logosConstMutation, ExprString(e.X), ExprString(base))
		x.invalidate()
		return true
	case !constRooted && x.mode() == variable:
		return false
	}
	x.mode_ = value
	typ := x.typ()
	if constRooted {
		base, _, _ := logosReadonlyBase(typ)
		typ = logosWithReadonlyDescendants(base)
	}
	x.typ_ = &Pointer{base: Default(typ)}
	if check.Info != nil {
		if check.Info.LogosAddrFresh == nil {
			check.Info.LogosAddrFresh = make(map[syntax.Expr]bool)
		}
		check.Info.LogosAddrFresh[e] = true
	}
	return true
}
