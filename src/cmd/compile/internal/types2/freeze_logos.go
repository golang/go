// Copyright 2026 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package types2

import (
	"cmd/compile/internal/syntax"
)

func (check *Checker) logosFreezeResult(call *syntax.CallExpr, x *operand) {
	if !check.conf.Logos || !x.isValid() || x.mode() != value {
		return
	}
	fun := syntax.Unparen(call.Fun)
	if idx, ok := fun.(*syntax.IndexExpr); ok {
		fun = syntax.Unparen(idx.X)
	}
	sel, ok := fun.(*syntax.SelectorExpr)
	if !ok {
		return
	}
	fn, ok := check.Info.Uses[sel.Sel].(*Func)
	if !ok || fn.Name() != "Freeze" {
		return
	}
	if pkg := fn.Pkg(); pkg == nil || pkg.Path() != "logos/rt" {
		return
	}
	t := x.typ()
	if t == nil {
		return
	}
	if _, ok := Unalias(t).(*Pointer); !ok {
		return
	}
	x.typ_ = &ConstView{Elem: t}
	check.updateExprType(x.expr, x.typ_, false)
}
