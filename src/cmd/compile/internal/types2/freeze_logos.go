// Copyright 2026 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package types2

import (
	"cmd/compile/internal/syntax"
)

// LoGo/s M5-S6: the .lgo view of logos/rt.Freeze (SPEC T-201). The Go
// declaration reads `func Freeze[T any](p *T) *T`; .lgo code sees
// `func Freeze[T any](p *T) const *T`, so the frozen result is a const
// view from the call on. Lowering erases const views (M5-S1), so the
// call needs no rewrite and Go importers keep reading the plain type
// from __.PKGDEF.
//
// A checker-side special case carries this, not facts: logos/rt is a
// Go package, and Go packages emit no const facts (T-Sec6), so a fact
// would need a new emitter plus a const overlay for Go packages that
// Go importers must never see. The hook wraps the already-instantiated
// result type, so inferred and explicitly instantiated calls agree,
// and it fires only in Logos mode, so plain Go keeps the declared
// signature. Upstream files take one hook line in callExpr.
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
