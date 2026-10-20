// Copyright 2026 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package types2

import (
	"cmd/compile/internal/syntax"
)

// LoGo/s M8-S5: the .lgo view of logos/sync Hot.Load (SPEC T-206). The
// Go declaration reads `func (h *Hot[T]) Load() *T`; .lgo code sees
// `Load() const *T`, so mutation through the loaded pointer fails with
// the const-view error, exactly like a store through rt.Freeze's
// result.
//
// Like logosFreezeResult, a checker-side special case carries this, not
// facts: logos/sync is a Go package, and Go packages emit no const
// facts (T-Sec6), so a fact would need a new emitter plus a const
// overlay for Go packages that Go importers must never see. The hook
// wraps the already-instantiated result type, so every instantiation
// agrees, and it fires only in Logos mode, so plain Go keeps the
// declared signature. Upstream files take one hook line in callExpr
// beside the M5-S6 hook.
func (check *Checker) logosHotLoadResult(call *syntax.CallExpr, x *operand) {
	if !check.conf.Logos || !x.isValid() || x.mode() != value {
		return
	}
	sel, ok := syntax.Unparen(call.Fun).(*syntax.SelectorExpr)
	if !ok {
		return
	}
	fn, ok := check.Info.Uses[sel.Sel].(*Func)
	if !ok || fn.Name() != "Load" {
		return
	}
	if pkg := fn.Pkg(); pkg == nil || pkg.Path() != "logos/sync" {
		return
	}
	sig, ok := fn.Type().(*Signature)
	if !ok || sig.Recv() == nil {
		return
	}
	recv := Unalias(sig.Recv().Type())
	if ptr, ok := recv.(*Pointer); ok {
		recv = Unalias(ptr.Elem())
	}
	named, ok := recv.(*Named)
	if !ok || named.Obj().Name() != "Hot" {
		return
	}
	if pkg := named.Obj().Pkg(); pkg == nil || pkg.Path() != "logos/sync" {
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
