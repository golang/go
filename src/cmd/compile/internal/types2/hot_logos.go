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
//
// The Update overlay below is the same idea for the callback: the Go
// declaration reads `Update(f func(old *T) *T)`; .lgo code sees
// `Update(f func(old const *T) *T)`, so ordinary assignability rejects
// a literal declaring `old *T` while a lambda takes the const
// parameter from its target and any store through `old` fails with the
// const-view error. Both hooks clone; neither mutates the imported
// signature, so plain Go and later calls keep the declared types.
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

// logosHotUpdateSig presents Hot.Update to .lgo callers with a const
// callback parameter (SPEC T-206): `Update(f func(old const *T) *T)`.
// It returns a clone of sig with the callback's first parameter wrapped
// in a const view, leaving the imported signature untouched so plain Go
// and unrelated calls keep `func(old *T) *T`. Calls that are not
// Hot.Update from logos/sync, or whose callback already takes a const
// view, come back unchanged.
func (check *Checker) logosHotUpdateSig(call *syntax.CallExpr, sig *Signature) *Signature {
	if !check.conf.Logos || sig == nil {
		return sig
	}
	sel, ok := syntax.Unparen(call.Fun).(*syntax.SelectorExpr)
	if !ok {
		return sig
	}
	fn, ok := check.Info.Uses[sel.Sel].(*Func)
	if !ok || fn.Name() != "Update" {
		return sig
	}
	if pkg := fn.Pkg(); pkg == nil || pkg.Path() != "logos/sync" {
		return sig
	}
	// The call signature of a bound method value carries no receiver
	// (selector stashes it in recvold), so the receiver check reads the
	// declared method type exactly like logosHotLoadResult does.
	msig, ok := fn.Type().(*Signature)
	if !ok || msig.Recv() == nil {
		return sig
	}
	recv := Unalias(msig.Recv().Type())
	if ptr, ok := recv.(*Pointer); ok {
		recv = Unalias(ptr.Elem())
	}
	named, ok := recv.(*Named)
	if !ok || named.Obj().Name() != "Hot" {
		return sig
	}
	if pkg := named.Obj().Pkg(); pkg == nil || pkg.Path() != "logos/sync" {
		return sig
	}
	if sig.params == nil || sig.params.Len() != 1 {
		return sig
	}
	cb, ok := Unalias(sig.params.At(0).Type()).(*Signature)
	if !ok || cb.params == nil || cb.params.Len() != 1 {
		return sig
	}
	if logosAsConstView(cb.params.At(0).Type()) != nil {
		return sig
	}
	// Update's callback takes exactly one parameter, so one cloned
	// parameter carries the const view; anything else keeps the
	// declared signature.
	first := *cb.params.At(0)
	first.setType(&ConstView{Elem: cb.params.At(0).Type()})
	cbCopy := *cb
	cbCopy.params = NewTuple(&first)
	outer := *sig.params.At(0)
	outer.setType(&cbCopy)
	outerCopy := *sig
	outerCopy.params = NewTuple(&outer)
	return &outerCopy
}
