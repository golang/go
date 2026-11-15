// Copyright 2026 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package types2

import (
	"cmd/compile/internal/syntax"
)

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
