// Copyright 2026 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package noder

import (
	"cmd/compile/internal/syntax"
	"cmd/compile/internal/types2"
)

func logosWriterNormalizeType(typ types2.Type) types2.Type {
normalize:
	for {
		switch t := typ.(type) {
		case *types2.ConstView:
			typ = t.Elem
		case *types2.ReadonlyDescendants:
			typ = t.Elem
		case *types2.Alias:
			if isGlobal(t.Obj()) {
				break normalize
			}
			typ = t.Rhs()
		default:
			break normalize
		}
	}
	return typ
}

func logosWriterIsNil(obj types2.Object, tv syntax.TypeAndValue) bool {
	_, isNil := obj.(*types2.Nil)
	return isNil || tv.IsNil()
}

func unconstView(t types2.Type) types2.Type {
	for {
		switch typ := types2.Unalias(t).(type) {
		case *types2.ConstView:
			t = typ.Elem
			continue
		case *types2.ReadonlyDescendants:
			t = typ.Elem
			continue
		}
		return t
	}
}

func (w *writer) logosWriterConvertExpr(src, dst types2.Type, expr syntax.Expr, implicit bool) (identical, done bool) {
	usrc := unconstView(src)
	udst := unconstView(dst)
	identical = dst == nil || types2.Identical(src, dst) || (implicit && types2.Identical(usrc, udst))
	if implicit && identical {
		w.expr(expr)
		return identical, true
	}
	if implicit && !types2.AssignableTo(src, dst) && !types2.AssignableTo(usrc, udst) {
		w.p.fatalf(expr, "%v is not assignable to %v", src, dst)
	}
	return identical, false
}
