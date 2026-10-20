// Copyright 2026 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package lint

import (
	"cmd/compile/internal/syntax"
	"cmd/compile/internal/types2"
)

// callOf returns e as a call expression, looking through parentheses.
func callOf(e syntax.Expr) *syntax.CallExpr {
	for {
		switch x := e.(type) {
		case *syntax.ParenExpr:
			e = x.X
		case *syntax.CallExpr:
			return x
		default:
			return nil
		}
	}
}

// exprList returns the elements of a tuple expression, or e itself.
func exprList(e syntax.Expr) []syntax.Expr {
	if l, ok := e.(*syntax.ListExpr); ok {
		return l.ElemList
	}
	return []syntax.Expr{e}
}

// isBlank reports whether e is the blank identifier.
func isBlank(e syntax.Expr) bool {
	n, ok := e.(*syntax.Name)
	return ok && n.Value == "_"
}

// resultTypes returns the result types of call. A conversion or a call
// with no checker type has none.
func resultTypes(call *syntax.CallExpr) []types2.Type {
	t := TypeOf(call)
	if t == nil {
		return nil
	}
	tuple, ok := t.(*types2.Tuple)
	if !ok {
		return []types2.Type{t}
	}
	out := make([]types2.Type, tuple.Len())
	for i := range out {
		out[i] = tuple.At(i).Type()
	}
	return out
}

// calleeFunc returns the function or method call invokes, or nil.
func calleeFunc(pass *Pass, call *syntax.CallExpr) *types2.Func {
	fun := call.Fun
	for {
		p, ok := fun.(*syntax.ParenExpr)
		if !ok {
			break
		}
		fun = p.X
	}
	var id *syntax.Name
	switch f := fun.(type) {
	case *syntax.Name:
		id = f
	case *syntax.SelectorExpr:
		id = f.Sel
	default:
		return nil
	}
	fn, _ := pass.Info.Uses[id].(*types2.Func)
	return fn
}

// isErrorType reports whether t is the predeclared error type.
func isErrorType(t types2.Type) bool {
	if t == nil {
		return false
	}
	named, ok := types2.Unalias(t).(*types2.Named)
	return ok && named.Obj().Pkg() == nil && named.Obj().Name() == "error"
}
