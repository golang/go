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
	call, _ := syntax.Unparen(e).(*syntax.CallExpr)
	return call
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

// calleeFunc returns the function or method call invokes, or nil. An
// explicit instantiation f[T] names the generic function f.
func calleeFunc(pass *Pass, call *syntax.CallExpr) *types2.Func {
	fun := syntax.Unparen(call.Fun)
	if inst, ok := fun.(*syntax.IndexExpr); ok {
		fun = syntax.Unparen(inst.X)
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

func isChainCall(pass *Pass, call *syntax.CallExpr) bool {
	sel, ok := syntax.Unparen(call.Fun).(*syntax.SelectorExpr)
	return ok && types2.LogosChainMethod(sel.Sel.Value) && pass.Info.Selections[sel] == nil && pass.Info.Uses[sel.Sel] == nil
}

func chainFuncs(pass *Pass, call *syntax.CallExpr) []syntax.Expr {
	if !isChainCall(pass, call) {
		return nil
	}
	return funcArgs(call)
}

func funcArgs(call *syntax.CallExpr) []syntax.Expr {
	var fns []syntax.Expr
	for _, arg := range call.ArgList {
		switch fn := syntax.Unparen(arg).(type) {
		case *syntax.LambdaExpr, *syntax.FuncLit:
			fns = append(fns, fn)
		}
	}
	return fns
}

func stageOf(pass *Pass, call *syntax.CallExpr) *syntax.CallExpr {
	for cur := call; isSegmentCall(cur); {
		recv := callOf(syntax.Unparen(cur.Fun).(*syntax.SelectorExpr).X)
		if recv == nil {
			return nil
		}
		if _, ok := types2.LogosGroupStage(pass.Pkg, pass.Info, recv); ok {
			return recv
		}
		cur = recv
	}
	return nil
}

func isSegmentCall(call *syntax.CallExpr) bool {
	sel, ok := syntax.Unparen(call.Fun).(*syntax.SelectorExpr)
	return ok && sel.Sel != nil && (sel.Sel.Value == "map" || sel.Sel.Value == "filter")
}

func onStage(pass *Pass, e syntax.Expr) bool {
	for call := callOf(e); call != nil; {
		if _, ok := types2.LogosGroupStage(pass.Pkg, pass.Info, call); ok {
			return true
		}
		sel, ok := syntax.Unparen(call.Fun).(*syntax.SelectorExpr)
		if !ok {
			return false
		}
		call = callOf(sel.X)
	}
	return false
}
