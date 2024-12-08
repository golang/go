// Copyright 2018 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package typeutil

import (
	"go/ast"
	"go/types"

	"golang.org/x/tools/internal/typeparams"
)

// Callee returns the named target of a function call, if any:
// a function, method, builtin, or variable.
//
// Functions and methods may potentially have type parameters.
func Callee(info *types.Info, call *ast.CallExpr) types.Object {
	fun := ast.Unparen(call.Fun)

	// Look through type instantiation if necessary.
	isInstance := false
	switch fun.(type) {
	case *ast.IndexExpr, *ast.IndexListExpr:
		// When extracting the callee from an *IndexExpr, we need to check that
		// it is a *types.Func and not a *types.Var.
		// Example: Don't match a slice m within the expression `m[0]()`.
		isInstance = true
		fun, _, _, _ = typeparams.UnpackIndexExpr(fun)
	}

	var obj types.Object
	switch fun := fun.(type) {
	case *ast.Ident:
		obj = info.Uses[fun] // type, var, builtin, or declared func
	case *ast.SelectorExpr:
		if sel, ok := info.Selections[fun]; ok {
			obj = sel.Obj() // method or field
		} else {
			obj = info.Uses[fun.Sel] // qualified identifier?
		}
	}
	if _, ok := obj.(*types.TypeName); ok {
		return nil // T(x) is a conversion, not a call
	}
	// A Func is required to match instantiations.
	if _, ok := obj.(*types.Func); isInstance && !ok {
		return nil // Was not a Func.
	}
	return obj
}

// StaticCallee returns the target (function or method) of a static function
// call, if any. It returns nil for calls to builtins.
//
// Note: for calls of instantiated functions and methods, StaticCallee returns
// the corresponding generic function or method on the generic type.
func StaticCallee(info *types.Info, call *ast.CallExpr) *types.Func {
	if f, ok := Callee(info, call).(*types.Func); ok && !interfaceMethod(f) {
		return f
	}
	return nil
}

func interfaceMethod(f *types.Func) bool {
	recv := f.Type().(*types.Signature).Recv()
	return recv != nil && types.IsInterface(recv.Type())
}
