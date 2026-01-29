// Copyright 2018 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package typeutil

import (
	"go/ast"
	"go/types"
	_ "unsafe" // for linkname
)

// Callee returns the named target of a function call, if any:
// a function, method, builtin, or variable.
// It returns nil for a T(x) conversion.
//
// Functions and methods may potentially have type parameters.
//
// Note: for calls of instantiated functions and methods, Callee returns
// the corresponding generic function or method on the generic type.
func Callee(info *types.Info, call *ast.CallExpr) types.Object {
	obj := info.Uses[usedIdent(info, call.Fun)]
	if obj == nil {
		return nil
	}
	if _, ok := obj.(*types.TypeName); ok {
		return nil
	}
	return obj
}

// StaticCallee returns the target (function or method) of a static function
// call, if any. It returns nil for calls to builtins.
//
// Note: for calls of instantiated functions and methods, StaticCallee returns
// the corresponding generic function or method on the generic type.
func StaticCallee(info *types.Info, call *ast.CallExpr) *types.Func {
	obj := info.Uses[usedIdent(info, call.Fun)]
	fn, _ := obj.(*types.Func)
	if fn == nil || interfaceMethod(fn) {
		return nil
	}
	return fn
}

// usedIdent is the implementation of [internal/typesinternal.UsedIdent].
// It returns the identifier associated with e.
// See typesinternal.UsedIdent for a fuller description.
// This function should live in typesinternal, but cannot because it would
// create an import cycle.
//
//go:linkname usedIdent golang.org/x/tools/go/types/typeutil.usedIdent
func usedIdent(info *types.Info, e ast.Expr) *ast.Ident {
	if info.Types == nil || info.Uses == nil {
		panic("one of info.Types or info.Uses is nil; both must be populated")
	}
	// Look through type instantiation if necessary.
	switch d := ast.Unparen(e).(type) {
	case *ast.IndexExpr:
		if info.Types[d.Index].IsType() {
			e = d.X
		}
	case *ast.IndexListExpr:
		e = d.X
	}

	switch e := ast.Unparen(e).(type) {
	// info.Uses always has the object we want, even for selector expressions.
	// We don't need info.Selections.
	// See go/types/recording.go:recordSelection.
	case *ast.Ident:
		return e
	case *ast.SelectorExpr:
		return e.Sel
	}
	return nil
}

// interfaceMethod reports whether its argument is a method of an interface.
// This function should live in typesinternal, but cannot because it would create an import cycle.
//
//go:linkname interfaceMethod golang.org/x/tools/go/types/typeutil.interfaceMethod
func interfaceMethod(f *types.Func) bool {
	recv := f.Signature().Recv()
	return recv != nil && types.IsInterface(recv.Type())
}
