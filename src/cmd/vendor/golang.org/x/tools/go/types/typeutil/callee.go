// Copyright 2018 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package typeutil

import (
	"go/ast"
	"go/types"

	"golang.org/x/tools/internal/typesinternal"
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
	return typesinternal.Callee(info, call)
}

// StaticCallee returns the target (function or method) of a static function
// call, if any. It returns nil for calls to builtins.
//
// Note: for calls of instantiated functions and methods, StaticCallee returns
// the corresponding generic function or method on the generic type.
func StaticCallee(info *types.Info, call *ast.CallExpr) *types.Func {
	return typesinternal.StaticCallee(info, call)
}
