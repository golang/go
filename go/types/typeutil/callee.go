// Copyright 2018 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package typeutil

import (
	"go/ast"
	"go/types"
)

// StaticCallee returns the target (function or method) of a static
// function call, if any. It returns nil for calls to builtin.
func StaticCallee(info *types.Info, call *ast.CallExpr) *types.Func {
	var obj types.Object
	switch fun := call.Fun.(type) {
	case *ast.Ident:
		obj = info.Uses[fun] // type, var, builtin, or declared func
	case *ast.SelectorExpr:
		if sel, ok := info.Selections[fun]; ok {
			obj = sel.Obj() // method or field
		} else {
			obj = info.Uses[fun.Sel] // qualified identifier?
		}
	}
	if f, ok := obj.(*types.Func); ok && !interfaceMethod(f) {
		return f
	}
	return nil
}

func interfaceMethod(f *types.Func) bool {
	recv := f.Type().(*types.Signature).Recv()
	return recv != nil && types.IsInterface(recv.Type())
}
