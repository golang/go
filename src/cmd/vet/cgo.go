// Copyright 2015 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// Check for invalid cgo pointer passing.
// This looks for code that uses cgo to call C code passing values
// whose types are almost always invalid according to the cgo pointer
// sharing rules.
// Specifically, it warns about attempts to pass a Go chan, map, func,
// or slice to C, either directly, or via a pointer, array, or struct.

package main

import (
	"go/ast"
	"go/token"
	"go/types"
)

func init() {
	register("cgocall",
		"check for types that may not be passed to cgo calls",
		checkCgoCall,
		callExpr)
}

func checkCgoCall(f *File, node ast.Node) {
	x := node.(*ast.CallExpr)

	// We are only looking for calls to functions imported from
	// the "C" package.
	sel, ok := x.Fun.(*ast.SelectorExpr)
	if !ok {
		return
	}
	id, ok := sel.X.(*ast.Ident)
	if !ok || id.Name != "C" {
		return
	}

	for _, arg := range x.Args {
		if !typeOKForCgoCall(cgoBaseType(f, arg)) {
			f.Badf(arg.Pos(), "possibly passing Go type with embedded pointer to C")
		}

		// Check for passing the address of a bad type.
		if conv, ok := arg.(*ast.CallExpr); ok && len(conv.Args) == 1 && f.hasBasicType(conv.Fun, types.UnsafePointer) {
			arg = conv.Args[0]
		}
		if u, ok := arg.(*ast.UnaryExpr); ok && u.Op == token.AND {
			if !typeOKForCgoCall(cgoBaseType(f, u.X)) {
				f.Badf(arg.Pos(), "possibly passing Go type with embedded pointer to C")
			}
		}
	}
}

// cgoBaseType tries to look through type conversions involving
// unsafe.Pointer to find the real type. It converts:
//   unsafe.Pointer(x) => x
//   *(*unsafe.Pointer)(unsafe.Pointer(&x)) => x
func cgoBaseType(f *File, arg ast.Expr) types.Type {
	switch arg := arg.(type) {
	case *ast.CallExpr:
		if len(arg.Args) == 1 && f.hasBasicType(arg.Fun, types.UnsafePointer) {
			return cgoBaseType(f, arg.Args[0])
		}
	case *ast.StarExpr:
		call, ok := arg.X.(*ast.CallExpr)
		if !ok || len(call.Args) != 1 {
			break
		}
		// Here arg is *f(v).
		t := f.pkg.types[call.Fun].Type
		if t == nil {
			break
		}
		ptr, ok := t.Underlying().(*types.Pointer)
		if !ok {
			break
		}
		// Here arg is *(*p)(v)
		elem, ok := ptr.Elem().Underlying().(*types.Basic)
		if !ok || elem.Kind() != types.UnsafePointer {
			break
		}
		// Here arg is *(*unsafe.Pointer)(v)
		call, ok = call.Args[0].(*ast.CallExpr)
		if !ok || len(call.Args) != 1 {
			break
		}
		// Here arg is *(*unsafe.Pointer)(f(v))
		if !f.hasBasicType(call.Fun, types.UnsafePointer) {
			break
		}
		// Here arg is *(*unsafe.Pointer)(unsafe.Pointer(v))
		u, ok := call.Args[0].(*ast.UnaryExpr)
		if !ok || u.Op != token.AND {
			break
		}
		// Here arg is *(*unsafe.Pointer)(unsafe.Pointer(&v))
		return cgoBaseType(f, u.X)
	}

	return f.pkg.types[arg].Type
}

// typeOKForCgoCall returns true if the type of arg is OK to pass to a
// C function using cgo. This is not true for Go types with embedded
// pointers.
func typeOKForCgoCall(t types.Type) bool {
	if t == nil {
		return true
	}
	switch t := t.Underlying().(type) {
	case *types.Chan, *types.Map, *types.Signature, *types.Slice:
		return false
	case *types.Pointer:
		return typeOKForCgoCall(t.Elem())
	case *types.Array:
		return typeOKForCgoCall(t.Elem())
	case *types.Struct:
		for i := 0; i < t.NumFields(); i++ {
			if !typeOKForCgoCall(t.Field(i).Type()) {
				return false
			}
		}
	}
	return true
}
