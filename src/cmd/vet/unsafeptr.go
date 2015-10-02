// Copyright 2014 The Go Authors.  All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// Check for invalid uintptr -> unsafe.Pointer conversions.

package main

import (
	"go/ast"
	"go/token"
	"go/types"
)

func init() {
	register("unsafeptr",
		"check for misuse of unsafe.Pointer",
		checkUnsafePointer,
		callExpr)
}

func checkUnsafePointer(f *File, node ast.Node) {
	x := node.(*ast.CallExpr)
	if len(x.Args) != 1 {
		return
	}
	if f.hasBasicType(x.Fun, types.UnsafePointer) && f.hasBasicType(x.Args[0], types.Uintptr) && !f.isSafeUintptr(x.Args[0]) {
		f.Badf(x.Pos(), "possible misuse of unsafe.Pointer")
	}
}

// isSafeUintptr reports whether x - already known to be a uintptr -
// is safe to convert to unsafe.Pointer. It is safe if x is itself derived
// directly from an unsafe.Pointer via conversion and pointer arithmetic
// or if x is the result of reflect.Value.Pointer or reflect.Value.UnsafeAddr
// or obtained from the Data field of a *reflect.SliceHeader or *reflect.StringHeader.
func (f *File) isSafeUintptr(x ast.Expr) bool {
	switch x := x.(type) {
	case *ast.ParenExpr:
		return f.isSafeUintptr(x.X)

	case *ast.SelectorExpr:
		switch x.Sel.Name {
		case "Data":
			// reflect.SliceHeader and reflect.StringHeader are okay,
			// but only if they are pointing at a real slice or string.
			// It's not okay to do:
			//	var x SliceHeader
			//	x.Data = uintptr(unsafe.Pointer(...))
			//	... use x ...
			//	p := unsafe.Pointer(x.Data)
			// because in the middle the garbage collector doesn't
			// see x.Data as a pointer and so x.Data may be dangling
			// by the time we get to the conversion at the end.
			// For now approximate by saying that *Header is okay
			// but Header is not.
			pt, ok := f.pkg.types[x.X].Type.(*types.Pointer)
			if ok {
				t, ok := pt.Elem().(*types.Named)
				if ok && t.Obj().Pkg().Path() == "reflect" {
					switch t.Obj().Name() {
					case "StringHeader", "SliceHeader":
						return true
					}
				}
			}
		}

	case *ast.CallExpr:
		switch len(x.Args) {
		case 0:
			// maybe call to reflect.Value.Pointer or reflect.Value.UnsafeAddr.
			sel, ok := x.Fun.(*ast.SelectorExpr)
			if !ok {
				break
			}
			switch sel.Sel.Name {
			case "Pointer", "UnsafeAddr":
				t, ok := f.pkg.types[sel.X].Type.(*types.Named)
				if ok && t.Obj().Pkg().Path() == "reflect" && t.Obj().Name() == "Value" {
					return true
				}
			}

		case 1:
			// maybe conversion of uintptr to unsafe.Pointer
			return f.hasBasicType(x.Fun, types.Uintptr) && f.hasBasicType(x.Args[0], types.UnsafePointer)
		}

	case *ast.BinaryExpr:
		switch x.Op {
		case token.ADD, token.SUB:
			return f.isSafeUintptr(x.X) && !f.isSafeUintptr(x.Y)
		}
	}
	return false
}
