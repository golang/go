// Copyright 2014 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// Package unsafeptr defines an Analyzer that checks for invalid
// conversions of uintptr to unsafe.Pointer.
package unsafeptr

import (
	"go/ast"
	"go/token"
	"go/types"

	"golang.org/x/tools/go/analysis"
	"golang.org/x/tools/go/analysis/passes/inspect"
	"golang.org/x/tools/go/ast/inspector"
)

const Doc = `check for invalid conversions of uintptr to unsafe.Pointer

The unsafeptr analyzer reports likely incorrect uses of unsafe.Pointer
to convert integers to pointers. A conversion from uintptr to
unsafe.Pointer is invalid if it implies that there is a uintptr-typed
word in memory that holds a pointer value, because that word will be
invisible to stack copying and to the garbage collector.`

var Analyzer = &analysis.Analyzer{
	Name:     "unsafeptr",
	Doc:      Doc,
	Requires: []*analysis.Analyzer{inspect.Analyzer},
	Run:      run,
}

func run(pass *analysis.Pass) (interface{}, error) {
	inspect := pass.ResultOf[inspect.Analyzer].(*inspector.Inspector)

	nodeFilter := []ast.Node{
		(*ast.CallExpr)(nil),
	}
	inspect.Preorder(nodeFilter, func(n ast.Node) {
		x := n.(*ast.CallExpr)
		if len(x.Args) != 1 {
			return
		}
		if hasBasicType(pass.TypesInfo, x.Fun, types.UnsafePointer) &&
			hasBasicType(pass.TypesInfo, x.Args[0], types.Uintptr) &&
			!isSafeUintptr(pass.TypesInfo, x.Args[0]) {
			pass.Reportf(x.Pos(), "possible misuse of unsafe.Pointer")
		}
	})
	return nil, nil
}

// isSafeUintptr reports whether x - already known to be a uintptr -
// is safe to convert to unsafe.Pointer. It is safe if x is itself derived
// directly from an unsafe.Pointer via conversion and pointer arithmetic
// or if x is the result of reflect.Value.Pointer or reflect.Value.UnsafeAddr
// or obtained from the Data field of a *reflect.SliceHeader or *reflect.StringHeader.
func isSafeUintptr(info *types.Info, x ast.Expr) bool {
	switch x := x.(type) {
	case *ast.ParenExpr:
		return isSafeUintptr(info, x.X)

	case *ast.SelectorExpr:
		if x.Sel.Name != "Data" {
			break
		}
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
		pt, ok := info.Types[x.X].Type.(*types.Pointer)
		if ok {
			t, ok := pt.Elem().(*types.Named)
			if ok && t.Obj().Pkg().Path() == "reflect" {
				switch t.Obj().Name() {
				case "StringHeader", "SliceHeader":
					return true
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
				t, ok := info.Types[sel.X].Type.(*types.Named)
				if ok && t.Obj().Pkg().Path() == "reflect" && t.Obj().Name() == "Value" {
					return true
				}
			}

		case 1:
			// maybe conversion of uintptr to unsafe.Pointer
			return hasBasicType(info, x.Fun, types.Uintptr) &&
				hasBasicType(info, x.Args[0], types.UnsafePointer)
		}

	case *ast.BinaryExpr:
		switch x.Op {
		case token.ADD, token.SUB, token.AND_NOT:
			return isSafeUintptr(info, x.X) && !isSafeUintptr(info, x.Y)
		}
	}
	return false
}

// hasBasicType reports whether x's type is a types.Basic with the given kind.
func hasBasicType(info *types.Info, x ast.Expr, kind types.BasicKind) bool {
	t := info.Types[x].Type
	if t != nil {
		t = t.Underlying()
	}
	b, ok := t.(*types.Basic)
	return ok && b.Kind() == kind
}
