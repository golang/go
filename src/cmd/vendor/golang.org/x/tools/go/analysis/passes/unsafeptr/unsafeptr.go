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
	"golang.org/x/tools/go/analysis/passes/internal/analysisutil"
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
		(*ast.StarExpr)(nil),
		(*ast.UnaryExpr)(nil),
	}
	inspect.Preorder(nodeFilter, func(n ast.Node) {
		switch x := n.(type) {
		case *ast.CallExpr:
			if len(x.Args) == 1 &&
				hasBasicType(pass.TypesInfo, x.Fun, types.UnsafePointer) &&
				hasBasicType(pass.TypesInfo, x.Args[0], types.Uintptr) &&
				!isSafeUintptr(pass.TypesInfo, x.Args[0]) {
				pass.ReportRangef(x, "possible misuse of unsafe.Pointer")
			}
		case *ast.StarExpr:
			if t := pass.TypesInfo.Types[x].Type; isReflectHeader(t) {
				pass.ReportRangef(x, "possible misuse of %s", t)
			}
		case *ast.UnaryExpr:
			if x.Op != token.AND {
				return
			}
			if t := pass.TypesInfo.Types[x.X].Type; isReflectHeader(t) {
				pass.ReportRangef(x, "possible misuse of %s", t)
			}
		}
	})
	return nil, nil
}

// isSafeUintptr reports whether x - already known to be a uintptr -
// is safe to convert to unsafe.Pointer.
func isSafeUintptr(info *types.Info, x ast.Expr) bool {
	// Check unsafe.Pointer safety rules according to
	// https://golang.org/pkg/unsafe/#Pointer.

	switch x := analysisutil.Unparen(x).(type) {
	case *ast.SelectorExpr:
		// "(6) Conversion of a reflect.SliceHeader or
		// reflect.StringHeader Data field to or from Pointer."
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
		if ok && isReflectHeader(pt.Elem()) {
			return true
		}

	case *ast.CallExpr:
		// "(5) Conversion of the result of reflect.Value.Pointer or
		// reflect.Value.UnsafeAddr from uintptr to Pointer."
		if len(x.Args) != 0 {
			break
		}
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
	}

	// "(3) Conversion of a Pointer to a uintptr and back, with arithmetic."
	return isSafeArith(info, x)
}

// isSafeArith reports whether x is a pointer arithmetic expression that is safe
// to convert to unsafe.Pointer.
func isSafeArith(info *types.Info, x ast.Expr) bool {
	switch x := analysisutil.Unparen(x).(type) {
	case *ast.CallExpr:
		// Base case: initial conversion from unsafe.Pointer to uintptr.
		return len(x.Args) == 1 &&
			hasBasicType(info, x.Fun, types.Uintptr) &&
			hasBasicType(info, x.Args[0], types.UnsafePointer)

	case *ast.BinaryExpr:
		// "It is valid both to add and to subtract offsets from a
		// pointer in this way. It is also valid to use &^ to round
		// pointers, usually for alignment."
		switch x.Op {
		case token.ADD, token.SUB, token.AND_NOT:
			// TODO(mdempsky): Match compiler
			// semantics. ADD allows a pointer on either
			// side; SUB and AND_NOT don't care about RHS.
			return isSafeArith(info, x.X) && !isSafeArith(info, x.Y)
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

// isReflectHeader reports whether t is reflect.SliceHeader or reflect.StringHeader.
func isReflectHeader(t types.Type) bool {
	if named, ok := t.(*types.Named); ok {
		if obj := named.Obj(); obj.Pkg() != nil && obj.Pkg().Path() == "reflect" {
			switch obj.Name() {
			case "SliceHeader", "StringHeader":
				return true
			}
		}
	}
	return false
}
