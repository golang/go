// Copyright 2013 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package importer

import (
	"fmt"
	"go/ast"

	"code.google.com/p/go.tools/go/exact"
	"code.google.com/p/go.tools/go/types"
)

// PackageInfo holds the ASTs and facts derived by the type-checker
// for a single package.
//
// Not mutated once constructed.
//
type PackageInfo struct {
	Pkg        *types.Package
	Importable bool        // true if 'import "Pkg.Path()"' would resolve to this
	Err        error       // non-nil if the package had static errors
	Files      []*ast.File // abstract syntax for the package's files
	types.Info             // type-checker deductions.
}

func (info *PackageInfo) String() string {
	return fmt.Sprintf("PackageInfo(%s)", info.Pkg.Path())
}

// TypeOf returns the type of expression e.
// Precondition: e belongs to the package's ASTs.
//
func (info *PackageInfo) TypeOf(e ast.Expr) types.Type {
	if t, ok := info.Types[e]; ok {
		return t
	}
	// Defining ast.Idents (id := expr) get only Ident callbacks
	// but not Expr callbacks.
	if id, ok := e.(*ast.Ident); ok {
		return info.ObjectOf(id).Type()
	}
	panic("no type for expression")
}

// ValueOf returns the value of expression e if it is a constant, nil
// otherwise.
// Precondition: e belongs to the package's ASTs.
//
func (info *PackageInfo) ValueOf(e ast.Expr) exact.Value {
	return info.Values[e]
}

// ObjectOf returns the typechecker object denoted by the specified id.
// Precondition: id belongs to the package's ASTs.
//
func (info *PackageInfo) ObjectOf(id *ast.Ident) types.Object {
	return info.Objects[id]
}

// IsType returns true iff expression e denotes a type.
// Precondition: e belongs to the package's ASTs.
//
// TODO(gri): move this into go/types.
//
func (info *PackageInfo) IsType(e ast.Expr) bool {
	switch e := e.(type) {
	case *ast.SelectorExpr: // pkg.Type
		if sel := info.Selections[e]; sel.Kind() == types.PackageObj {
			_, isType := sel.Obj().(*types.TypeName)
			return isType
		}
	case *ast.StarExpr: // *T
		return info.IsType(e.X)
	case *ast.Ident:
		_, isType := info.ObjectOf(e).(*types.TypeName)
		return isType
	case *ast.ArrayType, *ast.StructType, *ast.FuncType, *ast.InterfaceType, *ast.MapType, *ast.ChanType:
		return true
	case *ast.ParenExpr:
		return info.IsType(e.X)
	}
	return false
}

// TypeCaseVar returns the implicit variable created by a single-type
// case clause in a type switch, or nil if not found.
//
func (info *PackageInfo) TypeCaseVar(cc *ast.CaseClause) *types.Var {
	if v := info.Implicits[cc]; v != nil {
		return v.(*types.Var)
	}
	return nil
}

// ImportSpecPkg returns the PkgName for a given ImportSpec, possibly
// an implicit one for a dot-import or an import-without-rename.
// It returns nil if not found.
//
func (info *PackageInfo) ImportSpecPkg(spec *ast.ImportSpec) *types.PkgName {
	if spec.Name != nil {
		return info.ObjectOf(spec.Name).(*types.PkgName)
	}
	if p := info.Implicits[spec]; p != nil {
		return p.(*types.PkgName)
	}
	return nil
}
