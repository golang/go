// Copyright 2013 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package loader

import (
	"fmt"
	"go/ast"

	"code.google.com/p/go.tools/go/types"
)

// PackageInfo holds the ASTs and facts derived by the type-checker
// for a single package.
//
// Not mutated once exposed via the API.
//
type PackageInfo struct {
	Pkg                   *types.Package
	Importable            bool        // true if 'import "Pkg.Path()"' would resolve to this
	TransitivelyErrorFree bool        // true if Pkg and all its dependencies are free of errors
	Files                 []*ast.File // abstract syntax for the package's files
	Errors                []error     // non-nil if the package had errors
	types.Info                        // type-checker deductions.

	checker *types.Checker // transient type-checker state
}

func (info *PackageInfo) String() string {
	return fmt.Sprintf("PackageInfo(%s)", info.Pkg.Path())
}

// TODO(gri): move the methods below to types.Info.

// TypeOf returns the type of expression e, or nil if not found.
func (info *PackageInfo) TypeOf(e ast.Expr) types.Type {
	if t, ok := info.Types[e]; ok {
		return t.Type
	}
	// Idents appear only in Defs/Uses, not Types.
	if id, ok := e.(*ast.Ident); ok {
		return info.ObjectOf(id).Type()
	}
	return nil
}

// ObjectOf returns the type-checker object denoted by the specified
// id, or nil if not found.
//
// If id is an anonymous struct field, ObjectOf returns the field
// (*types.Var) it uses, not the type (*types.TypeName) it defines.
//
func (info *PackageInfo) ObjectOf(id *ast.Ident) types.Object {
	if obj, ok := info.Defs[id]; ok {
		return obj
	}
	return info.Uses[id]
}
