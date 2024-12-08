// Copyright 2023 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package testinggoroutine

import (
	"go/ast"
	"go/types"

	"golang.org/x/tools/internal/typeparams"
)

// AST and types utilities that not specific to testinggoroutines.

// localFunctionDecls returns a mapping from *types.Func to *ast.FuncDecl in files.
func localFunctionDecls(info *types.Info, files []*ast.File) func(*types.Func) *ast.FuncDecl {
	var fnDecls map[*types.Func]*ast.FuncDecl // computed lazily
	return func(f *types.Func) *ast.FuncDecl {
		if f != nil && fnDecls == nil {
			fnDecls = make(map[*types.Func]*ast.FuncDecl)
			for _, file := range files {
				for _, decl := range file.Decls {
					if fnDecl, ok := decl.(*ast.FuncDecl); ok {
						if fn, ok := info.Defs[fnDecl.Name].(*types.Func); ok {
							fnDecls[fn] = fnDecl
						}
					}
				}
			}
		}
		// TODO: set f = f.Origin() here.
		return fnDecls[f]
	}
}

// isMethodNamed returns true if f is a method defined
// in package with the path pkgPath with a name in names.
func isMethodNamed(f *types.Func, pkgPath string, names ...string) bool {
	if f == nil {
		return false
	}
	if f.Pkg() == nil || f.Pkg().Path() != pkgPath {
		return false
	}
	if f.Type().(*types.Signature).Recv() == nil {
		return false
	}
	for _, n := range names {
		if f.Name() == n {
			return true
		}
	}
	return false
}

func funcIdent(fun ast.Expr) *ast.Ident {
	switch fun := ast.Unparen(fun).(type) {
	case *ast.IndexExpr, *ast.IndexListExpr:
		x, _, _, _ := typeparams.UnpackIndexExpr(fun) // necessary?
		id, _ := x.(*ast.Ident)
		return id
	case *ast.Ident:
		return fun
	default:
		return nil
	}
}

// funcLitInScope returns a FuncLit that id is at least initially assigned to.
//
// TODO: This is closely tied to id.Obj which is deprecated.
func funcLitInScope(id *ast.Ident) *ast.FuncLit {
	// Compare to (*ast.Object).Pos().
	if id.Obj == nil {
		return nil
	}
	var rhs ast.Expr
	switch d := id.Obj.Decl.(type) {
	case *ast.AssignStmt:
		for i, x := range d.Lhs {
			if ident, isIdent := x.(*ast.Ident); isIdent && ident.Name == id.Name && i < len(d.Rhs) {
				rhs = d.Rhs[i]
			}
		}
	case *ast.ValueSpec:
		for i, n := range d.Names {
			if n.Name == id.Name && i < len(d.Values) {
				rhs = d.Values[i]
			}
		}
	}
	lit, _ := rhs.(*ast.FuncLit)
	return lit
}
