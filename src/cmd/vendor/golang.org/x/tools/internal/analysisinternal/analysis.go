// Copyright 2020 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// Package analysisinternal exposes internal-only fields from go/analysis.
package analysisinternal

import (
	"bytes"
	"fmt"
	"go/ast"
	"go/token"
	"go/types"
	"strings"

	"golang.org/x/tools/go/ast/astutil"
)

func TypeErrorEndPos(fset *token.FileSet, src []byte, start token.Pos) token.Pos {
	// Get the end position for the type error.
	offset, end := fset.PositionFor(start, false).Offset, start
	if offset >= len(src) {
		return end
	}
	if width := bytes.IndexAny(src[offset:], " \n,():;[]+-*"); width > 0 {
		end = start + token.Pos(width)
	}
	return end
}

func ZeroValue(fset *token.FileSet, f *ast.File, pkg *types.Package, typ types.Type) ast.Expr {
	under := typ
	if n, ok := typ.(*types.Named); ok {
		under = n.Underlying()
	}
	switch u := under.(type) {
	case *types.Basic:
		switch {
		case u.Info()&types.IsNumeric != 0:
			return &ast.BasicLit{Kind: token.INT, Value: "0"}
		case u.Info()&types.IsBoolean != 0:
			return &ast.Ident{Name: "false"}
		case u.Info()&types.IsString != 0:
			return &ast.BasicLit{Kind: token.STRING, Value: `""`}
		default:
			panic("unknown basic type")
		}
	case *types.Chan, *types.Interface, *types.Map, *types.Pointer, *types.Signature, *types.Slice:
		return ast.NewIdent("nil")
	case *types.Struct:
		texpr := typeExpr(fset, f, pkg, typ) // typ because we want the name here.
		if texpr == nil {
			return nil
		}
		return &ast.CompositeLit{
			Type: texpr,
		}
	case *types.Array:
		texpr := typeExpr(fset, f, pkg, u.Elem())
		if texpr == nil {
			return nil
		}
		return &ast.CompositeLit{
			Type: &ast.ArrayType{
				Elt: texpr,
				Len: &ast.BasicLit{Kind: token.INT, Value: fmt.Sprintf("%v", u.Len())},
			},
		}
	}
	return nil
}

func typeExpr(fset *token.FileSet, f *ast.File, pkg *types.Package, typ types.Type) ast.Expr {
	switch t := typ.(type) {
	case *types.Basic:
		switch t.Kind() {
		case types.UnsafePointer:
			return &ast.SelectorExpr{X: ast.NewIdent("unsafe"), Sel: ast.NewIdent("Pointer")}
		default:
			return ast.NewIdent(t.Name())
		}
	case *types.Named:
		if t.Obj().Pkg() == pkg {
			return ast.NewIdent(t.Obj().Name())
		}
		pkgName := t.Obj().Pkg().Name()
		// If the file already imports the package under another name, use that.
		for _, group := range astutil.Imports(fset, f) {
			for _, cand := range group {
				if strings.Trim(cand.Path.Value, `"`) == t.Obj().Pkg().Path() {
					if cand.Name != nil && cand.Name.Name != "" {
						pkgName = cand.Name.Name
					}
				}
			}
		}
		if pkgName == "." {
			return ast.NewIdent(t.Obj().Name())
		}
		return &ast.SelectorExpr{
			X:   ast.NewIdent(pkgName),
			Sel: ast.NewIdent(t.Obj().Name()),
		}
	default:
		return nil // TODO: anonymous structs, but who does that
	}
}

var GetTypeErrors = func(p interface{}) []types.Error { return nil }
var SetTypeErrors = func(p interface{}, errors []types.Error) {}

type TypeErrorPass string

const (
	NoNewVars      TypeErrorPass = "nonewvars"
	NoResultValues TypeErrorPass = "noresultvalues"
	UndeclaredName TypeErrorPass = "undeclaredname"
)
