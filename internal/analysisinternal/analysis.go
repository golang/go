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

var (
	GetTypeErrors func(p interface{}) []types.Error
	SetTypeErrors func(p interface{}, errors []types.Error)
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
		texpr := TypeExpr(fset, f, pkg, typ) // typ because we want the name here.
		if texpr == nil {
			return nil
		}
		return &ast.CompositeLit{
			Type: texpr,
		}
	case *types.Array:
		texpr := TypeExpr(fset, f, pkg, u.Elem())
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

func TypeExpr(fset *token.FileSet, f *ast.File, pkg *types.Package, typ types.Type) ast.Expr {
	switch t := typ.(type) {
	case *types.Basic:
		switch t.Kind() {
		case types.UnsafePointer:
			return &ast.SelectorExpr{X: ast.NewIdent("unsafe"), Sel: ast.NewIdent("Pointer")}
		default:
			return ast.NewIdent(t.Name())
		}
	case *types.Pointer:
		x := TypeExpr(fset, f, pkg, t.Elem())
		if x == nil {
			return nil
		}
		return &ast.UnaryExpr{
			Op: token.MUL,
			X:  x,
		}
	case *types.Array:
		elt := TypeExpr(fset, f, pkg, t.Elem())
		if elt == nil {
			return nil
		}
		return &ast.ArrayType{
			Len: &ast.BasicLit{
				Kind:  token.INT,
				Value: fmt.Sprintf("%d", t.Len()),
			},
			Elt: elt,
		}
	case *types.Slice:
		elt := TypeExpr(fset, f, pkg, t.Elem())
		if elt == nil {
			return nil
		}
		return &ast.ArrayType{
			Elt: elt,
		}
	case *types.Map:
		key := TypeExpr(fset, f, pkg, t.Key())
		value := TypeExpr(fset, f, pkg, t.Elem())
		if key == nil || value == nil {
			return nil
		}
		return &ast.MapType{
			Key:   key,
			Value: value,
		}
	case *types.Chan:
		dir := ast.ChanDir(t.Dir())
		if t.Dir() == types.SendRecv {
			dir = ast.SEND | ast.RECV
		}
		value := TypeExpr(fset, f, pkg, t.Elem())
		if value == nil {
			return nil
		}
		return &ast.ChanType{
			Dir:   dir,
			Value: value,
		}
	case *types.Signature:
		var params []*ast.Field
		for i := 0; i < t.Params().Len(); i++ {
			p := TypeExpr(fset, f, pkg, t.Params().At(i).Type())
			if p == nil {
				return nil
			}
			params = append(params, &ast.Field{
				Type: p,
				Names: []*ast.Ident{
					{
						Name: t.Params().At(i).Name(),
					},
				},
			})
		}
		var returns []*ast.Field
		for i := 0; i < t.Results().Len(); i++ {
			r := TypeExpr(fset, f, pkg, t.Results().At(i).Type())
			if r == nil {
				return nil
			}
			returns = append(returns, &ast.Field{
				Type: r,
			})
		}
		return &ast.FuncType{
			Params: &ast.FieldList{
				List: params,
			},
			Results: &ast.FieldList{
				List: returns,
			},
		}
	case *types.Named:
		if t.Obj().Pkg() == nil {
			return ast.NewIdent(t.Obj().Name())
		}
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

type TypeErrorPass string

const (
	NoNewVars      TypeErrorPass = "nonewvars"
	NoResultValues TypeErrorPass = "noresultvalues"
	UndeclaredName TypeErrorPass = "undeclaredname"
)

// StmtToInsertVarBefore returns the ast.Stmt before which we can safely insert a new variable.
// Some examples:
//
// Basic Example:
// z := 1
// y := z + x
// If x is undeclared, then this function would return `y := z + x`, so that we
// can insert `x := ` on the line before `y := z + x`.
//
// If stmt example:
// if z == 1 {
// } else if z == y {}
// If y is undeclared, then this function would return `if z == 1 {`, because we cannot
// insert a statement between an if and an else if statement. As a result, we need to find
// the top of the if chain to insert `y := ` before.
func StmtToInsertVarBefore(path []ast.Node) ast.Stmt {
	enclosingIndex := -1
	for i, p := range path {
		if _, ok := p.(ast.Stmt); ok {
			enclosingIndex = i
			break
		}
	}
	if enclosingIndex == -1 {
		return nil
	}
	enclosingStmt := path[enclosingIndex]
	switch enclosingStmt.(type) {
	case *ast.IfStmt:
		// The enclosingStmt is inside of the if declaration,
		// We need to check if we are in an else-if stmt and
		// get the base if statement.
		return baseIfStmt(path, enclosingIndex)
	case *ast.CaseClause:
		// Get the enclosing switch stmt if the enclosingStmt is
		// inside of the case statement.
		for i := enclosingIndex + 1; i < len(path); i++ {
			if node, ok := path[i].(*ast.SwitchStmt); ok {
				return node
			} else if node, ok := path[i].(*ast.TypeSwitchStmt); ok {
				return node
			}
		}
	}
	if len(path) <= enclosingIndex+1 {
		return enclosingStmt.(ast.Stmt)
	}
	// Check if the enclosing statement is inside another node.
	switch expr := path[enclosingIndex+1].(type) {
	case *ast.IfStmt:
		// Get the base if statement.
		return baseIfStmt(path, enclosingIndex+1)
	case *ast.ForStmt:
		if expr.Init == enclosingStmt || expr.Post == enclosingStmt {
			return expr
		}
	}
	return enclosingStmt.(ast.Stmt)
}

// baseIfStmt walks up the if/else-if chain until we get to
// the top of the current if chain.
func baseIfStmt(path []ast.Node, index int) ast.Stmt {
	stmt := path[index]
	for i := index + 1; i < len(path); i++ {
		if node, ok := path[i].(*ast.IfStmt); ok && node.Else == stmt {
			stmt = node
			continue
		}
		break
	}
	return stmt.(ast.Stmt)
}
