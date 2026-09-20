// Copyright 2026 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package specdoc

import (
	"fmt"
	"go/ast"
	"go/token"
	"strconv"

	"simd/archsimd/_gen/specgen"
	"simd/archsimd/_gen/specgen/specexpr"
)

// fieldListArgs translates an AST field list into a slice of specgen.Arg.
func fieldListArgs(fl *ast.FieldList) ([]specgen.Arg, error) {
	if fl == nil {
		return nil, nil
	}
	var args []specgen.Arg
	for _, f := range fl.List {
		typ, err := parseTypeExpr(f.Type)
		if err != nil {
			return nil, err
		}
		if len(f.Names) == 0 {
			args = append(args, specgen.Arg{Name: "", Type: typ})
			continue
		}
		for _, n := range f.Names {
			args = append(args, specgen.Arg{Name: n.Name, Type: typ})
		}
	}
	return args, nil
}

// parseTypeExpr translates an AST type expression into a specexpr.Type.
func parseTypeExpr(expr ast.Expr) (specexpr.Type, error) {
	switch t := expr.(type) {
	case *ast.Ident:
		return specexpr.ParseTypeName(t.Name)

	case *ast.StarExpr:
		elem, err := parseTypeExpr(t.X)
		if err != nil {
			return nil, err
		}
		return specexpr.Pointer{Elem: elem}, nil

	case *ast.ArrayType:
		elem, err := parseTypeExpr(t.Elt)
		if err != nil {
			return nil, err
		}
		if t.Len == nil {
			return specexpr.Slice{Elem: elem}, nil
		}
		lit, ok := t.Len.(*ast.BasicLit)
		if !ok || lit.Kind != token.INT {
			return nil, fmt.Errorf("array bound is not an integer literal: %T", t.Len)
		}
		n, err := strconv.Atoi(lit.Value)
		if err != nil || n < 0 {
			return nil, fmt.Errorf("invalid array bound %q: %w", lit.Value, err)
		}
		return specexpr.Array{Elem: elem, Len: specexpr.Int(n)}, nil

	case *ast.ParenExpr:
		return parseTypeExpr(t.X)

	default:
		return nil, fmt.Errorf("unexpected type expression: %T", expr)
	}
}
