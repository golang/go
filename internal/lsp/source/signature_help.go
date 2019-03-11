// Copyright 2018 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package source

import (
	"context"
	"fmt"
	"go/ast"
	"go/token"
	"go/types"

	"golang.org/x/tools/go/ast/astutil"
)

type SignatureInformation struct {
	Label           string
	Parameters      []ParameterInformation
	ActiveParameter int
}

type ParameterInformation struct {
	Label string
}

func SignatureHelp(ctx context.Context, f File, pos token.Pos) (*SignatureInformation, error) {
	fAST := f.GetAST(ctx)
	pkg := f.GetPackage(ctx)
	if pkg.IsIllTyped() {
		return nil, fmt.Errorf("package for %s is ill typed", f.URI())
	}

	// Find a call expression surrounding the query position.
	var callExpr *ast.CallExpr
	path, _ := astutil.PathEnclosingInterval(fAST, pos, pos)
	if path == nil {
		return nil, fmt.Errorf("cannot find node enclosing position")
	}
	for _, node := range path {
		if c, ok := node.(*ast.CallExpr); ok {
			callExpr = c
			break
		}
	}
	if callExpr == nil || callExpr.Fun == nil {
		return nil, fmt.Errorf("cannot find an enclosing function")
	}

	// Get the type information for the function corresponding to the call expression.
	var obj types.Object
	switch t := callExpr.Fun.(type) {
	case *ast.Ident:
		obj = pkg.GetTypesInfo().ObjectOf(t)
	case *ast.SelectorExpr:
		obj = pkg.GetTypesInfo().ObjectOf(t.Sel)
	default:
		return nil, fmt.Errorf("the enclosing function is malformed")
	}
	if obj == nil {
		return nil, fmt.Errorf("cannot resolve %s", callExpr.Fun)
	}
	// Find the signature corresponding to the object.
	var sig *types.Signature
	switch obj.(type) {
	case *types.Var:
		if underlying, ok := obj.Type().Underlying().(*types.Signature); ok {
			sig = underlying
		}
	case *types.Func:
		sig = obj.Type().(*types.Signature)
	}
	if sig == nil {
		return nil, fmt.Errorf("no function signatures found for %s", obj.Name())
	}
	pkgStringer := qualifier(fAST, pkg.GetTypes(), pkg.GetTypesInfo())
	var paramInfo []ParameterInformation
	for i := 0; i < sig.Params().Len(); i++ {
		param := sig.Params().At(i)
		label := types.TypeString(param.Type(), pkgStringer)
		if param.Name() != "" {
			label = fmt.Sprintf("%s %s", param.Name(), label)
		}
		paramInfo = append(paramInfo, ParameterInformation{
			Label: label,
		})
	}
	// Determine the query position relative to the number of parameters in the function.
	var activeParam int
	var start, end token.Pos
	for i, expr := range callExpr.Args {
		if start == token.NoPos {
			start = expr.Pos()
		}
		end = expr.End()
		if i < len(callExpr.Args)-1 {
			end = callExpr.Args[i+1].Pos() - 1 // comma
		}
		if start <= pos && pos <= end {
			break
		}
		activeParam++
		start = expr.Pos() + 1 // to account for commas
	}
	// Label for function, qualified by package name.
	label := obj.Name()
	if pkg := pkgStringer(obj.Pkg()); pkg != "" {
		label = pkg + "." + label
	}
	return &SignatureInformation{
		Label:           label + formatParams(sig.Params(), sig.Variadic(), pkgStringer),
		Parameters:      paramInfo,
		ActiveParameter: activeParam,
	}, nil
}
