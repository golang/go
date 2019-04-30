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
	"strings"

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
		if c, ok := node.(*ast.CallExpr); ok && pos >= c.Lparen && pos <= c.Rparen {
			callExpr = c
			break
		}
	}
	if callExpr == nil || callExpr.Fun == nil {
		return nil, fmt.Errorf("cannot find an enclosing function")
	}

	// Get the type information for the function being called.
	sigType := pkg.GetTypesInfo().TypeOf(callExpr.Fun)
	if sigType == nil {
		return nil, fmt.Errorf("cannot get type for Fun %[1]T (%[1]v)", callExpr.Fun)
	}

	sig, _ := sigType.Underlying().(*types.Signature)
	if sig == nil {
		return nil, fmt.Errorf("cannot find signature for Fun %[1]T (%[1]v)", callExpr.Fun)
	}

	qf := qualifier(fAST, pkg.GetTypes(), pkg.GetTypesInfo())
	var paramInfo []ParameterInformation
	for i := 0; i < sig.Params().Len(); i++ {
		param := sig.Params().At(i)
		label := types.TypeString(param.Type(), qf)
		if sig.Variadic() && i == sig.Params().Len()-1 {
			label = strings.Replace(label, "[]", "...", 1)
		}
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
	for _, expr := range callExpr.Args {
		if start == token.NoPos {
			start = expr.Pos()
		}
		end = expr.End()
		if start <= pos && pos <= end {
			break
		}

		// Don't advance the active parameter for the last parameter of a variadic function.
		if !sig.Variadic() || activeParam < sig.Params().Len()-1 {
			activeParam++
		}
		start = expr.Pos() + 1 // to account for commas
	}

	// Get the object representing the function, if available.
	// There is no object in certain cases such as calling a function returned by
	// a function (e.g. "foo()()").
	var obj types.Object
	switch t := callExpr.Fun.(type) {
	case *ast.Ident:
		obj = pkg.GetTypesInfo().ObjectOf(t)
	case *ast.SelectorExpr:
		obj = pkg.GetTypesInfo().ObjectOf(t.Sel)
	}

	var name string
	if obj != nil {
		name = obj.Name()
	} else {
		name = "func"
	}

	results, writeResultParens := formatResults(sig.Results(), qf)
	label, detail := formatFunction(name, formatParams(sig.Params(), sig.Variadic(), qf), results, writeResultParens)
	if sig.Results().Len() > 0 {
		label += " " + detail
	}
	return &SignatureInformation{
		Label:           label,
		Parameters:      paramInfo,
		ActiveParameter: activeParam,
	}, nil
}
