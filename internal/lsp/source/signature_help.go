// Copyright 2018 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package source

import (
	"context"
	"go/ast"
	"go/doc"
	"go/token"
	"go/types"

	"golang.org/x/tools/go/ast/astutil"
	"golang.org/x/tools/internal/lsp/protocol"
	"golang.org/x/tools/internal/telemetry/trace"
	errors "golang.org/x/xerrors"
)

type SignatureInformation struct {
	Label, Documentation string
	Parameters           []ParameterInformation
	ActiveParameter      int
}

type ParameterInformation struct {
	Label string
}

func SignatureHelp(ctx context.Context, view View, f File, pos protocol.Position) (*SignatureInformation, error) {
	ctx, done := trace.StartSpan(ctx, "source.SignatureHelp")
	defer done()

	_, cphs, err := view.CheckPackageHandles(ctx, f)
	if err != nil {
		return nil, err
	}
	cph, err := NarrowestCheckPackageHandle(cphs)
	if err != nil {
		return nil, err
	}
	pkg, err := cph.Check(ctx)
	if err != nil {
		return nil, err
	}
	ph, err := pkg.File(f.URI())
	if err != nil {
		return nil, err
	}
	file, m, _, err := ph.Cached()
	if err != nil {
		return nil, err
	}
	spn, err := m.PointSpan(pos)
	if err != nil {
		return nil, err
	}
	rng, err := spn.Range(m.Converter)
	if err != nil {
		return nil, err
	}
	// Find a call expression surrounding the query position.
	var callExpr *ast.CallExpr
	path, _ := astutil.PathEnclosingInterval(file, rng.Start, rng.Start)
	if path == nil {
		return nil, errors.Errorf("cannot find node enclosing position")
	}
FindCall:
	for _, node := range path {
		switch node := node.(type) {
		case *ast.CallExpr:
			if rng.Start >= node.Lparen && rng.Start <= node.Rparen {
				callExpr = node
				break FindCall
			}
		case *ast.FuncLit, *ast.FuncType:
			// The user is within an anonymous function,
			// which may be the parameter to the *ast.CallExpr.
			// Don't show signature help in this case.
			return nil, errors.Errorf("no signature help within a function declaration")
		}
	}
	if callExpr == nil || callExpr.Fun == nil {
		return nil, errors.Errorf("cannot find an enclosing function")
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

	// Handle builtin functions separately.
	if obj, ok := obj.(*types.Builtin); ok {
		return builtinSignature(ctx, view, callExpr, obj.Name(), rng.Start)
	}

	// Get the type information for the function being called.
	sigType := pkg.GetTypesInfo().TypeOf(callExpr.Fun)
	if sigType == nil {
		return nil, errors.Errorf("cannot get type for Fun %[1]T (%[1]v)", callExpr.Fun)
	}

	sig, _ := sigType.Underlying().(*types.Signature)
	if sig == nil {
		return nil, errors.Errorf("cannot find signature for Fun %[1]T (%[1]v)", callExpr.Fun)
	}

	qf := qualifier(file, pkg.GetTypes(), pkg.GetTypesInfo())
	params := formatParams(sig.Params(), sig.Variadic(), qf)
	results, writeResultParens := formatResults(sig.Results(), qf)
	activeParam := activeParameter(callExpr, sig.Params().Len(), sig.Variadic(), rng.Start)

	var (
		name    string
		comment *ast.CommentGroup
	)
	if obj != nil {
		node, err := objToNode(ctx, view, pkg, obj)
		if err != nil {
			return nil, err
		}
		rng, err := objToMappedRange(ctx, view, pkg, obj)
		if err != nil {
			return nil, err
		}
		decl := &Declaration{
			obj:         obj,
			mappedRange: rng,
			node:        node,
		}
		d, err := decl.hover(ctx)
		if err != nil {
			return nil, err
		}
		name = obj.Name()
		comment = d.comment
	} else {
		name = "func"
	}
	return signatureInformation(name, comment, params, results, writeResultParens, activeParam), nil
}

func builtinSignature(ctx context.Context, v View, callExpr *ast.CallExpr, name string, pos token.Pos) (*SignatureInformation, error) {
	obj := v.BuiltinPackage().Lookup(name)
	if obj == nil {
		return nil, errors.Errorf("no object for %s", name)
	}
	decl, ok := obj.Decl.(*ast.FuncDecl)
	if !ok {
		return nil, errors.Errorf("no function declaration for builtin: %s", name)
	}
	params, _ := formatFieldList(ctx, v, decl.Type.Params)
	results, writeResultParens := formatFieldList(ctx, v, decl.Type.Results)

	var (
		numParams int
		variadic  bool
	)
	if decl.Type.Params.List != nil {
		numParams = len(decl.Type.Params.List)
		lastParam := decl.Type.Params.List[numParams-1]
		if _, ok := lastParam.Type.(*ast.Ellipsis); ok {
			variadic = true
		}
	}
	activeParam := activeParameter(callExpr, numParams, variadic, pos)
	return signatureInformation(name, nil, params, results, writeResultParens, activeParam), nil
}

func signatureInformation(name string, comment *ast.CommentGroup, params, results []string, writeResultParens bool, activeParam int) *SignatureInformation {
	paramInfo := make([]ParameterInformation, 0, len(params))
	for _, p := range params {
		paramInfo = append(paramInfo, ParameterInformation{Label: p})
	}
	label := name + formatFunction(params, results, writeResultParens)
	var c string
	if comment != nil {
		c = doc.Synopsis(comment.Text())
	}
	return &SignatureInformation{
		Label:           label,
		Documentation:   c,
		Parameters:      paramInfo,
		ActiveParameter: activeParam,
	}
}

func activeParameter(callExpr *ast.CallExpr, numParams int, variadic bool, pos token.Pos) int {
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
		if !variadic || activeParam < numParams-1 {
			activeParam++
		}
		start = expr.Pos() + 1 // to account for commas
	}
	return activeParam
}
