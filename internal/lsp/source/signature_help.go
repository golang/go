// Copyright 2018 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package source

import (
	"context"
	"fmt"
	"go/ast"
	"go/doc"
	"go/token"
	"go/types"

	"golang.org/x/tools/go/ast/astutil"
	"golang.org/x/tools/internal/lsp/protocol"
	"golang.org/x/tools/internal/telemetry/trace"
	errors "golang.org/x/xerrors"
)

func SignatureHelp(ctx context.Context, snapshot Snapshot, fh FileHandle, pos protocol.Position) (*protocol.SignatureInformation, int, error) {
	ctx, done := trace.StartSpan(ctx, "source.SignatureHelp")
	defer done()

	pkg, pgh, err := getParsedFile(ctx, snapshot, fh, NarrowestPackageHandle)
	if err != nil {
		return nil, 0, fmt.Errorf("getting file for SignatureHelp: %v", err)
	}
	file, _, m, _, err := pgh.Cached()
	if err != nil {
		return nil, 0, err
	}
	spn, err := m.PointSpan(pos)
	if err != nil {
		return nil, 0, err
	}
	rng, err := spn.Range(m.Converter)
	if err != nil {
		return nil, 0, err
	}
	// Find a call expression surrounding the query position.
	var callExpr *ast.CallExpr
	path, _ := astutil.PathEnclosingInterval(file, rng.Start, rng.Start)
	if path == nil {
		return nil, 0, errors.Errorf("cannot find node enclosing position")
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
			return nil, 0, errors.Errorf("no signature help within a function declaration")
		}
	}
	if callExpr == nil || callExpr.Fun == nil {
		return nil, 0, errors.Errorf("cannot find an enclosing function")
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
		return builtinSignature(ctx, snapshot.View(), callExpr, obj.Name(), rng.Start)
	}

	// Get the type information for the function being called.
	sigType := pkg.GetTypesInfo().TypeOf(callExpr.Fun)
	if sigType == nil {
		return nil, 0, errors.Errorf("cannot get type for Fun %[1]T (%[1]v)", callExpr.Fun)
	}

	sig, _ := sigType.Underlying().(*types.Signature)
	if sig == nil {
		return nil, 0, errors.Errorf("cannot find signature for Fun %[1]T (%[1]v)", callExpr.Fun)
	}

	qf := qualifier(file, pkg.GetTypes(), pkg.GetTypesInfo())
	params := formatParams(ctx, snapshot, pkg, sig, qf)
	results, writeResultParens := formatResults(sig.Results(), qf)
	activeParam := activeParameter(callExpr.Args, sig.Params().Len(), sig.Variadic(), rng.Start)

	var (
		name    string
		comment *ast.CommentGroup
	)
	if obj != nil {
		node, err := objToNode(snapshot.View(), pkg, obj)
		if err != nil {
			return nil, 0, err
		}
		rng, err := objToMappedRange(snapshot.View(), pkg, obj)
		if err != nil {
			return nil, 0, err
		}
		decl := &Declaration{
			obj:         obj,
			mappedRange: rng,
			node:        node,
		}
		d, err := decl.hover(ctx)
		if err != nil {
			return nil, 0, err
		}
		name = obj.Name()
		comment = d.comment
	} else {
		name = "func"
	}
	return signatureInformation(name, comment, params, results, writeResultParens), activeParam, nil
}

func builtinSignature(ctx context.Context, v View, callExpr *ast.CallExpr, name string, pos token.Pos) (*protocol.SignatureInformation, int, error) {
	astObj, err := v.LookupBuiltin(ctx, name)
	if err != nil {
		return nil, 0, err
	}
	decl, ok := astObj.Decl.(*ast.FuncDecl)
	if !ok {
		return nil, 0, errors.Errorf("no function declaration for builtin: %s", name)
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
	activeParam := activeParameter(callExpr.Args, numParams, variadic, pos)
	return signatureInformation(name, nil, params, results, writeResultParens), activeParam, nil
}

func signatureInformation(name string, comment *ast.CommentGroup, params, results []string, writeResultParens bool) *protocol.SignatureInformation {
	paramInfo := make([]protocol.ParameterInformation, 0, len(params))
	for _, p := range params {
		paramInfo = append(paramInfo, protocol.ParameterInformation{Label: p})
	}
	label := name + formatFunction(params, results, writeResultParens)
	var c string
	if comment != nil {
		c = doc.Synopsis(comment.Text())
	}
	return &protocol.SignatureInformation{
		Label:         label,
		Documentation: c,
		Parameters:    paramInfo,
	}
}

func activeParameter(args []ast.Expr, numParams int, variadic bool, pos token.Pos) (activeParam int) {
	if len(args) == 0 {
		return 0
	}
	// First, check if the position is even in the range of the arguments.
	start, end := args[0].Pos(), args[len(args)-1].End()
	if !(start <= pos && pos <= end) {
		return 0
	}
	for _, expr := range args {
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
