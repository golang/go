// Copyright 2019 The Go Authors. All rights reserved.
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
	"golang.org/x/tools/internal/lsp/protocol"
	"golang.org/x/tools/internal/telemetry/log"
	"golang.org/x/tools/internal/telemetry/trace"
	errors "golang.org/x/xerrors"
)

func Highlight(ctx context.Context, snapshot Snapshot, fh FileHandle, pos protocol.Position) ([]protocol.Range, error) {
	ctx, done := trace.StartSpan(ctx, "source.Highlight")
	defer done()

	pkg, pgh, err := getParsedFile(ctx, snapshot, fh, WidestPackageHandle)
	if err != nil {
		return nil, fmt.Errorf("getting file for Highlight: %v", err)
	}
	file, m, _, err := pgh.Parse(ctx)
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
	path, _ := astutil.PathEnclosingInterval(file, rng.Start, rng.Start)
	if len(path) == 0 {
		return nil, errors.Errorf("no enclosing position found for %v:%v", int(pos.Line), int(pos.Character))
	}
	// If start==end for astutil.PathEnclosingInterval, the 1-char interval following start is used instead.
	// As a result, we might not get an exact match so we should check the 1-char interval to the left of the
	// passed in position to see if that is an exact match.
	if _, ok := path[0].(*ast.Ident); !ok {
		if p, _ := astutil.PathEnclosingInterval(file, rng.Start-1, rng.Start-1); p != nil {
			switch p[0].(type) {
			case *ast.Ident, *ast.SelectorExpr:
				path = p // use preceding ident/selector
			}
		}
	}

	switch path[0].(type) {
	case *ast.BasicLit:
		if len(path) > 1 {
			if _, ok := path[1].(*ast.ImportSpec); ok {
				return highlightImportUses(ctx, snapshot.View(), pkg, path)
			}
		}
		return highlightFuncControlFlow(ctx, snapshot.View(), pkg, path)
	case *ast.ReturnStmt, *ast.FuncDecl, *ast.FuncType:
		return highlightFuncControlFlow(ctx, snapshot.View(), pkg, path)
	case *ast.Ident:
		return highlightIdentifiers(ctx, snapshot.View(), pkg, path)
	case *ast.BranchStmt, *ast.ForStmt, *ast.RangeStmt:
		return highlightLoopControlFlow(ctx, snapshot.View(), pkg, path)
	}
	// If the cursor is in an unidentified area, return empty results.
	return nil, nil
}

func highlightFuncControlFlow(ctx context.Context, view View, pkg Package, path []ast.Node) ([]protocol.Range, error) {
	var enclosingFunc ast.Node
	var returnStmt *ast.ReturnStmt
	var resultsList *ast.FieldList
	inReturnList := false
Outer:
	// Reverse walk the path till we get to the func block.
	for i, n := range path {
		switch node := n.(type) {
		case *ast.KeyValueExpr:
			// If cursor is in a key: value expr, we don't want control flow highlighting
			return nil, nil
		case *ast.CallExpr:
			// If cusor is an arg in a callExpr, we don't want control flow highlighting.
			if i > 0 {
				for _, arg := range node.Args {
					if arg == path[i-1] {
						return nil, nil
					}
				}
			}
		case *ast.Field:
			inReturnList = true
		case *ast.FuncLit:
			enclosingFunc = n
			resultsList = node.Type.Results
			break Outer
		case *ast.FuncDecl:
			enclosingFunc = n
			resultsList = node.Type.Results
			break Outer
		case *ast.ReturnStmt:
			returnStmt = node
			// If the cursor is not directly in a *ast.ReturnStmt, then
			// we need to know if it is within one of the values that is being returned.
			inReturnList = inReturnList || path[0] != returnStmt
		}
	}
	// Cursor is not in a function.
	if enclosingFunc == nil {
		return nil, nil
	}
	// If the cursor is on a "return" or "func" keyword, we should highlight all of the exit
	// points of the function, including the "return" and "func" keywords.
	highlightAllReturnsAndFunc := path[0] == returnStmt || path[0] == enclosingFunc
	switch path[0].(type) {
	case *ast.Ident, *ast.BasicLit:
		// Cursor is in an identifier and not in a return statement or in the results list.
		if returnStmt == nil && !inReturnList {
			return nil, nil
		}
	case *ast.FuncType:
		highlightAllReturnsAndFunc = true
	}
	// The user's cursor may be within the return statement of a function,
	// or within the result section of a function's signature.
	// index := -1
	var nodes []ast.Node
	if returnStmt != nil {
		for _, n := range returnStmt.Results {
			nodes = append(nodes, n)
		}
	} else if resultsList != nil {
		for _, n := range resultsList.List {
			nodes = append(nodes, n)
		}
	}
	_, index := nodeAtPos(nodes, path[0].Pos())

	result := make(map[protocol.Range]bool)
	// Highlight the correct argument in the function declaration return types.
	if resultsList != nil && -1 < index && index < len(resultsList.List) {
		rng, err := nodeToProtocolRange(view, pkg, resultsList.List[index])
		if err != nil {
			log.Error(ctx, "Error getting range for node", err)
		} else {
			result[rng] = true
		}
	}
	// Add the "func" part of the func declaration.
	if highlightAllReturnsAndFunc {
		funcStmt, err := posToMappedRange(view, pkg, enclosingFunc.Pos(), enclosingFunc.Pos()+token.Pos(len("func")))
		if err != nil {
			return nil, err
		}
		rng, err := funcStmt.Range()
		if err != nil {
			return nil, err
		}
		result[rng] = true
	}
	// Traverse the AST to highlight the other relevant return statements in the function.
	ast.Inspect(enclosingFunc, func(n ast.Node) bool {
		// Don't traverse any other functions.
		switch n.(type) {
		case *ast.FuncDecl, *ast.FuncLit:
			return enclosingFunc == n
		}
		if n, ok := n.(*ast.ReturnStmt); ok {
			var toAdd ast.Node
			// Add the entire return statement, applies when highlight the word "return" or "func".
			if highlightAllReturnsAndFunc {
				toAdd = n
			}
			// Add the relevant field within the entire return statement.
			if -1 < index && index < len(n.Results) {
				toAdd = n.Results[index]
			}
			if toAdd != nil {
				rng, err := nodeToProtocolRange(view, pkg, toAdd)
				if err != nil {
					log.Error(ctx, "Error getting range for node", err)
				} else {
					result[rng] = true
				}
				return false
			}
		}
		return true
	})
	return rangeMapToSlice(result), nil
}

func highlightLoopControlFlow(ctx context.Context, view View, pkg Package, path []ast.Node) ([]protocol.Range, error) {
	var loop ast.Node
Outer:
	// Reverse walk the path till we get to the for loop.
	for _, n := range path {
		switch n.(type) {
		case *ast.ForStmt, *ast.RangeStmt:
			loop = n
			break Outer
		}
	}
	// Cursor is not in a for loop.
	if loop == nil {
		return nil, nil
	}
	result := make(map[protocol.Range]bool)
	// Add the for statement.
	forStmt, err := posToMappedRange(view, pkg, loop.Pos(), loop.Pos()+token.Pos(len("for")))
	if err != nil {
		return nil, err
	}
	rng, err := forStmt.Range()
	if err != nil {
		return nil, err
	}
	result[rng] = true

	ast.Inspect(loop, func(n ast.Node) bool {
		// Don't traverse any other for loops.
		switch n.(type) {
		case *ast.ForStmt, *ast.RangeStmt:
			return loop == n
		}
		// Add all branch statements in same scope as the identified one.
		if n, ok := n.(*ast.BranchStmt); ok {
			rng, err := nodeToProtocolRange(view, pkg, n)
			if err != nil {
				log.Error(ctx, "Error getting range for node", err)
				return false
			}
			result[rng] = true
		}
		return true
	})
	return rangeMapToSlice(result), nil
}

func highlightImportUses(ctx context.Context, view View, pkg Package, path []ast.Node) ([]protocol.Range, error) {
	result := make(map[protocol.Range]bool)
	basicLit, ok := path[0].(*ast.BasicLit)
	if !ok {
		return nil, errors.Errorf("highlightImportUses called with an ast.Node of type %T", basicLit)
	}

	ast.Inspect(path[len(path)-1], func(node ast.Node) bool {
		if imp, ok := node.(*ast.ImportSpec); ok && imp.Path == basicLit {
			if rng, err := nodeToProtocolRange(view, pkg, node); err == nil {
				result[rng] = true
				return false
			}
		}
		n, ok := node.(*ast.Ident)
		if !ok {
			return true
		}
		obj, ok := pkg.GetTypesInfo().ObjectOf(n).(*types.PkgName)
		if !ok {
			return true
		}
		if !strings.Contains(basicLit.Value, obj.Name()) {
			return true
		}
		if rng, err := nodeToProtocolRange(view, pkg, n); err == nil {
			result[rng] = true
		} else {
			log.Error(ctx, "Error getting range for node", err)
		}
		return false
	})
	return rangeMapToSlice(result), nil
}

func highlightIdentifiers(ctx context.Context, view View, pkg Package, path []ast.Node) ([]protocol.Range, error) {
	result := make(map[protocol.Range]bool)
	id, ok := path[0].(*ast.Ident)
	if !ok {
		return nil, errors.Errorf("highlightIdentifiers called with an ast.Node of type %T", id)
	}
	// Check if ident is inside return or func decl.
	if toAdd, err := highlightFuncControlFlow(ctx, view, pkg, path); toAdd != nil && err == nil {
		for _, r := range toAdd {
			result[r] = true
		}
	}

	// TODO: maybe check if ident is a reserved word, if true then don't continue and return results.

	idObj := pkg.GetTypesInfo().ObjectOf(id)
	pkgObj, isImported := idObj.(*types.PkgName)
	ast.Inspect(path[len(path)-1], func(node ast.Node) bool {
		if imp, ok := node.(*ast.ImportSpec); ok && isImported {
			if rng, err := highlightImport(view, pkg, pkgObj, imp); rng != nil && err == nil {
				result[*rng] = true
			}
		}
		n, ok := node.(*ast.Ident)
		if !ok {
			return true
		}
		if n.Name != id.Name {
			return false
		}
		if nObj := pkg.GetTypesInfo().ObjectOf(n); nObj != idObj {
			return false
		}
		if rng, err := nodeToProtocolRange(view, pkg, n); err == nil {
			result[rng] = true
		} else {
			log.Error(ctx, "Error getting range for node", err)
		}
		return false
	})
	return rangeMapToSlice(result), nil
}

func highlightImport(view View, pkg Package, obj *types.PkgName, imp *ast.ImportSpec) (*protocol.Range, error) {
	if imp.Name != nil || imp.Path == nil {
		return nil, nil
	}
	if !strings.Contains(imp.Path.Value, obj.Name()) {
		return nil, nil
	}
	rng, err := nodeToProtocolRange(view, pkg, imp.Path)
	if err != nil {
		return nil, err
	}
	return &rng, nil
}

func rangeMapToSlice(rangeMap map[protocol.Range]bool) []protocol.Range {
	var list []protocol.Range
	for i := range rangeMap {
		list = append(list, i)
	}
	return list
}
