// Copyright 2020 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// Package fillreturns defines an Analyzer that will attempt to
// automatically fill in a return statement that has missing
// values with zero value elements.
package fillreturns

import (
	"bytes"
	"go/ast"
	"go/format"
	"go/types"
	"regexp"
	"strconv"
	"strings"

	"golang.org/x/tools/go/analysis"
	"golang.org/x/tools/go/ast/astutil"
	"golang.org/x/tools/internal/analysisinternal"
)

const Doc = `suggested fixes for "wrong number of return values (want %d, got %d)"

This checker provides suggested fixes for type errors of the
type "wrong number of return values (want %d, got %d)". For example:
	func m() (int, string, *bool, error) {
		return
	}
will turn into
	func m() (int, string, *bool, error) {
		return 0, "", nil, nil
	}

This functionality is similar to https://github.com/sqs/goreturns.
`

var Analyzer = &analysis.Analyzer{
	Name:             "fillreturns",
	Doc:              Doc,
	Requires:         []*analysis.Analyzer{},
	Run:              run,
	RunDespiteErrors: true,
}

var wrongReturnNumRegex = regexp.MustCompile(`wrong number of return values \(want (\d+), got (\d+)\)`)

func run(pass *analysis.Pass) (interface{}, error) {
	errors := analysisinternal.GetTypeErrors(pass)
	// Filter out the errors that are not relevant to this analyzer.
	for _, typeErr := range errors {
		if !FixesError(typeErr.Msg) {
			continue
		}
		var file *ast.File
		for _, f := range pass.Files {
			if f.Pos() <= typeErr.Pos && typeErr.Pos <= f.End() {
				file = f
				break
			}
		}
		if file == nil {
			continue
		}

		// Get the end position of the error.
		var buf bytes.Buffer
		if err := format.Node(&buf, pass.Fset, file); err != nil {
			continue
		}
		typeErrEndPos := analysisinternal.TypeErrorEndPos(pass.Fset, buf.Bytes(), typeErr.Pos)

		// Get the path for the relevant range.
		path, _ := astutil.PathEnclosingInterval(file, typeErr.Pos, typeErrEndPos)
		if len(path) == 0 {
			return nil, nil
		}
		// Check to make sure the node of interest is a ReturnStmt.
		ret, ok := path[0].(*ast.ReturnStmt)
		if !ok {
			return nil, nil
		}

		// Get the function that encloses the ReturnStmt.
		var enclosingFunc *ast.FuncType
	Outer:
		for _, n := range path {
			switch node := n.(type) {
			case *ast.FuncLit:
				enclosingFunc = node.Type
				break Outer
			case *ast.FuncDecl:
				enclosingFunc = node.Type
				break Outer
			}
		}
		if enclosingFunc == nil {
			continue
		}
		numRetValues := len(ret.Results)
		typeInfo := pass.TypesInfo

		// skip if return value has a func call (whose multiple returns might be expanded)
		for _, expr := range ret.Results {
			e, ok := expr.(*ast.CallExpr)
			if !ok {
				continue
			}
			ident, ok := e.Fun.(*ast.Ident)
			if !ok || ident.Obj == nil {
				continue
			}
			fn, ok := ident.Obj.Decl.(*ast.FuncDecl)
			if !ok {
				continue
			}
			if len(fn.Type.Results.List) != 1 {
				continue
			}
			if typeInfo == nil {
				continue
			}
			if _, ok := typeInfo.TypeOf(e).(*types.Tuple); ok {
				continue
			}
		}

		// Fill in the missing arguments with zero-values.
		returnCount := 0
		zvs := make([]ast.Expr, len(enclosingFunc.Results.List))
		for i, result := range enclosingFunc.Results.List {
			zv := analysisinternal.ZeroValue(pass.Fset, file, pass.Pkg, typeInfo.TypeOf(result.Type))
			if zv == nil {
				return nil, nil
			}
			// We do not have any existing return values, fill in with zero-values.
			if returnCount >= numRetValues {
				zvs[i] = zv
				continue
			}
			// Compare the types to see if they are the same.
			current := ret.Results[returnCount]
			if equalTypes(typeInfo.TypeOf(current), typeInfo.TypeOf(result.Type)) {
				zvs[i] = current
				returnCount += 1
				continue
			}
			zvs[i] = zv
		}
		newRet := &ast.ReturnStmt{
			Return:  ret.Pos(),
			Results: zvs,
		}

		// Convert the new return statement ast to text.
		var newBuf bytes.Buffer
		if err := format.Node(&newBuf, pass.Fset, newRet); err != nil {
			return nil, err
		}

		pass.Report(analysis.Diagnostic{
			Pos:     typeErr.Pos,
			End:     typeErrEndPos,
			Message: typeErr.Msg,
			SuggestedFixes: []analysis.SuggestedFix{{
				Message: "Fill with empty values",
				TextEdits: []analysis.TextEdit{{
					Pos:     ret.Pos(),
					End:     ret.End(),
					NewText: newBuf.Bytes(),
				}},
			}},
		})
	}
	return nil, nil
}

func equalTypes(t1, t2 types.Type) bool {
	if t1 == t2 || types.Identical(t1, t2) {
		return true
	}
	// Code segment to help check for untyped equality from (golang/go#32146).
	if rhs, ok := t1.(*types.Basic); ok && rhs.Info()&types.IsUntyped > 0 {
		if lhs, ok := t2.Underlying().(*types.Basic); ok {
			return rhs.Info()&types.IsConstType == lhs.Info()&types.IsConstType
		}
	}
	// TODO: Figure out if we want to check for types.AssignableTo(t1, t2) || types.ConvertibleTo(t1, t2)
	return false
}

func FixesError(msg string) bool {
	matches := wrongReturnNumRegex.FindStringSubmatch(strings.TrimSpace(msg))
	if len(matches) < 3 {
		return false
	}
	wantNum, err := strconv.Atoi(matches[1])
	if err != nil {
		return false
	}
	gotNum, err := strconv.Atoi(matches[2])
	if err != nil {
		return false
	}
	// Logic for handling more return values than expected is hard.
	return wantNum >= gotNum
}
