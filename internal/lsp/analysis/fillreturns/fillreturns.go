// Copyright 2020 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// Package fillreturns defines an Analyzer that will attempt to
// automatically fill in a return statement that has missing
// values with zero value elements.
package fillreturns

import (
	"bytes"
	"fmt"
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
	info := pass.TypesInfo
	if info == nil {
		return nil, fmt.Errorf("nil TypeInfo")
	}

	errors := analysisinternal.GetTypeErrors(pass)
outer:
	for _, typeErr := range errors {
		// Filter out the errors that are not relevant to this analyzer.
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

		// Get the function type that encloses the ReturnStmt.
		var enclosingFunc *ast.FuncType
		for _, n := range path {
			switch node := n.(type) {
			case *ast.FuncLit:
				enclosingFunc = node.Type
			case *ast.FuncDecl:
				enclosingFunc = node.Type
			}
			if enclosingFunc != nil {
				break
			}
		}
		if enclosingFunc == nil {
			continue
		}

		// Find the function declaration that encloses the ReturnStmt.
		var outer *ast.FuncDecl
		for _, p := range path {
			if p, ok := p.(*ast.FuncDecl); ok {
				outer = p
				break
			}
		}
		if outer == nil {
			return nil, nil
		}

		// Skip any return statements that contain function calls with multiple return values.
		for _, expr := range ret.Results {
			e, ok := expr.(*ast.CallExpr)
			if !ok {
				continue
			}
			if tup, ok := info.TypeOf(e).(*types.Tuple); ok && tup.Len() > 1 {
				continue outer
			}
		}

		// Duplicate the return values to track which values have been matched.
		remaining := make([]ast.Expr, len(ret.Results))
		copy(remaining, ret.Results)

		fixed := make([]ast.Expr, len(enclosingFunc.Results.List))

		// For each value in the return function declaration, find the leftmost element
		// in the return statement that has the desired type. If no such element exits,
		// fill in the missing value with the appropriate "zero" value.
		var retTyps []types.Type
		for _, ret := range enclosingFunc.Results.List {
			retTyps = append(retTyps, info.TypeOf(ret.Type))
		}
		matches :=
			analysisinternal.FindMatchingIdents(retTyps, file, ret.Pos(), info, pass.Pkg)
		for i, retTyp := range retTyps {
			var match ast.Expr
			var idx int
			for j, val := range remaining {
				if !matchingTypes(info.TypeOf(val), retTyp) {
					continue
				}
				if !analysisinternal.IsZeroValue(val) {
					match, idx = val, j
					break
				}
				// If the current match is a "zero" value, we keep searching in
				// case we find a non-"zero" value match. If we do not find a
				// non-"zero" value, we will use the "zero" value.
				match, idx = val, j
			}

			if match != nil {
				fixed[i] = match
				remaining = append(remaining[:idx], remaining[idx+1:]...)
			} else {
				idents, ok := matches[retTyp]
				if !ok {
					return nil, fmt.Errorf("invalid return type: %v", retTyp)
				}
				// Find the identifer whose name is most similar to the return type.
				// If we do not find any identifer that matches the pattern,
				// generate a zero value.
				value := analysisinternal.FindBestMatch(retTyp.String(), idents)
				if value == nil {
					value = analysisinternal.ZeroValue(
						pass.Fset, file, pass.Pkg, retTyp)
				}
				if value == nil {
					return nil, nil
				}
				fixed[i] = value
			}
		}

		// Remove any non-matching "zero values" from the leftover values.
		var nonZeroRemaining []ast.Expr
		for _, expr := range remaining {
			if !analysisinternal.IsZeroValue(expr) {
				nonZeroRemaining = append(nonZeroRemaining, expr)
			}
		}
		// Append leftover return values to end of new return statement.
		fixed = append(fixed, nonZeroRemaining...)

		newRet := &ast.ReturnStmt{
			Return:  ret.Pos(),
			Results: fixed,
		}

		// Convert the new return statement AST to text.
		var newBuf bytes.Buffer
		if err := format.Node(&newBuf, pass.Fset, newRet); err != nil {
			return nil, err
		}

		pass.Report(analysis.Diagnostic{
			Pos:     typeErr.Pos,
			End:     typeErrEndPos,
			Message: typeErr.Msg,
			SuggestedFixes: []analysis.SuggestedFix{{
				Message: "Fill in return values",
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

func matchingTypes(want, got types.Type) bool {
	if want == got || types.Identical(want, got) {
		return true
	}
	// Code segment to help check for untyped equality from (golang/go#32146).
	if rhs, ok := want.(*types.Basic); ok && rhs.Info()&types.IsUntyped > 0 {
		if lhs, ok := got.Underlying().(*types.Basic); ok {
			return rhs.Info()&types.IsConstType == lhs.Info()&types.IsConstType
		}
	}
	return types.AssignableTo(want, got) || types.ConvertibleTo(want, got)
}

func FixesError(msg string) bool {
	matches := wrongReturnNumRegex.FindStringSubmatch(strings.TrimSpace(msg))
	if len(matches) < 3 {
		return false
	}
	if _, err := strconv.Atoi(matches[1]); err != nil {
		return false
	}
	if _, err := strconv.Atoi(matches[2]); err != nil {
		return false
	}
	return true
}
