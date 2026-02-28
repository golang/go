// Copyright 2015 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// Package unusedresult defines an analyzer that checks for unused
// results of calls to certain functions.
package unusedresult

// It is tempting to make this analysis inductive: for each function
// that tail-calls one of the functions that we check, check those
// functions too. However, just because you must use the result of
// fmt.Sprintf doesn't mean you need to use the result of every
// function that returns a formatted string: it may have other results
// and effects.

import (
	_ "embed"
	"go/ast"
	"go/token"
	"go/types"
	"sort"
	"strings"

	"golang.org/x/tools/go/analysis"
	"golang.org/x/tools/go/analysis/passes/inspect"
	"golang.org/x/tools/go/ast/inspector"
	"golang.org/x/tools/go/types/typeutil"
	"golang.org/x/tools/internal/analysis/analyzerutil"
	"golang.org/x/tools/internal/astutil"
)

//go:embed doc.go
var doc string

var Analyzer = &analysis.Analyzer{
	Name:     "unusedresult",
	Doc:      analyzerutil.MustExtractDoc(doc, "unusedresult"),
	URL:      "https://pkg.go.dev/golang.org/x/tools/go/analysis/passes/unusedresult",
	Requires: []*analysis.Analyzer{inspect.Analyzer},
	Run:      run,
}

// flags
var funcs, stringMethods stringSetFlag

func init() {
	// TODO(adonovan): provide a comment or declaration syntax to
	// allow users to add their functions to this set using facts.
	// For example:
	//
	//    func ignoringTheErrorWouldBeVeryBad() error {
	//      type mustUseResult struct{} // enables vet unusedresult check
	//      ...
	//    }
	//
	//    ignoringTheErrorWouldBeVeryBad() // oops
	//

	// List standard library functions here.
	// The context.With{Cancel,Deadline,Timeout} entries are
	// effectively redundant wrt the lostcancel analyzer.
	funcs = stringSetFlag{
		"context.WithCancel":      true,
		"context.WithDeadline":    true,
		"context.WithTimeout":     true,
		"context.WithValue":       true,
		"errors.New":              true,
		"fmt.Append":              true,
		"fmt.Appendf":             true,
		"fmt.Appendln":            true,
		"fmt.Errorf":              true,
		"fmt.Sprint":              true,
		"fmt.Sprintf":             true,
		"fmt.Sprintln":            true,
		"maps.All":                true,
		"maps.Clone":              true,
		"maps.Collect":            true,
		"maps.Equal":              true,
		"maps.EqualFunc":          true,
		"maps.Keys":               true,
		"maps.Values":             true,
		"slices.All":              true,
		"slices.AppendSeq":        true,
		"slices.Backward":         true,
		"slices.BinarySearch":     true,
		"slices.BinarySearchFunc": true,
		"slices.Chunk":            true,
		"slices.Clip":             true,
		"slices.Clone":            true,
		"slices.Collect":          true,
		"slices.Compact":          true,
		"slices.CompactFunc":      true,
		"slices.Compare":          true,
		"slices.CompareFunc":      true,
		"slices.Concat":           true,
		"slices.Contains":         true,
		"slices.ContainsFunc":     true,
		"slices.Delete":           true,
		"slices.DeleteFunc":       true,
		"slices.Equal":            true,
		"slices.EqualFunc":        true,
		"slices.Grow":             true,
		"slices.Index":            true,
		"slices.IndexFunc":        true,
		"slices.Insert":           true,
		"slices.IsSorted":         true,
		"slices.IsSortedFunc":     true,
		"slices.Max":              true,
		"slices.MaxFunc":          true,
		"slices.Min":              true,
		"slices.MinFunc":          true,
		"slices.Repeat":           true,
		"slices.Replace":          true,
		"slices.Sorted":           true,
		"slices.SortedFunc":       true,
		"slices.SortedStableFunc": true,
		"slices.Values":           true,
		"sort.Reverse":            true,
	}
	Analyzer.Flags.Var(&funcs, "funcs",
		"comma-separated list of functions whose results must be used")

	stringMethods.Set("Error,String")
	Analyzer.Flags.Var(&stringMethods, "stringmethods",
		"comma-separated list of names of methods of type func() string whose results must be used")
}

func run(pass *analysis.Pass) (any, error) {
	inspect := pass.ResultOf[inspect.Analyzer].(*inspector.Inspector)

	// Split functions into (pkg, name) pairs to save allocation later.
	pkgFuncs := make(map[[2]string]bool, len(funcs))
	for s := range funcs {
		if i := strings.LastIndexByte(s, '.'); i > 0 {
			pkgFuncs[[2]string{s[:i], s[i+1:]}] = true
		}
	}

	nodeFilter := []ast.Node{
		(*ast.ExprStmt)(nil),
	}
	inspect.Preorder(nodeFilter, func(n ast.Node) {
		call, ok := ast.Unparen(n.(*ast.ExprStmt).X).(*ast.CallExpr)
		if !ok {
			return // not a call statement
		}

		// Call to function or method?
		fn, ok := typeutil.Callee(pass.TypesInfo, call).(*types.Func)
		if !ok {
			return // e.g. var or builtin
		}
		if sig := fn.Signature(); sig.Recv() != nil {
			// method (e.g. foo.String())
			if types.Identical(sig, sigNoArgsStringResult) {
				if stringMethods[fn.Name()] {
					pass.ReportRangef(astutil.RangeOf(call.Pos(), call.Lparen),
						"result of (%s).%s call not used",
						sig.Recv().Type(), fn.Name())
				}
			}
		} else {
			// package-level function (e.g. fmt.Errorf)
			if pkgFuncs[[2]string{fn.Pkg().Path(), fn.Name()}] {
				pass.ReportRangef(astutil.RangeOf(call.Pos(), call.Lparen),
					"result of %s.%s call not used",
					fn.Pkg().Path(), fn.Name())
			}
		}
	})
	return nil, nil
}

// func() string
var sigNoArgsStringResult = types.NewSignatureType(nil, nil, nil, nil, types.NewTuple(types.NewParam(token.NoPos, nil, "", types.Typ[types.String])), false)

type stringSetFlag map[string]bool

func (ss *stringSetFlag) String() string {
	var items []string
	for item := range *ss {
		items = append(items, item)
	}
	sort.Strings(items)
	return strings.Join(items, ",")
}

func (ss *stringSetFlag) Set(s string) error {
	m := make(map[string]bool) // clobber previous value
	if s != "" {
		for name := range strings.SplitSeq(s, ",") {
			if name == "" {
				continue // TODO: report error? proceed?
			}
			m[name] = true
		}
	}
	*ss = m
	return nil
}
