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
	"golang.org/x/tools/go/analysis/passes/internal/analysisutil"
	"golang.org/x/tools/go/ast/inspector"
	"golang.org/x/tools/go/types/typeutil"
)

//go:embed doc.go
var doc string

var Analyzer = &analysis.Analyzer{
	Name:     "unusedresult",
	Doc:      analysisutil.MustExtractDoc(doc, "unusedresult"),
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
		"context.WithCancel":   true,
		"context.WithDeadline": true,
		"context.WithTimeout":  true,
		"context.WithValue":    true,
		"errors.New":           true,
		"fmt.Errorf":           true,
		"fmt.Sprint":           true,
		"fmt.Sprintf":          true,
		"slices.Clip":          true,
		"slices.Compact":       true,
		"slices.CompactFunc":   true,
		"slices.Delete":        true,
		"slices.DeleteFunc":    true,
		"slices.Grow":          true,
		"slices.Insert":        true,
		"slices.Replace":       true,
		"sort.Reverse":         true,
	}
	Analyzer.Flags.Var(&funcs, "funcs",
		"comma-separated list of functions whose results must be used")

	stringMethods.Set("Error,String")
	Analyzer.Flags.Var(&stringMethods, "stringmethods",
		"comma-separated list of names of methods of type func() string whose results must be used")
}

func run(pass *analysis.Pass) (interface{}, error) {
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
		if sig := fn.Type().(*types.Signature); sig.Recv() != nil {
			// method (e.g. foo.String())
			if types.Identical(sig, sigNoArgsStringResult) {
				if stringMethods[fn.Name()] {
					pass.Reportf(call.Lparen, "result of (%s).%s call not used",
						sig.Recv().Type(), fn.Name())
				}
			}
		} else {
			// package-level function (e.g. fmt.Errorf)
			if pkgFuncs[[2]string{fn.Pkg().Path(), fn.Name()}] {
				pass.Reportf(call.Lparen, "result of %s.%s call not used",
					fn.Pkg().Path(), fn.Name())
			}
		}
	})
	return nil, nil
}

// func() string
var sigNoArgsStringResult = types.NewSignature(nil, nil,
	types.NewTuple(types.NewParam(token.NoPos, nil, "", types.Typ[types.String])),
	false)

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
		for _, name := range strings.Split(s, ",") {
			if name == "" {
				continue // TODO: report error? proceed?
			}
			m[name] = true
		}
	}
	*ss = m
	return nil
}
