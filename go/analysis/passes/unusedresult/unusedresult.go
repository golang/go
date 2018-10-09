// Copyright 2015 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// Package unusedresult defines an analyer that checks for unused
// results of calls to certain pure functions.
package unusedresult

import (
	"go/ast"
	"go/token"
	"go/types"
	"sort"
	"strings"

	"golang.org/x/tools/go/analysis"
	"golang.org/x/tools/go/analysis/passes/inspect"
	"golang.org/x/tools/go/analysis/passes/internal/analysisutil"
	"golang.org/x/tools/go/ast/inspector"
)

// TODO(adonovan): make this analysis modular: export a mustUseResult
// fact for each function that tail-calls one of the functions that we
// check, and check those functions too.

var Analyzer = &analysis.Analyzer{
	Name: "unusedresult",
	Doc: `check for unused results of calls to some functions

Some functions like fmt.Errorf return a result and have no side effects,
so it is always a mistake to discard the result. This analyzer reports
calls to certain functions in which the result of the call is ignored.

The set of functions may be controlled using flags.`,
	Requires: []*analysis.Analyzer{inspect.Analyzer},
	Run:      run,
}

// flags
var funcs, stringMethods stringSetFlag

func init() {
	// TODO(adonovan): provide a comment syntax to allow users to
	// add their functions to this set using facts.
	funcs.Set("errors.New,fmt.Errorf,fmt.Sprintf,fmt.Sprint,sort.Reverse")
	Analyzer.Flags.Var(&funcs, "funcs",
		"comma-separated list of functions whose results must be used")

	stringMethods.Set("Error,String")
	Analyzer.Flags.Var(&stringMethods, "stringmethods",
		"comma-separated list of names of methods of type func() string whose results must be used")
}

func run(pass *analysis.Pass) (interface{}, error) {
	inspect := pass.ResultOf[inspect.Analyzer].(*inspector.Inspector)

	nodeFilter := []ast.Node{
		(*ast.ExprStmt)(nil),
	}
	inspect.Preorder(nodeFilter, func(n ast.Node) {
		call, ok := analysisutil.Unparen(n.(*ast.ExprStmt).X).(*ast.CallExpr)
		if !ok {
			return // not a call statement
		}
		fun := analysisutil.Unparen(call.Fun)

		if pass.TypesInfo.Types[fun].IsType() {
			return // a conversion, not a call
		}

		selector, ok := fun.(*ast.SelectorExpr)
		if !ok {
			return // neither a method call nor a qualified ident
		}

		sel, ok := pass.TypesInfo.Selections[selector]
		if ok && sel.Kind() == types.MethodVal {
			// method (e.g. foo.String())
			obj := sel.Obj().(*types.Func)
			sig := sel.Type().(*types.Signature)
			if types.Identical(sig, sigNoArgsStringResult) {
				if stringMethods[obj.Name()] {
					pass.Reportf(call.Lparen, "result of (%s).%s call not used",
						sig.Recv().Type(), obj.Name())
				}
			}
		} else if !ok {
			// package-qualified function (e.g. fmt.Errorf)
			obj := pass.TypesInfo.Uses[selector.Sel]
			if obj, ok := obj.(*types.Func); ok {
				qname := obj.Pkg().Path() + "." + obj.Name()
				if funcs[qname] {
					pass.Reportf(call.Lparen, "result of %v call not used", qname)
				}
			}
		}
	})
	return nil, nil
}

// func() string
var sigNoArgsStringResult = types.NewSignature(nil, nil,
	types.NewTuple(types.NewVar(token.NoPos, nil, "", types.Typ[types.String])),
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
