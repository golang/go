// Copyright 2016 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// Package httpresponse defines an Analyzer that checks for mistakes
// using HTTP responses.
package httpresponse

import (
	"go/ast"
	"go/types"

	"golang.org/x/tools/go/analysis"
	"golang.org/x/tools/go/analysis/passes/inspect"
	"golang.org/x/tools/go/ast/inspector"
	"golang.org/x/tools/internal/analysisinternal"
	"golang.org/x/tools/internal/typesinternal"
)

const Doc = `check for mistakes using HTTP responses

A common mistake when using the net/http package is to defer a function
call to close the http.Response Body before checking the error that
determines whether the response is valid:

	resp, err := http.Head(url)
	defer resp.Body.Close()
	if err != nil {
		log.Fatal(err)
	}
	// (defer statement belongs here)

This checker helps uncover latent nil dereference bugs by reporting a
diagnostic for such mistakes.`

var Analyzer = &analysis.Analyzer{
	Name:     "httpresponse",
	Doc:      Doc,
	URL:      "https://pkg.go.dev/golang.org/x/tools/go/analysis/passes/httpresponse",
	Requires: []*analysis.Analyzer{inspect.Analyzer},
	Run:      run,
}

func run(pass *analysis.Pass) (any, error) {
	inspect := pass.ResultOf[inspect.Analyzer].(*inspector.Inspector)

	// Fast path: if the package doesn't import net/http,
	// skip the traversal.
	if !analysisinternal.Imports(pass.Pkg, "net/http") {
		return nil, nil
	}

	nodeFilter := []ast.Node{
		(*ast.CallExpr)(nil),
	}
	inspect.WithStack(nodeFilter, func(n ast.Node, push bool, stack []ast.Node) bool {
		if !push {
			return true
		}
		call := n.(*ast.CallExpr)
		if !isHTTPFuncOrMethodOnClient(pass.TypesInfo, call) {
			return true // the function call is not related to this check.
		}

		// Find the innermost containing block, and get the list
		// of statements starting with the one containing call.
		stmts, ncalls := restOfBlock(stack)
		if len(stmts) < 2 {
			// The call to the http function is the last statement of the block.
			return true
		}

		// Skip cases in which the call is wrapped by another (#52661).
		// Example:  resp, err := checkError(http.Get(url))
		if ncalls > 1 {
			return true
		}

		asg, ok := stmts[0].(*ast.AssignStmt)
		if !ok {
			return true // the first statement is not assignment.
		}

		resp := rootIdent(asg.Lhs[0])
		if resp == nil {
			return true // could not find the http.Response in the assignment.
		}

		def, ok := stmts[1].(*ast.DeferStmt)
		if !ok {
			return true // the following statement is not a defer.
		}
		root := rootIdent(def.Call.Fun)
		if root == nil {
			return true // could not find the receiver of the defer call.
		}

		if resp.Obj == root.Obj {
			pass.ReportRangef(root, "using %s before checking for errors", resp.Name)
		}
		return true
	})
	return nil, nil
}

// isHTTPFuncOrMethodOnClient checks whether the given call expression is on
// either a function of the net/http package or a method of http.Client that
// returns (*http.Response, error).
func isHTTPFuncOrMethodOnClient(info *types.Info, expr *ast.CallExpr) bool {
	fun, _ := expr.Fun.(*ast.SelectorExpr)
	sig, _ := info.Types[fun].Type.(*types.Signature)
	if sig == nil {
		return false // the call is not of the form x.f()
	}

	res := sig.Results()
	if res.Len() != 2 {
		return false // the function called does not return two values.
	}
	isPtr, named := typesinternal.ReceiverNamed(res.At(0))
	if !isPtr || named == nil || !analysisinternal.IsTypeNamed(named, "net/http", "Response") {
		return false // the first return type is not *http.Response.
	}

	errorType := types.Universe.Lookup("error").Type()
	if !types.Identical(res.At(1).Type(), errorType) {
		return false // the second return type is not error
	}

	typ := info.Types[fun.X].Type
	if typ == nil {
		id, ok := fun.X.(*ast.Ident)
		return ok && id.Name == "http" // function in net/http package.
	}

	if analysisinternal.IsTypeNamed(typ, "net/http", "Client") {
		return true // method on http.Client.
	}
	ptr, ok := types.Unalias(typ).(*types.Pointer)
	return ok && analysisinternal.IsTypeNamed(ptr.Elem(), "net/http", "Client") // method on *http.Client.
}

// restOfBlock, given a traversal stack, finds the innermost containing
// block and returns the suffix of its statements starting with the current
// node, along with the number of call expressions encountered.
func restOfBlock(stack []ast.Node) ([]ast.Stmt, int) {
	var ncalls int
	for i := len(stack) - 1; i >= 0; i-- {
		if b, ok := stack[i].(*ast.BlockStmt); ok {
			for j, v := range b.List {
				if v == stack[i+1] {
					return b.List[j:], ncalls
				}
			}
			break
		}

		if _, ok := stack[i].(*ast.CallExpr); ok {
			ncalls++
		}
	}
	return nil, 0
}

// rootIdent finds the root identifier x in a chain of selections x.y.z, or nil if not found.
func rootIdent(n ast.Node) *ast.Ident {
	switch n := n.(type) {
	case *ast.SelectorExpr:
		return rootIdent(n.X)
	case *ast.Ident:
		return n
	default:
		return nil
	}
}
