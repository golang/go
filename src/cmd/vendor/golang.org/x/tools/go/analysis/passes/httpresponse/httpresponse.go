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
	Requires: []*analysis.Analyzer{inspect.Analyzer},
	Run:      run,
}

func run(pass *analysis.Pass) (interface{}, error) {
	inspect := pass.ResultOf[inspect.Analyzer].(*inspector.Inspector)

	// Fast path: if the package doesn't import net/http,
	// skip the traversal.
	if !imports(pass.Pkg, "net/http") {
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
		stmts := restOfBlock(stack)
		if len(stmts) < 2 {
			return true // the call to the http function is the last statement of the block.
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
	if ptr, ok := res.At(0).Type().(*types.Pointer); !ok || !isNamedType(ptr.Elem(), "net/http", "Response") {
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

	if isNamedType(typ, "net/http", "Client") {
		return true // method on http.Client.
	}
	ptr, ok := typ.(*types.Pointer)
	return ok && isNamedType(ptr.Elem(), "net/http", "Client") // method on *http.Client.
}

// restOfBlock, given a traversal stack, finds the innermost containing
// block and returns the suffix of its statements starting with the
// current node (the last element of stack).
func restOfBlock(stack []ast.Node) []ast.Stmt {
	for i := len(stack) - 1; i >= 0; i-- {
		if b, ok := stack[i].(*ast.BlockStmt); ok {
			for j, v := range b.List {
				if v == stack[i+1] {
					return b.List[j:]
				}
			}
			break
		}
	}
	return nil
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

// isNamedType reports whether t is the named type path.name.
func isNamedType(t types.Type, path, name string) bool {
	n, ok := t.(*types.Named)
	if !ok {
		return false
	}
	obj := n.Obj()
	return obj.Name() == name && obj.Pkg() != nil && obj.Pkg().Path() == path
}

func imports(pkg *types.Package, path string) bool {
	for _, imp := range pkg.Imports() {
		if imp.Path() == path {
			return true
		}
	}
	return false
}
