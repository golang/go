// Copyright 2016 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// This file contains the check for http.Response values being used before
// checking for errors.

package main

import (
	"go/ast"
	"go/types"
)

func init() {
	register("httpresponse",
		"check errors are checked before using an http Response",
		checkHTTPResponse, callExpr)
}

func checkHTTPResponse(f *File, node ast.Node) {
	// If http.Response or http.Client are not defined, skip this check.
	if httpResponseType == nil || httpClientType == nil {
		return
	}
	call := node.(*ast.CallExpr)
	if !isHTTPFuncOrMethodOnClient(f, call) {
		return // the function call is not related to this check.
	}

	finder := &blockStmtFinder{node: call}
	ast.Walk(finder, f.file)
	stmts := finder.stmts()
	if len(stmts) < 2 {
		return // the call to the http function is the last statement of the block.
	}

	asg, ok := stmts[0].(*ast.AssignStmt)
	if !ok {
		return // the first statement is not assignment.
	}
	resp := rootIdent(asg.Lhs[0])
	if resp == nil {
		return // could not find the http.Response in the assignment.
	}

	def, ok := stmts[1].(*ast.DeferStmt)
	if !ok {
		return // the following statement is not a defer.
	}
	root := rootIdent(def.Call.Fun)
	if root == nil {
		return // could not find the receiver of the defer call.
	}

	if resp.Obj == root.Obj {
		f.Badf(root.Pos(), "using %s before checking for errors", resp.Name)
	}
}

// isHTTPFuncOrMethodOnClient checks whether the given call expression is on
// either a function of the net/http package or a method of http.Client that
// returns (*http.Response, error).
func isHTTPFuncOrMethodOnClient(f *File, expr *ast.CallExpr) bool {
	fun, _ := expr.Fun.(*ast.SelectorExpr)
	sig, _ := f.pkg.types[fun].Type.(*types.Signature)
	if sig == nil {
		return false // the call is not on of the form x.f()
	}

	res := sig.Results()
	if res.Len() != 2 {
		return false // the function called does not return two values.
	}
	if ptr, ok := res.At(0).Type().(*types.Pointer); !ok || !types.Identical(ptr.Elem(), httpResponseType) {
		return false // the first return type is not *http.Response.
	}
	if !types.Identical(res.At(1).Type().Underlying(), errorType) {
		return false // the second return type is not error
	}

	typ := f.pkg.types[fun.X].Type
	if typ == nil {
		id, ok := fun.X.(*ast.Ident)
		return ok && id.Name == "http" // function in net/http package.
	}

	if types.Identical(typ, httpClientType) {
		return true // method on http.Client.
	}
	ptr, ok := typ.(*types.Pointer)
	return ok && types.Identical(ptr.Elem(), httpClientType) // method on *http.Client.
}

// blockStmtFinder is an ast.Visitor that given any ast node can find the
// statement containing it and its succeeding statements in the same block.
type blockStmtFinder struct {
	node  ast.Node       // target of search
	stmt  ast.Stmt       // innermost statement enclosing argument to Visit
	block *ast.BlockStmt // innermost block enclosing argument to Visit.
}

// Visit finds f.node performing a search down the ast tree.
// It keeps the last block statement and statement seen for later use.
func (f *blockStmtFinder) Visit(node ast.Node) ast.Visitor {
	if node == nil || f.node.Pos() < node.Pos() || f.node.End() > node.End() {
		return nil // not here
	}
	switch n := node.(type) {
	case *ast.BlockStmt:
		f.block = n
	case ast.Stmt:
		f.stmt = n
	}
	if f.node.Pos() == node.Pos() && f.node.End() == node.End() {
		return nil // found
	}
	return f // keep looking
}

// stmts returns the statements of f.block starting from the one including f.node.
func (f *blockStmtFinder) stmts() []ast.Stmt {
	for i, v := range f.block.List {
		if f.stmt == v {
			return f.block.List[i:]
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
