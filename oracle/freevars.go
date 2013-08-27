// Copyright 2013 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package oracle

import (
	"go/ast"

	"code.google.com/p/go.tools/go/types"
)

// freevars displays the lexical (not package-level) free variables of
// the selection.
//
// It treats A.B.C as a separate variable from A to reveal the parts
// of an aggregate type that are actually needed.
// This aids refactoring.
//
// TODO(adonovan): optionally display the free references to
// file/package scope objects, and to objects from other packages.
// Depending on where the resulting function abstraction will go,
// these might be interesting.  Perhaps group the results into three
// bands.
//
func freevars(o *oracle) (queryResult, error) {
	file := o.queryPath[len(o.queryPath)-1] // the enclosing file

	// The id and sel functions return non-nil if they denote an
	// object o or selection o.x.y that is referenced by the
	// selection but defined neither within the selection nor at
	// file scope, i.e. it is in the lexical environment.
	var id func(n *ast.Ident) types.Object
	var sel func(n *ast.SelectorExpr) types.Object

	sel = func(n *ast.SelectorExpr) types.Object {
		switch x := unparen(n.X).(type) {
		case *ast.SelectorExpr:
			return sel(x)
		case *ast.Ident:
			return id(x)
		}
		return nil
	}

	id = func(n *ast.Ident) types.Object {
		obj := o.queryPkgInfo.ObjectOf(n)
		if obj == nil {
			return nil // TODO(adonovan): fix: this fails for *types.Label.
			panic(o.errorf(n, "no types.Object for ast.Ident"))
		}
		if _, ok := obj.(*types.Package); ok {
			return nil // imported package
		}
		if n.Pos() == obj.Pos() {
			return nil // this ident is the definition, not a reference
		}
		if !(file.Pos() <= obj.Pos() && obj.Pos() <= file.End()) {
			return nil // not defined in this file
		}
		if obj.Parent() == nil {
			return nil // e.g. interface method  TODO(adonovan): what else?
		}
		if obj.Parent() == o.queryPkgInfo.Scopes[file] {
			return nil // defined at file scope
		}
		if o.startPos <= obj.Pos() && obj.Pos() <= o.endPos {
			return nil // defined within selection => not free
		}
		return obj
	}

	// Maps each reference that is free in the selection
	// to the object it refers to.
	freeRefs := make(map[string]freevarsRef)

	// Visit all the identifiers in the selected ASTs.
	ast.Inspect(o.queryPath[0], func(n ast.Node) bool {
		if n == nil {
			return true // popping DFS stack
		}

		// Is this node contained within the selection?
		// (freevars permits inexact selections,
		// like two stmts in a block.)
		if o.startPos <= n.Pos() && n.End() <= o.endPos {
			switch n := n.(type) {
			case *ast.Ident:
				if obj := id(n); obj != nil {
					freeRefs[o.printNode(n)] = freevarsRef{n, obj}
				}

			case *ast.SelectorExpr:
				if obj := sel(n); obj != nil {
					freeRefs[o.printNode(n)] = freevarsRef{n, obj}
					return false // don't descend
				}
			}
		}

		return true // descend
	})

	return &freevarsResult{
		refs: freeRefs,
	}, nil
}

type freevarsResult struct {
	refs map[string]freevarsRef
}

type freevarsRef struct {
	expr ast.Expr
	obj  types.Object
}

func (r *freevarsResult) display(o *oracle) {
	if len(r.refs) == 0 {
		o.printf(o, "No free identifers.")
		return
	}

	o.printf(o, "Free identifers:")
	for s, ref := range r.refs {
		typ := ref.obj.Type()
		if _, ok := ref.expr.(*ast.SelectorExpr); ok {
			typ = o.queryPkgInfo.TypeOf(ref.expr)
		}
		var kind string
		switch ref.obj.(type) {
		case *types.Var:
			kind = "var"
		case *types.Func:
			kind = "func"
		case *types.TypeName:
			kind = "type"
		case *types.Const:
			kind = "const"
		case *types.Label:
			kind = "label"
		default:
			panic(ref.obj)
		}
		o.printf(ref.obj, "%s %s %s", kind, s, typ)
	}
}
