// Copyright 2010 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package main

import (
	"go/ast"
	"go/token"
	"reflect"
)

type simplifier struct{}

func (s *simplifier) Visit(node ast.Node) ast.Visitor {
	switch n := node.(type) {
	case *ast.CompositeLit:
		// array, slice, and map composite literals may be simplified
		outer := n
		var eltType ast.Expr
		switch typ := outer.Type.(type) {
		case *ast.ArrayType:
			eltType = typ.Elt
		case *ast.MapType:
			eltType = typ.Value
		}

		if eltType != nil {
			typ := reflect.ValueOf(eltType)
			for i, x := range outer.Elts {
				px := &outer.Elts[i]
				// look at value of indexed/named elements
				if t, ok := x.(*ast.KeyValueExpr); ok {
					x = t.Value
					px = &t.Value
				}
				simplify(x)
				// if the element is a composite literal and its literal type
				// matches the outer literal's element type exactly, the inner
				// literal type may be omitted
				if inner, ok := x.(*ast.CompositeLit); ok {
					if match(nil, typ, reflect.ValueOf(inner.Type)) {
						inner.Type = nil
					}
				}
				// if the outer literal's element type is a pointer type *T
				// and the element is & of a composite literal of type T,
				// the inner &T may be omitted.
				if ptr, ok := eltType.(*ast.StarExpr); ok {
					if addr, ok := x.(*ast.UnaryExpr); ok && addr.Op == token.AND {
						if inner, ok := addr.X.(*ast.CompositeLit); ok {
							if match(nil, reflect.ValueOf(ptr.X), reflect.ValueOf(inner.Type)) {
								inner.Type = nil // drop T
								*px = inner      // drop &
							}
						}
					}
				}
			}

			// node was simplified - stop walk (there are no subnodes to simplify)
			return nil
		}

	case *ast.RangeStmt:
		// range of the form: for x, _ = range v {...}
		// can be simplified to: for x = range v {...}
		if n.Value != nil {
			if ident, ok := n.Value.(*ast.Ident); ok && ident.Name == "_" {
				n.Value = nil
			}
		}
	}

	return s
}

func simplify(node ast.Node) {
	var s simplifier
	ast.Walk(&s, node)
}
