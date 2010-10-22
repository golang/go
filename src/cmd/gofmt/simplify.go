// Copyright 2010 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package main

import (
	"go/ast"
	"reflect"
)


type compositeLitFinder struct{}

func (f *compositeLitFinder) Visit(node interface{}) ast.Visitor {
	if outer, ok := node.(*ast.CompositeLit); ok {
		// array, slice, and map composite literals may be simplified
		var eltType ast.Expr
		switch typ := outer.Type.(type) {
		case *ast.ArrayType:
			eltType = typ.Elt
		case *ast.MapType:
			eltType = typ.Value
		}

		if eltType != nil {
			typ := reflect.NewValue(eltType)
			for _, x := range outer.Elts {
				// look at value of indexed/named elements
				if t, ok := x.(*ast.KeyValueExpr); ok {
					x = t.Value
				}
				simplify(x)
				// if the element is a composite literal and its literal type
				// matches the outer literal's element type exactly, the inner
				// literal type may be omitted
				if inner, ok := x.(*ast.CompositeLit); ok {
					if match(nil, typ, reflect.NewValue(inner.Type)) {
						inner.Type = nil
					}
				}
			}

			// node was simplified - stop walk
			return nil
		}
	}

	// not a composite literal or not simplified - continue walk
	return f
}


func simplify(node interface{}) {
	var f compositeLitFinder
	ast.Walk(&f, node)
}
