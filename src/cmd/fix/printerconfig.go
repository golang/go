// Copyright 2012 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package main

import "go/ast"

func init() {
	register(printerconfigFix)
}

var printerconfigFix = fix{
	name: "printerconfig",
	date: "2012-12-11",
	f:    printerconfig,
	desc: `Add element keys to Config composite literals.`,
}

func printerconfig(f *ast.File) bool {
	if !imports(f, "go/printer") {
		return false
	}

	fixed := false
	walk(f, func(n interface{}) {
		cl, ok := n.(*ast.CompositeLit)
		if !ok {
			return
		}
		se, ok := cl.Type.(*ast.SelectorExpr)
		if !ok {
			return
		}
		if !isTopName(se.X, "printer") || se.Sel == nil {
			return
		}

		if ss := se.Sel.String(); ss == "Config" {
			for i, e := range cl.Elts {
				if _, ok := e.(*ast.KeyValueExpr); ok {
					break
				}
				switch i {
				case 0:
					cl.Elts[i] = &ast.KeyValueExpr{
						Key:   ast.NewIdent("Mode"),
						Value: e,
					}
				case 1:
					cl.Elts[i] = &ast.KeyValueExpr{
						Key:   ast.NewIdent("Tabwidth"),
						Value: e,
					}
				}
				fixed = true
			}
		}
	})
	return fixed
}
