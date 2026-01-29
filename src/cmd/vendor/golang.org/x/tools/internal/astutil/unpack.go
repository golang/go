// Copyright 2023 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package astutil

import (
	"go/ast"

	"golang.org/x/tools/internal/typeparams"
)

// UnpackRecv unpacks a receiver type expression, reporting whether it is a
// pointer receiver, along with the type name identifier and any receiver type
// parameter identifiers.
//
// Copied (with modifications) from go/types.
func UnpackRecv(rtyp ast.Expr) (ptr bool, rname *ast.Ident, tparams []*ast.Ident) {
L: // unpack receiver type
	// This accepts invalid receivers such as ***T and does not
	// work for other invalid receivers, but we don't care. The
	// validity of receiver expressions is checked elsewhere.
	for {
		switch t := rtyp.(type) {
		case *ast.ParenExpr:
			rtyp = t.X
		case *ast.StarExpr:
			ptr = true
			rtyp = t.X
		default:
			break L
		}
	}

	// unpack type parameters, if any
	switch rtyp.(type) {
	case *ast.IndexExpr, *ast.IndexListExpr:
		var indices []ast.Expr
		rtyp, _, indices, _ = typeparams.UnpackIndexExpr(rtyp)
		for _, arg := range indices {
			var par *ast.Ident
			switch arg := arg.(type) {
			case *ast.Ident:
				par = arg
			default:
				// ignore errors
			}
			if par == nil {
				par = &ast.Ident{NamePos: arg.Pos(), Name: "_"}
			}
			tparams = append(tparams, par)
		}
	}

	// unpack receiver name
	if name, _ := rtyp.(*ast.Ident); name != nil {
		rname = name
	}

	return
}
