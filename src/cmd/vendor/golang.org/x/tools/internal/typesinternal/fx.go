// Copyright 2025 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package typesinternal

import (
	"go/ast"
	"go/token"
	"go/types"
)

// NoEffects reports whether the expression has no side effects, i.e., it
// does not modify the memory state. This function is conservative: it may
// return false even when the expression has no effect.
func NoEffects(info *types.Info, expr ast.Expr) bool {
	noEffects := true
	ast.Inspect(expr, func(n ast.Node) bool {
		switch v := n.(type) {
		case nil, *ast.Ident, *ast.BasicLit, *ast.BinaryExpr, *ast.ParenExpr,
			*ast.SelectorExpr, *ast.IndexExpr, *ast.SliceExpr, *ast.TypeAssertExpr,
			*ast.StarExpr, *ast.CompositeLit,
			// non-expressions that may appear within expressions
			*ast.KeyValueExpr,
			*ast.FieldList,
			*ast.Field,
			*ast.Ellipsis,
			*ast.IndexListExpr:
			// No effect.

		case *ast.ArrayType,
			*ast.StructType,
			*ast.ChanType,
			*ast.FuncType,
			*ast.MapType,
			*ast.InterfaceType:
			// Type syntax: no effects, recursively.
			// Prune descent.
			return false

		case *ast.UnaryExpr:
			// Channel send <-ch has effects.
			if v.Op == token.ARROW {
				noEffects = false
			}

		case *ast.CallExpr:
			// Type conversion has no effects.
			if !info.Types[v.Fun].IsType() {
				if CallsPureBuiltin(info, v) {
					// A call such as len(e) has no effects of its
					// own, though the subexpression e might.
				} else {
					noEffects = false
				}
			}

		case *ast.FuncLit:
			// A FuncLit has no effects, but do not descend into it.
			return false

		default:
			// All other expressions have effects
			noEffects = false
		}

		return noEffects
	})
	return noEffects
}

// CallsPureBuiltin reports whether call is a call of a built-in
// function that is a pure computation over its operands (analogous to
// a + operator). Because it does not depend on program state, it may
// be evaluated at any point--though not necessarily at multiple
// points (consider new, make).
func CallsPureBuiltin(info *types.Info, call *ast.CallExpr) bool {
	if id, ok := ast.Unparen(call.Fun).(*ast.Ident); ok {
		if b, ok := info.ObjectOf(id).(*types.Builtin); ok {
			switch b.Name() {
			case "len", "cap", "complex", "imag", "real", "make", "new", "max", "min":
				return true
			}
			// Not: append clear close copy delete panic print println recover
		}
	}
	return false
}
