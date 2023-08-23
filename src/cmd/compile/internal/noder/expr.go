// Copyright 2021 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package noder

import (
	"fmt"

	"cmd/compile/internal/ir"
	"cmd/compile/internal/syntax"
)

// constExprOp returns an ir.Op that represents the outermost
// operation of the given constant expression. It's intended for use
// with ir.RawOrigExpr.
func constExprOp(expr syntax.Expr) ir.Op {
	switch expr := expr.(type) {
	default:
		panic(fmt.Sprintf("%s: unexpected expression: %T", expr.Pos(), expr))

	case *syntax.BasicLit:
		return ir.OLITERAL
	case *syntax.Name, *syntax.SelectorExpr:
		return ir.ONAME
	case *syntax.CallExpr:
		return ir.OCALL
	case *syntax.Operation:
		if expr.Y == nil {
			return unOps[expr.Op]
		}
		return binOps[expr.Op]
	}
}
