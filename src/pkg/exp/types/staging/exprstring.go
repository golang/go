// Copyright 2012 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package types

import (
	"bytes"
	"fmt"
	"go/ast"
)

// exprString returns a (simplified) string representation for an expression.
func exprString(expr ast.Expr) string {
	var buf bytes.Buffer
	writeExpr(&buf, expr)
	return buf.String()
}

// TODO(gri) Need to merge with typeString since some expressions are types (try: ([]int)(a))
func writeExpr(buf *bytes.Buffer, expr ast.Expr) {
	switch x := expr.(type) {
	case *ast.Ident:
		buf.WriteString(x.Name)

	case *ast.BasicLit:
		buf.WriteString(x.Value)

	case *ast.FuncLit:
		buf.WriteString("(func literal)")

	case *ast.CompositeLit:
		buf.WriteString("(composite literal)")

	case *ast.ParenExpr:
		buf.WriteByte('(')
		writeExpr(buf, x.X)
		buf.WriteByte(')')

	case *ast.SelectorExpr:
		writeExpr(buf, x.X)
		buf.WriteByte('.')
		buf.WriteString(x.Sel.Name)

	case *ast.IndexExpr:
		writeExpr(buf, x.X)
		buf.WriteByte('[')
		writeExpr(buf, x.Index)
		buf.WriteByte(']')

	case *ast.SliceExpr:
		writeExpr(buf, x.X)
		buf.WriteByte('[')
		if x.Low != nil {
			writeExpr(buf, x.Low)
		}
		buf.WriteByte(':')
		if x.High != nil {
			writeExpr(buf, x.High)
		}
		buf.WriteByte(']')

	case *ast.TypeAssertExpr:
		writeExpr(buf, x.X)
		buf.WriteString(".(...)")

	case *ast.CallExpr:
		writeExpr(buf, x.Fun)
		buf.WriteByte('(')
		for i, arg := range x.Args {
			if i > 0 {
				buf.WriteString(", ")
			}
			writeExpr(buf, arg)
		}
		buf.WriteByte(')')

	case *ast.StarExpr:
		buf.WriteByte('*')
		writeExpr(buf, x.X)

	case *ast.UnaryExpr:
		buf.WriteString(x.Op.String())
		writeExpr(buf, x.X)

	case *ast.BinaryExpr:
		// The AST preserves source-level parentheses so there is
		// no need to introduce parentheses here for correctness.
		writeExpr(buf, x.X)
		buf.WriteByte(' ')
		buf.WriteString(x.Op.String())
		buf.WriteByte(' ')
		writeExpr(buf, x.Y)

	default:
		fmt.Fprintf(buf, "<expr %T>", x)
	}
}
