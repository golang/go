package astutil

import "go/ast"

// Unparen returns e with any enclosing parentheses stripped.
func Unparen(e ast.Expr) ast.Expr {
	for {
		p, ok := e.(*ast.ParenExpr)
		if !ok {
			return e
		}
		e = p.X
	}
}
