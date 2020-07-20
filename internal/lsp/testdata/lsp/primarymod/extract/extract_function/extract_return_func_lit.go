package extract

import "go/ast"

func _() {
	ast.Inspect(ast.NewIdent("a"), func(n ast.Node) bool {
		if n == nil { //@mark(s0, "if")
			return true
		} //@mark(e0, "}")
		return false
	})
	//@extractfunc(s0, e0)
}
