package extract

import "go/ast"

func _() {
	ast.Inspect(ast.NewIdent("a"), func(n ast.Node) bool {
		if n == nil { //@mark(exSt11, "if")
			return true
		}
		return false //@mark(exEn11, "false")
	})
	//@extractfunc(exSt11, exEn11)
}
