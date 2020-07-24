package extract

import "go/ast"

func _() {
	ast.Inspect(ast.NewIdent("a"), func(n ast.Node) bool {
		if n == nil { //@mark(exSt4, "if")
			return true
		} //@mark(exEn4, "}")
		return false
	})
	//@extractfunc(exSt4, exEn4)
}
