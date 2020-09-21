package extract

import "go/ast"

func _() {
	x0 := 0
	if true {
		y := ast.CompositeLit{} //@suggestedfix("ast.CompositeLit{}", "refactor.extract")
	}
	if true {
		x1 := !false //@suggestedfix("!false", "refactor.extract")
	}
}
