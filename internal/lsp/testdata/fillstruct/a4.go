package fillstruct

import "go/ast"

type iStruct struct {
	X int
}

type sStruct struct {
	str string
}

type multiFill struct {
	num   int
	strin string
	arr   []int
}

type assignStruct struct {
	n ast.Node
}

func fill() {
	var x int
	var _ = iStruct{} //@suggestedfix("}", "refactor.rewrite")

	var s string
	var _ = sStruct{} //@suggestedfix("}", "refactor.rewrite")

	var n int
	_ = []int{}
	if true {
		arr := []int{1, 2}
	}
	var _ = multiFill{} //@suggestedfix("}", "refactor.rewrite")

	var node *ast.CompositeLit
	var _ = assignStruct{} //@suggestedfix("}", "refactor.rewrite")
}
