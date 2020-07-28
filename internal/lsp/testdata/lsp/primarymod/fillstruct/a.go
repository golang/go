// Copyright 2020 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package fillstruct

import (
	"go/ast"
	"go/token"

	"golang.org/x/tools/internal/lsp/fillstruct/data"
)

type basicStruct struct {
	foo int
}

var _ = basicStruct{} //@suggestedfix("}", "refactor.rewrite")

type twoArgStruct struct {
	foo int
	bar string
}

var _ = twoArgStruct{} //@suggestedfix("}", "refactor.rewrite")

type nestedStruct struct {
	bar   string
	basic basicStruct
}

var _ = nestedStruct{} //@suggestedfix("}", "refactor.rewrite")

var _ = data.B{} //@suggestedfix("}", "refactor.rewrite")

type typedStruct struct {
	m  map[string]int
	s  []int
	c  chan int
	c1 <-chan int
	a  [2]string
}

var _ = typedStruct{} //@suggestedfix("}", "refactor.rewrite")

type funStruct struct {
	fn func(i int) int
}

var _ = funStruct{} //@suggestedfix("}", "refactor.rewrite")

type funStructCompex struct {
	fn func(i int, s string) (string, int)
}

var _ = funStructCompex{} //@suggestedfix("}", "refactor.rewrite")

type funStructEmpty struct {
	fn func()
}

var _ = funStructEmpty{} //@suggestedfix("}", "refactor.rewrite")

type Foo struct {
	A int
}

type Bar struct {
	X *Foo
	Y *Foo
}

var _ = Bar{} //@suggestedfix("}", "refactor.rewrite")

type importedStruct struct {
	m  map[*ast.CompositeLit]ast.Field
	s  []ast.BadExpr
	a  [3]token.Token
	c  chan ast.EmptyStmt
	fn func(ast_decl ast.DeclStmt) ast.Ellipsis
	st ast.CompositeLit
}

var _ = importedStruct{} //@suggestedfix("}", "refactor.rewrite")

type pointerBuiltinStruct struct {
	b *bool
	s *string
	i *int
}

var _ = pointerBuiltinStruct{} //@suggestedfix("}", "refactor.rewrite")

var _ = []ast.BasicLit{
	{}, //@suggestedfix("}", "refactor.rewrite")
}

var _ = []ast.BasicLit{{}} //@suggestedfix("}", "refactor.rewrite")

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
