// Copyright 2020 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package fillstruct

import (
	data "b"
	"go/ast"
	"go/token"
)

type emptyStruct struct{}

var _ = emptyStruct{}

type basicStruct struct {
	foo int
}

var _ = basicStruct{} // want ""

type twoArgStruct struct {
	foo int
	bar string
}

var _ = twoArgStruct{} // want ""

var _ = twoArgStruct{ // want ""
	bar: "bar",
}

type nestedStruct struct {
	bar   string
	basic basicStruct
}

var _ = nestedStruct{} // want ""

var _ = data.B{} // want ""

type typedStruct struct {
	m  map[string]int
	s  []int
	c  chan int
	c1 <-chan int
	a  [2]string
}

var _ = typedStruct{} // want ""

type funStruct struct {
	fn func(i int) int
}

var _ = funStruct{} // want ""

type funStructCompex struct {
	fn func(i int, s string) (string, int)
}

var _ = funStructCompex{} // want ""

type funStructEmpty struct {
	fn func()
}

var _ = funStructEmpty{} // want ""

type Foo struct {
	A int
}

type Bar struct {
	X *Foo
	Y *Foo
}

var _ = Bar{} // want ""

type importedStruct struct {
	m  map[*ast.CompositeLit]ast.Field
	s  []ast.BadExpr
	a  [3]token.Token
	c  chan ast.EmptyStmt
	fn func(ast_decl ast.DeclStmt) ast.Ellipsis
	st ast.CompositeLit
}

var _ = importedStruct{} // want ""

type pointerBuiltinStruct struct {
	b *bool
	s *string
	i *int
}

var _ = pointerBuiltinStruct{} // want ""

var _ = []ast.BasicLit{
	{}, // want ""
}

var _ = []ast.BasicLit{{}, // want ""
}
