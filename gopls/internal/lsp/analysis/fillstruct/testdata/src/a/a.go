// Copyright 2020 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package fillstruct

import (
	data "b"
	"go/ast"
	"go/token"
	"unsafe"
)

type emptyStruct struct{}

var _ = emptyStruct{}

type basicStruct struct {
	foo int
}

var _ = basicStruct{} // want `Fill basicStruct`

type twoArgStruct struct {
	foo int
	bar string
}

var _ = twoArgStruct{} // want `Fill twoArgStruct`

var _ = twoArgStruct{ // want `Fill twoArgStruct`
	bar: "bar",
}

type nestedStruct struct {
	bar   string
	basic basicStruct
}

var _ = nestedStruct{} // want `Fill nestedStruct`

var _ = data.B{} // want `Fill b.B`

type typedStruct struct {
	m  map[string]int
	s  []int
	c  chan int
	c1 <-chan int
	a  [2]string
}

var _ = typedStruct{} // want `Fill typedStruct`

type funStruct struct {
	fn func(i int) int
}

var _ = funStruct{} // want `Fill funStruct`

type funStructComplex struct {
	fn func(i int, s string) (string, int)
}

var _ = funStructComplex{} // want `Fill funStructComplex`

type funStructEmpty struct {
	fn func()
}

var _ = funStructEmpty{} // want `Fill funStructEmpty`

type Foo struct {
	A int
}

type Bar struct {
	X *Foo
	Y *Foo
}

var _ = Bar{} // want `Fill Bar`

type importedStruct struct {
	m  map[*ast.CompositeLit]ast.Field
	s  []ast.BadExpr
	a  [3]token.Token
	c  chan ast.EmptyStmt
	fn func(ast_decl ast.DeclStmt) ast.Ellipsis
	st ast.CompositeLit
}

var _ = importedStruct{} // want `Fill importedStruct`

type pointerBuiltinStruct struct {
	b *bool
	s *string
	i *int
}

var _ = pointerBuiltinStruct{} // want `Fill pointerBuiltinStruct`

var _ = []ast.BasicLit{
	{}, // want `Fill go/ast.BasicLit`
}

var _ = []ast.BasicLit{{}, // want "go/ast.BasicLit"
}

type unsafeStruct struct {
	foo unsafe.Pointer
}

var _ = unsafeStruct{} // want `Fill unsafeStruct`
