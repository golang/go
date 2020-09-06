package fillstruct

import (
	"go/ast"
	"go/token"
)

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
