//go:build go1.28

package parser

import (
	"go/ast"
	"go/token"
)

const preciseLiteralEnd = true

func basicLit(pos, end token.Pos, kind token.Token, value string) *ast.BasicLit {
	return &ast.BasicLit{ValuePos: pos, ValueEnd: end, Kind: kind, Value: value}
}
