//go:build !go1.28

package parser

import (
	"go/ast"
	"go/token"
)

const preciseLiteralEnd = false

func basicLit(pos, end token.Pos, kind token.Token, value string) *ast.BasicLit {
	return &ast.BasicLit{ValuePos: pos, Kind: kind, Value: value}
}
