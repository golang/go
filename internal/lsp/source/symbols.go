// Copyright 2019 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package source

import (
	"bytes"
	"context"
	"go/ast"
	"go/format"
	"go/token"

	"golang.org/x/tools/internal/span"
)

type SymbolKind int

const (
	PackageSymbol SymbolKind = iota
	StructSymbol
	VariableSymbol
	ConstantSymbol
	FunctionSymbol
	MethodSymbol
)

type Symbol struct {
	Name     string
	Detail   string
	Span     span.Span
	Kind     SymbolKind
	Children []Symbol
}

func DocumentSymbols(ctx context.Context, f File) []Symbol {
	var symbols []Symbol
	fset := f.GetFileSet(ctx)
	astFile := f.GetAST(ctx)
	for _, decl := range astFile.Decls {
		switch decl := decl.(type) {
		case *ast.FuncDecl:
			symbols = append(symbols, funcSymbol(decl, fset))
		case *ast.GenDecl:
			for _, spec := range decl.Specs {
				switch spec := spec.(type) {
				case *ast.ImportSpec:
					symbols = append(symbols, importSymbol(spec, fset))
				case *ast.TypeSpec:
					symbols = append(symbols, typeSymbol(spec, fset))
				case *ast.ValueSpec:
					for _, name := range spec.Names {
						symbols = append(symbols, varSymbol(decl, name, fset))
					}
				}
			}
		}
	}
	return symbols
}

func funcSymbol(decl *ast.FuncDecl, fset *token.FileSet) Symbol {
	s := Symbol{
		Name: decl.Name.String(),
		Kind: FunctionSymbol,
	}

	if decl.Recv != nil {
		s.Kind = MethodSymbol
	}

	span, err := nodeSpan(decl, fset)
	if err == nil {
		s.Span = span
	}
	buf := &bytes.Buffer{}
	if err := format.Node(buf, fset, decl); err == nil {
		s.Detail = buf.String()
	}
	return s
}

func importSymbol(spec *ast.ImportSpec, fset *token.FileSet) Symbol {
	s := Symbol{
		Name:   spec.Path.Value,
		Kind:   PackageSymbol,
		Detail: "import " + spec.Path.Value,
	}
	span, err := nodeSpan(spec, fset)
	if err == nil {
		s.Span = span
	}
	return s
}

func typeSymbol(spec *ast.TypeSpec, fset *token.FileSet) Symbol {
	s := Symbol{
		Name: spec.Name.String(),
		Kind: StructSymbol,
	}
	span, err := nodeSpan(spec, fset)
	if err == nil {
		s.Span = span
	}
	return s
}

func varSymbol(decl *ast.GenDecl, name *ast.Ident, fset *token.FileSet) Symbol {
	s := Symbol{
		Name: name.Name,
		Kind: VariableSymbol,
	}

	if decl.Tok == token.CONST {
		s.Kind = ConstantSymbol
	}

	span, err := nodeSpan(name, fset)
	if err == nil {
		s.Span = span
	}

	return s
}

func nodeSpan(n ast.Node, fset *token.FileSet) (span.Span, error) {
	r := span.NewRange(fset, n.Pos(), n.End())
	return r.Span()
}
