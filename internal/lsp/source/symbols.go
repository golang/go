// Copyright 2019 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package source

import (
	"context"
	"fmt"
	"go/ast"
	"go/token"
	"go/types"

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
	InterfaceSymbol
)

type Symbol struct {
	Name          string
	Detail        string
	Span          span.Span
	SelectionSpan span.Span
	Kind          SymbolKind
	Children      []Symbol
}

func DocumentSymbols(ctx context.Context, f File) []Symbol {
	fset := f.GetFileSet(ctx)
	file := f.GetAST(ctx)
	pkg := f.GetPackage(ctx)
	info := pkg.GetTypesInfo()
	q := qualifier(file, pkg.GetTypes(), info)

	var symbols []Symbol
	for _, decl := range file.Decls {
		switch decl := decl.(type) {
		case *ast.FuncDecl:
			if obj := info.ObjectOf(decl.Name); obj != nil {
				symbols = append(symbols, funcSymbol(decl, obj, fset, q))
			}
		case *ast.GenDecl:
			for _, spec := range decl.Specs {
				switch spec := spec.(type) {
				case *ast.TypeSpec:
					if obj := info.ObjectOf(spec.Name); obj != nil {
						symbols = append(symbols, typeSymbol(spec, obj, fset, q))
					}
				case *ast.ValueSpec:
					for _, name := range spec.Names {
						if obj := info.ObjectOf(name); obj != nil {
							symbols = append(symbols, varSymbol(decl, name, obj, fset, q))
						}
					}
				}
			}
		}
	}
	return symbols
}

func funcSymbol(decl *ast.FuncDecl, obj types.Object, fset *token.FileSet, q types.Qualifier) Symbol {
	s := Symbol{
		Name: obj.Name(),
		Kind: FunctionSymbol,
	}
	if span, err := nodeSpan(decl, fset); err == nil {
		s.Span = span
	}
	if span, err := nodeSpan(decl.Name, fset); err == nil {
		s.SelectionSpan = span
	}
	sig, _ := obj.Type().(*types.Signature)
	if sig != nil {
		if sig.Recv() != nil {
			s.Kind = MethodSymbol
		}
		s.Detail += "("
		for i := 0; i < sig.Params().Len(); i++ {
			if i > 0 {
				s.Detail += ", "
			}
			param := sig.Params().At(i)
			label := types.TypeString(param.Type(), q)
			if param.Name() != "" {
				label = fmt.Sprintf("%s %s", param.Name(), label)
			}
			s.Detail += label
		}
		s.Detail += ")"
	}
	return s
}

func typeSymbol(spec *ast.TypeSpec, obj types.Object, fset *token.FileSet, q types.Qualifier) Symbol {
	s := Symbol{
		Name: obj.Name(),
		Kind: StructSymbol,
	}
	if types.IsInterface(obj.Type()) {
		s.Kind = InterfaceSymbol
	}
	if span, err := nodeSpan(spec, fset); err == nil {
		s.Span = span
	}
	if span, err := nodeSpan(spec.Name, fset); err == nil {
		s.SelectionSpan = span
	}
	s.Detail, _ = formatType(obj.Type(), q)
	return s
}

func varSymbol(decl ast.Node, name *ast.Ident, obj types.Object, fset *token.FileSet, q types.Qualifier) Symbol {
	s := Symbol{
		Name: obj.Name(),
		Kind: VariableSymbol,
	}
	if _, ok := obj.(*types.Const); ok {
		s.Kind = ConstantSymbol
	}
	if span, err := nodeSpan(decl, fset); err == nil {
		s.Span = span
	}
	if span, err := nodeSpan(name, fset); err == nil {
		s.SelectionSpan = span
	}
	s.Detail = types.TypeString(obj.Type(), q)
	return s
}

func nodeSpan(n ast.Node, fset *token.FileSet) (span.Span, error) {
	r := span.NewRange(fset, n.Pos(), n.End())
	return r.Span()
}
