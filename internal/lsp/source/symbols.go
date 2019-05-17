// Copyright 2019 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package source

import (
	"context"
	"errors"
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
	NumberSymbol
	StringSymbol
	BooleanSymbol
	FieldSymbol
)

type Symbol struct {
	Name          string
	Detail        string
	Span          span.Span
	SelectionSpan span.Span
	Kind          SymbolKind
	Children      []Symbol
}

func DocumentSymbols(ctx context.Context, f GoFile) []Symbol {
	fset := f.FileSet()
	file := f.GetAST(ctx)
	if file == nil {
		return nil
	}
	pkg := f.GetPackage(ctx)
	if pkg == nil || pkg.IsIllTyped() {
		return nil
	}
	info := pkg.GetTypesInfo()
	q := qualifier(file, pkg.GetTypes(), info)

	methodsToReceiver := make(map[types.Type][]Symbol)
	symbolsToReceiver := make(map[types.Type]int)
	var symbols []Symbol
	for _, decl := range file.Decls {
		switch decl := decl.(type) {
		case *ast.FuncDecl:
			if obj := info.ObjectOf(decl.Name); obj != nil {
				if fs := funcSymbol(decl, obj, fset, q); fs.Kind == MethodSymbol {
					// Store methods separately, as we want them to appear as children
					// of the corresponding type (which we may not have seen yet).
					rtype := obj.Type().(*types.Signature).Recv().Type()
					methodsToReceiver[rtype] = append(methodsToReceiver[rtype], fs)
				} else {
					symbols = append(symbols, fs)
				}
			}
		case *ast.GenDecl:
			for _, spec := range decl.Specs {
				switch spec := spec.(type) {
				case *ast.TypeSpec:
					if obj := info.ObjectOf(spec.Name); obj != nil {
						ts := typeSymbol(info, spec, obj, fset, q)
						symbols = append(symbols, ts)
						symbolsToReceiver[obj.Type()] = len(symbols) - 1
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

	// Attempt to associate methods to the corresponding type symbol.
	for typ, methods := range methodsToReceiver {
		if ptr, ok := typ.(*types.Pointer); ok {
			typ = ptr.Elem()
		}

		if i, ok := symbolsToReceiver[typ]; ok {
			symbols[i].Children = append(symbols[i].Children, methods...)
		} else {
			// The type definition for the receiver of these methods was not in the document.
			symbols = append(symbols, methods...)
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

func setKind(s *Symbol, typ types.Type, q types.Qualifier) {
	switch typ := typ.Underlying().(type) {
	case *types.Interface:
		s.Kind = InterfaceSymbol
	case *types.Struct:
		s.Kind = StructSymbol
	case *types.Signature:
		s.Kind = FunctionSymbol
		if typ.Recv() != nil {
			s.Kind = MethodSymbol
		}
	case *types.Named:
		setKind(s, typ.Underlying(), q)
	case *types.Basic:
		i := typ.Info()
		switch {
		case i&types.IsNumeric != 0:
			s.Kind = NumberSymbol
		case i&types.IsBoolean != 0:
			s.Kind = BooleanSymbol
		case i&types.IsString != 0:
			s.Kind = StringSymbol
		}
	default:
		s.Kind = VariableSymbol
	}
}

func typeSymbol(info *types.Info, spec *ast.TypeSpec, obj types.Object, fset *token.FileSet, q types.Qualifier) Symbol {
	s := Symbol{Name: obj.Name()}
	s.Detail, _ = formatType(obj.Type(), q)
	setKind(&s, obj.Type(), q)

	if span, err := nodeSpan(spec, fset); err == nil {
		s.Span = span
	}
	if span, err := nodeSpan(spec.Name, fset); err == nil {
		s.SelectionSpan = span
	}

	t, objIsStruct := obj.Type().Underlying().(*types.Struct)
	st, specIsStruct := spec.Type.(*ast.StructType)
	if objIsStruct && specIsStruct {
		for i := 0; i < t.NumFields(); i++ {
			f := t.Field(i)
			child := Symbol{Name: f.Name(), Kind: FieldSymbol}
			child.Detail, _ = formatType(f.Type(), q)

			spanNode, selectionNode := nodesForStructField(i, st)
			if span, err := nodeSpan(spanNode, fset); err == nil {
				child.Span = span
			}
			if span, err := nodeSpan(selectionNode, fset); err == nil {
				child.SelectionSpan = span
			}

			s.Children = append(s.Children, child)
		}
	}

	ti, objIsInterface := obj.Type().Underlying().(*types.Interface)
	ai, specIsInterface := spec.Type.(*ast.InterfaceType)
	if objIsInterface && specIsInterface {
		for i := 0; i < ti.NumExplicitMethods(); i++ {
			method := ti.ExplicitMethod(i)
			child := Symbol{
				Name: method.Name(),
				Kind: MethodSymbol,
			}

			var spanNode, selectionNode ast.Node
		Methods:
			for _, f := range ai.Methods.List {
				for _, id := range f.Names {
					if id.Name == method.Name() {
						spanNode, selectionNode = f, id
						break Methods
					}
				}
			}
			if span, err := nodeSpan(spanNode, fset); err == nil {
				child.Span = span
			}
			if span, err := nodeSpan(selectionNode, fset); err == nil {
				child.SelectionSpan = span
			}
			s.Children = append(s.Children, child)
		}

		for i := 0; i < ti.NumEmbeddeds(); i++ {
			embedded := ti.EmbeddedType(i)
			nt, isNamed := embedded.(*types.Named)
			if !isNamed {
				continue
			}

			child := Symbol{Name: types.TypeString(embedded, q)}
			setKind(&child, embedded, q)
			var spanNode, selectionNode ast.Node
		Embeddeds:
			for _, f := range ai.Methods.List {
				if len(f.Names) > 0 {
					continue
				}

				if t := info.TypeOf(f.Type); types.Identical(nt, t) {
					spanNode, selectionNode = f, f.Type
					break Embeddeds
				}
			}

			if span, err := nodeSpan(spanNode, fset); err == nil {
				child.Span = span
			}
			if span, err := nodeSpan(selectionNode, fset); err == nil {
				child.SelectionSpan = span
			}
			s.Children = append(s.Children, child)
		}
	}
	return s
}

func nodesForStructField(i int, st *ast.StructType) (span, selection ast.Node) {
	j := 0
	for _, field := range st.Fields.List {
		if len(field.Names) == 0 {
			if i == j {
				return field, field.Type
			}
			j++
			continue
		}
		for _, name := range field.Names {
			if i == j {
				return field, name
			}
			j++
		}
	}
	return nil, nil
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
	if n == nil {
		return span.Span{}, errors.New("no span for nil node")
	}
	r := span.NewRange(fset, n.Pos(), n.End())
	return r.Span()
}
