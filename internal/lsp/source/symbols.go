// Copyright 2019 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package source

import (
	"context"
	"fmt"
	"go/ast"
	"go/types"

	"golang.org/x/tools/internal/lsp/protocol"
	"golang.org/x/tools/internal/telemetry/trace"
)

func DocumentSymbols(ctx context.Context, view View, f GoFile) ([]protocol.DocumentSymbol, error) {
	ctx, done := trace.StartSpan(ctx, "source.DocumentSymbols")
	defer done()

	cphs, err := f.CheckPackageHandles(ctx)
	if err != nil {
		return nil, err
	}
	cph := NarrowestCheckPackageHandle(cphs)
	pkg, err := cph.Check(ctx)
	if err != nil {
		return nil, err
	}
	ph, err := pkg.File(f.URI())
	if err != nil {
		return nil, err
	}
	file, m, _, err := ph.Cached(ctx)
	if err != nil {
		return nil, err
	}

	info := pkg.GetTypesInfo()
	q := qualifier(file, pkg.GetTypes(), info)

	methodsToReceiver := make(map[types.Type][]protocol.DocumentSymbol)
	symbolsToReceiver := make(map[types.Type]int)
	var symbols []protocol.DocumentSymbol
	for _, decl := range file.Decls {
		switch decl := decl.(type) {
		case *ast.FuncDecl:
			if obj := info.ObjectOf(decl.Name); obj != nil {
				if fs := funcSymbol(ctx, view, m, decl, obj, q); fs.Kind == protocol.Method {
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
						ts := typeSymbol(ctx, view, m, info, spec, obj, q)
						symbols = append(symbols, ts)
						symbolsToReceiver[obj.Type()] = len(symbols) - 1
					}
				case *ast.ValueSpec:
					for _, name := range spec.Names {
						if obj := info.ObjectOf(name); obj != nil {
							symbols = append(symbols, varSymbol(ctx, view, m, decl, name, obj, q))
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
	return symbols, nil
}

func funcSymbol(ctx context.Context, view View, m *protocol.ColumnMapper, decl *ast.FuncDecl, obj types.Object, q types.Qualifier) protocol.DocumentSymbol {
	s := protocol.DocumentSymbol{
		Name: obj.Name(),
		Kind: protocol.Function,
	}
	if span, err := nodeToProtocolRange(ctx, view, m, decl); err == nil {
		s.Range = span
	}
	if span, err := nodeToProtocolRange(ctx, view, m, decl.Name); err == nil {
		s.SelectionRange = span
	}
	sig, _ := obj.Type().(*types.Signature)
	if sig != nil {
		if sig.Recv() != nil {
			s.Kind = protocol.Method
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

func setKind(s *protocol.DocumentSymbol, typ types.Type, q types.Qualifier) {
	switch typ := typ.Underlying().(type) {
	case *types.Interface:
		s.Kind = protocol.Interface
	case *types.Struct:
		s.Kind = protocol.Struct
	case *types.Signature:
		s.Kind = protocol.Function
		if typ.Recv() != nil {
			s.Kind = protocol.Method
		}
	case *types.Named:
		setKind(s, typ.Underlying(), q)
	case *types.Basic:
		i := typ.Info()
		switch {
		case i&types.IsNumeric != 0:
			s.Kind = protocol.Number
		case i&types.IsBoolean != 0:
			s.Kind = protocol.Boolean
		case i&types.IsString != 0:
			s.Kind = protocol.String
		}
	default:
		s.Kind = protocol.Variable
	}
}

func typeSymbol(ctx context.Context, view View, m *protocol.ColumnMapper, info *types.Info, spec *ast.TypeSpec, obj types.Object, q types.Qualifier) protocol.DocumentSymbol {
	s := protocol.DocumentSymbol{
		Name: obj.Name(),
	}
	s.Detail, _ = formatType(obj.Type(), q)
	setKind(&s, obj.Type(), q)

	if span, err := nodeToProtocolRange(ctx, view, m, spec); err == nil {
		s.Range = span
	}
	if span, err := nodeToProtocolRange(ctx, view, m, spec.Name); err == nil {
		s.SelectionRange = span
	}
	t, objIsStruct := obj.Type().Underlying().(*types.Struct)
	st, specIsStruct := spec.Type.(*ast.StructType)
	if objIsStruct && specIsStruct {
		for i := 0; i < t.NumFields(); i++ {
			f := t.Field(i)
			child := protocol.DocumentSymbol{
				Name: f.Name(),
				Kind: protocol.Field,
			}
			child.Detail, _ = formatType(f.Type(), q)

			spanNode, selectionNode := nodesForStructField(i, st)
			if span, err := nodeToProtocolRange(ctx, view, m, spanNode); err == nil {
				child.Range = span
			}
			if span, err := nodeToProtocolRange(ctx, view, m, selectionNode); err == nil {
				child.SelectionRange = span
			}
			s.Children = append(s.Children, child)
		}
	}

	ti, objIsInterface := obj.Type().Underlying().(*types.Interface)
	ai, specIsInterface := spec.Type.(*ast.InterfaceType)
	if objIsInterface && specIsInterface {
		for i := 0; i < ti.NumExplicitMethods(); i++ {
			method := ti.ExplicitMethod(i)
			child := protocol.DocumentSymbol{
				Name: method.Name(),
				Kind: protocol.Method,
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
			if span, err := nodeToProtocolRange(ctx, view, m, spanNode); err == nil {
				child.Range = span
			}
			if span, err := nodeToProtocolRange(ctx, view, m, selectionNode); err == nil {
				child.SelectionRange = span
			}
			s.Children = append(s.Children, child)
		}

		for i := 0; i < ti.NumEmbeddeds(); i++ {
			embedded := ti.EmbeddedType(i)
			nt, isNamed := embedded.(*types.Named)
			if !isNamed {
				continue
			}

			child := protocol.DocumentSymbol{
				Name: types.TypeString(embedded, q),
			}
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
			if rng, err := nodeToProtocolRange(ctx, view, m, spanNode); err == nil {
				child.Range = rng
			}
			if span, err := nodeToProtocolRange(ctx, view, m, selectionNode); err == nil {
				child.SelectionRange = span
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

func varSymbol(ctx context.Context, view View, m *protocol.ColumnMapper, decl ast.Node, name *ast.Ident, obj types.Object, q types.Qualifier) protocol.DocumentSymbol {
	s := protocol.DocumentSymbol{
		Name: obj.Name(),
		Kind: protocol.Variable,
	}
	if _, ok := obj.(*types.Const); ok {
		s.Kind = protocol.Constant
	}
	if rng, err := nodeToProtocolRange(ctx, view, m, decl); err == nil {
		s.Range = rng
	}
	if span, err := nodeToProtocolRange(ctx, view, m, name); err == nil {
		s.SelectionRange = span
	}
	s.Detail = types.TypeString(obj.Type(), q)
	return s
}
