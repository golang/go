// Copyright 2019 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package source

import (
	"context"
	"fmt"
	"go/ast"
	"go/types"

	"golang.org/x/tools/internal/event"
	"golang.org/x/tools/internal/lsp/protocol"
)

func DocumentSymbols(ctx context.Context, snapshot Snapshot, fh FileHandle) ([]protocol.DocumentSymbol, error) {
	ctx, done := event.Start(ctx, "source.DocumentSymbols")
	defer done()

	pkg, pgh, err := getParsedFile(ctx, snapshot, fh, NarrowestPackageHandle)
	if err != nil {
		return nil, fmt.Errorf("getting file for DocumentSymbols: %w", err)
	}
	file, _, _, _, err := pgh.Cached()
	if err != nil {
		return nil, err
	}

	info := pkg.GetTypesInfo()
	q := qualifier(file, pkg.GetTypes(), info)

	symbolsToReceiver := make(map[types.Type]int)
	var symbols []protocol.DocumentSymbol
	for _, decl := range file.Decls {
		switch decl := decl.(type) {
		case *ast.FuncDecl:
			if obj := info.ObjectOf(decl.Name); obj != nil {
				fs, err := funcSymbol(snapshot.View(), pkg, decl, obj, q)
				if err != nil {
					return nil, err
				}
				// If function is a method, prepend the type of the method.
				if fs.Kind == protocol.Method {
					rtype := obj.Type().(*types.Signature).Recv().Type()
					fs.Name = fmt.Sprintf("(%s).%s", types.TypeString(rtype, q), fs.Name)
				}
				symbols = append(symbols, fs)
			}
		case *ast.GenDecl:
			for _, spec := range decl.Specs {
				switch spec := spec.(type) {
				case *ast.TypeSpec:
					if obj := info.ObjectOf(spec.Name); obj != nil {
						ts, err := typeSymbol(snapshot.View(), pkg, info, spec, obj, q)
						if err != nil {
							return nil, err
						}
						symbols = append(symbols, ts)
						symbolsToReceiver[obj.Type()] = len(symbols) - 1
					}
				case *ast.ValueSpec:
					for _, name := range spec.Names {
						if obj := info.ObjectOf(name); obj != nil {
							vs, err := varSymbol(snapshot.View(), pkg, decl, name, obj, q)
							if err != nil {
								return nil, err
							}
							symbols = append(symbols, vs)
						}
					}
				}
			}
		}
	}
	return symbols, nil
}

func funcSymbol(view View, pkg Package, decl *ast.FuncDecl, obj types.Object, q types.Qualifier) (protocol.DocumentSymbol, error) {
	s := protocol.DocumentSymbol{
		Name: obj.Name(),
		Kind: protocol.Function,
	}
	var err error
	s.Range, err = nodeToProtocolRange(view, pkg, decl)
	if err != nil {
		return protocol.DocumentSymbol{}, err
	}
	s.SelectionRange, err = nodeToProtocolRange(view, pkg, decl.Name)
	if err != nil {
		return protocol.DocumentSymbol{}, err
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
	return s, nil
}

func typeSymbol(view View, pkg Package, info *types.Info, spec *ast.TypeSpec, obj types.Object, qf types.Qualifier) (protocol.DocumentSymbol, error) {
	s := protocol.DocumentSymbol{
		Name: obj.Name(),
	}
	s.Detail, _ = formatType(obj.Type(), qf)
	s.Kind = typeToKind(obj.Type())

	var err error
	s.Range, err = nodeToProtocolRange(view, pkg, spec)
	if err != nil {
		return protocol.DocumentSymbol{}, err
	}
	s.SelectionRange, err = nodeToProtocolRange(view, pkg, spec.Name)
	if err != nil {
		return protocol.DocumentSymbol{}, err
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
			child.Detail, _ = formatType(f.Type(), qf)

			spanNode, selectionNode := nodesForStructField(i, st)
			if span, err := nodeToProtocolRange(view, pkg, spanNode); err == nil {
				child.Range = span
			}
			if span, err := nodeToProtocolRange(view, pkg, selectionNode); err == nil {
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
			child.Range, err = nodeToProtocolRange(view, pkg, spanNode)
			if err != nil {
				return protocol.DocumentSymbol{}, err
			}
			child.SelectionRange, err = nodeToProtocolRange(view, pkg, selectionNode)
			if err != nil {
				return protocol.DocumentSymbol{}, err
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
				Name: types.TypeString(embedded, qf),
			}
			child.Kind = typeToKind(embedded)
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
			child.Range, err = nodeToProtocolRange(view, pkg, spanNode)
			if err != nil {
				return protocol.DocumentSymbol{}, err
			}
			child.SelectionRange, err = nodeToProtocolRange(view, pkg, selectionNode)
			if err != nil {
				return protocol.DocumentSymbol{}, err
			}
			s.Children = append(s.Children, child)
		}
	}
	return s, nil
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

func varSymbol(view View, pkg Package, decl ast.Node, name *ast.Ident, obj types.Object, q types.Qualifier) (protocol.DocumentSymbol, error) {
	s := protocol.DocumentSymbol{
		Name: obj.Name(),
		Kind: protocol.Variable,
	}
	if _, ok := obj.(*types.Const); ok {
		s.Kind = protocol.Constant
	}
	var err error
	s.Range, err = nodeToProtocolRange(view, pkg, decl)
	if err != nil {
		return protocol.DocumentSymbol{}, err
	}
	s.SelectionRange, err = nodeToProtocolRange(view, pkg, name)
	if err != nil {
		return protocol.DocumentSymbol{}, err
	}
	s.Detail = types.TypeString(obj.Type(), q)
	return s, nil
}
