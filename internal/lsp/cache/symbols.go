// Copyright 2021 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package cache

import (
	"context"
	"go/ast"
	"go/parser"
	"go/token"
	"go/types"
	"strings"

	"golang.org/x/tools/internal/lsp/lsppos"
	"golang.org/x/tools/internal/lsp/protocol"
	"golang.org/x/tools/internal/lsp/source"
	"golang.org/x/tools/internal/memoize"
)

type symbolHandle struct {
	handle *memoize.Handle

	fh source.FileHandle

	// key is the hashed key for the package.
	key symbolHandleKey
}

// symbolData contains the data produced by extracting symbols from a file.
type symbolData struct {
	symbols []source.Symbol
	err     error
}

type symbolHandleKey string

func (s *snapshot) buildSymbolHandle(ctx context.Context, fh source.FileHandle) *symbolHandle {
	if h := s.getSymbolHandle(fh.URI()); h != nil {
		return h
	}
	key := symbolHandleKey(fh.FileIdentity().Hash)
	h := s.generation.Bind(key, func(ctx context.Context, arg memoize.Arg) interface{} {
		snapshot := arg.(*snapshot)
		data := &symbolData{}
		data.symbols, data.err = symbolize(ctx, snapshot, fh)
		return data
	}, nil)

	sh := &symbolHandle{
		handle: h,
		fh:     fh,
		key:    key,
	}
	return s.addSymbolHandle(sh)
}

// symbolize extracts symbols from a file. It uses a parsed file already
// present in the cache but otherwise does not populate the cache.
func symbolize(ctx context.Context, snapshot *snapshot, fh source.FileHandle) ([]source.Symbol, error) {
	src, err := fh.Read()
	if err != nil {
		return nil, err
	}

	var (
		file     *ast.File
		fileDesc *token.File
	)

	// If the file has already been fully parsed through the cache, we can just
	// use the result.
	key := parseKey{file: fh.FileIdentity(), mode: source.ParseFull}
	if pgh := snapshot.getGoFile(key); pgh != nil {
		cached := pgh.handle.Cached(snapshot.generation)
		if cached != nil {
			cached := cached.(*parseGoData)
			if cached.parsed != nil {
				file = cached.parsed.File
				fileDesc = cached.parsed.Tok
			}
		}
	}

	// Otherwise, we parse the file ourselves. Notably we don't use parseGo here,
	// so that we can avoid parsing comments and can skip object resolution,
	// which has a meaningful impact on performance. Neither comments nor objects
	// are necessary for symbol construction.
	if file == nil {
		fset := token.NewFileSet()
		file, err = parser.ParseFile(fset, fh.URI().Filename(), src, skipObjectResolution)
		if file == nil {
			return nil, err
		}
		fileDesc = fset.File(file.Package)
	}

	w := &symbolWalker{
		mapper: lsppos.NewTokenMapper(src, fileDesc),
	}

	w.fileDecls(file.Decls)

	return w.symbols, w.firstError
}

type symbolWalker struct {
	mapper *lsppos.TokenMapper // for computing positions

	symbols    []source.Symbol
	firstError error
}

func (w *symbolWalker) atNode(node ast.Node, name string, kind protocol.SymbolKind, path ...*ast.Ident) {
	var b strings.Builder
	for _, ident := range path {
		if ident != nil {
			b.WriteString(ident.Name)
			b.WriteString(".")
		}
	}
	b.WriteString(name)

	rng, err := w.mapper.Range(node.Pos(), node.End())
	if err != nil {
		w.error(err)
		return
	}
	sym := source.Symbol{
		Name:  b.String(),
		Kind:  kind,
		Range: rng,
	}
	w.symbols = append(w.symbols, sym)
}

func (w *symbolWalker) error(err error) {
	if err != nil && w.firstError == nil {
		w.firstError = err
	}
}

func (w *symbolWalker) fileDecls(decls []ast.Decl) {
	for _, decl := range decls {
		switch decl := decl.(type) {
		case *ast.FuncDecl:
			kind := protocol.Function
			var recv *ast.Ident
			if decl.Recv.NumFields() > 0 {
				kind = protocol.Method
				recv = unpackRecv(decl.Recv.List[0].Type)
			}
			w.atNode(decl.Name, decl.Name.Name, kind, recv)
		case *ast.GenDecl:
			for _, spec := range decl.Specs {
				switch spec := spec.(type) {
				case *ast.TypeSpec:
					kind := guessKind(spec)
					w.atNode(spec.Name, spec.Name.Name, kind)
					w.walkType(spec.Type, spec.Name)
				case *ast.ValueSpec:
					for _, name := range spec.Names {
						kind := protocol.Variable
						if decl.Tok == token.CONST {
							kind = protocol.Constant
						}
						w.atNode(name, name.Name, kind)
					}
				}
			}
		}
	}
}

func guessKind(spec *ast.TypeSpec) protocol.SymbolKind {
	switch spec.Type.(type) {
	case *ast.InterfaceType:
		return protocol.Interface
	case *ast.StructType:
		return protocol.Struct
	case *ast.FuncType:
		return protocol.Function
	}
	return protocol.Class
}

func unpackRecv(rtyp ast.Expr) *ast.Ident {
	// Extract the receiver identifier. Lifted from go/types/resolver.go
L:
	for {
		switch t := rtyp.(type) {
		case *ast.ParenExpr:
			rtyp = t.X
		case *ast.StarExpr:
			rtyp = t.X
		default:
			break L
		}
	}
	if name, _ := rtyp.(*ast.Ident); name != nil {
		return name
	}
	return nil
}

// walkType processes symbols related to a type expression. path is path of
// nested type identifiers to the type expression.
func (w *symbolWalker) walkType(typ ast.Expr, path ...*ast.Ident) {
	switch st := typ.(type) {
	case *ast.StructType:
		for _, field := range st.Fields.List {
			w.walkField(field, protocol.Field, protocol.Field, path...)
		}
	case *ast.InterfaceType:
		for _, field := range st.Methods.List {
			w.walkField(field, protocol.Interface, protocol.Method, path...)
		}
	}
}

// walkField processes symbols related to the struct field or interface method.
//
// unnamedKind and namedKind are the symbol kinds if the field is resp. unnamed
// or named. path is the path of nested identifiers containing the field.
func (w *symbolWalker) walkField(field *ast.Field, unnamedKind, namedKind protocol.SymbolKind, path ...*ast.Ident) {
	if len(field.Names) == 0 {
		switch typ := field.Type.(type) {
		case *ast.SelectorExpr:
			// embedded qualified type
			w.atNode(field, typ.Sel.Name, unnamedKind, path...)
		default:
			w.atNode(field, types.ExprString(field.Type), unnamedKind, path...)
		}
	}
	for _, name := range field.Names {
		w.atNode(name, name.Name, namedKind, path...)
		w.walkType(field.Type, append(path, name)...)
	}
}
