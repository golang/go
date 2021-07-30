// Copyright 2021 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package cache

import (
	"context"
	"go/ast"
	"go/token"
	"go/types"
	"strings"

	"golang.org/x/tools/internal/lsp/protocol"
	"golang.org/x/tools/internal/lsp/source"
	"golang.org/x/tools/internal/memoize"
	"golang.org/x/tools/internal/span"
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

// symbolize extracts symbols from a file. It does not parse the file through the cache.
func symbolize(ctx context.Context, snapshot *snapshot, fh source.FileHandle) ([]source.Symbol, error) {
	var w symbolWalker
	fset := token.NewFileSet() // don't use snapshot.FileSet, as that would needlessly leak memory.
	data := parseGo(ctx, fset, fh, source.ParseFull)
	if data.parsed != nil && data.parsed.File != nil {
		w.curFile = data.parsed
		w.curURI = protocol.URIFromSpanURI(data.parsed.URI)
		w.fileDecls(data.parsed.File.Decls)
	}
	return w.symbols, w.firstError
}

type symbolWalker struct {
	curFile    *source.ParsedGoFile
	pkgName    string
	curURI     protocol.DocumentURI
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

	rng, err := fileRange(w.curFile, node.Pos(), node.End())
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

func fileRange(pgf *source.ParsedGoFile, start, end token.Pos) (protocol.Range, error) {
	s, err := span.FileSpan(pgf.Tok, pgf.Mapper.Converter, start, end)
	if err != nil {
		return protocol.Range{}, nil
	}
	return pgf.Mapper.Range(s)
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
