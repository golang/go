// Copyright 2020 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package source

import (
	"context"
	"go/ast"
	"go/token"
	"go/types"
	"strings"

	"golang.org/x/tools/internal/lsp/fuzzy"
	"golang.org/x/tools/internal/lsp/protocol"
	"golang.org/x/tools/internal/telemetry/log"
	"golang.org/x/tools/internal/telemetry/trace"
)

func WorkspaceSymbols(ctx context.Context, views []View, query string) ([]protocol.SymbolInformation, error) {
	ctx, done := trace.StartSpan(ctx, "source.WorkspaceSymbols")
	defer done()

	seen := make(map[string]struct{})
	var symbols []protocol.SymbolInformation
	for _, view := range views {
		knownPkgs, err := view.Snapshot().KnownPackages(ctx)
		if err != nil {
			return nil, err
		}
		matcher := makeMatcher(view.Options().Matcher, query)
		for _, ph := range knownPkgs {
			pkg, err := ph.Check(ctx)
			if err != nil {
				return nil, err
			}
			if _, ok := seen[pkg.PkgPath()]; ok {
				continue
			}
			seen[pkg.PkgPath()] = struct{}{}
			for _, fh := range pkg.CompiledGoFiles() {
				file, _, _, _, err := fh.Cached()
				if err != nil {
					return nil, err
				}
				for _, si := range findSymbol(file.Decls, pkg.GetTypesInfo(), matcher) {
					rng, err := nodeToProtocolRange(view, pkg, si.node)
					if err != nil {
						log.Error(ctx, "Error getting range for node", err)
						continue
					}
					symbols = append(symbols, protocol.SymbolInformation{
						Name: si.name,
						Kind: si.kind,
						Location: protocol.Location{
							URI:   protocol.URIFromSpanURI(fh.File().Identity().URI),
							Range: rng,
						},
					})
				}
			}
		}
	}
	return symbols, nil
}

type symbolInformation struct {
	name string
	kind protocol.SymbolKind
	node ast.Node
}

type matcherFunc func(string) bool

func makeMatcher(m Matcher, query string) matcherFunc {
	switch m {
	case Fuzzy:
		fm := fuzzy.NewMatcher(query)
		return func(s string) bool {
			return fm.Score(s) > 0
		}
	case CaseSensitive:
		return func(s string) bool {
			return strings.Contains(s, query)
		}
	default:
		q := strings.ToLower(query)
		return func(s string) bool {
			return strings.Contains(strings.ToLower(s), q)
		}
	}
}

func findSymbol(decls []ast.Decl, info *types.Info, matcher matcherFunc) []symbolInformation {
	var result []symbolInformation
	for _, decl := range decls {
		switch decl := decl.(type) {
		case *ast.FuncDecl:
			if matcher(decl.Name.Name) {
				kind := protocol.Function
				if decl.Recv != nil {
					kind = protocol.Method
				}
				result = append(result, symbolInformation{
					name: decl.Name.Name,
					kind: kind,
					node: decl.Name,
				})
			}
		case *ast.GenDecl:
			for _, spec := range decl.Specs {
				switch spec := spec.(type) {
				case *ast.TypeSpec:
					if matcher(spec.Name.Name) {
						result = append(result, symbolInformation{
							name: spec.Name.Name,
							kind: typeToKind(info.TypeOf(spec.Type)),
							node: spec.Name,
						})
					}
					switch st := spec.Type.(type) {
					case *ast.StructType:
						for _, field := range st.Fields.List {
							result = append(result, findFieldSymbol(field, protocol.Field, matcher)...)
						}
					case *ast.InterfaceType:
						for _, field := range st.Methods.List {
							kind := protocol.Method
							if len(field.Names) == 0 {
								kind = protocol.Interface
							}
							result = append(result, findFieldSymbol(field, kind, matcher)...)
						}
					}
				case *ast.ValueSpec:
					for _, name := range spec.Names {
						if matcher(name.Name) {
							kind := protocol.Variable
							if decl.Tok == token.CONST {
								kind = protocol.Constant
							}
							result = append(result, symbolInformation{
								name: name.Name,
								kind: kind,
								node: name,
							})
						}
					}
				}
			}
		}
	}
	return result
}

func typeToKind(typ types.Type) protocol.SymbolKind {
	switch typ := typ.Underlying().(type) {
	case *types.Interface:
		return protocol.Interface
	case *types.Struct:
		return protocol.Struct
	case *types.Signature:
		if typ.Recv() != nil {
			return protocol.Method
		}
		return protocol.Function
	case *types.Named:
		return typeToKind(typ.Underlying())
	case *types.Basic:
		i := typ.Info()
		switch {
		case i&types.IsNumeric != 0:
			return protocol.Number
		case i&types.IsBoolean != 0:
			return protocol.Boolean
		case i&types.IsString != 0:
			return protocol.String
		}
	}
	return protocol.Variable
}

func findFieldSymbol(field *ast.Field, kind protocol.SymbolKind, matcher matcherFunc) []symbolInformation {
	var result []symbolInformation

	if len(field.Names) == 0 {
		name := types.ExprString(field.Type)
		if matcher(name) {
			result = append(result, symbolInformation{
				name: name,
				kind: kind,
				node: field,
			})
		}
		return result
	}

	for _, name := range field.Names {
		if matcher(name.Name) {
			result = append(result, symbolInformation{
				name: name.Name,
				kind: kind,
				node: name,
			})
		}
	}

	return result
}
