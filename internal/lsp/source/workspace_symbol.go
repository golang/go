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

	"golang.org/x/tools/internal/event"
	"golang.org/x/tools/internal/lsp/fuzzy"
	"golang.org/x/tools/internal/lsp/protocol"
)

const maxSymbols = 100

// WorkspaceSymbols matches symbols across views using the given query,
// according to the SymbolMatcher matcher.
//
// The workspace symbol method is defined in the spec as follows:
//
//  > The workspace symbol request is sent from the client to the server to
//  > list project-wide symbols matching the query string.
//
// It is unclear what "project-wide" means here, but given the parameters of
// workspace/symbol do not include any workspace identifier, then it has to be
// assumed that "project-wide" means "across all workspaces".  Hence why
// WorkspaceSymbols receives the views []View.
//
// However, it then becomes unclear what it would mean to call WorkspaceSymbols
// with a different configured SymbolMatcher per View. Therefore we assume that
// Session level configuration will define the SymbolMatcher to be used for the
// WorkspaceSymbols method.
func WorkspaceSymbols(ctx context.Context, matcherType SymbolMatcher, style SymbolStyle, views []View, query string) ([]protocol.SymbolInformation, error) {
	ctx, done := event.Start(ctx, "source.WorkspaceSymbols")
	defer done()
	if query == "" {
		return nil, nil
	}

	queryMatcher := makeQueryMatcher(matcherType, query)
	seen := make(map[string]struct{})
	var symbols []protocol.SymbolInformation
outer:
	for _, view := range views {
		knownPkgs, err := view.Snapshot().KnownPackages(ctx)
		if err != nil {
			return nil, err
		}
		for _, ph := range knownPkgs {
			pkg, err := ph.Check(ctx)
			symbolMatcher := makePackageSymbolMatcher(style, pkg, queryMatcher)
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
				for _, si := range findSymbol(file.Decls, pkg.GetTypesInfo(), symbolMatcher) {
					mrng, err := posToMappedRange(view, pkg, si.node.Pos(), si.node.End())
					if err != nil {
						event.Error(ctx, "Error getting mapped range for node", err)
						continue
					}
					rng, err := mrng.Range()
					if err != nil {
						event.Error(ctx, "Error getting range from mapped range", err)
						continue
					}
					symbols = append(symbols, protocol.SymbolInformation{
						Name: si.name,
						Kind: si.kind,
						Location: protocol.Location{
							URI:   protocol.URIFromSpanURI(mrng.URI()),
							Range: rng,
						},
						ContainerName: pkg.PkgPath(),
					})
					if len(symbols) > maxSymbols {
						break outer
					}
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

func makeQueryMatcher(m SymbolMatcher, query string) matcherFunc {
	switch m {
	case SymbolFuzzy:
		fm := fuzzy.NewMatcher(query)
		return func(s string) bool {
			return fm.Score(s) > 0
		}
	case SymbolCaseSensitive:
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

// packageSymbolMatcher matches (possibly partially) qualified symbols within a
// package scope.
//
// The given symbolizer controls how symbol names are extracted from the
// package scope.
type packageSymbolMatcher struct {
	queryMatcher matcherFunc
	pkg          Package
	symbolize    symbolizer
}

// symbolMatch returns the package symbol for name that matches the underlying
// query, or the empty string if no match is found.
func (s packageSymbolMatcher) symbolMatch(name string) string {
	return s.symbolize(name, s.pkg, s.queryMatcher)
}

func makePackageSymbolMatcher(style SymbolStyle, pkg Package, matcher matcherFunc) func(string) string {
	var s symbolizer
	switch style {
	case DynamicSymbols:
		s = dynamicSymbolMatch
	case FullyQualifiedSymbols:
		s = fullyQualifiedSymbolMatch
	default:
		s = packageSymbolMatch
	}
	return packageSymbolMatcher{queryMatcher: matcher, pkg: pkg, symbolize: s}.symbolMatch
}

// A symbolizer returns a qualified symbol match for the unqualified name
// within pkg, if one exists, or the empty string if no match is found.
type symbolizer func(name string, pkg Package, m matcherFunc) string

func fullyQualifiedSymbolMatch(name string, pkg Package, matcher matcherFunc) string {
	// TODO: this should probably include pkg.Name() as well.
	fullyQualified := pkg.PkgPath() + "." + name
	if matcher(fullyQualified) {
		return fullyQualified
	}
	return ""
}

func dynamicSymbolMatch(name string, pkg Package, matcher matcherFunc) string {
	pkgQualified := pkg.Name() + "." + name
	if match := shortestMatch(pkgQualified, matcher); match != "" {
		return match
	}
	fullyQualified := pkg.PkgPath() + "." + name
	if match := shortestMatch(fullyQualified, matcher); match != "" {
		return match
	}
	return ""
}

func packageSymbolMatch(name string, pkg Package, matcher matcherFunc) string {
	qualified := pkg.Name() + "." + name
	if matcher(qualified) {
		return qualified
	}
	return ""
}

func shortestMatch(fullPath string, matcher func(string) bool) string {
	pathParts := strings.Split(fullPath, "/")
	dottedParts := strings.Split(pathParts[len(pathParts)-1], ".")
	// First match the smallest package identifier.
	if m := matchRight(dottedParts, ".", matcher); m != "" {
		return m
	}
	// Then match the shortest subpath.
	return matchRight(pathParts, "/", matcher)
}

func matchRight(parts []string, sep string, matcher func(string) bool) string {
	for i := 0; i < len(parts); i++ {
		path := strings.Join(parts[len(parts)-1-i:], sep)
		if matcher(path) {
			return path
		}
	}
	return ""
}

func findSymbol(decls []ast.Decl, info *types.Info, symbolMatch func(string) string) []symbolInformation {
	var result []symbolInformation
	for _, decl := range decls {
		switch decl := decl.(type) {
		case *ast.FuncDecl:
			fn := decl.Name.Name
			kind := protocol.Function
			if decl.Recv != nil {
				kind = protocol.Method
				switch typ := decl.Recv.List[0].Type.(type) {
				case *ast.StarExpr:
					fn = typ.X.(*ast.Ident).Name + "." + fn
				case *ast.Ident:
					fn = typ.Name + "." + fn
				}
			}
			if m := symbolMatch(fn); m != "" {
				result = append(result, symbolInformation{
					name: m,
					kind: kind,
					node: decl.Name,
				})
			}
		case *ast.GenDecl:
			for _, spec := range decl.Specs {
				switch spec := spec.(type) {
				case *ast.TypeSpec:
					target := spec.Name.Name
					if m := symbolMatch(target); m != "" {
						result = append(result, symbolInformation{
							name: m,
							kind: typeToKind(info.TypeOf(spec.Type)),
							node: spec.Name,
						})
					}
					switch st := spec.Type.(type) {
					case *ast.StructType:
						for _, field := range st.Fields.List {
							result = append(result, findFieldSymbol(field, protocol.Field, symbolMatch, target)...)
						}
					case *ast.InterfaceType:
						for _, field := range st.Methods.List {
							kind := protocol.Method
							if len(field.Names) == 0 {
								kind = protocol.Interface
							}
							result = append(result, findFieldSymbol(field, kind, symbolMatch, target)...)
						}
					}
				case *ast.ValueSpec:
					for _, name := range spec.Names {
						if m := symbolMatch(name.Name); m != "" {
							kind := protocol.Variable
							if decl.Tok == token.CONST {
								kind = protocol.Constant
							}
							result = append(result, symbolInformation{
								name: m,
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

func findFieldSymbol(field *ast.Field, kind protocol.SymbolKind, symbolMatch func(string) string, prefix string) []symbolInformation {
	var result []symbolInformation

	if len(field.Names) == 0 {
		name := types.ExprString(field.Type)
		target := prefix + "." + name
		if m := symbolMatch(target); m != "" {
			result = append(result, symbolInformation{
				name: m,
				kind: kind,
				node: field,
			})
		}
		return result
	}

	for _, name := range field.Names {
		target := prefix + "." + name.Name
		if m := symbolMatch(target); m != "" {
			result = append(result, symbolInformation{
				name: m,
				kind: kind,
				node: name,
			})
		}
	}

	return result
}
