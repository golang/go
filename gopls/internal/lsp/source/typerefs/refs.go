// Copyright 2023 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package typerefs

import (
	"fmt"
	"go/ast"
	"go/token"
	"sort"
	"strings"

	"golang.org/x/tools/gopls/internal/lsp/source"
	"golang.org/x/tools/internal/typeparams"
)

// declInfo holds information about a single declaration.
type declInfo struct {
	node    ast.Node        // "declaring node" for this decl, to be traversed
	tparams map[string]bool // names declared by type parameters within this declaration
	file    *ast.File       // file containing this decl, for local imports
}

// Refs analyzes local syntax of the provided ParsedGoFiles to extract
// references between types used in top-level declarations and other
// declarations.
//
// The provided pkgIndex is used for efficient representation of references,
// and must be used to unpack the resulting references.
//
// See the package documentation for more details as to what a ref does (and
// does not) represent.
func Refs(pgfs []*source.ParsedGoFile, id source.PackageID, imports map[source.ImportPath]*source.Metadata, pkgIndex *PackageIndex) map[string][]Ref {
	var (
		// decls collects declaration nodes that collectively define the type of
		// each name in the package scope.
		//
		//  - For valid code, there may be multiple declarations recorded when that
		//    name is a type that has methods.
		//  - For invalid code, there may also be multiple declarations recorded due
		//    to duplicate declarations.
		//
		// In either case, the algorithm is the same: we walk all declarations for
		// each name to collect referring identifiers.
		decls = make(map[string][]*declInfo)

		// localImports holds local import information, per file. The value is a
		// slice because multiple packages may be referenced by a given name in the
		// presence of type errors (or multiple dot imports, which are keyed by
		// ".").
		localImports = make(map[*ast.File]map[string][]source.PackageID)
	)

	// Scan top-level declarations once to collect local import names and
	// declInfo for each non-import declaration.
	for _, pgf := range pgfs {
		file := pgf.File
		fileImports := make(map[string][]source.PackageID)
		localImports[file] = fileImports

		for _, d := range file.Decls {
			switch d := d.(type) {
			case *ast.GenDecl:
				switch d.Tok {
				case token.IMPORT:
					// Record local import names for this file.
					for _, spec := range d.Specs {
						spec := spec.(*ast.ImportSpec)
						path := source.UnquoteImportPath(spec)
						if path == "" {
							continue
						}
						dep := imports[path]
						if dep == nil {
							// Note here that we don't try to "guess" the name of an import
							// based on e.g. its importPath. Doing so would only result in
							// edges that don't go anywhere.
							continue
						}
						name := string(dep.Name)
						if spec.Name != nil {
							if spec.Name.Name == "_" {
								continue
							}
							name = spec.Name.Name // possibly "."
						}
						fileImports[name] = append(fileImports[name], dep.ID)
					}

				case token.TYPE:
					for _, spec := range d.Specs {
						spec := spec.(*ast.TypeSpec)
						name := spec.Name.Name
						if name == "_" {
							continue
						}
						decls[name] = append(decls[name], &declInfo{
							node:    spec,
							tparams: tparamsMap(typeparams.ForTypeSpec(spec)),
							file:    file,
						})
					}

				case token.VAR, token.CONST:
					for _, spec := range d.Specs {
						spec := spec.(*ast.ValueSpec)
						for _, name := range spec.Names {
							if name.Name == "_" {
								continue
							}
							decls[name.Name] = append(decls[name.Name], &declInfo{node: spec, file: file})
						}
					}
				}

			case *ast.FuncDecl:
				if d.Name.Name == "_" {
					continue
				}
				// This check for NumFields() > 0 is consistent with go/types, which
				// reports an error but treats the declaration like a normal function
				// when Recv is non-nil but empty (as in func () f()).
				if d.Recv.NumFields() > 0 {
					// Method. Associate it with the receiver.
					_, id, tparams := unpackRecv(d.Recv.List[0].Type)
					methodInfo := &declInfo{
						node: d,
						file: file,
					}
					if len(tparams) > 0 {
						methodInfo.tparams = make(map[string]bool)
						for _, tparam := range tparams {
							if tparam.Name != "_" {
								methodInfo.tparams[tparam.Name] = true
							}
						}
					}
					decls[id.Name] = append(decls[id.Name], methodInfo)
				} else {
					// Non-method.
					decls[d.Name.Name] = append(decls[d.Name.Name], &declInfo{
						node:    d,
						tparams: tparamsMap(typeparams.ForFuncType(d.Type)),
						file:    file,
					})
				}
			}
		}
	}

	// mappedRefs maps each name in this package to the set
	// of (pkg, name) pairs it references.
	mappedRefs := make(map[string]map[source.PackageID]map[string]bool)
	for name, infos := range decls {
		// recordEdge records the (id, name)->(id2, name) edge.
		recordEdge := func(id2 source.PackageID, name2 string) {
			pkgRefs, ok := mappedRefs[name]
			if !ok {
				pkgRefs = make(map[source.PackageID]map[string]bool)
				mappedRefs[name] = pkgRefs
			}
			names, ok := pkgRefs[id2]
			if !ok {
				names = make(map[string]bool)
				pkgRefs[id2] = names
			}
			names[name2] = true
		}

		for _, info := range infos {
			fileImports := localImports[info.file]

			// Visit each reference to name or name.sel.
			visitDeclOrSpec(info.node, func(name, sel string) {
				if info.tparams[name] {
					return
				}

				// If name is declared in the package scope, record an edge whether or
				// not sel is empty. A field or method selector may affect the type of
				// the current decl via initializers:
				//
				//  package p
				//  var x = y.F
				//  var y = struct {F int}{}
				if _, ok := decls[name]; ok {
					recordEdge(id, name)
				} else if token.IsExported(name) {
					// Only record an edge to dot-imported packages if there was no edge
					// to a local name. This assumes that there are no duplicate declarations.
					for _, depID := range fileImports["."] {
						// Conservatively, assume that this name comes from every
						// dot-imported package.
						recordEdge(depID, name)
					}
				}

				// Record an edge to an import if it matches the name, even if that
				// name collides with a package level name. Unlike the case of dotted
				// imports, we know the package is invalid here, and choose to fail
				// conservatively.
				if sel != "" && token.IsExported(sel) {
					for _, depID := range fileImports[name] {
						recordEdge(depID, sel)
					}
				}
			})
		}
	}

	if trace {
		fmt.Printf("%s\n", id)
		var names []string
		for name := range mappedRefs {
			names = append(names, name)
		}
		sort.Strings(names)
		for _, name := range names {
			fmt.Printf("\t-> %s\n", name)
			for id2, pkgRefs := range mappedRefs[name] {
				var ns []string
				for n := range pkgRefs {
					ns = append(ns, n)
				}
				sort.Strings(ns)
				fmt.Printf("\t\t-> %s.{%s}\n", id2, strings.Join(ns, ", "))
			}
		}
	}

	edges := make(map[string][]Ref)
	for name, pkgRefs := range mappedRefs {
		for id, names := range pkgRefs {
			for name2 := range names {
				edges[name] = append(edges[name], Ref{pkgIndex.idx(id), name2})
			}
		}
	}
	return edges
}

// tparamsMap returns a set recording each name declared by the provided field
// list. It so happens that we only care about names declared by type parameter
// lists.
func tparamsMap(tparams *ast.FieldList) map[string]bool {
	if tparams == nil || len(tparams.List) == 0 {
		return nil
	}
	m := make(map[string]bool)
	for _, f := range tparams.List {
		for _, name := range f.Names {
			if name.Name != "_" {
				m[name.Name] = true
			}
		}
	}
	return m
}

// A refVisitor visits referring identifiers and dotted identifiers.
//
// For a referring identifier I, name="I" and sel="". For a dotted identifier
// q.I, name="q" and sel="I".
type refVisitor = func(name, sel string)

// visitDeclOrSpec visits referring idents or dotted idents that may affect
// the type of the declaration at the given node, which must be an ast.Decl or
// ast.Spec.
func visitDeclOrSpec(node ast.Node, f refVisitor) {
	// Declarations
	switch n := node.(type) {
	// ImportSpecs should not appear here, and will panic in the default case.

	case *ast.ValueSpec:
		// Skip Doc, Names, Comments, which do not affect the decl type.
		// Initializers only affect the type of a value spec if the type is unset.
		if n.Type != nil {
			visitExpr(n.Type, f)
		} else { // only need to walk expr list if type is nil
			visitExprList(n.Values, f)
		}

	case *ast.TypeSpec:
		// Skip Doc, Name, and Comment, which do not affect the decl type.
		if tparams := typeparams.ForTypeSpec(n); tparams != nil {
			visitFieldList(tparams, f)
		}
		visitExpr(n.Type, f)

	case *ast.BadDecl:
		// nothing to do

	// We should not reach here with a GenDecl, so panic below in the default case.

	case *ast.FuncDecl:
		// Skip Doc, Name, and Body, which do not affect the type.
		// Recv is handled by Refs: methods are associated with their type.
		visitExpr(n.Type, f)

	default:
		panic(fmt.Sprintf("unexpected node type %T", node))
	}
}

// visitExpr visits referring idents and dotted idents that may affect the
// type of expr.
//
// visitExpr can't reliably distinguish a dotted ident pkg.X from a
// selection expr.f or T.method.
func visitExpr(expr ast.Expr, f refVisitor) {
	switch n := expr.(type) {
	// These four cases account for about two thirds of all nodes,
	// so we place them first to shorten the common control paths.
	// (See go.dev/cl/480915.)
	case *ast.Ident:
		f(n.Name, "")

	case *ast.BasicLit:
		// nothing to do

	case *ast.SelectorExpr:
		if ident, ok := n.X.(*ast.Ident); ok {
			f(ident.Name, n.Sel.Name)
		} else {
			visitExpr(n.X, f)
			// Skip n.Sel as we don't care about which field or method is selected,
			// as we'll have recorded an edge to all declarations relevant to the
			// receiver type via visiting n.X above.
		}

	case *ast.CallExpr:
		visitExpr(n.Fun, f)
		visitExprList(n.Args, f) // args affect types for unsafe.Sizeof or builtins or generics

	// Expressions
	case *ast.Ellipsis:
		if n.Elt != nil {
			visitExpr(n.Elt, f)
		}

	case *ast.FuncLit:
		visitExpr(n.Type, f)
		// Skip Body, which does not affect the type.

	case *ast.CompositeLit:
		if n.Type != nil {
			visitExpr(n.Type, f)
		}
		// Skip Elts, which do not affect the type.

	case *ast.ParenExpr:
		visitExpr(n.X, f)

	case *ast.IndexExpr:
		visitExpr(n.X, f)
		visitExpr(n.Index, f) // may affect type for instantiations

	case *typeparams.IndexListExpr:
		visitExpr(n.X, f)
		for _, index := range n.Indices {
			visitExpr(index, f) // may affect the type for instantiations
		}

	case *ast.SliceExpr:
		visitExpr(n.X, f)
		// skip Low, High, and Max, which do not affect type.

	case *ast.TypeAssertExpr:
		// Skip X, as it doesn't actually affect the resulting type of the type
		// assertion.
		if n.Type != nil {
			visitExpr(n.Type, f)
		}

	case *ast.StarExpr:
		visitExpr(n.X, f)

	case *ast.UnaryExpr:
		visitExpr(n.X, f)

	case *ast.BinaryExpr:
		visitExpr(n.X, f)
		visitExpr(n.Y, f)

	case *ast.KeyValueExpr:
		panic("unreachable") // unreachable, as we don't descend into elts of composite lits.

	case *ast.ArrayType:
		if n.Len != nil {
			visitExpr(n.Len, f)
		}
		visitExpr(n.Elt, f)

	case *ast.StructType:
		visitFieldList(n.Fields, f)

	case *ast.FuncType:
		if tparams := typeparams.ForFuncType(n); tparams != nil {
			visitFieldList(tparams, f)
		}
		if n.Params != nil {
			visitFieldList(n.Params, f)
		}
		if n.Results != nil {
			visitFieldList(n.Results, f)
		}

	case *ast.InterfaceType:
		visitFieldList(n.Methods, f)

	case *ast.MapType:
		visitExpr(n.Key, f)
		visitExpr(n.Value, f)

	case *ast.ChanType:
		visitExpr(n.Value, f)

	case *ast.BadExpr:
		// nothing to do

	default:
		panic(fmt.Sprintf("ast.Walk: unexpected node type %T", n))
	}
}

func visitExprList(list []ast.Expr, f refVisitor) {
	for _, x := range list {
		visitExpr(x, f)
	}
}

func visitFieldList(n *ast.FieldList, f refVisitor) {
	for _, field := range n.List {
		visitExpr(field.Type, f)
	}
}

// Copied (with modifications) from go/types.
func unpackRecv(rtyp ast.Expr) (ptr bool, rname *ast.Ident, tparams []*ast.Ident) {
L: // unpack receiver type
	// This accepts invalid receivers such as ***T and does not
	// work for other invalid receivers, but we don't care. The
	// validity of receiver expressions is checked elsewhere.
	for {
		switch t := rtyp.(type) {
		case *ast.ParenExpr:
			rtyp = t.X
		case *ast.StarExpr:
			ptr = true
			rtyp = t.X
		default:
			break L
		}
	}

	// unpack type parameters, if any
	switch rtyp.(type) {
	case *ast.IndexExpr, *typeparams.IndexListExpr:
		var indices []ast.Expr
		rtyp, _, indices, _ = typeparams.UnpackIndexExpr(rtyp)
		for _, arg := range indices {
			var par *ast.Ident
			switch arg := arg.(type) {
			case *ast.Ident:
				par = arg
			default:
				// ignore errors
			}
			if par == nil {
				par = &ast.Ident{NamePos: arg.Pos(), Name: "_"}
			}
			tparams = append(tparams, par)
		}
	}

	// unpack receiver name
	if name, _ := rtyp.(*ast.Ident); name != nil {
		rname = name
	}

	return
}
