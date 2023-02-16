// Copyright 2019 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package cache

import (
	"context"
	"fmt"
	"go/ast"
	"go/scanner"
	"go/token"
	"go/types"

	"golang.org/x/tools/go/types/objectpath"
	"golang.org/x/tools/gopls/internal/lsp/protocol"
	"golang.org/x/tools/gopls/internal/lsp/source"
	"golang.org/x/tools/gopls/internal/lsp/source/methodsets"
	"golang.org/x/tools/gopls/internal/lsp/source/xrefs"
	"golang.org/x/tools/gopls/internal/span"
	"golang.org/x/tools/internal/memoize"
)

// Convenient local aliases for typed strings.
type (
	PackageID   = source.PackageID
	PackagePath = source.PackagePath
	PackageName = source.PackageName
	ImportPath  = source.ImportPath
)

// A Package is the union of snapshot-local information (Metadata) and shared
// type-checking information (a syntaxPackage).
//
// TODO(rfindley): for now, we do not persist the post-processing of
// loadDiagnostics, because the value of the snapshot.packages map  is just the
// package handle. Fix this.
type Package struct {
	m   *source.Metadata
	pkg *syntaxPackage
}

// syntaxPackage contains parse trees and type information for a package.
type syntaxPackage struct {
	// -- identifiers --
	id   PackageID
	mode source.ParseMode

	// -- outputs --
	fset            *token.FileSet // for now, same as the snapshot's FileSet
	goFiles         []*source.ParsedGoFile
	compiledGoFiles []*source.ParsedGoFile
	diagnostics     []*source.Diagnostic
	parseErrors     []scanner.ErrorList
	typeErrors      []types.Error
	types           *types.Package
	typesInfo       *types.Info
	importMap       *importMap        // required for DependencyTypes (until we have shallow export data)
	hasFixedFiles   bool              // if true, AST was sufficiently mangled that we should hide type errors
	xrefs           []byte            // serializable index of outbound cross-references
	analyses        memoize.Store     // maps analyzer.Name to Promise[actionResult]
	methodsets      *methodsets.Index // index of method sets of package-level types
}

func (p *Package) String() string { return string(p.m.ID) }

func (p *Package) Metadata() *source.Metadata { return p.m }

// An importMap is an mapping from source.PackagePath to types.Package
// of the dependencies of a syntaxPackage. Once constructed (by calls
// to union) it is never subsequently modified.
type importMap struct {
	// Concretely, it is a node that contains one element of the
	// mapping and whose deps are edges in DAG that comprises the
	// rest of the mapping. This design optimizes union over get.
	//
	// TODO(adonovan): simplify when we use shallow export data.
	// At that point it becomes a simple lookup in the importers
	// map, which should be retained in syntaxPackage.
	// (Otherwise this implementation could expose types.Packages
	// that represent an old state that has since changed, but
	// not in a way that is consequential to a downstream package.)

	types *types.Package // map entry for types.Path => types
	deps  []*importMap   // all other entries
}

func (m *importMap) union(dep *importMap) { m.deps = append(m.deps, dep) }

func (m *importMap) get(path source.PackagePath, seen map[*importMap]bool) *types.Package {
	if !seen[m] {
		seen[m] = true
		if source.PackagePath(m.types.Path()) == path {
			return m.types
		}
		for _, dep := range m.deps {
			if pkg := dep.get(path, seen); pkg != nil {
				return pkg
			}
		}
	}
	return nil
}

// A loadScope defines a package loading scope for use with go/packages.
//
// TODO(rfindley): move this to load.go.
type loadScope interface {
	aScope()
}

type (
	fileLoadScope    span.URI // load packages containing a file (including command-line-arguments)
	packageLoadScope string   // load a specific package (the value is its PackageID)
	moduleLoadScope  string   // load packages in a specific module
	viewLoadScope    span.URI // load the workspace
)

// Implement the loadScope interface.
func (fileLoadScope) aScope()    {}
func (packageLoadScope) aScope() {}
func (moduleLoadScope) aScope()  {}
func (viewLoadScope) aScope()    {}

func (p *Package) ParseMode() source.ParseMode {
	return p.pkg.mode
}

func (p *Package) CompiledGoFiles() []*source.ParsedGoFile {
	return p.pkg.compiledGoFiles
}

func (p *Package) File(uri span.URI) (*source.ParsedGoFile, error) {
	return p.pkg.File(uri)
}

func (pkg *syntaxPackage) File(uri span.URI) (*source.ParsedGoFile, error) {
	for _, cgf := range pkg.compiledGoFiles {
		if cgf.URI == uri {
			return cgf, nil
		}
	}
	for _, gf := range pkg.goFiles {
		if gf.URI == uri {
			return gf, nil
		}
	}
	return nil, fmt.Errorf("no parsed file for %s in %v", uri, pkg.id)
}

func (p *Package) GetSyntax() []*ast.File {
	var syntax []*ast.File
	for _, pgf := range p.pkg.compiledGoFiles {
		syntax = append(syntax, pgf.File)
	}
	return syntax
}

func (p *Package) FileSet() *token.FileSet {
	return p.pkg.fset
}

func (p *Package) GetTypes() *types.Package {
	return p.pkg.types
}

func (p *Package) GetTypesInfo() *types.Info {
	return p.pkg.typesInfo
}

// DependencyTypes returns the type checker's symbol for the specified
// package. It returns nil if path is not among the transitive
// dependencies of p, or if no symbols from that package were
// referenced during the type-checking of p.
func (p *Package) DependencyTypes(path source.PackagePath) *types.Package {
	return p.pkg.importMap.get(path, make(map[*importMap]bool))
}

func (p *Package) HasParseErrors() bool {
	return len(p.pkg.parseErrors) != 0
}

func (p *Package) HasTypeErrors() bool {
	return len(p.pkg.typeErrors) != 0
}

func (p *Package) DiagnosticsForFile(ctx context.Context, s source.Snapshot, uri span.URI) ([]*source.Diagnostic, error) {
	var diags []*source.Diagnostic
	for _, diag := range p.m.Diagnostics {
		if diag.URI == uri {
			diags = append(diags, diag)
		}
	}
	for _, diag := range p.pkg.diagnostics {
		if diag.URI == uri {
			diags = append(diags, diag)
		}
	}

	return diags, nil
}

// ReferencesTo returns the location of each reference within package p
// to one of the target objects denoted by the pair (package path, object path).
func (p *Package) ReferencesTo(targets map[PackagePath]map[objectpath.Path]struct{}) []protocol.Location {
	// TODO(adonovan): In future, p.xrefs will be retrieved from a
	// section of the cache file produced by type checking.
	// (Other sections will include the package's export data,
	// "implements" relations, exported symbols, etc.)
	// For now we just hang it off the pkg.
	return xrefs.Lookup(p.m, p.pkg.xrefs, targets)
}

func (p *Package) MethodSetsIndex() *methodsets.Index {
	// TODO(adonovan): In future, p.methodsets will be retrieved from a
	// section of the cache file produced by type checking.
	return p.pkg.methodsets
}
