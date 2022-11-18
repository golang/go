// Copyright 2019 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package cache

import (
	"fmt"
	"go/ast"
	"go/scanner"
	"go/token"
	"go/types"

	"golang.org/x/mod/module"
	"golang.org/x/tools/gopls/internal/lsp/source"
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

// pkg contains parse trees and type information for a package.
type pkg struct {
	m               *source.Metadata
	mode            source.ParseMode
	fset            *token.FileSet // for now, same as the snapshot's FileSet
	goFiles         []*source.ParsedGoFile
	compiledGoFiles []*source.ParsedGoFile
	diagnostics     []*source.Diagnostic
	deps            map[PackageID]*pkg // use m.DepsBy{Pkg,Imp}Path to look up ID
	version         *module.Version    // may be nil; may differ from m.Module.Version
	parseErrors     []scanner.ErrorList
	typeErrors      []types.Error
	types           *types.Package
	typesInfo       *types.Info
	hasFixedFiles   bool // if true, AST was sufficiently mangled that we should hide type errors

	analyses memoize.Store // maps analyzer.Name to Promise[actionResult]
}

func (p *pkg) String() string { return string(p.ID()) }

// A loadScope defines a package loading scope for use with go/packages.
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

func (p *pkg) ID() PackageID        { return p.m.ID }
func (p *pkg) Name() PackageName    { return p.m.Name }
func (p *pkg) PkgPath() PackagePath { return p.m.PkgPath }

func (p *pkg) ParseMode() source.ParseMode {
	return p.mode
}

func (p *pkg) CompiledGoFiles() []*source.ParsedGoFile {
	return p.compiledGoFiles
}

func (p *pkg) File(uri span.URI) (*source.ParsedGoFile, error) {
	for _, cgf := range p.compiledGoFiles {
		if cgf.URI == uri {
			return cgf, nil
		}
	}
	for _, gf := range p.goFiles {
		if gf.URI == uri {
			return gf, nil
		}
	}
	return nil, fmt.Errorf("no parsed file for %s in %v", uri, p.m.ID)
}

func (p *pkg) GetSyntax() []*ast.File {
	var syntax []*ast.File
	for _, pgf := range p.compiledGoFiles {
		syntax = append(syntax, pgf.File)
	}
	return syntax
}

func (p *pkg) FileSet() *token.FileSet {
	return p.fset
}

func (p *pkg) GetTypes() *types.Package {
	return p.types
}

func (p *pkg) GetTypesInfo() *types.Info {
	return p.typesInfo
}

func (p *pkg) GetTypesSizes() types.Sizes {
	return p.m.TypesSizes
}

func (p *pkg) ForTest() string {
	return string(p.m.ForTest)
}

// DirectDep returns the directly imported dependency of this package,
// given its PackagePath.  (If you have an ImportPath, e.g. a string
// from an import declaration, use ResolveImportPath instead.
// They may differ in case of vendoring.)
func (p *pkg) DirectDep(pkgPath PackagePath) (source.Package, error) {
	if id, ok := p.m.DepsByPkgPath[pkgPath]; ok {
		if imp := p.deps[id]; imp != nil {
			return imp, nil
		}
	}
	return nil, fmt.Errorf("package does not import package with path %s", pkgPath)
}

// ResolveImportPath returns the directly imported dependency of this package,
// given its ImportPath. See also DirectDep.
func (p *pkg) ResolveImportPath(importPath ImportPath) (source.Package, error) {
	if id, ok := p.m.DepsByImpPath[importPath]; ok && id != "" {
		if imp := p.deps[id]; imp != nil {
			return imp, nil
		}
	}
	return nil, fmt.Errorf("package does not import %s", importPath)
}

func (p *pkg) Imports() []source.Package {
	var result []source.Package // unordered
	for _, dep := range p.deps {
		result = append(result, dep)
	}
	return result
}

func (p *pkg) Version() *module.Version {
	return p.version
}

func (p *pkg) HasListOrParseErrors() bool {
	return len(p.m.Errors) != 0 || len(p.parseErrors) != 0
}

func (p *pkg) HasTypeErrors() bool {
	return len(p.typeErrors) != 0
}
