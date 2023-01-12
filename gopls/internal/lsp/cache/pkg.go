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
	"strings"

	"golang.org/x/tools/go/types/objectpath"
	"golang.org/x/tools/gopls/internal/lsp/protocol"
	"golang.org/x/tools/gopls/internal/lsp/source"
	"golang.org/x/tools/gopls/internal/lsp/source/methodsets"
	"golang.org/x/tools/gopls/internal/lsp/source/xrefs"
	"golang.org/x/tools/gopls/internal/span"
	"golang.org/x/tools/internal/event"
	"golang.org/x/tools/internal/event/tag"
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
	m               *source.Metadata
	pkg             *syntaxPackage
	loadDiagnostics *memoize.Promise // post-processed errors from loading
}

func newPackage(m *source.Metadata, pkg *syntaxPackage) *Package {
	p := &Package{
		m:   m,
		pkg: pkg,
	}
	if len(m.Errors) > 0 || len(m.DepsErrors) > 0 {
		p.loadDiagnostics = memoize.NewPromise(fmt.Sprintf("loadDiagnostics(%s)", m.ID), func(ctx context.Context, arg interface{}) interface{} {
			s := arg.(*snapshot)
			var diags []*source.Diagnostic
			for _, packagesErr := range p.m.Errors {
				// Filter out parse errors from go list. We'll get them when we
				// actually parse, and buggy overlay support may generate spurious
				// errors. (See TestNewModule_Issue38207.)
				if strings.Contains(packagesErr.Msg, "expected '") {
					continue
				}
				pkgDiags, err := goPackagesErrorDiagnostics(packagesErr, p.pkg, p.m.LoadDir)
				if err != nil {
					// There are certain cases where the go command returns invalid
					// positions, so we cannot panic or even bug.Reportf here.
					event.Error(ctx, "unable to compute positions for list errors", err, tag.Package.Of(string(p.m.ID)))
					continue
				}
				diags = append(diags, pkgDiags...)
			}

			// TODO(rfindley): this is buggy: an insignificant change to a modfile
			// (or an unsaved modfile) could affect the position of deps errors,
			// without invalidating the package.
			depsDiags, err := s.depsErrors(ctx, p.pkg, p.m.DepsErrors)
			if err != nil {
				if ctx.Err() == nil {
					// TODO(rfindley): consider making this a bug.Reportf. depsErrors should
					// not normally fail.
					event.Error(ctx, "unable to compute deps errors", err, tag.Package.Of(string(p.m.ID)))
				}
				return nil
			}
			diags = append(diags, depsDiags...)
			return diags
		})
	}
	return p
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
	hasFixedFiles   bool              // if true, AST was sufficiently mangled that we should hide type errors
	xrefs           []byte            // serializable index of outbound cross-references
	analyses        memoize.Store     // maps analyzer.Name to Promise[actionResult]
	methodsets      *methodsets.Index // index of method sets of package-level types
}

func (p *Package) String() string { return string(p.m.ID) }

func (p *Package) Metadata() *source.Metadata {
	return p.m
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

func (p *Package) HasParseErrors() bool {
	return len(p.pkg.parseErrors) != 0
}

func (p *Package) HasTypeErrors() bool {
	return len(p.pkg.typeErrors) != 0
}

func (p *Package) DiagnosticsForFile(ctx context.Context, s source.Snapshot, uri span.URI) ([]*source.Diagnostic, error) {
	var diags []*source.Diagnostic
	for _, diag := range p.pkg.diagnostics {
		if diag.URI == uri {
			diags = append(diags, diag)
		}
	}

	if p.loadDiagnostics != nil {
		res, err := p.loadDiagnostics.Get(ctx, s)
		if err != nil {
			return nil, err
		}
		for _, diag := range res.([]*source.Diagnostic) {
			if diag.URI == uri {
				diags = append(diags, diag)
			}
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
