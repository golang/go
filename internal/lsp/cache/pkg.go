// Copyright 2019 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package cache

import (
	"go/ast"
	"go/types"

	"golang.org/x/mod/module"
	"golang.org/x/tools/internal/lsp/source"
	"golang.org/x/tools/internal/span"
	errors "golang.org/x/xerrors"
)

// pkg contains the type information needed by the source package.
type pkg struct {
	m                    *metadata
	mode                 source.ParseMode
	goFiles              []*source.ParsedGoFile
	compiledGoFiles      []*source.ParsedGoFile
	diagnostics          []*source.Diagnostic
	imports              map[packagePath]*pkg
	version              *module.Version
	typeErrors           []types.Error
	types                *types.Package
	typesInfo            *types.Info
	typesSizes           types.Sizes
	hasListOrParseErrors bool
	hasTypeErrors        bool
}

// Declare explicit types for package paths, names, and IDs to ensure that we
// never use an ID where a path belongs, and vice versa. If we confused these,
// it would result in confusing errors because package IDs often look like
// package paths.
type (
	packageID   string
	packagePath string
	packageName string
)

// Declare explicit types for files and directories to distinguish between the two.
type (
	fileURI         span.URI
	moduleLoadScope string
	viewLoadScope   span.URI
)

func (p *pkg) ID() string {
	return string(p.m.id)
}

func (p *pkg) Name() string {
	return string(p.m.name)
}

func (p *pkg) PkgPath() string {
	return string(p.m.pkgPath)
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
	return nil, errors.Errorf("no parsed file for %s in %v", uri, p.m.id)
}

func (p *pkg) GetSyntax() []*ast.File {
	var syntax []*ast.File
	for _, pgf := range p.compiledGoFiles {
		syntax = append(syntax, pgf.File)
	}
	return syntax
}

func (p *pkg) GetTypes() *types.Package {
	return p.types
}

func (p *pkg) GetTypesInfo() *types.Info {
	return p.typesInfo
}

func (p *pkg) GetTypesSizes() types.Sizes {
	return p.typesSizes
}

func (p *pkg) IsIllTyped() bool {
	return p.types == nil || p.typesInfo == nil || p.typesSizes == nil
}

func (p *pkg) ForTest() string {
	return string(p.m.forTest)
}

func (p *pkg) GetImport(pkgPath string) (source.Package, error) {
	if imp := p.imports[packagePath(pkgPath)]; imp != nil {
		return imp, nil
	}
	// Don't return a nil pointer because that still satisfies the interface.
	return nil, errors.Errorf("no imported package for %s", pkgPath)
}

func (p *pkg) MissingDependencies() []string {
	// We don't invalidate metadata for import deletions, so check the package
	// imports via the *types.Package. Only use metadata if p.types is nil.
	if p.types == nil {
		var md []string
		for i := range p.m.missingDeps {
			md = append(md, string(i))
		}
		return md
	}
	var md []string
	for _, pkg := range p.types.Imports() {
		if _, ok := p.m.missingDeps[packagePath(pkg.Path())]; ok {
			md = append(md, pkg.Path())
		}
	}
	return md
}

func (p *pkg) Imports() []source.Package {
	var result []source.Package
	for _, imp := range p.imports {
		result = append(result, imp)
	}
	return result
}

func (p *pkg) Version() *module.Version {
	return p.version
}

func (p *pkg) HasListOrParseErrors() bool {
	return p.hasListOrParseErrors
}

func (p *pkg) HasTypeErrors() bool {
	return p.hasTypeErrors
}
