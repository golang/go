// Copyright 2019 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package cache

import (
	"context"
	"fmt"
	"go/ast"
	"go/parser"
	"go/scanner"
	"go/token"
	"go/types"

	"golang.org/x/tools/go/analysis"
	"golang.org/x/tools/go/packages"
	"golang.org/x/tools/internal/lsp/source"
	"golang.org/x/tools/internal/span"
)

func (v *view) parse(ctx context.Context, file source.File) ([]packages.Error, error) {
	v.mcache.mu.Lock()
	defer v.mcache.mu.Unlock()

	// Apply any queued-up content changes.
	if err := v.applyContentChanges(ctx); err != nil {
		return nil, err
	}

	f, ok := file.(*goFile)
	if !ok {
		return nil, fmt.Errorf("not a go file: %v", file.URI())
	}

	// If the package for the file has not been invalidated by the application
	// of the pending changes, there is no need to continue.
	if f.isPopulated() {
		return nil, nil
	}
	// Check if the file's imports have changed. If they have, update the
	// metadata by calling packages.Load.
	if errs, err := v.checkMetadata(ctx, f); err != nil {
		return errs, err
	}
	if f.meta == nil {
		return nil, fmt.Errorf("no metadata found for %v", f.filename())
	}
	imp := &importer{
		view: v,
		seen: make(map[string]struct{}),
		ctx:  ctx,
		fset: f.FileSet(),
	}
	// Start prefetching direct imports.
	for importPath := range f.meta.children {
		go imp.Import(importPath)
	}
	// Type-check package.
	pkg, err := imp.typeCheck(f.meta.pkgPath)
	if pkg == nil || pkg.GetTypes() == nil {
		return nil, err
	}

	// If we still have not found the package for the file, something is wrong.
	if f.pkg == nil {
		return nil, fmt.Errorf("parse: no package found for %v", f.filename())
	}
	return nil, nil
}

func (v *view) checkMetadata(ctx context.Context, f *goFile) ([]packages.Error, error) {
	if v.reparseImports(ctx, f, f.filename()) {
		cfg := v.buildConfig()
		pkgs, err := packages.Load(cfg, fmt.Sprintf("file=%s", f.filename()))
		if len(pkgs) == 0 {
			if err == nil {
				err = fmt.Errorf("%s: no packages found", f.filename())
			}
			// Return this error as a diagnostic to the user.
			return []packages.Error{
				{
					Msg:  err.Error(),
					Kind: packages.ListError,
				},
			}, err
		}
		for _, pkg := range pkgs {
			// If the package comes back with errors from `go list`, don't bother
			// type-checking it.
			if len(pkg.Errors) > 0 {
				return pkg.Errors, fmt.Errorf("package %s has errors, skipping type-checking", pkg.PkgPath)
			}
			v.link(ctx, pkg.PkgPath, pkg, nil)
		}
	}
	return nil, nil
}

// reparseImports reparses a file's import declarations to determine if they
// have changed.
func (v *view) reparseImports(ctx context.Context, f *goFile, filename string) bool {
	if f.meta == nil {
		return true
	}
	// Get file content in case we don't already have it.
	f.read(ctx)
	parsed, _ := parser.ParseFile(f.FileSet(), filename, f.content, parser.ImportsOnly)
	if parsed == nil {
		return true
	}
	// If the package name has changed, re-run `go list`.
	if f.meta.name != parsed.Name.Name {
		return true
	}
	// If the package's imports have changed, re-run `go list`.
	if len(f.imports) != len(parsed.Imports) {
		return true
	}
	for i, importSpec := range f.imports {
		if importSpec.Path.Value != f.imports[i].Path.Value {
			return true
		}
	}
	return false
}

func (v *view) link(ctx context.Context, pkgPath string, pkg *packages.Package, parent *metadata) *metadata {
	m, ok := v.mcache.packages[pkgPath]
	if !ok {
		m = &metadata{
			pkgPath:    pkgPath,
			id:         pkg.ID,
			typesSizes: pkg.TypesSizes,
			parents:    make(map[string]bool),
			children:   make(map[string]bool),
		}
		v.mcache.packages[pkgPath] = m
	}
	// Reset any field that could have changed across calls to packages.Load.
	m.name = pkg.Name
	m.files = pkg.CompiledGoFiles
	for _, filename := range m.files {
		if f, _ := v.getFile(span.FileURI(filename)); f != nil {
			gof, ok := f.(*goFile)
			if !ok {
				v.Session().Logger().Errorf(ctx, "not a go file: %v", f.URI())
				continue
			}
			gof.meta = m
		}
	}
	// Connect the import graph.
	if parent != nil {
		m.parents[parent.pkgPath] = true
		parent.children[pkgPath] = true
	}
	for importPath, importPkg := range pkg.Imports {
		if _, ok := m.children[importPath]; !ok {
			v.link(ctx, importPath, importPkg, m)
		}
	}
	// Clear out any imports that have been removed.
	for importPath := range m.children {
		if _, ok := pkg.Imports[importPath]; !ok {
			delete(m.children, importPath)
			if child, ok := v.mcache.packages[importPath]; ok {
				delete(child.parents, pkgPath)
			}
		}
	}
	return m
}

type importer struct {
	view *view

	// seen maintains the set of previously imported packages.
	// If we have seen a package that is already in this map, we have a circular import.
	seen map[string]struct{}

	ctx  context.Context
	fset *token.FileSet
}

func (imp *importer) Import(pkgPath string) (*types.Package, error) {
	if _, ok := imp.seen[pkgPath]; ok {
		return nil, fmt.Errorf("circular import detected")
	}
	imp.view.pcache.mu.Lock()
	e, ok := imp.view.pcache.packages[pkgPath]
	if ok {
		// cache hit
		imp.view.pcache.mu.Unlock()
		// wait for entry to become ready
		<-e.ready
	} else {
		// cache miss
		e = &entry{ready: make(chan struct{})}
		imp.view.pcache.packages[pkgPath] = e
		imp.view.pcache.mu.Unlock()

		// This goroutine becomes responsible for populating
		// the entry and broadcasting its readiness.
		e.pkg, e.err = imp.typeCheck(pkgPath)
		close(e.ready)
	}
	if e.err != nil {
		return nil, e.err
	}
	return e.pkg.types, nil
}

func (imp *importer) typeCheck(pkgPath string) (*pkg, error) {
	meta, ok := imp.view.mcache.packages[pkgPath]
	if !ok {
		return nil, fmt.Errorf("no metadata for %v", pkgPath)
	}
	// Use the default type information for the unsafe package.
	var typ *types.Package
	if meta.pkgPath == "unsafe" {
		typ = types.Unsafe
	} else {
		typ = types.NewPackage(meta.pkgPath, meta.name)
	}
	pkg := &pkg{
		id:         meta.id,
		pkgPath:    meta.pkgPath,
		files:      meta.files,
		imports:    make(map[string]*pkg),
		types:      typ,
		typesSizes: meta.typesSizes,
		typesInfo: &types.Info{
			Types:      make(map[ast.Expr]types.TypeAndValue),
			Defs:       make(map[*ast.Ident]types.Object),
			Uses:       make(map[*ast.Ident]types.Object),
			Implicits:  make(map[ast.Node]types.Object),
			Selections: make(map[*ast.SelectorExpr]*types.Selection),
			Scopes:     make(map[ast.Node]*types.Scope),
		},
		analyses: make(map[*analysis.Analyzer]*analysisEntry),
	}
	appendError := func(err error) {
		imp.view.appendPkgError(pkg, err)
	}
	files, errs := imp.parseFiles(meta.files)
	for _, err := range errs {
		appendError(err)
	}
	pkg.syntax = files

	// Handle circular imports by copying previously seen imports.
	seen := make(map[string]struct{})
	for k, v := range imp.seen {
		seen[k] = v
	}
	seen[pkgPath] = struct{}{}

	cfg := &types.Config{
		Error: appendError,
		Importer: &importer{
			view: imp.view,
			seen: seen,
			ctx:  imp.ctx,
			fset: imp.fset,
		},
	}
	check := types.NewChecker(cfg, imp.fset, pkg.types, pkg.typesInfo)
	check.Files(pkg.syntax)

	// Add every file in this package to our cache.
	imp.view.cachePackage(imp.ctx, pkg, meta)

	return pkg, nil
}

func (v *view) cachePackage(ctx context.Context, pkg *pkg, meta *metadata) {
	for _, file := range pkg.GetSyntax() {
		// TODO: If a file is in multiple packages, which package do we store?
		if !file.Pos().IsValid() {
			v.Session().Logger().Errorf(ctx, "invalid position for file %v", file.Name)
			continue
		}
		tok := v.Session().Cache().FileSet().File(file.Pos())
		if tok == nil {
			v.Session().Logger().Errorf(ctx, "no token.File for %v", file.Name)
			continue
		}
		fURI := span.FileURI(tok.Name())
		f, err := v.getFile(fURI)
		if err != nil {
			v.Session().Logger().Errorf(ctx, "no file: %v", err)
			continue
		}
		gof, ok := f.(*goFile)
		if !ok {
			v.Session().Logger().Errorf(ctx, "not a go file: %v", f.URI())
			continue
		}
		gof.token = tok
		gof.ast = file
		gof.imports = gof.ast.Imports
		gof.pkg = pkg
	}

	v.pcache.mu.Lock()
	defer v.pcache.mu.Unlock()

	// Cache the entry for this package.
	// All dependencies are cached through calls to *imp.Import.
	e := &entry{
		pkg:   pkg,
		err:   nil,
		ready: make(chan struct{}),
	}
	close(e.ready)
	v.pcache.packages[pkg.pkgPath] = e

	// Set imports of package to correspond to cached packages.
	// We lock the package cache, but we shouldn't get any inconsistencies
	// because we are still holding the lock on the view.
	for importPath := range meta.children {
		if importEntry, ok := v.pcache.packages[importPath]; ok {
			pkg.imports[importPath] = importEntry.pkg
		}
	}
}

func (v *view) appendPkgError(pkg *pkg, err error) {
	if err == nil {
		return
	}
	var errs []packages.Error
	switch err := err.(type) {
	case *scanner.Error:
		errs = append(errs, packages.Error{
			Pos:  err.Pos.String(),
			Msg:  err.Msg,
			Kind: packages.ParseError,
		})
	case scanner.ErrorList:
		// The first parser error is likely the root cause of the problem.
		if err.Len() > 0 {
			errs = append(errs, packages.Error{
				Pos:  err[0].Pos.String(),
				Msg:  err[0].Msg,
				Kind: packages.ParseError,
			})
		}
	case types.Error:
		errs = append(errs, packages.Error{
			Pos:  v.Session().Cache().FileSet().Position(err.Pos).String(),
			Msg:  err.Msg,
			Kind: packages.TypeError,
		})
	}
	pkg.errors = append(pkg.errors, errs...)
}
