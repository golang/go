// Copyright 2019 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package cache

import (
	"context"
	"fmt"

	"golang.org/x/tools/go/packages"
	"golang.org/x/tools/internal/lsp/source"
	"golang.org/x/tools/internal/span"
)

func (v *view) loadParseTypecheck(ctx context.Context, f *goFile) ([]packages.Error, error) {
	v.mcache.mu.Lock()
	defer v.mcache.mu.Unlock()

	// If the AST for this file is trimmed, and we are explicitly type-checking it,
	// don't ignore function bodies.
	if f.astIsTrimmed() {
		v.pcache.mu.Lock()
		f.invalidateAST()
		v.pcache.mu.Unlock()
	}

	// Get the metadata for the file.
	meta, errs, err := v.checkMetadata(ctx, f)
	if err != nil {
		return errs, err
	}
	if meta == nil {
		return nil, nil
	}
	for id, m := range meta {
		imp := &importer{
			view:          v,
			seen:          make(map[packageID]struct{}),
			ctx:           ctx,
			fset:          v.session.cache.FileSet(),
			topLevelPkgID: id,
		}
		// Start prefetching direct imports.
		for importID := range m.children {
			go imp.getPkg(importID)
		}
		// Type-check package.
		pkg, err := imp.getPkg(imp.topLevelPkgID)
		if err != nil {
			return nil, err
		}
		if pkg == nil || pkg.IsIllTyped() {
			return nil, fmt.Errorf("loadParseTypecheck: %s is ill typed", m.pkgPath)
		}
	}
	if len(f.pkgs) == 0 {
		return nil, fmt.Errorf("no packages found for %s", f.URI())
	}
	return nil, nil
}

func sameSet(x, y map[packagePath]struct{}) bool {
	if len(x) != len(y) {
		return false
	}
	for k := range x {
		if _, ok := y[k]; !ok {
			return false
		}
	}
	return true
}

// checkMetadata determines if we should run go/packages.Load for this file.
// If yes, update the metadata for the file and its package.
func (v *view) checkMetadata(ctx context.Context, f *goFile) (map[packageID]*metadata, []packages.Error, error) {
	f.mu.Lock()
	defer f.mu.Unlock()

	// Save the metadata's current missing imports, if any.
	originalMissingImports := f.missingImports

	if !v.parseImports(ctx, f) {
		return f.meta, nil, nil
	}

	// Reset the file's metadata and type information if we are re-running `go list`.
	for k := range f.meta {
		delete(f.meta, k)
	}
	for k := range f.pkgs {
		delete(f.pkgs, k)
	}

	pkgs, err := packages.Load(v.buildConfig(), fmt.Sprintf("file=%s", f.filename()))
	if len(pkgs) == 0 {
		if err == nil {
			err = fmt.Errorf("go/packages.Load: no packages found for %s", f.filename())
		}
		// Return this error as a diagnostic to the user.
		return nil, []packages.Error{
			{
				Msg:  err.Error(),
				Kind: packages.UnknownError,
			},
		}, err
	}

	// Clear missing imports.
	for k := range f.missingImports {
		delete(f.missingImports, k)
	}
	for _, pkg := range pkgs {
		// If the package comes back with errors from `go list`,
		// don't bother type-checking it.
		if len(pkg.Errors) > 0 {
			return nil, pkg.Errors, fmt.Errorf("package %s has errors, skipping type-checking", pkg.PkgPath)
		}
		for importPath, importPkg := range pkg.Imports {
			// If we encounter a package we cannot import, mark it as missing.
			if importPkg.PkgPath != "unsafe" && len(importPkg.CompiledGoFiles) == 0 {
				if f.missingImports == nil {
					f.missingImports = make(map[packagePath]struct{})
				}
				f.missingImports[packagePath(importPath)] = struct{}{}
			}
		}
		// Build the import graph for this package.
		v.link(ctx, packagePath(pkg.PkgPath), pkg, nil)
	}

	// If `go list` failed to get data for the file in question (this should never happen).
	if len(f.meta) == 0 {
		return nil, nil, fmt.Errorf("loadParseTypecheck: no metadata found for %v", f.filename())
	}

	// If we have already seen these missing imports before, and we still have type information,
	// there is no need to continue.
	if sameSet(originalMissingImports, f.missingImports) && len(f.pkgs) != 0 {
		return nil, nil, nil
	}

	return f.meta, nil, nil
}

// reparseImports reparses a file's package and import declarations to
// determine if they have changed.
func (v *view) parseImports(ctx context.Context, f *goFile) bool {
	if len(f.meta) == 0 || len(f.missingImports) > 0 {
		return true
	}
	// Get file content in case we don't already have it.
	parsed, _ := v.session.cache.ParseGoHandle(f.Handle(ctx), source.ParseHeader).Parse(ctx)
	if parsed == nil {
		return true
	}
	// TODO: Add support for re-running `go list` when the package name changes.

	// If the package's imports have changed, re-run `go list`.
	if len(f.imports) != len(parsed.Imports) {
		return true
	}

	for i, importSpec := range f.imports {
		if importSpec.Path.Value != parsed.Imports[i].Path.Value {
			return true
		}
	}
	return false
}

func (v *view) link(ctx context.Context, pkgPath packagePath, pkg *packages.Package, parent *metadata) *metadata {
	id := packageID(pkg.ID)
	m, ok := v.mcache.packages[id]

	// If a file was added or deleted we need to invalidate the package cache
	// so relevant packages get parsed and type-checked again.
	if ok && !filenamesIdentical(m.files, pkg.CompiledGoFiles) {
		v.pcache.mu.Lock()
		v.remove(id, make(map[packageID]struct{}))
		v.pcache.mu.Unlock()
	}

	// If we haven't seen this package before.
	if !ok {
		m = &metadata{
			pkgPath:    pkgPath,
			id:         id,
			typesSizes: pkg.TypesSizes,
			parents:    make(map[packageID]bool),
			children:   make(map[packageID]bool),
		}
		v.mcache.packages[id] = m
		v.mcache.ids[pkgPath] = id
	}
	// Reset any field that could have changed across calls to packages.Load.
	m.name = pkg.Name
	m.files = pkg.CompiledGoFiles
	for _, filename := range m.files {
		if f, _ := v.getFile(span.FileURI(filename)); f != nil {
			if gof, ok := f.(*goFile); ok {
				if gof.meta == nil {
					gof.meta = make(map[packageID]*metadata)
				}
				gof.meta[m.id] = m
			} else {
				v.Session().Logger().Errorf(ctx, "not a Go file: %s", f.URI())
			}
		}
	}
	// Connect the import graph.
	if parent != nil {
		m.parents[parent.id] = true
		parent.children[id] = true
	}
	for importPath, importPkg := range pkg.Imports {
		if _, ok := m.children[packageID(importPkg.ID)]; !ok {
			v.link(ctx, packagePath(importPath), importPkg, m)
		}
	}
	// Clear out any imports that have been removed.
	for importID := range m.children {
		child, ok := v.mcache.packages[importID]
		if !ok {
			continue
		}
		importPath := string(child.pkgPath)
		if _, ok := pkg.Imports[importPath]; ok {
			continue
		}
		delete(m.children, importID)
		delete(child.parents, id)
	}
	return m
}

// filenamesIdentical reports whether two sets of file names are identical.
func filenamesIdentical(oldFiles, newFiles []string) bool {
	if len(oldFiles) != len(newFiles) {
		return false
	}
	oldByName := make(map[string]struct{}, len(oldFiles))
	for _, filename := range oldFiles {
		oldByName[filename] = struct{}{}
	}
	for _, newFilename := range newFiles {
		if _, found := oldByName[newFilename]; !found {
			return false
		}
		delete(oldByName, newFilename)
	}
	return len(oldByName) == 0
}
