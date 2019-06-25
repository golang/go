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
		f.invalidateAST(ctx)
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
			go imp.getPkg(ctx, importID)
		}
		// Type-check package.
		pkg, err := imp.getPkg(ctx, imp.topLevelPkgID)
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
	if !v.runGopackages(ctx, f) {
		return f.meta, nil, nil
	}

	// Check if the context has been canceled before calling packages.Load.
	if ctx.Err() != nil {
		return nil, nil, ctx.Err()
	}

	pkgs, err := packages.Load(v.Config(), fmt.Sprintf("file=%s", f.filename()))
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
	// Track missing imports as we look at the package's errors.
	missingImports := make(map[packagePath]struct{})
	for _, pkg := range pkgs {
		// If the package comes back with errors from `go list`,
		// don't bother type-checking it.
		if len(pkg.Errors) > 0 {
			return nil, pkg.Errors, fmt.Errorf("package %s has errors, skipping type-checking", pkg.PkgPath)
		}
		// Build the import graph for this package.
		if err := v.link(ctx, packagePath(pkg.PkgPath), pkg, nil, missingImports); err != nil {
			return nil, nil, err
		}
	}
	m, err := validateMetadata(ctx, missingImports, f)
	if err != nil {
		return nil, nil, err
	}
	return m, nil, nil
}

func validateMetadata(ctx context.Context, missingImports map[packagePath]struct{}, f *goFile) (map[packageID]*metadata, error) {
	f.mu.Lock()
	defer f.mu.Unlock()

	// If `go list` failed to get data for the file in question (this should never happen).
	if len(f.meta) == 0 {
		return nil, fmt.Errorf("loadParseTypecheck: no metadata found for %v", f.filename())
	}

	// If we have already seen these missing imports before, and we have type information,
	// there is no need to continue.
	if sameSet(missingImports, f.missingImports) && len(f.pkgs) != 0 {
		return nil, nil
	}
	// Otherwise, update the missing imports map.
	f.missingImports = missingImports
	return f.meta, nil
}

// reparseImports reparses a file's package and import declarations to
// determine if they have changed.
func (v *view) runGopackages(ctx context.Context, f *goFile) (result bool) {
	f.mu.Lock()
	defer func() {
		// Clear metadata if we are intending to re-run go/packages.
		if result {
			// Reset the file's metadata and type information if we are re-running `go list`.
			for k := range f.meta {
				delete(f.meta, k)
			}
			for k := range f.pkgs {
				delete(f.pkgs, k)
			}
		}

		defer f.mu.Unlock()
	}()

	if len(f.meta) == 0 || len(f.missingImports) > 0 {
		return true
	}
	// Get file content in case we don't already have it.
	parsed, _ := v.session.cache.ParseGoHandle(f.Handle(ctx), source.ParseHeader).Parse(ctx)
	if parsed == nil {
		return true
	}
	// Check if the package's name has changed, by checking if this is a filename
	// we already know about, and if so, check if its package name has changed.
	for _, m := range f.meta {
		for _, filename := range m.files {
			if filename == f.URI().Filename() {
				if m.name != parsed.Name.Name {
					return true
				}
			}
		}
	}
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

func (v *view) link(ctx context.Context, pkgPath packagePath, pkg *packages.Package, parent *metadata, missingImports map[packagePath]struct{}) error {
	id := packageID(pkg.ID)
	m, ok := v.mcache.packages[id]

	// If a file was added or deleted we need to invalidate the package cache
	// so relevant packages get parsed and type-checked again.
	if ok && !filenamesIdentical(m.files, pkg.CompiledGoFiles) {
		v.pcache.mu.Lock()
		v.remove(ctx, id, make(map[packageID]struct{}))
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
		f, err := v.getFile(ctx, span.FileURI(filename))
		if err != nil {
			v.session.log.Errorf(ctx, "no file %s: %v", filename, err)
		}
		gof, ok := f.(*goFile)
		if !ok {
			v.session.log.Errorf(ctx, "not a Go file: %s", f.URI())
		}
		if gof.meta == nil {
			gof.meta = make(map[packageID]*metadata)
		}
		gof.meta[m.id] = m
	}
	// Connect the import graph.
	if parent != nil {
		m.parents[parent.id] = true
		parent.children[id] = true
	}
	for importPath, importPkg := range pkg.Imports {
		importPkgPath := packagePath(importPath)
		if importPkgPath == pkgPath {
			return fmt.Errorf("cycle detected in %s", importPath)
		}
		// Don't remember any imports with significant errors.
		if importPkgPath != "unsafe" && len(pkg.CompiledGoFiles) == 0 {
			missingImports[importPkgPath] = struct{}{}
			continue
		}
		if _, ok := m.children[packageID(importPkg.ID)]; !ok {
			if err := v.link(ctx, importPkgPath, importPkg, m, missingImports); err != nil {
				v.session.log.Errorf(ctx, "error in dependency %s: %v", importPkgPath, err)
			}
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
	return nil
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
