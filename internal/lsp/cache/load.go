// Copyright 2019 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package cache

import (
	"context"
	"fmt"

	"golang.org/x/tools/go/packages"
	"golang.org/x/tools/internal/lsp/source"
	"golang.org/x/tools/internal/lsp/telemetry"
	"golang.org/x/tools/internal/span"
	"golang.org/x/tools/internal/telemetry/log"
	"golang.org/x/tools/internal/telemetry/tag"
	"golang.org/x/tools/internal/telemetry/trace"
	errors "golang.org/x/xerrors"
)

func (view *view) loadParseTypecheck(ctx context.Context, f *goFile, fh source.FileHandle) error {
	ctx, done := trace.StartSpan(ctx, "cache.view.loadParseTypeCheck", telemetry.URI.Of(f.URI()))
	defer done()

	meta, err := view.load(ctx, f, fh)
	if err != nil {
		return err
	}
	for _, m := range meta {
		imp := &importer{
			view:              view,
			config:            view.Config(ctx),
			seen:              make(map[packageID]struct{}),
			topLevelPackageID: m.id,
		}
		cph, err := imp.checkPackageHandle(ctx, m)
		if err != nil {
			log.Error(ctx, "loadParseTypeCheck: failed to get CheckPackageHandle", err, telemetry.Package.Of(m.id))
			continue
		}
		// Cache this package on the file object, since all dependencies are cached in the Import function.
		if err := imp.cachePackage(ctx, cph); err != nil {
			log.Error(ctx, "loadParseTypeCheck: failed to cache package", err, telemetry.Package.Of(m.id))
			continue
		}
	}
	return nil
}

func (view *view) load(ctx context.Context, f *goFile, fh source.FileHandle) ([]*metadata, error) {
	ctx, done := trace.StartSpan(ctx, "cache.view.load", telemetry.URI.Of(f.URI()))
	defer done()

	view.mu.Lock()
	defer view.mu.Unlock()

	view.mcache.mu.Lock()
	defer view.mcache.mu.Unlock()

	var toDelete []packageID
	f.mu.Lock()
	for id, cph := range f.cphs {
		if cph != nil {
			toDelete = append(toDelete, id)
		}
	}
	f.mu.Unlock()

	// If the AST for this file is trimmed, and we are explicitly type-checking it,
	// don't ignore function bodies.
	if f.wrongParseMode(ctx, fh, source.ParseFull) {
		// Remove the package and all of its reverse dependencies from the cache.
		for _, id := range toDelete {
			f.view.remove(ctx, id, map[packageID]struct{}{})
		}
	}

	// Get the metadata for the file.
	meta, err := view.checkMetadata(ctx, f, fh)
	if err != nil {
		return nil, err
	}
	if len(meta) == 0 {
		return nil, fmt.Errorf("no package metadata found for %s", f.URI())
	}
	return meta, nil
}

// checkMetadata determines if we should run go/packages.Load for this file.
// If yes, update the metadata for the file and its package.
func (v *view) checkMetadata(ctx context.Context, f *goFile, fh source.FileHandle) (metadata []*metadata, err error) {
	// Check if we need to re-run go/packages before loading the package.
	var runGopackages bool
	func() {
		f.mu.Lock()
		defer f.mu.Unlock()

		runGopackages, err = v.shouldRunGopackages(ctx, f, fh)
		metadata = f.metadata()
	}()
	if err != nil {
		return nil, err
	}

	// The package metadata is correct as-is, so just return it.
	if !runGopackages {
		return metadata, nil
	}

	// Don't bother running go/packages if the context has been canceled.
	if ctx.Err() != nil {
		return nil, ctx.Err()
	}

	ctx, done := trace.StartSpan(ctx, "packages.Load", telemetry.File.Of(f.filename()))
	defer done()

	pkgs, err := packages.Load(v.Config(ctx), fmt.Sprintf("file=%s", f.filename()))
	if len(pkgs) == 0 {
		if err == nil {
			err = errors.Errorf("go/packages.Load: no packages found for %s", f.filename())
		}
		// Return this error as a diagnostic to the user.
		return nil, err
	}
	// Track missing imports as we look at the package's errors.
	missingImports := make(map[packagePath]struct{})

	// Clear metadata since we are re-running go/packages.
	// Reset the file's metadata and type information if we are re-running `go list`.
	f.mu.Lock()
	for k := range f.meta {
		delete(f.meta, k)
	}
	for k := range f.cphs {
		delete(f.cphs, k)
	}
	f.mu.Unlock()

	log.Print(ctx, "go/packages.Load", tag.Of("packages", len(pkgs)))
	for _, pkg := range pkgs {
		log.Print(ctx, "go/packages.Load", tag.Of("package", pkg.PkgPath), tag.Of("files", pkg.CompiledGoFiles))
		// Build the import graph for this package.
		if err := v.link(ctx, &importGraph{
			pkgPath:        packagePath(pkg.PkgPath),
			pkg:            pkg,
			parent:         nil,
			missingImports: make(map[packagePath]struct{}),
		}); err != nil {
			return nil, err
		}
	}
	m, err := validateMetadata(ctx, missingImports, f)
	if err != nil {
		return nil, err
	}
	return m, nil
}

func validateMetadata(ctx context.Context, missingImports map[packagePath]struct{}, f *goFile) ([]*metadata, error) {
	f.mu.Lock()
	defer f.mu.Unlock()

	// If `go list` failed to get data for the file in question (this should never happen).
	if len(f.meta) == 0 {
		return nil, errors.Errorf("loadParseTypecheck: no metadata found for %v", f.filename())
	}

	// If we have already seen these missing imports before, and we have type information,
	// there is no need to continue.
	if sameSet(missingImports, f.missingImports) && len(f.cphs) != 0 {
		return nil, nil
	}

	// Otherwise, update the missing imports map.
	f.missingImports = missingImports

	return f.metadata(), nil
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

// shouldRunGopackages reparses a file's package and import declarations to
// determine if they have changed.
// It assumes that the caller holds the lock on the f.mu lock.
func (v *view) shouldRunGopackages(ctx context.Context, f *goFile, fh source.FileHandle) (result bool, err error) {
	if len(f.meta) == 0 || len(f.missingImports) > 0 {
		return true, nil
	}
	// Get file content in case we don't already have it.
	parsed, _, _, err := v.session.cache.ParseGoHandle(fh, source.ParseHeader).Parse(ctx)
	if err != nil {
		return false, err
	}
	// Check if the package's name has changed, by checking if this is a filename
	// we already know about, and if so, check if its package name has changed.
	for _, m := range f.meta {
		for _, uri := range m.files {
			if span.CompareURI(uri, f.URI()) == 0 {
				if m.name != parsed.Name.Name {
					return true, nil
				}
			}
		}
	}
	// If the package's imports have changed, re-run `go list`.
	if len(f.imports) != len(parsed.Imports) {
		return true, nil
	}
	for i, importSpec := range f.imports {
		if importSpec.Path.Value != parsed.Imports[i].Path.Value {
			return true, nil
		}
	}
	return false, nil
}

type importGraph struct {
	pkgPath        packagePath
	pkg            *packages.Package
	parent         *metadata
	missingImports map[packagePath]struct{}
}

func (v *view) link(ctx context.Context, g *importGraph) error {
	// Recreate the metadata rather than reusing it to avoid locking.
	m := &metadata{
		id:         packageID(g.pkg.ID),
		pkgPath:    g.pkgPath,
		name:       g.pkg.Name,
		typesSizes: g.pkg.TypesSizes,
		errors:     g.pkg.Errors,
	}
	for _, filename := range g.pkg.CompiledGoFiles {
		m.files = append(m.files, span.FileURI(filename))

		// Call the unlocked version of getFile since we are holding the view's mutex.
		f, err := v.getFile(ctx, span.FileURI(filename), source.Go)
		if err != nil {
			log.Error(ctx, "no file", err, telemetry.File.Of(filename))
			continue
		}
		gof, ok := f.(*goFile)
		if !ok {
			log.Error(ctx, "not a Go file", nil, telemetry.File.Of(filename))
			continue
		}
		// Cache the metadata for this file.
		gof.mu.Lock()
		if gof.meta == nil {
			gof.meta = make(map[packageID]*metadata)
		}
		gof.meta[m.id] = m
		gof.mu.Unlock()
	}

	// Preserve the import graph.
	if original, ok := v.mcache.packages[m.id]; ok {
		m.children = original.children
		m.parents = original.parents
	}
	if m.children == nil {
		m.children = make(map[packageID]*metadata)
	}
	if m.parents == nil {
		m.parents = make(map[packageID]bool)
	}

	// Add the metadata to the cache.
	v.mcache.packages[m.id] = m

	// Connect the import graph.
	if g.parent != nil {
		m.parents[g.parent.id] = true
		g.parent.children[m.id] = m
	}
	for importPath, importPkg := range g.pkg.Imports {
		importPkgPath := packagePath(importPath)
		if importPkgPath == g.pkgPath {
			return fmt.Errorf("cycle detected in %s", importPath)
		}
		// Don't remember any imports with significant errors.
		if importPkgPath != "unsafe" && len(importPkg.CompiledGoFiles) == 0 {
			g.missingImports[importPkgPath] = struct{}{}
			continue
		}
		if _, ok := m.children[packageID(importPkg.ID)]; !ok {
			if err := v.link(ctx, &importGraph{
				pkgPath:        importPkgPath,
				pkg:            importPkg,
				parent:         m,
				missingImports: g.missingImports,
			}); err != nil {
				log.Error(ctx, "error in dependency", err)
			}
		}
	}
	// Clear out any imports that have been removed since the package was last loaded.
	for importID := range m.children {
		child, ok := v.mcache.packages[importID]
		if !ok {
			continue
		}
		importPath := string(child.pkgPath)
		if _, ok := g.pkg.Imports[importPath]; ok {
			continue
		}
		delete(m.children, importID)
		delete(child.parents, m.id)
	}
	return nil
}
