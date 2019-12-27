// Copyright 2019 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package cache

import (
	"context"
	"fmt"
	"go/types"

	"golang.org/x/tools/go/packages"
	"golang.org/x/tools/internal/lsp/source"
	"golang.org/x/tools/internal/lsp/telemetry"
	"golang.org/x/tools/internal/span"
	"golang.org/x/tools/internal/telemetry/log"
	"golang.org/x/tools/internal/telemetry/tag"
	"golang.org/x/tools/internal/telemetry/trace"
	errors "golang.org/x/xerrors"
)

type metadata struct {
	id              packageID
	pkgPath         packagePath
	name            string
	goFiles         []span.URI
	compiledGoFiles []span.URI
	typesSizes      types.Sizes
	errors          []packages.Error
	deps            []packageID
	missingDeps     map[packagePath]struct{}

	// config is the *packages.Config associated with the loaded package.
	config *packages.Config
}

func (s *snapshot) load(ctx context.Context, scope interface{}) ([]*metadata, error) {
	var query string
	switch scope := scope.(type) {
	case packageID:
		query = string(scope)
	case fileURI:
		query = fmt.Sprintf("file=%s", span.URI(scope).Filename())
	case directoryURI:
		filename := span.URI(scope).Filename()
		query = fmt.Sprintf("%s/...", filename)
		// Simplify the query if it will be run in the requested directory.
		// This ensures compatibility with Go 1.12 that doesn't allow
		// <directory>/... in GOPATH mode.
		if s.view.folder.Filename() == filename {
			query = "./..."
		}
	}

	ctx, done := trace.StartSpan(ctx, "cache.view.load", telemetry.Query.Of(query))
	defer done()

	cfg := s.view.Config(ctx)
	pkgs, err := packages.Load(cfg, query)

	// If the context was canceled, return early.
	// Otherwise, we might be type-checking an incomplete result.
	if err == context.Canceled {
		return nil, err
	}

	log.Print(ctx, "go/packages.Load", tag.Of("query", query), tag.Of("packages", len(pkgs)))
	if len(pkgs) == 0 {
		if err == nil {
			err = errors.Errorf("no packages found for query %s", query)
		}
	}
	if err != nil {
		return nil, err
	}
	return s.updateMetadata(ctx, scope, pkgs, cfg)
}

// shouldLoad reparses a file's package and import declarations to
// determine if they have changed.
func (c *cache) shouldLoad(ctx context.Context, s *snapshot, originalFH, currentFH source.FileHandle) bool {
	if originalFH == nil {
		return true
	}
	// If the file is a mod file, we should always load.
	if originalFH.Identity().Kind == currentFH.Identity().Kind && currentFH.Identity().Kind == source.Mod {
		return true
	}

	// Get the original and current parsed files in order to check package name and imports.
	original, _, _, originalErr := c.ParseGoHandle(originalFH, source.ParseHeader).Parse(ctx)
	current, _, _, currentErr := c.ParseGoHandle(currentFH, source.ParseHeader).Parse(ctx)
	if originalErr != nil || currentErr != nil {
		return (originalErr == nil) != (currentErr == nil)
	}

	// Check if the package's metadata has changed. The cases handled are:
	//
	//    1. A package's name has changed
	//    2. A file's imports have changed
	//
	if original.Name.Name != current.Name.Name {
		return true
	}
	// If the package's imports have increased, definitely re-run `go list`.
	if len(original.Imports) < len(current.Imports) {
		return true
	}
	importSet := make(map[string]struct{})
	for _, importSpec := range original.Imports {
		importSet[importSpec.Path.Value] = struct{}{}
	}
	// If any of the current imports were not in the original imports.
	for _, importSpec := range current.Imports {
		if _, ok := importSet[importSpec.Path.Value]; !ok {
			return true
		}
	}
	return false
}

func (s *snapshot) updateMetadata(ctx context.Context, scope interface{}, pkgs []*packages.Package, cfg *packages.Config) ([]*metadata, error) {
	var results []*metadata
	for _, pkg := range pkgs {
		if _, isDir := scope.(directoryURI); !isDir || s.view.Options().VerboseOutput {
			log.Print(ctx, "go/packages.Load", tag.Of("package", pkg.PkgPath), tag.Of("files", pkg.CompiledGoFiles))
		}
		// Handle golang/go#36292 by ignoring packages with no sources and no errors.
		if len(pkg.GoFiles) == 0 && len(pkg.CompiledGoFiles) == 0 && len(pkg.Errors) == 0 {
			continue
		}
		// Set the metadata for this package.
		if err := s.updateImports(ctx, packagePath(pkg.PkgPath), pkg, cfg, map[packageID]struct{}{}); err != nil {
			return nil, err
		}
		m := s.getMetadata(packageID(pkg.ID))
		if m != nil {
			results = append(results, m)
		}
	}

	// Rebuild the import graph when the metadata is updated.
	s.clearAndRebuildImportGraph()

	if len(results) == 0 {
		return nil, errors.Errorf("no metadata for %s", scope)
	}
	return results, nil
}

func (s *snapshot) updateImports(ctx context.Context, pkgPath packagePath, pkg *packages.Package, cfg *packages.Config, seen map[packageID]struct{}) error {
	id := packageID(pkg.ID)
	if _, ok := seen[id]; ok {
		return errors.Errorf("import cycle detected: %q", id)
	}
	// Recreate the metadata rather than reusing it to avoid locking.
	m := &metadata{
		id:         id,
		pkgPath:    pkgPath,
		name:       pkg.Name,
		typesSizes: pkg.TypesSizes,
		errors:     pkg.Errors,
		config:     cfg,
	}

	for _, filename := range pkg.CompiledGoFiles {
		uri := span.FileURI(filename)
		m.compiledGoFiles = append(m.compiledGoFiles, uri)
		s.addID(uri, m.id)
	}
	for _, filename := range pkg.GoFiles {
		uri := span.FileURI(filename)
		m.goFiles = append(m.goFiles, uri)
		s.addID(uri, m.id)
	}

	seen[id] = struct{}{}
	copied := make(map[packageID]struct{})
	for k, v := range seen {
		copied[k] = v
	}
	for importPath, importPkg := range pkg.Imports {
		importPkgPath := packagePath(importPath)
		importID := packageID(importPkg.ID)

		m.deps = append(m.deps, importID)

		// Don't remember any imports with significant errors.
		if importPkgPath != "unsafe" && len(importPkg.CompiledGoFiles) == 0 {
			if m.missingDeps == nil {
				m.missingDeps = make(map[packagePath]struct{})
			}
			m.missingDeps[importPkgPath] = struct{}{}
			continue
		}
		dep := s.getMetadata(importID)
		if dep == nil {
			if err := s.updateImports(ctx, importPkgPath, importPkg, cfg, copied); err != nil {
				log.Error(ctx, "error in dependency", err)
			}
		}
	}

	// Add the metadata to the cache.
	s.setMetadata(m)

	return nil
}
