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
	id          packageID
	pkgPath     packagePath
	name        string
	files       []span.URI
	typesSizes  types.Sizes
	errors      []packages.Error
	deps        []packageID
	missingDeps map[packagePath]struct{}

	// config is the *packages.Config associated with the loaded package.
	config *packages.Config
}

func (s *snapshot) load(ctx context.Context, uri span.URI) ([]*metadata, error) {
	ctx, done := trace.StartSpan(ctx, "cache.view.load", telemetry.URI.Of(uri))
	defer done()

	cfg := s.view.Config(ctx)
	pkgs, err := packages.Load(cfg, fmt.Sprintf("file=%s", uri.Filename()))

	// If the context was canceled, return early.
	// Otherwise, we might be type-checking an incomplete result.
	if err == context.Canceled {
		return nil, errors.Errorf("no metadata for %s: %v", uri.Filename(), err)
	}

	log.Print(ctx, "go/packages.Load", tag.Of("packages", len(pkgs)))
	if len(pkgs) == 0 {
		if err == nil {
			err = errors.Errorf("go/packages.Load: no packages found for %s", uri)
		}
		// Return this error as a diagnostic to the user.
		return nil, err
	}
	m, prevMissingImports, err := s.updateMetadata(ctx, uri, pkgs, cfg)
	if err != nil {
		return nil, err
	}
	meta, err := validateMetadata(ctx, m, prevMissingImports)
	if err != nil {
		return nil, err
	}
	return meta, nil
}

func validateMetadata(ctx context.Context, metadata []*metadata, prevMissingImports map[packageID]map[packagePath]struct{}) ([]*metadata, error) {
	// If we saw incorrect metadata for this package previously, don't both rechecking it.
	for _, m := range metadata {
		if len(m.missingDeps) > 0 {
			prev, ok := prevMissingImports[m.id]
			// There are missing imports that we previously hadn't seen before.
			if !ok {
				return metadata, nil
			}
			// The set of missing imports has changed.
			if !sameSet(prev, m.missingDeps) {
				return metadata, nil
			}
		} else {
			// There are no missing imports.
			return metadata, nil
		}
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

// shouldLoad reparses a file's package and import declarations to
// determine if they have changed.
func (c *cache) shouldLoad(ctx context.Context, s *snapshot, originalFH, currentFH source.FileHandle) bool {
	if originalFH == nil {
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
	// If the package's imports have changed, re-run `go list`.
	if len(original.Imports) != len(current.Imports) {
		return true
	}
	for i, importSpec := range original.Imports {
		// TODO: Handle the case where the imports have just been re-ordered.
		if importSpec.Path.Value != current.Imports[i].Path.Value {
			return true
		}
	}
	return false
}

func (s *snapshot) updateMetadata(ctx context.Context, uri span.URI, pkgs []*packages.Package, cfg *packages.Config) ([]*metadata, map[packageID]map[packagePath]struct{}, error) {
	// Clear metadata since we are re-running go/packages.
	prevMissingImports := make(map[packageID]map[packagePath]struct{})
	m := s.getMetadataForURI(uri)

	for _, m := range m {
		if len(m.missingDeps) > 0 {
			prevMissingImports[m.id] = m.missingDeps
		}
	}

	var results []*metadata
	for _, pkg := range pkgs {
		log.Print(ctx, "go/packages.Load", tag.Of("package", pkg.PkgPath), tag.Of("files", pkg.CompiledGoFiles))

		// Set the metadata for this package.
		if err := s.updateImports(ctx, packagePath(pkg.PkgPath), pkg, cfg); err != nil {
			return nil, nil, err
		}
		m := s.getMetadata(packageID(pkg.ID))
		if m != nil {
			results = append(results, m)
		}
	}

	// Rebuild the import graph when the metadata is updated.
	s.clearAndRebuildImportGraph()

	if len(results) == 0 {
		return nil, nil, errors.Errorf("no metadata for %s", uri)
	}
	return results, prevMissingImports, nil
}

func (s *snapshot) updateImports(ctx context.Context, pkgPath packagePath, pkg *packages.Package, cfg *packages.Config) error {
	// Recreate the metadata rather than reusing it to avoid locking.
	m := &metadata{
		id:         packageID(pkg.ID),
		pkgPath:    pkgPath,
		name:       pkg.Name,
		typesSizes: pkg.TypesSizes,
		errors:     pkg.Errors,
		config:     cfg,
	}
	for _, filename := range pkg.CompiledGoFiles {
		uri := span.FileURI(filename)
		m.files = append(m.files, uri)

		s.addID(uri, m.id)
	}

	// Add the metadata to the cache.
	s.setMetadata(m)

	for importPath, importPkg := range pkg.Imports {
		importPkgPath := packagePath(importPath)
		importID := packageID(importPkg.ID)

		if importPkgPath == pkgPath {
			return errors.Errorf("cycle detected in %s", importPath)
		}
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
			if err := s.updateImports(ctx, importPkgPath, importPkg, cfg); err != nil {
				log.Error(ctx, "error in dependency", err)
			}
		}
	}
	return nil
}
