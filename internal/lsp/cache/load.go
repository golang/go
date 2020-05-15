// Copyright 2019 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package cache

import (
	"context"
	"fmt"
	"go/types"
	"sort"
	"strings"

	"golang.org/x/tools/go/packages"
	"golang.org/x/tools/internal/event"
	"golang.org/x/tools/internal/lsp/debug/tag"
	"golang.org/x/tools/internal/lsp/source"
	"golang.org/x/tools/internal/packagesinternal"
	"golang.org/x/tools/internal/span"
	errors "golang.org/x/xerrors"
)

type metadata struct {
	id              packageID
	pkgPath         packagePath
	name            string
	goFiles         []span.URI
	compiledGoFiles []span.URI
	forTest         packagePath
	typesSizes      types.Sizes
	errors          []packages.Error
	deps            []packageID
	missingDeps     map[packagePath]struct{}
	module          *packages.Module

	// config is the *packages.Config associated with the loaded package.
	config *packages.Config
}

func (s *snapshot) load(ctx context.Context, scopes ...interface{}) error {
	var query []string
	var containsDir bool // for logging
	for _, scope := range scopes {
		switch scope := scope.(type) {
		case packagePath:
			if scope == "command-line-arguments" {
				panic("attempted to load command-line-arguments")
			}
			// The only time we pass package paths is when we're doing a
			// partial workspace load. In those cases, the paths came back from
			// go list and should already be GOPATH-vendorized when appropriate.
			query = append(query, string(scope))
		case fileURI:
			query = append(query, fmt.Sprintf("file=%s", span.URI(scope).Filename()))
		case directoryURI:
			filename := span.URI(scope).Filename()
			q := fmt.Sprintf("%s/...", filename)
			// Simplify the query if it will be run in the requested directory.
			// This ensures compatibility with Go 1.12 that doesn't allow
			// <directory>/... in GOPATH mode.
			if s.view.folder.Filename() == filename {
				q = "./..."
			}
			query = append(query, q)
		case viewLoadScope:
			// If we are outside of GOPATH, a module, or some other known
			// build system, don't load subdirectories.
			if !s.view.hasValidBuildConfiguration {
				query = append(query, "./")
			} else {
				query = append(query, "./...")
			}
		default:
			panic(fmt.Sprintf("unknown scope type %T", scope))
		}
		switch scope.(type) {
		case directoryURI, viewLoadScope:
			containsDir = true
		}
	}
	sort.Strings(query) // for determinism

	ctx, done := event.Start(ctx, "cache.view.load", tag.Query.Of(query))
	defer done()

	cfg := s.Config(ctx)
	pkgs, err := packages.Load(cfg, query...)

	// If the context was canceled, return early. Otherwise, we might be
	// type-checking an incomplete result. Check the context directly,
	// because go/packages adds extra information to the error.
	if ctx.Err() != nil {
		return ctx.Err()
	}
	if err != nil {
		event.Error(ctx, "go/packages.Load", err, tag.Snapshot.Of(s.ID()), tag.Directory.Of(cfg.Dir), tag.Query.Of(query), tag.PackageCount.Of(len(pkgs)))
	} else {
		event.Log(ctx, "go/packages.Load", tag.Snapshot.Of(s.ID()), tag.Directory.Of(cfg.Dir), tag.Query.Of(query), tag.PackageCount.Of(len(pkgs)))
	}
	if len(pkgs) == 0 {
		return err
	}
	for _, pkg := range pkgs {
		if !containsDir || s.view.Options().VerboseOutput {
			event.Log(ctx, "go/packages.Load", tag.Snapshot.Of(s.ID()), tag.PackagePath.Of(pkg.PkgPath), tag.Files.Of(pkg.CompiledGoFiles))
		}
		// Ignore packages with no sources, since we will never be able to
		// correctly invalidate that metadata.
		if len(pkg.GoFiles) == 0 && len(pkg.CompiledGoFiles) == 0 {
			continue
		}
		// Special case for the builtin package, as it has no dependencies.
		if pkg.PkgPath == "builtin" {
			if err := s.view.buildBuiltinPackage(ctx, pkg.GoFiles); err != nil {
				return err
			}
			continue
		}
		// Skip test main packages.
		if isTestMain(pkg, s.view.gocache) {
			continue
		}
		// Set the metadata for this package.
		m, err := s.setMetadata(ctx, packagePath(pkg.PkgPath), pkg, cfg, map[packageID]struct{}{})
		if err != nil {
			return err
		}
		if _, err := s.buildPackageHandle(ctx, m.id, source.ParseFull); err != nil {
			return err
		}
	}
	// Rebuild the import graph when the metadata is updated.
	s.clearAndRebuildImportGraph()

	return nil
}

func (s *snapshot) setMetadata(ctx context.Context, pkgPath packagePath, pkg *packages.Package, cfg *packages.Config, seen map[packageID]struct{}) (*metadata, error) {
	id := packageID(pkg.ID)
	if _, ok := seen[id]; ok {
		return nil, errors.Errorf("import cycle detected: %q", id)
	}
	// Recreate the metadata rather than reusing it to avoid locking.
	m := &metadata{
		id:         id,
		pkgPath:    pkgPath,
		name:       pkg.Name,
		forTest:    packagePath(packagesinternal.GetForTest(pkg)),
		typesSizes: pkg.TypesSizes,
		errors:     pkg.Errors,
		config:     cfg,
		module:     pkg.Module,
	}

	for _, filename := range pkg.CompiledGoFiles {
		uri := span.URIFromPath(filename)
		m.compiledGoFiles = append(m.compiledGoFiles, uri)
		s.addID(uri, m.id)
	}
	for _, filename := range pkg.GoFiles {
		uri := span.URIFromPath(filename)
		m.goFiles = append(m.goFiles, uri)
		s.addID(uri, m.id)
	}

	copied := map[packageID]struct{}{
		id: {},
	}
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
		if s.getMetadata(importID) == nil {
			if _, err := s.setMetadata(ctx, importPkgPath, importPkg, cfg, copied); err != nil {
				event.Error(ctx, "error in dependency", err)
			}
		}
	}

	// Add the metadata to the cache.
	s.mu.Lock()
	defer s.mu.Unlock()

	// TODO: We should make sure not to set duplicate metadata,
	// and instead panic here. This can be done by making sure not to
	// reset metadata information for packages we've already seen.
	if original, ok := s.metadata[m.id]; ok {
		m = original
	} else {
		s.metadata[m.id] = m
	}

	// Set the workspace packages. If any of the package's files belong to the
	// view, then the package is considered to be a workspace package.
	for _, uri := range append(m.compiledGoFiles, m.goFiles...) {
		// If the package's files are in this view, mark it as a workspace package.
		if s.view.contains(uri) {
			// A test variant of a package can only be loaded directly by loading
			// the non-test variant with -test. Track the import path of the non-test variant.
			if m.forTest != "" {
				s.workspacePackages[m.id] = m.forTest
			} else {
				s.workspacePackages[m.id] = pkgPath
			}
			break
		}
	}
	return m, nil
}

func isTestMain(pkg *packages.Package, gocache string) bool {
	// Test mains must have an import path that ends with ".test".
	if !strings.HasSuffix(pkg.PkgPath, ".test") {
		return false
	}
	// Test main packages are always named "main".
	if pkg.Name != "main" {
		return false
	}
	// Test mains always have exactly one GoFile that is in the build cache.
	if len(pkg.GoFiles) > 1 {
		return false
	}
	if !strings.HasPrefix(pkg.GoFiles[0], gocache) {
		return false
	}
	return true
}
