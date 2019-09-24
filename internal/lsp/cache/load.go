// Copyright 2019 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package cache

import (
	"context"
	"fmt"
	"go/ast"

	"golang.org/x/tools/go/packages"
	"golang.org/x/tools/internal/lsp/source"
	"golang.org/x/tools/internal/lsp/telemetry"
	"golang.org/x/tools/internal/span"
	"golang.org/x/tools/internal/telemetry/log"
	"golang.org/x/tools/internal/telemetry/tag"
	"golang.org/x/tools/internal/telemetry/trace"
	errors "golang.org/x/xerrors"
)

func (v *view) loadParseTypecheck(ctx context.Context, f *goFile, fh source.FileHandle) ([]source.CheckPackageHandle, error) {
	ctx, done := trace.StartSpan(ctx, "cache.view.loadParseTypeCheck", telemetry.URI.Of(f.URI()))
	defer done()

	meta, err := v.load(ctx, f, fh)
	if err != nil {
		return nil, err
	}
	// If load has explicitly returns nil metadata and no error,
	// it means that we should not re-typecheck the packages.
	if meta == nil {
		return nil, nil
	}
	var (
		cphs    []*checkPackageHandle
		results []source.CheckPackageHandle
	)
	for _, m := range meta {
		imp := &importer{
			view:              v,
			config:            v.Config(ctx),
			seen:              make(map[packageID]struct{}),
			topLevelPackageID: m.id,
		}
		cph, err := imp.checkPackageHandle(ctx, m)
		if err != nil {
			return nil, err
		}
		for _, ph := range cph.files {
			if err := v.cachePerFile(ctx, ph); err != nil {
				return nil, err
			}
		}
		cphs = append(cphs, cph)
		results = append(results, cph)
	}
	// Cache the package type information for the top-level package.
	v.updatePackages(cphs)
	return results, nil
}

func (v *view) cachePerFile(ctx context.Context, ph source.ParseGoHandle) error {
	file, _, _, err := ph.Parse(ctx)
	if err != nil {
		return err
	}
	f, err := v.GetFile(ctx, ph.File().Identity().URI)
	if err != nil {
		return err
	}
	gof, ok := f.(*goFile)
	if !ok {
		return errors.Errorf("%s is not a Go file", ph.File().Identity().URI)
	}
	gof.mu.Lock()
	gof.imports = file.Imports
	gof.mu.Unlock()
	return nil
}

func (view *view) load(ctx context.Context, f *goFile, fh source.FileHandle) ([]*metadata, error) {
	ctx, done := trace.StartSpan(ctx, "cache.view.load", telemetry.URI.Of(f.URI()))
	defer done()

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
func (v *view) checkMetadata(ctx context.Context, f *goFile, fh source.FileHandle) ([]*metadata, error) {
	var shouldRunGopackages bool

	m := v.getMetadata(fh.Identity().URI)
	if len(m) == 0 {
		shouldRunGopackages = true
	}
	// Get file content in case we don't already have it.
	parsed, _, _, err := v.session.cache.ParseGoHandle(fh, source.ParseHeader).Parse(ctx)
	if err != nil {
		return nil, err
	}
	// Check if we need to re-run go/packages before loading the package.
	shouldRunGopackages = shouldRunGopackages || v.shouldRunGopackages(ctx, f, parsed, m)

	// The package metadata is correct as-is, so just return it.
	if !shouldRunGopackages {
		return m, nil
	}

	// Don't bother running go/packages if the context has been canceled.
	if ctx.Err() != nil {
		return nil, ctx.Err()
	}

	ctx, done := trace.StartSpan(ctx, "packages.Load", telemetry.File.Of(f.filename()))
	defer done()

	pkgs, err := packages.Load(v.Config(ctx), fmt.Sprintf("file=%s", f.filename()))
	log.Print(ctx, "go/packages.Load", tag.Of("packages", len(pkgs)))

	if len(pkgs) == 0 {
		if err == nil {
			err = errors.Errorf("go/packages.Load: no packages found for %s", f.filename())
		}
		// Return this error as a diagnostic to the user.
		return nil, err
	}
	m, prevMissingImports, err := v.updateMetadata(ctx, f.URI(), pkgs)
	if err != nil {
		return nil, err
	}
	meta, err := validateMetadata(ctx, m, prevMissingImports)
	if err != nil {
		return nil, err
	}
	return meta, nil
}

// shouldRunGopackages reparses a file's package and import declarations to
// determine if they have changed.
// It assumes that the caller holds the lock on the f.mu lock.
func (v *view) shouldRunGopackages(ctx context.Context, f *goFile, file *ast.File, metadata []*metadata) bool {
	f.mu.Lock()
	defer f.mu.Unlock()

	// Check if the package's name has changed, by checking if this is a filename
	// we already know about, and if so, check if its package name has changed.
	for _, m := range metadata {
		for _, uri := range m.files {
			if span.CompareURI(uri, f.URI()) == 0 {
				if m.name != file.Name.Name {
					return true
				}
			}
		}
	}
	// If the package's imports have changed, re-run `go list`.
	if len(f.imports) != len(file.Imports) {
		return true
	}
	for i, importSpec := range f.imports {
		if importSpec.Path.Value != file.Imports[i].Path.Value {
			return true
		}
	}
	return false
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
