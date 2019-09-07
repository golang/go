// Copyright 2019 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package cache

import (
	"context"
	"go/ast"
	"sync"

	"golang.org/x/tools/internal/lsp/source"
	"golang.org/x/tools/internal/lsp/telemetry"
	"golang.org/x/tools/internal/span"
	"golang.org/x/tools/internal/telemetry/log"
	errors "golang.org/x/xerrors"
)

// goFile holds all of the information we know about a Go file.
type goFile struct {
	fileBase

	// mu protects all mutable state of the Go file,
	// which can be modified during type-checking.
	mu sync.Mutex

	// missingImports is the set of unresolved imports for this package.
	// It contains any packages with `go list` errors.
	missingImports map[packagePath]struct{}

	// justOpened indicates that the file has just been opened.
	// We re-run go/packages.Load on just opened files to make sure
	// that we know about all of their packages.
	justOpened bool

	imports []*ast.ImportSpec

	pkgs map[packageID]source.CheckPackageHandle
	meta map[packageID]*metadata
}

// metadata assumes that the caller holds the f.mu lock.
func (f *goFile) metadata() []*metadata {
	result := make([]*metadata, 0, len(f.meta))
	for _, m := range f.meta {
		result = append(result, m)
	}
	return result
}

func (cache *cache) cachedAST(fh source.FileHandle, mode source.ParseMode) (*ast.File, error) {
	for _, m := range []source.ParseMode{
		source.ParseHeader,
		source.ParseExported,
		source.ParseFull,
	} {
		if m < mode {
			continue
		}
		if v, ok := cache.store.Cached(parseKey{
			file: fh.Identity(),
			mode: m,
		}).(*parseGoData); ok {
			return v.ast, v.err
		}
	}
	return nil, nil
}

func (f *goFile) GetPackages(ctx context.Context) ([]source.Package, error) {
	cphs, err := f.GetCheckPackageHandles(ctx)
	if err != nil {
		return nil, err
	}
	var pkgs []source.Package
	for _, cph := range cphs {
		pkg, err := cph.Check(ctx)
		if err != nil {
			log.Error(ctx, "failed to check package", err)
		}
		pkgs = append(pkgs, pkg)
	}
	if len(pkgs) == 0 {
		return nil, errors.Errorf("no packages for %s", f.URI())
	}
	return pkgs, nil
}

func (f *goFile) GetPackage(ctx context.Context) (source.Package, error) {
	cph, err := f.GetCheckPackageHandle(ctx)
	if err != nil {
		return nil, err
	}
	return cph.Check(ctx)
}

func (f *goFile) GetCheckPackageHandles(ctx context.Context) ([]source.CheckPackageHandle, error) {
	ctx = telemetry.File.With(ctx, f.URI())
	fh := f.Handle(ctx)

	if f.isDirty(ctx, fh) || f.wrongParseMode(ctx, fh, source.ParseFull) {
		if err := f.view.loadParseTypecheck(ctx, f, fh); err != nil {
			return nil, err
		}
	}

	f.mu.Lock()
	defer f.mu.Unlock()

	var cphs []source.CheckPackageHandle
	for _, cph := range f.pkgs {
		cphs = append(cphs, cph)
	}
	if len(cphs) == 0 {
		return nil, errors.Errorf("no CheckPackageHandles for %s", f.URI())
	}
	return cphs, nil
}

func (f *goFile) GetCheckPackageHandle(ctx context.Context) (source.CheckPackageHandle, error) {
	cphs, err := f.GetCheckPackageHandles(ctx)
	if err != nil {
		return nil, err
	}
	return bestCheckPackageHandle(f.URI(), cphs)
}

func (f *goFile) GetCachedPackage(ctx context.Context) (source.Package, error) {
	f.mu.Lock()
	var cphs []source.CheckPackageHandle
	for _, cph := range f.pkgs {
		cphs = append(cphs, cph)
	}
	f.mu.Unlock()

	if len(cphs) == 0 {
		return nil, errors.Errorf("no CheckPackageHandles for %s", f.URI())
	}

	cph, err := bestCheckPackageHandle(f.URI(), cphs)
	if err != nil {
		return nil, err
	}
	return cph.Cached(ctx)
}

func (f *goFile) GetCachedPackages(ctx context.Context) ([]source.Package, error) {
	f.mu.Lock()
	defer f.mu.Unlock()

	var pkgs []source.Package
	for _, cph := range f.pkgs {
		pkg, err := cph.Cached(ctx)
		if err != nil {
			return nil, err
		}
		pkgs = append(pkgs, pkg)
	}
	if len(pkgs) == 0 {
		return nil, errors.Errorf("no CheckPackageHandles for %s", f.URI())
	}
	return pkgs, nil
}

// bestCheckPackageHandle picks the "narrowest" package for a given file.
//
// By "narrowest" package, we mean the package with the fewest number of files
// that includes the given file. This solves the problem of test variants,
// as the test will have more files than the non-test package.
func bestCheckPackageHandle(uri span.URI, cphs []source.CheckPackageHandle) (source.CheckPackageHandle, error) {
	var result source.CheckPackageHandle
	for _, cph := range cphs {
		if result == nil || len(cph.Files()) < len(result.Files()) {
			result = cph
		}
	}
	if result == nil {
		return nil, errors.Errorf("no CheckPackageHandle for %s", uri)
	}
	return result, nil
}

func (f *goFile) wrongParseMode(ctx context.Context, fh source.FileHandle, mode source.ParseMode) bool {
	f.mu.Lock()
	defer f.mu.Unlock()

	for _, cph := range f.pkgs {
		for _, ph := range cph.Files() {
			if fh.Identity() == ph.File().Identity() {
				return ph.Mode() < mode
			}
		}
	}
	return true
}

func (f *goFile) Builtin() (*ast.File, bool) {
	builtinPkg := f.View().BuiltinPackage()
	for filename, file := range builtinPkg.Files {
		if filename == f.URI().Filename() {
			return file, true
		}
	}
	return nil, false
}

// isDirty is true if the file needs to be type-checked.
// It assumes that the file's view's mutex is held by the caller.
func (f *goFile) isDirty(ctx context.Context, fh source.FileHandle) bool {
	f.mu.Lock()
	defer f.mu.Unlock()

	// If the the file has just been opened,
	// it may be part of more packages than we are aware of.
	//
	// Note: This must be the first case, otherwise we may not reset the value of f.justOpened.
	if f.justOpened {
		f.meta = make(map[packageID]*metadata)
		f.pkgs = make(map[packageID]source.CheckPackageHandle)
		f.justOpened = false
		return true
	}
	if len(f.meta) == 0 || len(f.pkgs) == 0 {
		return true
	}
	if len(f.missingImports) > 0 {
		return true
	}
	for _, cph := range f.pkgs {
		for _, file := range cph.Files() {
			// There is a type-checked package for the current file handle.
			if file.File().Identity() == fh.Identity() {
				return false
			}
		}
	}
	return true
}

func (f *goFile) GetActiveReverseDeps(ctx context.Context) (files []source.GoFile) {
	seen := make(map[packageID]struct{}) // visited packages
	results := make(map[*goFile]struct{})

	f.view.mu.Lock()
	defer f.view.mu.Unlock()

	f.view.mcache.mu.Lock()
	defer f.view.mcache.mu.Unlock()

	for _, m := range f.metadata() {
		f.view.reverseDeps(ctx, seen, results, m.id)
		for f := range results {
			if f == nil {
				continue
			}
			// Don't return any of the active files in this package.
			f.mu.Lock()
			_, ok := f.meta[m.id]
			f.mu.Unlock()
			if ok {
				continue
			}

			files = append(files, f)
		}
	}
	return files
}

func (v *view) reverseDeps(ctx context.Context, seen map[packageID]struct{}, results map[*goFile]struct{}, id packageID) {
	if _, ok := seen[id]; ok {
		return
	}
	seen[id] = struct{}{}
	m, ok := v.mcache.packages[id]
	if !ok {
		return
	}
	for _, uri := range m.files {
		// Call unlocked version of getFile since we hold the lock on the view.
		if f, err := v.getFile(ctx, uri); err == nil && v.session.IsOpen(uri) {
			results[f.(*goFile)] = struct{}{}
		}
	}
	for parentID := range m.parents {
		v.reverseDeps(ctx, seen, results, parentID)
	}
}
