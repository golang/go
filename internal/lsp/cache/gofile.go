// Copyright 2019 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package cache

import (
	"context"
	"fmt"
	"go/ast"
	"go/token"
	"sync"

	"golang.org/x/tools/internal/lsp/source"
	"golang.org/x/tools/internal/lsp/telemetry"
	"golang.org/x/tools/internal/lsp/telemetry/log"
	"golang.org/x/tools/internal/span"
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

	pkgs map[packageID]*pkg
	meta map[packageID]*metadata
}

type astFile struct {
	uri       span.URI
	file      *ast.File
	err       error // parse errors
	ph        source.ParseGoHandle
	isTrimmed bool
}

func (f *goFile) GetToken(ctx context.Context) (*token.File, error) {
	file, err := f.GetAST(ctx, source.ParseFull)
	if file == nil {
		return nil, err
	}
	tok := f.view.session.cache.fset.File(file.Pos())
	if tok == nil {
		return nil, fmt.Errorf("no token.File for %s", f.URI())
	}
	return tok, nil
}

func (f *goFile) GetAST(ctx context.Context, mode source.ParseMode) (*ast.File, error) {
	f.view.mu.Lock()
	defer f.view.mu.Unlock()
	ctx = telemetry.File.With(ctx, f.URI())

	if f.isDirty(ctx) || f.wrongParseMode(ctx, mode) {
		if _, err := f.view.loadParseTypecheck(ctx, f); err != nil {
			return nil, fmt.Errorf("GetAST: unable to check package for %s: %v", f.URI(), err)
		}
	}
	fh := f.Handle(ctx)
	// Check for a cached AST first, in case getting a trimmed version would actually cause a re-parse.
	for _, m := range []source.ParseMode{
		source.ParseHeader,
		source.ParseExported,
		source.ParseFull,
	} {
		if m < mode {
			continue
		}
		if v, ok := f.view.session.cache.store.Cached(parseKey{
			file: fh.Identity(),
			mode: m,
		}).(*parseGoData); ok {
			return v.ast, v.err
		}
	}
	ph := f.view.session.cache.ParseGoHandle(fh, mode)
	return ph.Parse(ctx)
}

func (f *goFile) GetPackages(ctx context.Context) []source.Package {
	f.view.mu.Lock()
	defer f.view.mu.Unlock()
	ctx = telemetry.File.With(ctx, f.URI())

	if f.isDirty(ctx) || f.wrongParseMode(ctx, source.ParseFull) {
		if errs, err := f.view.loadParseTypecheck(ctx, f); err != nil {
			log.Error(ctx, "unable to check package", err, telemetry.File)

			// Create diagnostics for errors if we are able to.
			if len(errs) > 0 {
				return []source.Package{&pkg{errors: errs}}
			}
			return nil
		}
	}

	f.mu.Lock()
	defer f.mu.Unlock()

	var pkgs []source.Package
	for _, pkg := range f.pkgs {
		pkgs = append(pkgs, pkg)
	}
	return pkgs
}

func (f *goFile) GetPackage(ctx context.Context) source.Package {
	pkgs := f.GetPackages(ctx)
	var result source.Package

	// Pick the "narrowest" package, i.e. the package with the fewest number of files.
	// This solves the problem of test variants,
	// as the test will have more files than the non-test package.
	for _, pkg := range pkgs {
		if result == nil || len(pkg.GetFilenames()) < len(result.GetFilenames()) {
			result = pkg
		}
	}
	return result
}

func (f *goFile) wrongParseMode(ctx context.Context, mode source.ParseMode) bool {
	f.mu.Lock()
	defer f.mu.Unlock()

	fh := f.Handle(ctx)
	for _, pkg := range f.pkgs {
		for _, ph := range pkg.files {
			if fh.Identity() == ph.File().Identity() {
				return ph.Mode() < mode
			}
		}
	}
	return false
}

// isDirty is true if the file needs to be type-checked.
// It assumes that the file's view's mutex is held by the caller.
func (f *goFile) isDirty(ctx context.Context) bool {
	f.mu.Lock()
	defer f.mu.Unlock()

	// If the the file has just been opened,
	// it may be part of more packages than we are aware of.
	//
	// Note: This must be the first case, otherwise we may not reset the value of f.justOpened.
	if f.justOpened {
		f.meta = make(map[packageID]*metadata)
		f.pkgs = make(map[packageID]*pkg)
		f.justOpened = false
		return true
	}
	if len(f.meta) == 0 || len(f.pkgs) == 0 {
		return true
	}
	if len(f.missingImports) > 0 {
		return true
	}
	fh := f.Handle(ctx)
	for _, pkg := range f.pkgs {
		for _, file := range pkg.files {
			// There is a type-checked package for the current file handle.
			if file.File().Identity() == fh.Identity() {
				return false
			}
		}
	}
	return true
}

func (f *goFile) GetActiveReverseDeps(ctx context.Context) []source.GoFile {
	pkg := f.GetPackage(ctx)
	if pkg == nil {
		return nil
	}

	f.view.mu.Lock()
	defer f.view.mu.Unlock()

	f.view.mcache.mu.Lock()
	defer f.view.mcache.mu.Unlock()

	id := packageID(pkg.ID())

	seen := make(map[packageID]struct{}) // visited packages
	results := make(map[*goFile]struct{})
	f.view.reverseDeps(ctx, seen, results, id)

	var files []source.GoFile
	for rd := range results {
		if rd == nil {
			continue
		}
		// Don't return any of the active files in this package.
		if _, ok := rd.pkgs[id]; ok {
			continue
		}
		files = append(files, rd)
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
	for _, filename := range m.files {
		uri := span.FileURI(filename)
		if f, err := v.getFile(ctx, uri); err == nil && v.session.IsOpen(uri) {
			results[f.(*goFile)] = struct{}{}
		}
	}
	for parentID := range m.parents {
		v.reverseDeps(ctx, seen, results, parentID)
	}
}
