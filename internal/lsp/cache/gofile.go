// Copyright 2019 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package cache

import (
	"context"
	"go/ast"
	"go/token"
	"sync"

	"golang.org/x/tools/internal/lsp/source"
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

	ast  *astFile
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

func (f *goFile) GetToken(ctx context.Context) *token.File {
	f.view.mu.Lock()
	defer f.view.mu.Unlock()

	if f.isDirty() || f.astIsTrimmed() {
		if _, err := f.view.loadParseTypecheck(ctx, f); err != nil {
			f.View().Session().Logger().Errorf(ctx, "unable to check package for %s: %v", f.URI(), err)
			return nil
		}
	}
	f.mu.Lock()
	defer f.mu.Unlock()

	if unexpectedAST(ctx, f) {
		return nil
	}
	return f.token
}

func (f *goFile) GetAnyAST(ctx context.Context) *ast.File {
	f.view.mu.Lock()
	defer f.view.mu.Unlock()

	if f.isDirty() {
		if _, err := f.view.loadParseTypecheck(ctx, f); err != nil {
			f.View().Session().Logger().Errorf(ctx, "unable to check package for %s: %v", f.URI(), err)
			return nil
		}
	}

	f.mu.Lock()
	defer f.mu.Unlock()

	if f.ast == nil {
		return nil
	}
	return f.ast.file
}

func (f *goFile) GetAST(ctx context.Context) *ast.File {
	f.view.mu.Lock()
	defer f.view.mu.Unlock()

	if f.isDirty() || f.astIsTrimmed() {
		if _, err := f.view.loadParseTypecheck(ctx, f); err != nil {
			f.View().Session().Logger().Errorf(ctx, "unable to check package for %s: %v", f.URI(), err)
			return nil
		}
	}
	f.mu.Lock()
	defer f.mu.Unlock()

	if unexpectedAST(ctx, f) {
		return nil
	}
	return f.ast.file
}

func (f *goFile) GetPackages(ctx context.Context) []source.Package {
	f.view.mu.Lock()
	defer f.view.mu.Unlock()

	if f.isDirty() || f.astIsTrimmed() {
		if errs, err := f.view.loadParseTypecheck(ctx, f); err != nil {
			f.View().Session().Logger().Errorf(ctx, "unable to check package for %s: %v", f.URI(), err)

			// Create diagnostics for errors if we are able to.
			if len(errs) > 0 {
				return []source.Package{&pkg{errors: errs}}
			}
			return nil
		}
	}

	f.mu.Lock()
	defer f.mu.Unlock()

	if unexpectedAST(ctx, f) {
		return nil
	}
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

func unexpectedAST(ctx context.Context, f *goFile) bool {
	// If the AST comes back nil, something has gone wrong.
	if f.ast == nil {
		f.View().Session().Logger().Errorf(ctx, "expected full AST for %s, returned nil", f.URI())
		return true
	}
	// If the AST comes back trimmed, something has gone wrong.
	if f.ast.isTrimmed {
		f.View().Session().Logger().Errorf(ctx, "expected full AST for %s, returned trimmed", f.URI())
		return true
	}
	return false
}

// isDirty is true if the file needs to be type-checked.
// It assumes that the file's view's mutex is held by the caller.
func (f *goFile) isDirty() bool {
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
	return f.token == nil || f.ast == nil
}

func (f *goFile) astIsTrimmed() bool {
	f.mu.Lock()
	defer f.mu.Unlock()

	return f.ast != nil && f.ast.isTrimmed
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
