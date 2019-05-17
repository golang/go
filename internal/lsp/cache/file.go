// Copyright 2018 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package cache

import (
	"context"
	"go/ast"
	"go/token"
	"path/filepath"
	"strings"

	"golang.org/x/tools/internal/lsp/source"
	"golang.org/x/tools/internal/span"
)

// viewFile extends source.File with helper methods for the view package.
type viewFile interface {
	source.File
	filename() string
	addURI(uri span.URI) int
}

// fileBase holds the common functionality for all files.
// It is intended to be embedded in the file implementations
type fileBase struct {
	uris  []span.URI
	fname string

	view  *view
	fc    *source.FileContent
	token *token.File
}

// goFile holds all the information we know about a go file.
type goFile struct {
	fileBase

	ast     *ast.File
	pkg     *pkg
	meta    *metadata
	imports []*ast.ImportSpec
}

func basename(filename string) string {
	return strings.ToLower(filepath.Base(filename))
}

func (f *fileBase) URI() span.URI {
	return f.uris[0]
}

func (f *fileBase) filename() string {
	return f.fname
}

// View returns the view associated with the file.
func (f *fileBase) View() source.View {
	return f.view
}

// Content returns the contents of the file, reading it from file system if needed.
func (f *fileBase) Content(ctx context.Context) *source.FileContent {
	f.view.mu.Lock()
	defer f.view.mu.Unlock()
	f.read(ctx)
	return f.fc
}

func (f *fileBase) FileSet() *token.FileSet {
	return f.view.Session().Cache().FileSet()
}

func (f *goFile) GetToken(ctx context.Context) *token.File {
	f.view.mu.Lock()
	defer f.view.mu.Unlock()
	if f.token == nil || len(f.view.contentChanges) > 0 {
		if _, err := f.view.parse(ctx, f); err != nil {
			f.View().Session().Logger().Errorf(ctx, "unable to check package for %s: %v", f.URI(), err)
			return nil
		}
	}
	return f.token
}

func (f *goFile) GetAST(ctx context.Context) *ast.File {
	f.view.mu.Lock()
	defer f.view.mu.Unlock()

	if f.ast == nil || len(f.view.contentChanges) > 0 {
		if _, err := f.view.parse(ctx, f); err != nil {
			f.View().Session().Logger().Errorf(ctx, "unable to check package for %s: %v", f.URI(), err)
			return nil
		}
	}
	return f.ast
}

func (f *goFile) GetPackage(ctx context.Context) source.Package {
	f.view.mu.Lock()
	defer f.view.mu.Unlock()

	if f.pkg == nil || len(f.view.contentChanges) > 0 {
		if errs, err := f.view.parse(ctx, f); err != nil {
			f.View().Session().Logger().Errorf(ctx, "unable to check package for %s: %v", f.URI(), err)

			// Create diagnostics for errors if we are able to.
			if len(errs) > 0 {
				return &pkg{errors: errs}
			}
			return nil
		}
	}
	return f.pkg
}

// read is the internal part of Content. It assumes that the caller is
// holding the mutex of the file's view.
func (f *fileBase) read(ctx context.Context) {
	if err := ctx.Err(); err != nil {
		f.fc = &source.FileContent{Error: err}
		return
	}
	if f.fc != nil {
		if len(f.view.contentChanges) == 0 {
			return
		}

		f.view.mcache.mu.Lock()
		err := f.view.applyContentChanges(ctx)
		f.view.mcache.mu.Unlock()

		if err != nil {
			f.fc = &source.FileContent{Error: err}
			return
		}
	}
	// We don't know the content yet, so read it.
	f.fc = f.view.Session().ReadFile(f.URI())
}

// isPopulated returns true if all of the computed fields of the file are set.
func (f *goFile) isPopulated() bool {
	return f.ast != nil && f.token != nil && f.pkg != nil && f.meta != nil && f.imports != nil
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

	seen := make(map[string]struct{}) // visited packages
	results := make(map[*goFile]struct{})
	f.view.reverseDeps(ctx, seen, results, pkg.PkgPath())

	var files []source.GoFile
	for rd := range results {
		if rd == nil {
			continue
		}
		// Don't return any of the active files in this package.
		if rd.pkg != nil && rd.pkg == pkg {
			continue
		}
		files = append(files, rd)
	}
	return files
}

func (v *view) reverseDeps(ctx context.Context, seen map[string]struct{}, results map[*goFile]struct{}, pkgPath string) {
	if _, ok := seen[pkgPath]; ok {
		return
	}
	seen[pkgPath] = struct{}{}
	m, ok := v.mcache.packages[pkgPath]
	if !ok {
		return
	}
	for _, filename := range m.files {
		uri := span.FileURI(filename)
		if f, err := v.getFile(uri); err == nil && v.session.IsOpen(uri) {
			results[f.(*goFile)] = struct{}{}
		}
	}
	for parentPkgPath := range m.parents {
		v.reverseDeps(ctx, seen, results, parentPkgPath)
	}
}
