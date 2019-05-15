// Copyright 2018 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package cache

import (
	"context"
	"go/ast"
	"go/token"
	"io/ioutil"
	"path/filepath"
	"strings"

	"golang.org/x/tools/internal/lsp/source"
	"golang.org/x/tools/internal/span"
)

// file holds all the information we know about a file.
type file struct {
	uris     []span.URI
	filename string
	basename string

	view    *view
	active  bool
	content []byte
	ast     *ast.File
	token   *token.File
	pkg     *pkg
	meta    *metadata
	imports []*ast.ImportSpec
}

func basename(filename string) string {
	return strings.ToLower(filepath.Base(filename))
}

func (f *file) URI() span.URI {
	return f.uris[0]
}

// View returns the view associated with the file.
func (f *file) View() source.View {
	return f.view
}

// GetContent returns the contents of the file, reading it from file system if needed.
func (f *file) GetContent(ctx context.Context) []byte {
	f.view.mu.Lock()
	defer f.view.mu.Unlock()

	if ctx.Err() == nil {
		f.read(ctx)
	}

	return f.content
}

func (f *file) GetFileSet(ctx context.Context) *token.FileSet {
	return f.view.config.Fset
}

func (f *file) GetToken(ctx context.Context) *token.File {
	f.view.mu.Lock()
	defer f.view.mu.Unlock()

	if f.token == nil || len(f.view.contentChanges) > 0 {
		if _, err := f.view.parse(ctx, f); err != nil {
			return nil
		}
	}
	return f.token
}

func (f *file) GetAST(ctx context.Context) *ast.File {
	f.view.mu.Lock()
	defer f.view.mu.Unlock()

	if f.ast == nil || len(f.view.contentChanges) > 0 {
		if _, err := f.view.parse(ctx, f); err != nil {
			return nil
		}
	}
	return f.ast
}

func (f *file) GetPackage(ctx context.Context) source.Package {
	f.view.mu.Lock()
	defer f.view.mu.Unlock()

	if f.pkg == nil || len(f.view.contentChanges) > 0 {
		if errs, err := f.view.parse(ctx, f); err != nil {
			// Create diagnostics for errors if we are able to.
			if len(errs) > 0 {
				return &pkg{errors: errs}
			}
			return nil
		}
	}
	return f.pkg
}

// read is the internal part of GetContent. It assumes that the caller is
// holding the mutex of the file's view.
func (f *file) read(ctx context.Context) {
	if f.content != nil {
		if len(f.view.contentChanges) == 0 {
			return
		}

		f.view.mcache.mu.Lock()
		err := f.view.applyContentChanges(ctx)
		f.view.mcache.mu.Unlock()

		if err == nil {
			return
		}
	}
	// We might have the content saved in an overlay.
	if content, ok := f.view.config.Overlay[f.filename]; ok {
		f.content = content
		return
	}
	// We don't know the content yet, so read it.
	content, err := ioutil.ReadFile(f.filename)
	if err != nil {
		f.view.Logger().Errorf(ctx, "unable to read file %s: %v", f.filename, err)
		return
	}
	f.content = content
}

// isPopulated returns true if all of the computed fields of the file are set.
func (f *file) isPopulated() bool {
	return f.ast != nil && f.token != nil && f.pkg != nil && f.meta != nil && f.imports != nil
}

func (f *file) GetActiveReverseDeps(ctx context.Context) []source.File {
	pkg := f.GetPackage(ctx)
	if pkg == nil {
		return nil
	}

	f.view.mu.Lock()
	defer f.view.mu.Unlock()

	f.view.mcache.mu.Lock()
	defer f.view.mcache.mu.Unlock()

	seen := make(map[string]struct{}) // visited packages
	results := make(map[*file]struct{})
	f.view.reverseDeps(ctx, seen, results, pkg.PkgPath())

	files := make([]source.File, 0, len(results))
	for rd := range results {
		if rd == nil {
			continue
		}
		// Don't return any of the active file's in this package.
		if rd.pkg != nil && rd.pkg == pkg {
			continue
		}
		files = append(files, rd)
	}
	return files
}

func (v *view) reverseDeps(ctx context.Context, seen map[string]struct{}, results map[*file]struct{}, pkgPath string) {
	if _, ok := seen[pkgPath]; ok {
		return
	}
	seen[pkgPath] = struct{}{}
	m, ok := v.mcache.packages[pkgPath]
	if !ok {
		return
	}
	for _, filename := range m.files {
		if f, err := v.getFile(span.FileURI(filename)); err == nil && f.active {
			results[f] = struct{}{}
		}
	}
	for parentPkgPath := range m.parents {
		v.reverseDeps(ctx, seen, results, parentPkgPath)
	}
}
