// Copyright 2018 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package cache

import (
	"context"
	"go/ast"
	"go/token"
	"io/ioutil"

	"golang.org/x/tools/go/packages"
	"golang.org/x/tools/internal/lsp/source"
)

// File holds all the information we know about a file.
type File struct {
	URI     source.URI
	view    *View
	active  bool
	content []byte
	ast     *ast.File
	token   *token.File
	pkg     *packages.Package
	meta    *metadata
	imports []*ast.ImportSpec
}

// GetContent returns the contents of the file, reading it from file system if needed.
func (f *File) GetContent(ctx context.Context) []byte {
	f.view.mu.Lock()
	defer f.view.mu.Unlock()

	if ctx.Err() == nil {
		f.read(ctx)
	}

	return f.content
}

func (f *File) GetFileSet(ctx context.Context) *token.FileSet {
	return f.view.Config.Fset
}

func (f *File) GetToken(ctx context.Context) *token.File {
	f.view.mu.Lock()
	defer f.view.mu.Unlock()

	if f.token == nil || len(f.view.contentChanges) > 0 {
		if err := f.view.parse(ctx, f.URI); err != nil {
			return nil
		}
	}
	return f.token
}

func (f *File) GetAST(ctx context.Context) *ast.File {
	f.view.mu.Lock()
	defer f.view.mu.Unlock()

	if f.ast == nil || len(f.view.contentChanges) > 0 {
		if err := f.view.parse(ctx, f.URI); err != nil {
			return nil
		}
	}
	return f.ast
}

func (f *File) GetPackage(ctx context.Context) *packages.Package {
	f.view.mu.Lock()
	defer f.view.mu.Unlock()

	if f.pkg == nil || len(f.view.contentChanges) > 0 {
		if err := f.view.parse(ctx, f.URI); err != nil {
			return nil
		}
	}
	return f.pkg
}

// read is the internal part of GetContent. It assumes that the caller is
// holding the mutex of the file's view.
func (f *File) read(ctx context.Context) {
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
	// We don't know the content yet, so read it.
	filename, err := f.URI.Filename()
	if err != nil {
		return
	}
	content, err := ioutil.ReadFile(filename)
	if err != nil {
		return
	}
	f.content = content
}

// isPopulated returns true if all of the computed fields of the file are set.
func (f *File) isPopulated() bool {
	return f.ast != nil && f.token != nil && f.pkg != nil && f.meta != nil && f.imports != nil
}
