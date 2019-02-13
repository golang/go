// Copyright 2018 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package cache

import (
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
}

// GetContent returns the contents of the file, reading it from file system if needed.
func (f *File) GetContent() []byte {
	f.view.mu.Lock()
	defer f.view.mu.Unlock()
	f.read()
	return f.content
}

func (f *File) GetFileSet() *token.FileSet {
	return f.view.Config.Fset
}

func (f *File) GetToken() *token.File {
	f.view.mu.Lock()
	defer f.view.mu.Unlock()
	if f.token == nil {
		if err := f.view.parse(f.URI); err != nil {
			return nil
		}
	}
	return f.token
}

func (f *File) GetAST() *ast.File {
	f.view.mu.Lock()
	defer f.view.mu.Unlock()
	if f.ast == nil {
		if err := f.view.parse(f.URI); err != nil {
			return nil
		}
	}
	return f.ast
}

func (f *File) GetPackage() *packages.Package {
	f.view.mu.Lock()
	defer f.view.mu.Unlock()
	if f.pkg == nil {
		if err := f.view.parse(f.URI); err != nil {
			return nil
		}
	}
	return f.pkg
}

// read is the internal part of Read that presumes the lock is already held
func (f *File) read() {
	if f.content != nil {
		return
	}
	// we don't know the content yet, so read it
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
