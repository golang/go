// Copyright 2018 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package cache

import (
	"fmt"
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

// Read returns the contents of the file, reading it from file system if needed.
func (f *File) Read() ([]byte, error) {
	f.view.mu.Lock()
	defer f.view.mu.Unlock()
	return f.read()
}

func (f *File) GetFileSet() (*token.FileSet, error) {
	if f.view.Config.Fset == nil {
		return nil, fmt.Errorf("no fileset for file view config")
	}
	return f.view.Config.Fset, nil
}

func (f *File) GetToken() (*token.File, error) {
	f.view.mu.Lock()
	defer f.view.mu.Unlock()
	if f.token == nil {
		if err := f.view.parse(f.URI); err != nil {
			return nil, err
		}
		if f.token == nil {
			return nil, fmt.Errorf("failed to find or parse %v", f.URI)
		}
	}
	return f.token, nil
}

func (f *File) GetAST() (*ast.File, error) {
	f.view.mu.Lock()
	defer f.view.mu.Unlock()
	if f.ast == nil {
		if err := f.view.parse(f.URI); err != nil {
			return nil, err
		}
	}
	return f.ast, nil
}

func (f *File) GetPackage() (*packages.Package, error) {
	f.view.mu.Lock()
	defer f.view.mu.Unlock()
	if f.pkg == nil {
		if err := f.view.parse(f.URI); err != nil {
			return nil, err
		}
	}
	return f.pkg, nil
}

// read is the internal part of Read that presumes the lock is already held
func (f *File) read() ([]byte, error) {
	if f.content != nil {
		return f.content, nil
	}
	// we don't know the content yet, so read it
	filename, err := f.URI.Filename()
	if err != nil {
		return nil, err
	}
	content, err := ioutil.ReadFile(filename)
	if err != nil {
		return nil, err
	}
	f.content = content
	return f.content, nil
}
