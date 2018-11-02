// Copyright 2018 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package source

import (
	"fmt"
	"go/token"
	"sync"

	"golang.org/x/tools/go/packages"
	"golang.org/x/tools/internal/lsp/protocol"
)

type View struct {
	mu sync.Mutex // protects all mutable state of the view

	Config *packages.Config

	files map[protocol.DocumentURI]*File
}

func NewView() *View {
	return &View{
		Config: &packages.Config{
			Mode:  packages.LoadSyntax,
			Fset:  token.NewFileSet(),
			Tests: true,
		},
		files: make(map[protocol.DocumentURI]*File),
	}
}

// GetFile returns a File for the given uri.
// It will always succeed, adding the file to the managed set if needed.
func (v *View) GetFile(uri protocol.DocumentURI) *File {
	v.mu.Lock()
	f, found := v.files[uri]
	if !found {
		f := &File{URI: uri}
		v.files[f.URI] = f
	}
	v.mu.Unlock()
	return f
}

// TypeCheck type-checks the package for the given package path.
func (v *View) TypeCheck(uri protocol.DocumentURI) (*packages.Package, error) {
	v.mu.Lock()
	defer v.mu.Unlock()
	path, err := FromURI(uri)
	if err != nil {
		return nil, err
	}
	pkgs, err := packages.Load(v.Config, fmt.Sprintf("file=%s", path))
	if len(pkgs) == 0 {
		return nil, err
	}
	pkg := pkgs[0]
	return pkg, nil
}
