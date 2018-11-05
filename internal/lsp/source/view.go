// Copyright 2018 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package source

import (
	"fmt"
	"go/token"
	"sync"

	"golang.org/x/tools/go/packages"
)

type View struct {
	mu sync.Mutex // protects all mutable state of the view

	Config *packages.Config

	files map[URI]*File
}

func NewView() *View {
	return &View{
		Config: &packages.Config{
			Mode:    packages.LoadSyntax,
			Fset:    token.NewFileSet(),
			Tests:   true,
			Overlay: make(map[string][]byte),
		},
		files: make(map[URI]*File),
	}
}

// GetFile returns a File for the given uri.
// It will always succeed, adding the file to the managed set if needed.
func (v *View) GetFile(uri URI) *File {
	v.mu.Lock()
	f := v.getFile(uri)
	v.mu.Unlock()
	return f
}

// getFile is the unlocked internal implementation of GetFile.
func (v *View) getFile(uri URI) *File {
	f, found := v.files[uri]
	if !found {
		f = &File{
			URI:  uri,
			view: v,
		}
		v.files[f.URI] = f
	}
	return f
}

func (v *View) parse(uri URI) error {
	path, err := uri.Filename()
	if err != nil {
		return err
	}
	pkgs, err := packages.Load(v.Config, fmt.Sprintf("file=%s", path))
	if len(pkgs) == 0 {
		if err == nil {
			err = fmt.Errorf("no packages found for %s", path)
		}
		return err
	}
	for _, pkg := range pkgs {
		// add everything we find to the files cache
		for _, fAST := range pkg.Syntax {
			// if a file was in multiple packages, which token/ast/pkg do we store
			fToken := v.Config.Fset.File(fAST.Pos())
			fURI := ToURI(fToken.Name())
			f := v.getFile(fURI)
			f.token = fToken
			f.ast = fAST
			f.pkg = pkg
		}
	}
	return nil
}
