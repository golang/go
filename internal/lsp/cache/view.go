// Copyright 2018 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package cache

import (
	"context"
	"fmt"
	"go/token"
	"sync"

	"golang.org/x/tools/go/packages"
	"golang.org/x/tools/internal/lsp/source"
)

type View struct {
	mu sync.Mutex // protects all mutable state of the view

	Config packages.Config

	files map[source.URI]*File
}

// NewView creates a new View, given a root path and go/packages configuration.
// If config is nil, one is created with the directory set to the rootPath.
func NewView(config *packages.Config) *View {
	return &View{
		Config: *config,
		files:  make(map[source.URI]*File),
	}
}

func (v *View) FileSet() *token.FileSet {
	return v.Config.Fset
}

// SetContent sets the overlay contents for a file. A nil content value will
// remove the file from the active set and revert it to its on-disk contents.
func (v *View) SetContent(ctx context.Context, uri source.URI, content []byte) (source.View, error) {
	v.mu.Lock()
	defer v.mu.Unlock()

	f := v.getFile(uri)
	f.content = content

	// Resetting the contents invalidates the ast, token, and pkg fields.
	f.ast = nil
	f.token = nil
	f.pkg = nil

	// We might need to update the overlay.
	switch {
	case f.active && content == nil:
		// The file was active, so we need to forget its content.
		f.active = false
		if filename, err := f.URI.Filename(); err == nil {
			delete(f.view.Config.Overlay, filename)
		}
		f.content = nil
	case content != nil:
		// This is an active overlay, so we update the map.
		f.active = true
		if filename, err := f.URI.Filename(); err == nil {
			f.view.Config.Overlay[filename] = f.content
		}
	}

	// TODO(rstambler): We should really return a new, updated view.
	return v, nil
}

// GetFile returns a File for the given URI. It will always succeed because it
// adds the file to the managed set if needed.
func (v *View) GetFile(ctx context.Context, uri source.URI) (source.File, error) {
	v.mu.Lock()
	f := v.getFile(uri)
	v.mu.Unlock()
	return f, nil
}

// getFile is the unlocked internal implementation of GetFile.
func (v *View) getFile(uri source.URI) *File {
	f, found := v.files[uri]
	if !found {
		f = &File{
			URI:  uri,
			view: v,
		}
		v.files[uri] = f
	}
	return f
}

func (v *View) parse(uri source.URI) error {
	path, err := uri.Filename()
	if err != nil {
		return err
	}
	pkgs, err := packages.Load(&v.Config, fmt.Sprintf("file=%s", path))
	if len(pkgs) == 0 {
		if err == nil {
			err = fmt.Errorf("no packages found for %s", path)
		}
		return err
	}

	for _, pkg := range pkgs {
		if len(pkg.Syntax) == 0 {
			return fmt.Errorf("no syntax trees for %s", pkg.PkgPath)
		}
		// Add every file in this package to our cache.
		for _, fAST := range pkg.Syntax {
			// TODO: If a file is in multiple packages, which package do we store?
			fToken := v.Config.Fset.File(fAST.Pos())
			fURI := source.ToURI(fToken.Name())
			f := v.getFile(fURI)
			f.token = fToken
			f.ast = fAST
			f.pkg = pkg
		}
	}
	return nil
}
