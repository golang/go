// Copyright 2018 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package cache

import (
	"context"
	"go/token"
	"sync"

	"golang.org/x/tools/go/packages"
	"golang.org/x/tools/internal/lsp/source"
)

type View struct {
	// mu protects all mutable state of the view.
	mu sync.Mutex

	// Config is the configuration used for the view's interaction with the
	// go/packages API. It is shared across all views.
	Config packages.Config

	// files caches information for opened files in a view.
	files map[source.URI]*File

	// mcache caches metadata for the packages of the opened files in a view.
	mcache *metadataCache

	analysisCache *source.AnalysisCache
}

type metadataCache struct {
	mu       sync.Mutex
	packages map[string]*metadata
}

type metadata struct {
	id, pkgPath, name string
	files             []string
	parents, children map[string]bool
}

func NewView(config *packages.Config) *View {
	return &View{
		Config: *config,
		files:  make(map[source.URI]*File),
		mcache: &metadataCache{
			packages: make(map[string]*metadata),
		},
	}
}

func (v *View) FileSet() *token.FileSet {
	return v.Config.Fset
}

func (v *View) GetAnalysisCache() *source.AnalysisCache {
	v.analysisCache = source.NewAnalysisCache()
	return v.analysisCache
}

func (v *View) copyView() *View {
	return &View{
		Config: v.Config,
		files:  make(map[source.URI]*File),
		mcache: v.mcache,
	}
}

// SetContent sets the overlay contents for a file. A nil content value will
// remove the file from the active set and revert it to its on-disk contents.
func (v *View) SetContent(ctx context.Context, uri source.URI, content []byte) (source.View, error) {
	v.mu.Lock()
	defer v.mu.Unlock()

	newView := v.copyView()

	for fURI, f := range v.files {
		newView.files[fURI] = &File{
			URI:     fURI,
			view:    newView,
			active:  f.active,
			content: f.content,
			ast:     f.ast,
			token:   f.token,
			pkg:     f.pkg,
			meta:    f.meta,
			imports: f.imports,
		}
	}

	f := newView.getFile(uri)
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

	return newView, nil
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
